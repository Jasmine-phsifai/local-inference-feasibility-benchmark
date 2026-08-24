"""Compare private ASR outputs when no trusted transcript exists."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import math
import re
import secrets
import struct
import unicodedata
import uuid
import wave
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from .asr_agreement_public_protocol import (
    CHARACTER_COUNT_BUCKET_UPPER_BOUNDS as _CHARACTER_COUNT_BUCKET_UPPER_BOUNDS,
    FRACTION_BUCKET_LOWER_BOUNDS as _FRACTION_BUCKET_LOWER_BOUNDS,
    INTERPRETATION as _ASR_INTERPRETATION,
    MINIMUM_EXACT_AGGREGATE_DENOMINATOR as _MINIMUM_EXACT_AGGREGATE_DENOMINATOR,
    PUBLIC_PRIVACY as _ASR_PRIVACY,
    public_event_sha256 as _protocol_public_event_sha256,
    validate_public_event as _validate_protocol_public_event,
)
from .event_journal import (
    append_event_once,
    locked_file_bytes,
    locked_journal_bytes,
    read_journal_bytes,
)
from .fingerprint import fingerprint_json
from .journal_integrity import (
    SUSTAINED_START_EVENT,
    SUSTAINED_TERMINAL_EVENTS,
    SustainedJournalSnapshot,
    capture_sustained_journal_snapshot,
)
from .load_sustained_workload import load_sustained_workload_from_bytes
from .project_paths import SUSTAINED_EVENTS_PATH, SUSTAINED_REGISTRY_PATH
from .private_records_commitment import (
    PRIVATE_RECORDS_COMMITMENT_SCHEME,
    PRIVATE_RECORDS_PROVENANCE_FIELDS as _PROVENANCE_FIELDS,
    verify_private_records_bytes_commitment,
)
from .validate_public_summary import validate_public_summary


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_CANDIDATES = 8
_MAX_WORKLOAD_ITEMS = 4_096
_MAX_WORKLOAD_MANIFEST_BYTES = 1_048_576
_MAX_RECORDS_PER_CANDIDATE = 4_096
_MAX_RECORD_FILE_BYTES = 16 * 1_048_576
_MAX_PROVENANCE_BYTES = 65_536
_MAX_REGISTRY_BYTES = 2 * 1_048_576
_MAX_PREDICTION_CHARACTERS = 200_000
_MAX_TOTAL_PREDICTION_CHARACTERS = 2_000_000
# Three 20-minute, ten-chunk lecture sources require about 100 million cells.
# Keep the exact dynamic program bounded while admitting that measured cohort.
_MAX_TOTAL_EDIT_CELLS = 150_000_000
_MAX_TOTAL_PCM_FRAME_COUNT = 24 * 60 * 60 * 16_000
_MAX_PCM_WAV_BYTES = 7200 * 16_000 * 2 + 1_048_576
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})
_APPEND_TICKET_KEY = secrets.token_bytes(32)
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")
_MIXED_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z]+(?:-[a-z]+)*|[0-9]+(?:\.[0-9]+)?",
    re.IGNORECASE,
)


class _DuplicateJsonKey(ValueError):
    pass


class _AuthorizedAsrPublicEvent(dict):
    """Carry a non-serialized in-process ticket from scoring to publication."""

    def __init__(self, event: dict, *, append_ticket: str):
        super().__init__(event)
        self.append_ticket = append_ticket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        help="Repeat NAME=private-records.jsonl for at least two candidates.",
    )
    parser.add_argument("--append-journal", type=Path)
    args = parser.parse_args()

    event = score_asr_agreement(
        workload_path=args.workload,
        candidate_record_paths=_parse_candidate_specs(args.candidate),
    )
    if args.append_journal is not None:
        _append_public_event_once(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def score_asr_agreement(
    *,
    workload_path: Path,
    candidate_record_paths: dict[str, Path],
    sustained_events_path: Path | None = None,
) -> dict:
    """Return privacy-bounded agreement evidence without treating it as truth."""

    _validate_candidate_specs(candidate_record_paths)
    resolved_registry_path = SUSTAINED_REGISTRY_PATH.resolve(strict=True)
    registry_bytes = _read_bounded_bytes(
        resolved_registry_path,
        maximum_bytes=_MAX_REGISTRY_BYTES,
        description="ASR agreement sustained registry",
    )
    registered_sources = _load_registered_asr_sources(
        resolved_registry_path,
        registry_bytes=registry_bytes,
    )
    resolved_workload_path = workload_path.resolve(strict=True)
    workload_bytes = _read_bounded_bytes(
        resolved_workload_path,
        maximum_bytes=_MAX_WORKLOAD_MANIFEST_BYTES,
        description="ASR agreement workload",
    )
    workload_document = _decode_json_object_bytes(
        workload_bytes,
        description="ASR agreement workload",
    )
    if workload_document.get("task") != "asr":
        raise ValueError("ASR agreement requires an ASR workload")
    if workload_document.get("workload_class") != "private_course":
        raise ValueError("ASR agreement accepts only the private_course workload class")
    raw_items = workload_document.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("ASR agreement workload requires nonempty items")
    if len(raw_items) > _MAX_WORKLOAD_ITEMS:
        raise ValueError("ASR agreement workload item budget exceeded")

    workload = load_sustained_workload_from_bytes(
        workload_bytes,
        manifest_path=resolved_workload_path,
        expected_task="asr",
    )
    _require_distinct_workload_content(
        workload["items"],
        content_bindings=workload["item_content_bindings"],
    )
    items = _load_items(workload["items"])
    small_private_cohort = len(items) < _MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    if sustained_events_path is None:
        sustained_events_path = SUSTAINED_EVENTS_PATH
    resolved_sustained_events_path = sustained_events_path.resolve(strict=True)
    sustained_snapshot = capture_sustained_journal_snapshot(
        resolved_sustained_events_path
    )
    sustained_sources, invalidated_attempt_ids, corrected_attempt_ids = _load_sustained_source_events(
        sustained_snapshot
    )

    source_bundles = []
    resolved_record_paths: set[Path] = set()
    resolved_provenance_paths: set[Path] = set()
    attempt_ids: set[str] = set()
    attempt_keys: set[str] = set()
    source_identities: set[tuple[str, int]] = set()
    for alias, path in candidate_record_paths.items():
        resolved_path = path.resolve(strict=True)
        if resolved_path in resolved_record_paths:
            raise ValueError("ASR agreement candidates require distinct record files")
        resolved_record_paths.add(resolved_path)

        provenance_path = resolved_path.with_name("records-provenance.json").resolve(
            strict=True
        )
        if provenance_path in resolved_provenance_paths:
            raise ValueError(
                "ASR agreement candidates require distinct provenance sidecars"
            )
        resolved_provenance_paths.add(provenance_path)
        records_bytes = _read_bounded_bytes(
            resolved_path,
            maximum_bytes=_MAX_RECORD_FILE_BYTES,
            description="ASR agreement record file",
        )
        provenance = _verify_records_provenance(
            resolved_path,
            records_bytes=records_bytes,
            provenance_path=provenance_path,
            workload_fingerprint=workload["fingerprint"],
        )
        registered_source = _validate_registered_source(
            provenance,
            registered_sources=registered_sources,
        )
        source_identity = (
            registered_source["candidate_id"],
            registered_source["config_index"],
        )
        if source_identity in source_identities:
            raise ValueError(
                "ASR agreement requires distinct candidate/config identities"
            )
        source_identities.add(source_identity)
        attempt_id = provenance["attempt_id"]
        if attempt_id in attempt_ids:
            raise ValueError("ASR agreement candidates require distinct attempts")
        attempt_ids.add(attempt_id)
        attempt_key = provenance["attempt_key"]
        if attempt_key in attempt_keys:
            raise ValueError("ASR agreement candidates require distinct attempt keys")
        attempt_keys.add(attempt_key)

        candidate_records = _read_records(records_bytes, set(items))
        _validate_source_status(
            provenance["status"],
            expected_ids=set(items),
            records=candidate_records,
        )
        _validate_sustained_source_event(
            provenance,
            records_bytes=records_bytes,
            records=candidate_records,
            expected_ids=set(items),
            sustained_sources=sustained_sources,
            invalidated_attempt_ids=invalidated_attempt_ids,
            corrected_attempt_ids=corrected_attempt_ids,
            workload_summary=workload["public_summary"],
        )
        source_bundles.append(
            {
                "alias": alias,
                "provenance": provenance,
                "registered_source": registered_source,
                "records": candidate_records,
            }
        )

    source_bundles.sort(
        key=lambda bundle: (
            bundle["registered_source"]["candidate_id"],
            bundle["registered_source"]["config_index"],
        )
    )
    records: dict[int, dict[str, dict]] = {}
    provenances: dict[int, dict] = {}
    registered_by_evidence_id: dict[int, dict] = {}
    for candidate_evidence_id, bundle in enumerate(source_bundles, start=1):
        records[candidate_evidence_id] = bundle["records"]
        provenances[candidate_evidence_id] = bundle["provenance"]
        registered_by_evidence_id[candidate_evidence_id] = bundle[
            "registered_source"
        ]

    candidate_metrics = [
        {
            "candidate_evidence_id": candidate_evidence_id,
            **_candidate_metrics(
                items,
                records[candidate_evidence_id],
                small_private_cohort=small_private_cohort,
            ),
        }
        for candidate_evidence_id in sorted(records)
    ]
    pair_metrics = []
    edit_budget = [_MAX_TOTAL_EDIT_CELLS]
    pair_evidence_id = len(records) + 1
    for left_id, right_id in combinations(sorted(records), 2):
        pair_metrics.append(
            {
                "pair_evidence_id": pair_evidence_id,
                "left_candidate_evidence_id": left_id,
                "right_candidate_evidence_id": right_id,
                **_pair_metrics(
                    items,
                    records[left_id],
                    records[right_id],
                    small_private_cohort=small_private_cohort,
                    edit_budget=edit_budget,
                ),
            }
        )
        pair_evidence_id += 1

    metrics = validate_public_summary(
        {
            "sample_count": len(items),
            "candidate_count": len(records),
            "small_private_cohort": small_private_cohort,
            "candidates": candidate_metrics,
            "pairs": pair_metrics,
        }
    )
    event = {
        "event": "asr_agreement_scored",
        "candidate_id": "private_course_asr_agreement",
        "protocol": "asr-text-agreement-v10",
        "scorer_fingerprint": _scorer_fingerprint(
            registry_bytes=registry_bytes,
            registry_path=resolved_registry_path,
        ),
        "source_authority_fingerprint": _asr_authority_fingerprint(
            registry_bytes=registry_bytes,
            sustained_events_sha256=sustained_snapshot.contents_sha256,
        ),
        "workload_class": "private_course",
        "source_candidates": [
            {
                "candidate_evidence_id": candidate_evidence_id,
                "candidate_id": registered_by_evidence_id[candidate_evidence_id][
                    "candidate_id"
                ],
                "status": provenances[candidate_evidence_id]["status"],
                "config_index": registered_by_evidence_id[candidate_evidence_id][
                    "config_index"
                ],
                "config_fingerprint": fingerprint_json(
                    registered_by_evidence_id[candidate_evidence_id]["config"]
                ),
            }
            for candidate_evidence_id in sorted(provenances)
        ],
        "privacy": {
            **_ASR_PRIVACY,
            "workload_meets_minimum_exact_aggregate_denominator": (
                not small_private_cohort
            ),
        },
        "interpretation": dict(_ASR_INTERPRETATION),
        "metrics": metrics,
    }
    event["public_event_sha256"] = _public_event_sha256(event)
    result = {
        **event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
    }
    _validate_asr_public_event(result, registry_bytes=registry_bytes)
    _require_asr_authorities_unchanged(
        registry_path=resolved_registry_path,
        registry_bytes=registry_bytes,
        sustained_events_path=resolved_sustained_events_path,
        sustained_snapshot=sustained_snapshot,
    )
    return _AuthorizedAsrPublicEvent(
        result,
        append_ticket=_asr_append_ticket(result),
    )


def _validate_candidate_specs(candidate_record_paths: object) -> None:
    if not isinstance(candidate_record_paths, dict):
        raise ValueError("ASR agreement candidates must be a mapping")
    if not 2 <= len(candidate_record_paths) <= _MAX_CANDIDATES:
        raise ValueError("ASR agreement requires two to eight candidates")
    if any(
        type(candidate_id) is not str
        or _CANDIDATE_ID.fullmatch(candidate_id) is None
        or not isinstance(path, Path)
        for candidate_id, path in candidate_record_paths.items()
    ):
        raise ValueError(
            "ASR agreement candidate IDs and record paths must be bounded public values"
        )


def _load_registered_asr_sources(
    registry_path: Path,
    *,
    registry_bytes: bytes | None = None,
) -> dict[str, dict]:
    if registry_bytes is None:
        registry_bytes = _read_bounded_bytes(
            registry_path,
            maximum_bytes=_MAX_REGISTRY_BYTES,
            description="ASR agreement sustained registry",
        )
    registry = _decode_json_object_bytes(
        registry_bytes,
        description="ASR agreement sustained registry",
    )
    candidates = registry.get("candidates") if isinstance(registry, dict) else None
    if (
        type(registry.get("schema_version")) is not int
        or registry["schema_version"] != 1
        or registry.get("protocol") != "sustained-process-v1"
        or not isinstance(candidates, list)
    ):
        raise ValueError("ASR agreement sustained registry is invalid")

    registered_sources: dict[str, dict] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("task") != "asr":
            continue
        candidate_id = candidate.get("id")
        configs = candidate.get("configs")
        if (
            type(candidate_id) is not str
            or _CANDIDATE_ID.fullmatch(candidate_id) is None
            or candidate_id in registered_sources
            or not isinstance(configs, list)
            or not configs
            or any(not isinstance(config, dict) for config in configs)
        ):
            raise ValueError("ASR agreement sustained registry is invalid")
        registered_sources[candidate_id] = candidate
    if not registered_sources:
        raise ValueError("ASR agreement sustained registry contains no ASR sources")
    return registered_sources


def _validate_registered_source(
    provenance: dict,
    *,
    registered_sources: dict[str, dict],
) -> dict:
    candidate_id = provenance["candidate_id"]
    candidate = registered_sources.get(candidate_id)
    config_index = provenance["config_index"]
    if candidate is None:
        raise ValueError("ASR agreement source candidate is not registered")
    configs = candidate["configs"]
    retired_indices = candidate.get("retired_config_indices", [])
    allowed_phases = candidate.get("allowed_phases")
    config = configs[config_index] if config_index < len(configs) else None
    config_phases = config.get("phases") if isinstance(config, dict) else None
    if (
        "status" in candidate
        or not isinstance(retired_indices, list)
        or any(type(index) is not int or index < 0 for index in retired_indices)
        or len(retired_indices) != len(set(retired_indices))
        or any(index >= len(configs) for index in retired_indices)
        or config_index >= len(configs)
        or config_index in retired_indices
        or not _json_values_equal(provenance["config"], config)
        or (
            allowed_phases is not None
            and (
                not isinstance(allowed_phases, list)
                or any(type(phase) is not str or not phase for phase in allowed_phases)
                or len(allowed_phases) != len(set(allowed_phases))
                or "quality" not in allowed_phases
            )
        )
        or (
            config_phases is not None
            and (
                not isinstance(config_phases, list)
                or any(type(phase) is not str or not phase for phase in config_phases)
                or len(config_phases) != len(set(config_phases))
                or "quality" not in config_phases
            )
        )
    ):
        raise ValueError(
            "ASR agreement source candidate/config does not match the registry"
        )
    return {
        "candidate_id": candidate_id,
        "config_index": config_index,
        "config": config,
    }


def _require_distinct_workload_content(
    items: list[dict],
    *,
    content_bindings: dict[str, dict],
) -> None:
    if not isinstance(content_bindings, dict):
        raise ValueError("ASR agreement workload content bindings are invalid")
    remaining_frames = [_MAX_TOTAL_PCM_FRAME_COUNT]
    decoded_pcm_sha256 = [
        _canonical_pcm_sha256(
            item,
            content_binding=content_bindings.get(item["id"]),
            remaining_frames=remaining_frames,
        )
        for item in items
    ]
    if len(decoded_pcm_sha256) != len(set(decoded_pcm_sha256)):
        raise ValueError("ASR agreement requires distinct workload item content")


def _canonical_pcm_sha256(
    item: dict,
    *,
    content_binding: object,
    remaining_frames: list[int],
) -> str:
    raw_wav = _read_bound_workload_bytes(Path(item["path"]), content_binding)
    digest = hashlib.sha256()
    digest.update(b"asr-canonical-pcm16le-mono-16000-v1\0")
    try:
        with wave.open(io.BytesIO(raw_wav), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            if (
                channels != 1
                or sample_width != 2
                or sample_rate != 16_000
                or reader.getcomptype() != "NONE"
                or frame_count <= 0
                or frame_count > remaining_frames[0]
                or not math.isclose(
                    frame_count / sample_rate,
                    float(item["duration_seconds"]),
                    rel_tol=0.0,
                    abs_tol=1 / sample_rate,
                )
            ):
                raise ValueError("ASR agreement requires bounded PCM16 WAV inputs")
            digest.update(
                struct.pack(
                    ">IIII",
                    sample_rate,
                    channels,
                    sample_width,
                    frame_count,
                )
            )
            consumed_frames = 0
            while consumed_frames < frame_count:
                requested_frames = min(65_536, frame_count - consumed_frames)
                frames = reader.readframes(requested_frames)
                if len(frames) != requested_frames * sample_width * channels:
                    raise ValueError("ASR agreement PCM input is truncated")
                consumed_frames += requested_frames
                digest.update(frames)
            if reader.readframes(1):
                raise ValueError("ASR agreement PCM frame count is inconsistent")
    except (EOFError, OSError, wave.Error) as error:
        raise ValueError("ASR agreement requires bounded PCM16 WAV inputs") from error
    remaining_frames[0] -= frame_count
    return digest.hexdigest()


def _read_bound_workload_bytes(path: Path, content_binding: object) -> bytes:
    if not isinstance(content_binding, dict):
        raise ValueError("ASR agreement workload content binding is invalid")
    size_bytes = content_binding.get("size_bytes")
    content_sha256 = content_binding.get("content_sha256")
    if (
        type(size_bytes) is not int
        or not 0 < size_bytes <= _MAX_PCM_WAV_BYTES
        or type(content_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
    ):
        raise ValueError("ASR agreement workload content binding is invalid")
    try:
        with path.open("rb") as handle:
            snapshot = handle.read(size_bytes + 1)
    except OSError as error:
        raise ValueError("ASR agreement workload content is unavailable") from error
    if (
        len(snapshot) != size_bytes
        or hashlib.sha256(snapshot).hexdigest() != content_sha256
    ):
        raise ValueError("ASR agreement workload content changed after binding")
    return snapshot


def _load_sustained_source_events(
    snapshot: SustainedJournalSnapshot,
) -> tuple[dict[str, dict], set[str], set[str]]:
    lifecycle: dict[str, dict[str, list[tuple[int, dict]]]] = {}
    for position, event in enumerate(snapshot.events):
        event_name = event.get("event")
        if (
            event_name != SUSTAINED_START_EVENT
            and event_name not in SUSTAINED_TERMINAL_EVENTS
        ):
            continue
        attempt_id = event.get("attempt_id")
        if type(attempt_id) is not str:
            continue
        kind = "start" if event_name == SUSTAINED_START_EVENT else "terminal"
        lifecycle.setdefault(attempt_id, {"start": [], "terminal": []})[kind].append(
            (position, event)
        )
    sources = {}
    for attempt_id, events in lifecycle.items():
        if (
            len(events["start"]) == 1
            and len(events["terminal"]) == 1
            and events["start"][0][0] < events["terminal"][0][0]
        ):
            sources[attempt_id] = {
                "start": events["start"][0][1],
                "terminal": events["terminal"][0][1],
            }
    return (
        sources,
        set(snapshot.invalidated_attempt_ids),
        set(snapshot.corrected_attempt_ids),
    )


def _validate_sustained_source_event(
    provenance: dict,
    *,
    records_bytes: bytes,
    records: dict[str, dict],
    expected_ids: set[str],
    sustained_sources: dict[str, dict],
    invalidated_attempt_ids: set[str],
    corrected_attempt_ids: set[str],
    workload_summary: dict,
) -> None:
    attempt_id = provenance["attempt_id"]
    if attempt_id in invalidated_attempt_ids:
        raise ValueError("ASR agreement source attempt is invalidated")
    if attempt_id in corrected_attempt_ids:
        raise ValueError("ASR agreement source attempt has an active correction")
    lifecycle = sustained_sources.get(attempt_id)
    if lifecycle is None:
        raise ValueError("ASR agreement source attempt is absent from the journal")
    expected_status = {
        "succeeded": ("sustained_attempt_succeeded", "complete"),
        "partial_failure": ("sustained_attempt_partial", "partial_failure"),
        "all_failed": ("sustained_attempt_failed", "all_failed"),
    }[provenance["status"]]
    for event in (lifecycle["start"], lifecycle["terminal"]):
        for key in (
            "protocol",
            "candidate_id",
            "task",
            "config_index",
            "phase",
            "target_wall_seconds",
            "trial_index",
            "code_fingerprint",
            "environment_fingerprint",
            "controller_environment_fingerprint",
            "execution_policy_fingerprint",
        ):
            if not _json_values_equal(event.get(key), provenance[key]):
                raise ValueError("ASR agreement source journal identity mismatch")
        if not _json_values_equal(event.get("config"), provenance["config"]):
            raise ValueError("ASR agreement source journal config mismatch")
        if not _json_values_equal(event.get("workload"), workload_summary):
            raise ValueError("ASR agreement source journal workload mismatch")
        if (
            event.get("private_records_commitment_scheme")
            != PRIVATE_RECORDS_COMMITMENT_SCHEME
        ):
            raise ValueError("ASR agreement source commitment scheme mismatch")
    terminal = lifecycle["terminal"]
    result = terminal.get("result")
    successful_record_count = sum(
        records.get(sample_id, {}).get("success") is True
        for sample_id in expected_ids
    )
    expected_counts = {
        "attempted": len(records),
        "completed": successful_record_count,
        "failed": len(records) - successful_record_count,
    }
    counts = result.get("counts") if isinstance(result, dict) else None
    if (
        terminal.get("event") != expected_status[0]
        or not isinstance(result, dict)
        or result.get("status") != expected_status[1]
        or result.get("candidate_id") != provenance["candidate_id"]
        or result.get("task") != "asr"
        or result.get("workload_class") != "private_course"
        or not isinstance(counts, dict)
        or any(type(counts.get(key)) is not int for key in expected_counts)
        or any(counts.get(key) != value for key, value in expected_counts.items())
    ):
        raise ValueError("ASR agreement source journal outcome mismatch")
    try:
        verify_private_records_bytes_commitment(
            records_bytes,
            provenance,
            records_sha256=provenance.get("records_sha256"),
            private_commitment=provenance.get("private_records_commitment"),
            public_commitment=terminal.get("private_artifact_commitment"),
        )
    except (OSError, ValueError) as error:
        raise ValueError("ASR agreement source commitment mismatch") from error


def _json_values_equal(left: object, right: object) -> bool:
    try:
        left_json = json.dumps(
            left,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        right_json = json.dumps(
            right,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False
    return left_json == right_json


def _parse_candidate_specs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        candidate_id, separator, raw_path = value.partition("=")
        if (
            not separator
            or _CANDIDATE_ID.fullmatch(candidate_id) is None
            or not raw_path
            or candidate_id in result
        ):
            raise ValueError("candidate values must be unique NAME=PATH pairs")
        result[candidate_id] = Path(raw_path)
    if not 2 <= len(result) <= _MAX_CANDIDATES:
        raise ValueError("ASR agreement requires two to eight candidates")
    return result


def _load_items(raw_items: object) -> dict[str, dict]:
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("ASR agreement workload requires nonempty items")
    if len(raw_items) > _MAX_WORKLOAD_ITEMS:
        raise ValueError("ASR agreement workload item budget exceeded")
    items = {}
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("ASR agreement workload items must be objects")
        sample_id = item.get("id")
        duration = item.get("duration_seconds")
        expected_speech = item.get("expected_speech", True)
        if (
            not isinstance(sample_id, str)
            or sample_id in items
            or isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration <= 0
            or type(expected_speech) is not bool
        ):
            raise ValueError("ASR agreement workload item is invalid")
        items[sample_id] = {
            "duration_seconds": float(duration),
            "expected_speech": expected_speech,
        }
    return items


def _read_records(records_bytes: bytes, expected_ids: set[str]) -> dict[str, dict]:
    try:
        lines = records_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("ASR agreement record file is not UTF-8") from error
    records = {}
    total_prediction_characters = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if len(records) >= _MAX_RECORDS_PER_CANDIDATE:
            raise ValueError("ASR agreement record count budget exceeded")
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise ValueError(
                f"invalid ASR agreement record at line {line_number}"
            ) from error
        if not isinstance(record, dict):
            raise ValueError(f"invalid ASR agreement record at line {line_number}")
        sample_id = record.get("sample_id")
        success = record.get("success")
        prediction = record.get("prediction")
        if (
            type(sample_id) is not str
            or sample_id not in expected_ids
            or sample_id in records
            or type(success) is not bool
            or (success and type(prediction) is not str)
            or (
                not success
                and "prediction" in record
                and type(prediction) is not str
            )
            or (
                type(prediction) is str
                and len(prediction) > _MAX_PREDICTION_CHARACTERS
            )
        ):
            raise ValueError(f"invalid ASR agreement record at line {line_number}")
        if not success and "prediction" not in record:
            record = {**record, "prediction": ""}
            prediction = ""
        total_prediction_characters += len(prediction)
        if total_prediction_characters > _MAX_TOTAL_PREDICTION_CHARACTERS:
            raise ValueError("ASR agreement total prediction budget exceeded")
        records[sample_id] = record
    return records


def _verify_records_provenance(
    records_path: Path,
    *,
    records_bytes: bytes,
    provenance_path: Path | None = None,
    workload_fingerprint: str,
) -> dict:
    if provenance_path is None:
        provenance_path = records_path.with_name("records-provenance.json").resolve(
            strict=True
        )
    provenance = _read_bounded_json_object(
        provenance_path,
        maximum_bytes=_MAX_PROVENANCE_BYTES,
        description="ASR agreement records provenance",
    )
    candidate_id = provenance.get("candidate_id")
    if (
        set(provenance) != _PROVENANCE_FIELDS
        or type(provenance.get("schema_version")) is not int
        or provenance["schema_version"] != 1
        or provenance.get("protocol") != "sustained-process-v1"
        or type(provenance.get("status")) is not str
        or provenance["status"] not in _SOURCE_STATUSES
        or provenance.get("task") != "asr"
        or provenance.get("phase") != "quality"
        or provenance.get("workload_class") != "private_course"
        or provenance.get("workload_fingerprint") != workload_fingerprint
        or provenance.get("records_sha256")
        != hashlib.sha256(records_bytes).hexdigest()
        or type(candidate_id) is not str
        or _CANDIDATE_ID.fullmatch(candidate_id) is None
        or not isinstance(provenance.get("config"), dict)
        or not isinstance(provenance.get("private_records_commitment"), dict)
        or isinstance(provenance.get("target_wall_seconds"), bool)
        or not isinstance(provenance.get("target_wall_seconds"), (int, float))
        or not math.isfinite(float(provenance["target_wall_seconds"]))
        or not 1 <= float(provenance["target_wall_seconds"]) <= 7200
    ):
        raise ValueError("ASR agreement records provenance is invalid")
    attempt_id = provenance.get("attempt_id")
    try:
        parsed_attempt_id = uuid.UUID(attempt_id)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("ASR agreement records provenance is invalid") from error
    if str(parsed_attempt_id) != attempt_id:
        raise ValueError("ASR agreement records provenance is invalid")
    for key in (
        "attempt_key",
        "code_fingerprint",
        "environment_fingerprint",
        "controller_environment_fingerprint",
        "execution_policy_fingerprint",
    ):
        if (
            type(provenance.get(key)) is not str
            or re.fullmatch(r"[0-9a-f]{16}", provenance[key]) is None
        ):
            raise ValueError("ASR agreement records provenance is invalid")
    for key in ("config_index", "trial_index"):
        if type(provenance.get(key)) is not int or provenance[key] < 0:
            raise ValueError("ASR agreement records provenance is invalid")
    for key in ("workload_fingerprint", "records_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", provenance[key]) is None:
            raise ValueError("ASR agreement records provenance is invalid")
    return provenance


def _validate_source_status(
    source_status: str,
    *,
    expected_ids: set[str],
    records: dict[str, dict],
) -> None:
    successful_sample_count = sum(
        records.get(sample_id, {}).get("success") is True
        for sample_id in expected_ids
    )
    unavailable_sample_count = len(expected_ids) - successful_sample_count
    expected_status = (
        "succeeded"
        if unavailable_sample_count == 0
        else "all_failed" if successful_sample_count == 0 else "partial_failure"
    )
    if source_status != expected_status:
        raise ValueError("ASR agreement source status does not match records")


def _candidate_metrics(
    items: dict[str, dict],
    records: dict[str, dict],
    *,
    small_private_cohort: bool,
) -> dict:
    successful_sample_count = 0
    explicit_failed_record_count = 0
    missing_record_count = 0
    any_failed_output_nonempty = False
    speech_sample_count = sum(item["expected_speech"] for item in items.values())
    successful_speech_sample_count = 0
    successful_speech_nonempty_count = 0
    near_silence_sample_count = len(items) - speech_sample_count
    successful_near_silence_sample_count = 0
    successful_near_silence_nonempty_count = 0
    successful_near_silence_seconds = 0.0
    successful_near_silence_characters = 0
    successful_character_counts = []
    successful_repetition = []
    for sample_id, item in items.items():
        record = records.get(sample_id)
        if record is None:
            missing_record_count += 1
            continue
        value = record["prediction"]
        normalized = _normalized_characters(value)
        if not record["success"]:
            explicit_failed_record_count += 1
            any_failed_output_nonempty = any_failed_output_nonempty or bool(normalized)
            continue

        successful_sample_count += 1
        successful_character_counts.append(len(normalized))
        successful_repetition.append(_repeated_trigram_ratio(_mixed_tokens(value)))
        if item["expected_speech"]:
            successful_speech_sample_count += 1
            successful_speech_nonempty_count += bool(normalized)
        else:
            successful_near_silence_sample_count += 1
            successful_near_silence_nonempty_count += bool(normalized)
            successful_near_silence_seconds += item["duration_seconds"]
            successful_near_silence_characters += len(normalized)

    total_successful_characters = sum(successful_character_counts)
    exact_aggregates_published = (
        not small_private_cohort
        and successful_sample_count >= _MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    )
    near_silence_exact_aggregates_published = (
        not small_private_cohort
        and successful_near_silence_sample_count
        >= _MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    )
    return {
        "availability": {
            "attempted_sample_count": len(items),
            "successful_sample_count": successful_sample_count,
            "unavailable_sample_count": len(items) - successful_sample_count,
        },
        "successful_output_metrics": {
            "sample_denominator": successful_sample_count,
            "speech_sample_denominator": successful_speech_sample_count,
            "near_silence_sample_denominator": (
                successful_near_silence_sample_count
            ),
            "exact_character_aggregates_published": exact_aggregates_published,
            "near_silence_exact_character_aggregates_published": (
                near_silence_exact_aggregates_published
            ),
            "speech_sample_count": speech_sample_count,
            "successful_speech_sample_count": successful_speech_sample_count,
            "successful_speech_nonempty_count": successful_speech_nonempty_count,
            "near_silence_sample_count": near_silence_sample_count,
            "successful_near_silence_sample_count": (
                successful_near_silence_sample_count
            ),
            "successful_near_silence_nonempty_count": (
                successful_near_silence_nonempty_count
            ),
            "normalized_character_count_bucket": _character_count_bucket(
                total_successful_characters
            ),
            "near_silence_normalized_character_count_bucket": (
                _character_count_bucket(successful_near_silence_characters)
            ),
            "repeated_trigram_observed": any(
                value > 0.0 for value in successful_repetition
            ),
            "near_silence_successful_seconds": (
                successful_near_silence_seconds
                if near_silence_exact_aggregates_published
                else None
            ),
            "near_silence_characters_per_minute": (
                successful_near_silence_characters
                / (successful_near_silence_seconds / 60.0)
                if (
                    near_silence_exact_aggregates_published
                    and successful_near_silence_seconds
                )
                else 0.0 if near_silence_exact_aggregates_published else None
            ),
            "mean_normalized_character_count": (
                total_successful_characters / successful_sample_count
                if exact_aggregates_published and successful_sample_count
                else 0.0 if exact_aggregates_published else None
            ),
            "mean_observed_repeated_trigram_ratio": (
                sum(successful_repetition) / successful_sample_count
                if exact_aggregates_published and successful_sample_count
                else 0.0 if exact_aggregates_published else None
            ),
        },
        "failed_output_diagnostics": {
            "explicit_failed_record_count": explicit_failed_record_count,
            "missing_record_count": missing_record_count,
            "any_explicit_failed_output_nonempty": any_failed_output_nonempty,
        },
    }


def _pair_metrics(
    items: dict[str, dict],
    left_records: dict[str, dict],
    right_records: dict[str, dict],
    *,
    small_private_cohort: bool,
    edit_budget: list[int],
) -> dict:
    total_distance = 0
    total_denominator = 0
    exact_match_count = 0
    one_empty_disagreement_count = 0
    comparable_sample_count = 0
    unavailable_sample_count = 0
    length_agreements = []
    for sample_id in items:
        left_record = left_records.get(sample_id)
        right_record = right_records.get(sample_id)
        if not (
            left_record
            and right_record
            and left_record["success"]
            and right_record["success"]
        ):
            unavailable_sample_count += 1
            continue
        comparable_sample_count += 1
        left = _normalized_characters(left_record["prediction"])
        right = _normalized_characters(right_record["prediction"])
        denominator = max(len(left), len(right))
        total_distance += _bounded_levenshtein(left, right, edit_budget)
        total_denominator += denominator
        exact_match_count += left == right
        one_empty_disagreement_count += bool(left) != bool(right)
        length_agreements.append(
            min(len(left), len(right)) / denominator if denominator else 1.0
        )

    normalized_similarity = (
        1.0 - total_distance / total_denominator
        if total_denominator
        else 1.0 if comparable_sample_count else None
    )
    mean_length_agreement = (
        sum(length_agreements) / comparable_sample_count
        if comparable_sample_count
        else None
    )
    exact_aggregates_published = (
        not small_private_cohort
        and comparable_sample_count >= _MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    )
    return {
        "availability": {
            "comparable_sample_count": comparable_sample_count,
            "unavailable_sample_count": unavailable_sample_count,
        },
        "successful_output_agreement": {
            "sample_denominator": comparable_sample_count,
            "exact_character_aggregates_published": exact_aggregates_published,
            "normalized_character_similarity_is_character_micro_weighted": True,
            "normalized_character_denominator": (
                total_denominator if exact_aggregates_published else None
            ),
            "normalized_character_similarity": (
                normalized_similarity if exact_aggregates_published else None
            ),
            "normalized_character_similarity_bucket": _fraction_bucket(
                normalized_similarity
            ),
            "mean_length_agreement": (
                mean_length_agreement if exact_aggregates_published else None
            ),
            "mean_length_agreement_bucket": _fraction_bucket(
                mean_length_agreement
            ),
            "exact_match_count": (
                exact_match_count if exact_aggregates_published else None
            ),
            "one_empty_disagreement_count": (
                one_empty_disagreement_count
                if exact_aggregates_published
                else None
            ),
            "any_exact_match": exact_match_count > 0,
            "all_comparable_exact_matches": (
                comparable_sample_count > 0
                and exact_match_count == comparable_sample_count
            ),
            "any_one_empty_disagreement": one_empty_disagreement_count > 0,
        },
    }


def _character_count_bucket(count: int) -> int:
    for bucket, upper_bound in enumerate(_CHARACTER_COUNT_BUCKET_UPPER_BOUNDS):
        if count <= upper_bound:
            return bucket
    return len(_CHARACTER_COUNT_BUCKET_UPPER_BOUNDS)


def _fraction_bucket(value: float | None) -> int | None:
    if value is None:
        return None
    if value == 1.0:
        return len(_FRACTION_BUCKET_LOWER_BOUNDS) - 1
    for bucket in range(len(_FRACTION_BUCKET_LOWER_BOUNDS) - 1):
        if value < _FRACTION_BUCKET_LOWER_BOUNDS[bucket + 1]:
            return bucket
    return len(_FRACTION_BUCKET_LOWER_BOUNDS) - 2


def _read_bounded_json_object(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
) -> dict:
    raw = _read_bounded_bytes(
        path,
        maximum_bytes=maximum_bytes,
        description=description,
    )
    return _decode_json_object_bytes(raw, description=description)


def _decode_json_object_bytes(raw: bytes, *, description: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{description} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} is invalid")
    return value


def _read_bounded_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
) -> bytes:
    try:
        with path.open("rb") as handle:
            value = handle.read(maximum_bytes + 1)
    except OSError as error:
        raise ValueError(f"{description} is unavailable") from error
    if len(value) > maximum_bytes:
        raise ValueError(f"{description} byte budget exceeded")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_nonfinite_json_constant(constant: str):
    raise ValueError(f"non-finite ASR agreement JSON constant: {constant}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite ASR agreement JSON number: {value}")
    return parsed


def _mixed_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize(
        "NFKC",
        _SENSEVOICE_TAG.sub("", value),
    ).casefold()
    return _MIXED_TOKEN.findall(normalized)


def _normalized_characters(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC",
        _SENSEVOICE_TAG.sub("", value),
    ).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _repeated_trigram_ratio(tokens: list[str]) -> float:
    if len(tokens) < 3:
        return 0.0
    trigrams = [
        tuple(tokens[index : index + 3])
        for index in range(len(tokens) - 2)
    ]
    return (len(trigrams) - len(set(trigrams))) / len(trigrams)


def _levenshtein(left, right) -> int:
    if left == right:
        return 0
    if not left or not right:
        return max(len(left), len(right))
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _bounded_levenshtein(left: str, right: str, edit_budget: list[int]) -> int:
    if left == right or not left or not right:
        return _levenshtein(left, right)
    cells = len(left) * len(right)
    if cells > edit_budget[0]:
        raise ValueError("ASR agreement edit-distance budget exceeded")
    edit_budget[0] -= cells
    return _levenshtein(left, right)


def _public_event_sha256(event: dict) -> str:
    return _protocol_public_event_sha256(event)


def _append_public_event_once(path: Path, event: dict) -> bool:
    if (
        not isinstance(event, _AuthorizedAsrPublicEvent)
        or type(getattr(event, "append_ticket", None)) is not str
        or not hmac.compare_digest(event.append_ticket, _asr_append_ticket(event))
    ):
        raise ValueError("ASR agreement event was not authorized by this scoring process")
    try:
        sustained_events_path = SUSTAINED_EVENTS_PATH.resolve(strict=True)
        registry_path = SUSTAINED_REGISTRY_PATH.resolve(strict=True)
        output_path = path.resolve()
        if _paths_identify_same_file(output_path, sustained_events_path):
            raise ValueError(
                "ASR agreement output journal cannot be the sustained journal"
            )
        if _paths_identify_same_file(output_path, registry_path):
            raise ValueError(
                "ASR agreement output journal cannot be the sustained registry"
            )
        with locked_file_bytes(registry_path) as registry_bytes:
            if len(registry_bytes) > _MAX_REGISTRY_BYTES:
                raise ValueError("ASR agreement sustained registry is too large")
            with locked_journal_bytes(sustained_events_path) as journal_bytes:
                _validate_asr_public_event(event, registry_bytes=registry_bytes)
                current_authority = _asr_authority_fingerprint(
                    registry_bytes=registry_bytes,
                    sustained_events_sha256=hashlib.sha256(journal_bytes).hexdigest(),
                )
                if event["source_authority_fingerprint"] != current_authority:
                    raise ValueError(
                        "ASR agreement public authority changed before append"
                    )
                return append_event_once(path, event)
    except OSError as error:
        raise ValueError("ASR agreement sustained journal is unavailable") from error


def _paths_identify_same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.samefile(right)
    except OSError:
        return False


def _validate_asr_public_event(event: object, *, registry_bytes: bytes) -> dict:
    try:
        registered_sources = _load_registered_asr_sources(
            SUSTAINED_REGISTRY_PATH,
            registry_bytes=registry_bytes,
        )
    except ValueError as error:
        raise ValueError("ASR agreement public event is invalid") from error

    def source_matches_registry(source: dict) -> bool:
        candidate_id = source.get("candidate_id")
        config_index = source.get("config_index")
        candidate = registered_sources.get(candidate_id)
        if (
            not isinstance(candidate, dict)
            or type(config_index) is not int
            or config_index >= len(candidate["configs"])
            or source["config_fingerprint"]
            != fingerprint_json(candidate["configs"][config_index])
        ):
            return False
        try:
            _validate_registered_source(
                {
                    "candidate_id": candidate_id,
                    "config_index": config_index,
                    "config": candidate["configs"][config_index],
                },
                registered_sources=registered_sources,
            )
        except ValueError:
            return False
        return True

    return _validate_protocol_public_event(
        event,
        source_matches_registry=source_matches_registry,
    )


def _asr_authority_fingerprint(
    *,
    registry_bytes: bytes,
    sustained_events_sha256: str,
) -> str:
    return fingerprint_json(
        {
            "protocol": "asr-agreement-public-authority-v1",
            "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "sustained_events_sha256": sustained_events_sha256,
        }
    )


def _asr_append_ticket(event: dict) -> str:
    identity = b"asr-agreement-append-ticket-v1\0" + json.dumps(
        event,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    return hmac.new(_APPEND_TICKET_KEY, identity, hashlib.sha256).hexdigest()


def _require_asr_authorities_unchanged(
    *,
    registry_path: Path,
    registry_bytes: bytes,
    sustained_events_path: Path,
    sustained_snapshot: SustainedJournalSnapshot,
) -> None:
    current_registry = _read_bounded_bytes(
        registry_path,
        maximum_bytes=_MAX_REGISTRY_BYTES,
        description="ASR agreement sustained registry",
    )
    try:
        current_journal_sha256 = hashlib.sha256(
            read_journal_bytes(sustained_events_path)
        ).hexdigest()
    except OSError as error:
        raise ValueError("ASR agreement sustained journal is unavailable") from error
    if (
        current_registry != registry_bytes
        or current_journal_sha256 != sustained_snapshot.contents_sha256
    ):
        raise ValueError("ASR agreement public authority changed during scoring")


def _scorer_fingerprint(
    *,
    registry_bytes: bytes | None = None,
    registry_path: Path | None = None,
) -> str:
    """Bind the event to all repository-owned code used to produce it."""

    if registry_path is None:
        registry_path = SUSTAINED_REGISTRY_PATH
    resolved_registry_path = registry_path.resolve(strict=True)
    if registry_bytes is None:
        registry_bytes = _read_bounded_bytes(
            resolved_registry_path,
            maximum_bytes=_MAX_REGISTRY_BYTES,
            description="ASR agreement sustained registry",
        )
    dependencies = []
    for index, path in enumerate(_scorer_dependency_paths()):
        resolved_path = path.resolve(strict=True)
        contents = (
            registry_bytes
            if resolved_path == resolved_registry_path
            else resolved_path.read_bytes()
        )
        dependencies.append(
            {
                "dependency_index": index,
                "filename": resolved_path.name,
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    if not any(
        path.resolve(strict=True) == resolved_registry_path
        for path in _scorer_dependency_paths()
    ):
        dependencies.append(
            {
                "dependency_index": len(dependencies),
                "filename": resolved_registry_path.name,
                "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            }
        )
    return fingerprint_json(dependencies)


def _scorer_dependency_paths(module_path: Path | None = None) -> list[Path]:
    if module_path is None:
        module_path = Path(__file__).resolve()
    return [
        module_path,
        module_path.with_name("asr_agreement_public_protocol.py"),
        module_path.with_name("event_journal.py"),
        module_path.with_name("fingerprint.py"),
        module_path.with_name("journal_integrity.py"),
        module_path.with_name("load_registry.py"),
        module_path.with_name("load_sustained_workload.py"),
        module_path.with_name("project_paths.py"),
        module_path.with_name("private_records_commitment.py"),
        module_path.with_name("validate_public_summary.py"),
        SUSTAINED_REGISTRY_PATH,
    ]


if __name__ == "__main__":
    main()
