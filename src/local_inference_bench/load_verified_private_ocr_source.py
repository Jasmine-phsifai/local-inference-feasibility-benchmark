"""Load one HMAC-bound private OCR source from sustained evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .event_journal import read_journal_bytes
from .fingerprint import fingerprint_json
from .journal_integrity import (
    SUSTAINED_START_EVENT,
    SUSTAINED_TERMINAL_EVENTS,
    capture_sustained_journal_snapshot,
)
from .private_records_commitment import (
    PRIVATE_RECORDS_COMMITMENT_SCHEME,
    PRIVATE_RECORDS_PROVENANCE_FIELDS as _PROVENANCE_FIELDS,
    verify_private_records_bytes_commitment,
)
from .project_paths import SUSTAINED_EVENTS_PATH, SUSTAINED_REGISTRY_PATH


_PUBLIC_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LOWER_FINGERPRINT = re.compile(r"^[0-9a-f]{16}$")
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})
_MAX_RECORD_BYTES = 64 * 1024 * 1024
_MAX_PROVENANCE_BYTES = 65_536
_MAX_REGISTRY_BYTES = 2 * 1024 * 1024
_MAX_RECORD_COUNT = 100_000
_MAX_LINES_PER_RECORD = 10_000
_MAX_LINE_CHARACTERS = 10_000
_MAX_TOTAL_CHARACTERS = 1_000_000


class _DuplicateJsonKey(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedPrivateOcrAuthoritySnapshot:
    sustained_events_path: Path
    registry_path: Path
    sustained_events: tuple[dict, ...]
    invalidated_attempt_ids: frozenset[str]
    corrected_attempt_ids: frozenset[str]
    sustained_events_sha256: str
    registry_bytes: bytes


def capture_verified_private_ocr_authority(
    *,
    sustained_events_path: Path | None = None,
    registry_path: Path | None = None,
) -> VerifiedPrivateOcrAuthoritySnapshot:
    """Capture the exact journal and registry authority used by multiple sources."""

    if sustained_events_path is None:
        sustained_events_path = SUSTAINED_EVENTS_PATH
    if registry_path is None:
        registry_path = SUSTAINED_REGISTRY_PATH
    resolved_events_path = sustained_events_path.resolve(strict=True)
    resolved_registry_path = registry_path.resolve(strict=True)
    journal = capture_sustained_journal_snapshot(resolved_events_path)
    registry_bytes = _read_bounded_bytes(
        resolved_registry_path,
        maximum_bytes=_MAX_REGISTRY_BYTES,
        description="sustained registry",
    )
    _decode_json_object(registry_bytes, description="sustained registry")
    return VerifiedPrivateOcrAuthoritySnapshot(
        sustained_events_path=resolved_events_path,
        registry_path=resolved_registry_path,
        sustained_events=journal.events,
        invalidated_attempt_ids=journal.invalidated_attempt_ids,
        corrected_attempt_ids=journal.corrected_attempt_ids,
        sustained_events_sha256=journal.contents_sha256,
        registry_bytes=registry_bytes,
    )


def verify_private_ocr_authority_is_current(
    snapshot: VerifiedPrivateOcrAuthoritySnapshot,
    *,
    sustained_events_bytes: bytes | None = None,
    registry_bytes: bytes | None = None,
) -> None:
    """Fail when either public authority changed after the captured validation point."""

    if not isinstance(snapshot, VerifiedPrivateOcrAuthoritySnapshot):
        raise ValueError("private OCR authority snapshot is invalid")
    if sustained_events_bytes is None:
        try:
            sustained_events_bytes = read_journal_bytes(snapshot.sustained_events_path)
        except OSError as error:
            raise ValueError("sustained journal authority is unavailable") from error
    if type(sustained_events_bytes) is not bytes:
        raise ValueError("sustained journal authority is unavailable")
    journal_sha256 = hashlib.sha256(sustained_events_bytes).hexdigest()
    if registry_bytes is None:
        registry_bytes = _read_bounded_bytes(
            snapshot.registry_path,
            maximum_bytes=_MAX_REGISTRY_BYTES,
            description="sustained registry",
        )
    elif type(registry_bytes) is not bytes:
        raise ValueError("sustained registry authority is unavailable")
    if (
        journal_sha256 != snapshot.sustained_events_sha256
        or registry_bytes != snapshot.registry_bytes
    ):
        raise ValueError("private OCR public authority changed during validation")


def validate_public_ocr_score_sources(
    snapshot: VerifiedPrivateOcrAuthoritySnapshot,
    sources: list[dict],
    *,
    sustained_events_bytes: bytes | None = None,
    registry_bytes: bytes | None = None,
) -> None:
    """Bind privacy-safe score identities to active registry configs at append time."""

    if not isinstance(snapshot, VerifiedPrivateOcrAuthoritySnapshot) or not isinstance(
        sources, list
    ):
        raise ValueError("private OCR score registry authority is invalid")
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("private OCR score registry authority is invalid")
        config = _load_active_registered_config(
            snapshot.registry_bytes,
            candidate_id=source.get("candidate_id"),
            config_index=source.get("config_index"),
        )
        if source.get("config_fingerprint") != fingerprint_json(config):
            raise ValueError("private OCR score source does not match the active registry")
    verify_private_ocr_authority_is_current(
        snapshot,
        sustained_events_bytes=sustained_events_bytes,
        registry_bytes=registry_bytes,
    )


def load_verified_private_ocr_source(
    records_path: Path,
    *,
    expected_workload_fingerprint: str | None = None,
    expected_workload_summary: dict | None = None,
    sustained_events_path: Path | None = None,
    registry_path: Path | None = None,
    authority_snapshot: VerifiedPrivateOcrAuthoritySnapshot | None = None,
) -> dict:
    """Return immutable OCR records only after every private binding verifies."""

    if sustained_events_path is None:
        sustained_events_path = SUSTAINED_EVENTS_PATH
    if registry_path is None:
        registry_path = SUSTAINED_REGISTRY_PATH
    if authority_snapshot is None:
        authority_snapshot = capture_verified_private_ocr_authority(
            sustained_events_path=sustained_events_path,
            registry_path=registry_path,
        )
    elif (
        authority_snapshot.sustained_events_path
        != sustained_events_path.resolve(strict=True)
        or authority_snapshot.registry_path != registry_path.resolve(strict=True)
    ):
        raise ValueError("private OCR authority snapshot paths do not match")
    resolved_records_path = records_path.resolve(strict=True)
    records_bytes = _read_bounded_bytes(
        resolved_records_path,
        maximum_bytes=_MAX_RECORD_BYTES,
        description="private OCR records",
    )
    records = _parse_ocr_records(records_bytes)
    if not records:
        raise ValueError("private OCR source contains no records")
    provenance_path = resolved_records_path.with_name("records-provenance.json").resolve(
        strict=True
    )
    provenance_bytes = _read_bounded_bytes(
        provenance_path,
        maximum_bytes=_MAX_PROVENANCE_BYTES,
        description="private OCR provenance",
    )
    provenance = _decode_json_object(
        provenance_bytes,
        description="private OCR provenance",
    )
    _validate_provenance(
        provenance,
        records_bytes=records_bytes,
        expected_workload_fingerprint=expected_workload_fingerprint,
    )
    registered_source = _validate_registered_source(
        provenance,
        registry_bytes=authority_snapshot.registry_bytes,
    )
    lifecycle = _load_exact_lifecycle(
        authority_snapshot,
        attempt_id=provenance["attempt_id"],
    )
    workload_summary = _validate_lifecycle(
        lifecycle,
        provenance=provenance,
        records=records,
        records_bytes=records_bytes,
        expected_workload_summary=expected_workload_summary,
    )
    verify_private_ocr_authority_is_current(authority_snapshot)
    return {
        "records_path": resolved_records_path,
        "records_bytes": records_bytes,
        "records": records,
        "provenance_bytes": provenance_bytes,
        "provenance": provenance,
        "registered_source": registered_source,
        "workload_summary": workload_summary,
    }


def _validate_provenance(
    provenance: dict,
    *,
    records_bytes: bytes,
    expected_workload_fingerprint: str | None,
) -> None:
    candidate_id = provenance.get("candidate_id")
    target_wall_seconds = provenance.get("target_wall_seconds")
    workload_fingerprint = provenance.get("workload_fingerprint")
    if (
        set(provenance) != _PROVENANCE_FIELDS
        or type(provenance.get("schema_version")) is not int
        or provenance["schema_version"] != 1
        or provenance.get("protocol") != "sustained-process-v1"
        or type(provenance.get("status")) is not str
        or provenance["status"] not in _SOURCE_STATUSES
        or provenance.get("task") != "ocr"
        or provenance.get("phase") != "quality"
        or provenance.get("workload_class") != "private_course"
        or type(candidate_id) is not str
        or _PUBLIC_ID.fullmatch(candidate_id) is None
        or not isinstance(provenance.get("config"), dict)
        or type(provenance.get("config_index")) is not int
        or provenance["config_index"] < 0
        or type(provenance.get("trial_index")) is not int
        or provenance["trial_index"] < 0
        or isinstance(target_wall_seconds, bool)
        or not isinstance(target_wall_seconds, (int, float))
        or not math.isfinite(float(target_wall_seconds))
        or not 1 <= float(target_wall_seconds) <= 7200
        or type(workload_fingerprint) is not str
        or _LOWER_SHA256.fullmatch(workload_fingerprint) is None
        or (
            expected_workload_fingerprint is not None
            and workload_fingerprint != expected_workload_fingerprint
        )
        or provenance.get("records_sha256")
        != hashlib.sha256(records_bytes).hexdigest()
        or not isinstance(provenance.get("private_records_commitment"), dict)
    ):
        raise ValueError("private OCR provenance is invalid")
    try:
        parsed_attempt_id = str(uuid.UUID(provenance.get("attempt_id")))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("private OCR provenance is invalid") from error
    if parsed_attempt_id != provenance.get("attempt_id"):
        raise ValueError("private OCR provenance is invalid")
    for field in (
        "attempt_key",
        "code_fingerprint",
        "environment_fingerprint",
        "controller_environment_fingerprint",
        "execution_policy_fingerprint",
    ):
        if (
            type(provenance.get(field)) is not str
            or _LOWER_FINGERPRINT.fullmatch(provenance[field]) is None
        ):
            raise ValueError("private OCR provenance is invalid")


def _validate_registered_source(provenance: dict, *, registry_bytes: bytes) -> dict:
    config = _load_active_registered_config(
        registry_bytes,
        candidate_id=provenance["candidate_id"],
        config_index=provenance["config_index"],
    )
    if not _json_values_equal(config, provenance["config"]):
        raise ValueError("private OCR source candidate/config is not active for quality")
    return {
        "candidate_id": provenance["candidate_id"],
        "config_index": provenance["config_index"],
        "config": config,
    }


def _load_active_registered_config(
    registry_bytes: bytes,
    *,
    candidate_id: object,
    config_index: object,
) -> dict:
    registry = _decode_json_object(registry_bytes, description="sustained registry")
    candidates = registry.get("candidates")
    if (
        type(registry.get("schema_version")) is not int
        or registry["schema_version"] != 1
        or registry.get("protocol") != "sustained-process-v1"
        or not isinstance(candidates, list)
    ):
        raise ValueError("sustained registry is invalid")
    matches = [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and candidate.get("id") == candidate_id
        and candidate.get("task") == "ocr"
    ]
    if len(matches) != 1:
        raise ValueError("private OCR source candidate is not registered")
    candidate = matches[0]
    configs = candidate.get("configs")
    if type(config_index) is not int or config_index < 0:
        raise ValueError("private OCR source candidate/config is not active for quality")
    retired_indices = candidate.get("retired_config_indices", [])
    allowed_phases = candidate.get("allowed_phases")
    config = (
        configs[config_index]
        if isinstance(configs, list) and config_index < len(configs)
        else None
    )
    config_phases = config.get("phases") if isinstance(config, dict) else None
    if (
        "status" in candidate
        or not isinstance(retired_indices, list)
        or any(type(index) is not int or index < 0 for index in retired_indices)
        or len(retired_indices) != len(set(retired_indices))
        or not isinstance(configs, list)
        or any(index >= len(configs) for index in retired_indices)
        or not isinstance(config, dict)
        or config_index in retired_indices
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
        raise ValueError("private OCR source candidate/config is not active for quality")
    return config


def _load_exact_lifecycle(
    authority_snapshot: VerifiedPrivateOcrAuthoritySnapshot,
    *,
    attempt_id: str,
) -> dict:
    if attempt_id in authority_snapshot.invalidated_attempt_ids:
        raise ValueError("private OCR source attempt is invalidated")
    if attempt_id in authority_snapshot.corrected_attempt_ids:
        raise ValueError("private OCR source attempt has an active correction")
    starts = []
    terminals = []
    for position, event in enumerate(authority_snapshot.sustained_events):
        if event.get("attempt_id") != attempt_id:
            continue
        if event.get("event") == SUSTAINED_START_EVENT:
            starts.append((position, event))
        elif event.get("event") in SUSTAINED_TERMINAL_EVENTS:
            terminals.append((position, event))
    if len(starts) != 1 or len(terminals) != 1:
        raise ValueError("private OCR source lifecycle is incomplete or ambiguous")
    if starts[0][0] >= terminals[0][0]:
        raise ValueError("private OCR source lifecycle order is invalid")
    return {"start": starts[0][1], "terminal": terminals[0][1]}


def _validate_lifecycle(
    lifecycle: dict,
    *,
    provenance: dict,
    records: dict[str, dict],
    records_bytes: bytes,
    expected_workload_summary: dict | None,
) -> dict:
    start = lifecycle["start"]
    terminal = lifecycle["terminal"]
    identity_fields = (
        "protocol",
        "attempt_id",
        "candidate_id",
        "task",
        "config",
        "config_index",
        "phase",
        "target_wall_seconds",
        "trial_index",
        "code_fingerprint",
        "environment_fingerprint",
        "controller_environment_fingerprint",
        "execution_policy_fingerprint",
    )
    for event in (start, terminal):
        if (
            any(
                not _json_values_equal(event.get(field), provenance.get(field))
                for field in identity_fields
            )
            or "attempt_key" in event
            or "workload_fingerprint" in event
            or event.get("private_records_commitment_scheme")
            != PRIVATE_RECORDS_COMMITMENT_SCHEME
        ):
            raise ValueError("private OCR source journal identity mismatch")
    workload_summary = start.get("workload")
    if (
        not isinstance(workload_summary, dict)
        or not _json_values_equal(terminal.get("workload"), workload_summary)
        or (
            expected_workload_summary is not None
            and not _json_values_equal(workload_summary, expected_workload_summary)
        )
        or workload_summary.get("workload_class") != "private_course"
        or type(workload_summary.get("item_count")) is not int
        or workload_summary["item_count"] <= 0
        or isinstance(workload_summary.get("total_duration_seconds"), bool)
        or not isinstance(
            workload_summary.get("total_duration_seconds"),
            (int, float),
        )
        or not math.isfinite(float(workload_summary["total_duration_seconds"]))
        or float(workload_summary["total_duration_seconds"]) < 0
    ):
        raise ValueError("private OCR source journal workload mismatch")
    item_count = workload_summary["item_count"]
    if len(records) > item_count:
        raise ValueError("private OCR source has more records than workload items")
    completed = sum(record["success"] for record in records.values())
    failed = len(records) - completed
    missing = item_count - len(records)
    derived_status = (
        "succeeded"
        if completed == item_count
        else "all_failed" if completed == 0 else "partial_failure"
    )
    if provenance["status"] != derived_status:
        raise ValueError("private OCR source status does not match its records")
    expected_terminal = {
        "succeeded": ("sustained_attempt_succeeded", "complete"),
        "partial_failure": ("sustained_attempt_partial", "partial_failure"),
        "all_failed": ("sustained_attempt_failed", "all_failed"),
    }[provenance["status"]]
    result = terminal.get("result")
    counts = result.get("counts") if isinstance(result, dict) else None
    status_counts_match = {
        "succeeded": failed == 0 and missing == 0,
        "partial_failure": completed > 0 and failed > 0,
        "all_failed": completed == 0 and failed > 0,
    }[provenance["status"]]
    if (
        terminal.get("event") != expected_terminal[0]
        or not isinstance(result, dict)
        or result.get("status") != expected_terminal[1]
        or result.get("candidate_id") != provenance["candidate_id"]
        or result.get("task") != "ocr"
        or result.get("workload_class") != "private_course"
        or not isinstance(counts, dict)
        or type(counts.get("attempted")) is not int
        or type(counts.get("completed")) is not int
        or type(counts.get("failed")) is not int
        or counts.get("attempted") != len(records)
        or counts.get("completed") != completed
        or counts.get("failed") != failed
        or not status_counts_match
    ):
        raise ValueError("private OCR source journal outcome mismatch")
    try:
        verify_private_records_bytes_commitment(
            records_bytes,
            provenance,
            records_sha256=provenance.get("records_sha256"),
            private_commitment=provenance.get("private_records_commitment"),
            public_commitment=terminal.get("private_artifact_commitment"),
        )
    except (OSError, ValueError) as error:
        raise ValueError("private OCR source commitment mismatch") from error
    return workload_summary


def _parse_ocr_records(records_bytes: bytes) -> dict[str, dict]:
    try:
        lines = records_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("private OCR records are not UTF-8") from error
    records = {}
    total_characters = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if len(records) >= _MAX_RECORD_COUNT:
            raise ValueError("private OCR record count budget exceeded")
        try:
            record = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=lambda constant: _reject_json_constant(constant),
                parse_float=_parse_finite_json_float,
            )
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise ValueError(f"invalid private OCR record at line {line_number}") from error
        sample_id = record.get("sample_id") if isinstance(record, dict) else None
        success = record.get("success") if isinstance(record, dict) else None
        output_lines = record.get("lines") if isinstance(record, dict) else None
        if (
            type(sample_id) is not str
            or _PUBLIC_ID.fullmatch(sample_id) is None
            or sample_id in records
            or type(success) is not bool
            or (success and not isinstance(output_lines, list))
            or (output_lines is not None and not isinstance(output_lines, list))
            or len(output_lines or []) > _MAX_LINES_PER_RECORD
        ):
            raise ValueError(f"invalid private OCR record at line {line_number}")
        for output_line in output_lines or []:
            text = output_line.get("text") if isinstance(output_line, dict) else None
            if type(text) is not str or len(text) > _MAX_LINE_CHARACTERS:
                raise ValueError(f"invalid private OCR record at line {line_number}")
            total_characters += len(text)
            if total_characters > _MAX_TOTAL_CHARACTERS:
                raise ValueError("private OCR record text budget exceeded")
        records[sample_id] = record
    return records


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
    return _decode_json_object(raw, description=description)


def _decode_json_object(raw: bytes, *, description: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=lambda constant: _reject_json_constant(constant),
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


def _json_values_equal(left: object, right: object) -> bool:
    try:
        return json.dumps(
            left,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ) == json.dumps(
            right,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return False


def _reject_json_constant(constant: str):
    raise ValueError(f"non-finite JSON constant is invalid: {constant}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is invalid: {value}")
    return parsed


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result
