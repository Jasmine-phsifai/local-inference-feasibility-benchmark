"""Aggregate HMAC-anchored blind OCR v10 judgments into safe evidence."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import re
import secrets
import uuid
from collections import Counter
from pathlib import Path

from local_inference_bench.event_journal import (
    append_event_once,
    locked_file_bytes,
    locked_journal_bytes,
    read_journal_bytes,
)
from local_inference_bench.fingerprint import fingerprint_json
from local_inference_bench.load_verified_private_ocr_source import (
    VerifiedPrivateOcrAuthoritySnapshot,
    capture_verified_private_ocr_authority,
    load_verified_private_ocr_source,
    validate_public_ocr_score_sources,
    verify_private_ocr_authority_is_current,
)
from local_inference_bench.private_records_commitment import (
    PRIVATE_RECORDS_COMMITMENT_SCHEME,
)
from local_inference_bench.project_paths import (
    QUALITY_EVENTS_PATH,
    SUSTAINED_EVENTS_PATH,
    SUSTAINED_REGISTRY_PATH,
)
from local_inference_bench.validate_public_summary import validate_public_summary
from local_inference_bench.verified_blind_ocr_protocol import (
    COMMITMENT_SCHEME,
    PROTOCOL,
    SCORE_INTERPRETATION,
    SCORE_SOURCE_FIELDS,
    packet_commitment_payload,
    public_event_sha256,
    utc_timestamp_now,
    validate_preparation_event,
    validate_score_event,
)
from scripts.prepare_verified_blind_ocr_comparison import (
    PRODUCER_PATHS as PREPARATION_PRODUCER_PATHS,
    _preparation_event,
)


AGGREGATE_PROTOCOL = PROTOCOL
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGGREGATION_PRODUCER_PATHS = (
    "scripts/aggregate_verified_blind_ocr_judgments.py",
    "src/local_inference_bench/validate_public_summary.py",
)
_MAX_MAPPING_BYTES = 8 * 1024 * 1024
_MAX_PACKET_BYTES = 16 * 1024 * 1024
_MAX_JUDGMENT_BYTES = 4 * 1024 * 1024
_MAX_QUALITY_JOURNAL_BYTES = 64 * 1024 * 1024
_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_PUBLIC_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_BLIND_ID = re.compile(r"^blind_[0-9]{3}$")
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{16}$")
_RECORD_STATUSES = frozenset({"available", "failed", "unavailable"})
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})
_ALLOWED_ERROR_CODES = frozenset(
    {
        "false_positive",
        "formula_or_code",
        "garble",
        "missing_text",
        "reading_order",
        "small_text",
    }
)
_EXPECTED_INSTRUCTIONS = {
    "winner_values": ["A", "B", "tie"],
    "severity_scale": [0, 1, 2, 3],
    "allowed_error_codes": sorted(_ALLOWED_ERROR_CODES),
    "option_status_values": ["available", "failed", "unavailable"],
}
_MAPPING_FIELDS = frozenset(
    {
        "schema_version",
        "protocol",
        "preparation_id",
        "variant_ids",
        "samples",
        "source_bindings",
        "preparation_producer_sha256",
        "packet_fingerprint",
        "private_packet_commitment",
    }
)
_PACKET_FIELDS = frozenset(
    {"schema_version", "protocol", "preparation_id", "instructions", "samples"}
)
_SAMPLE_FIELDS = frozenset(
    {"sample_id", "source_id", "identities", "record_statuses", "image_sha256"}
)
_PACKET_SAMPLE_FIELDS = frozenset(
    {"sample_id", "image_path", "options", "option_statuses"}
)
_SOURCE_BINDING_FIELDS = frozenset(
    {"variant_id", "attempt_id", "private_artifact_commitment"}
)
_JUDGMENT_FIELDS = frozenset(
    {"schema_version", "protocol", "packet_fingerprint", "samples"}
)
_JUDGMENT_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "winner",
        "a_severity",
        "b_severity",
        "a_usable",
        "b_usable",
        "a_error_codes",
        "b_error_codes",
    }
)
_PUBLIC_SOURCE_FIELDS = SCORE_SOURCE_FIELDS
_INTERPRETATION = SCORE_INTERPRETATION
_APPEND_TICKET_KEY = secrets.token_bytes(32)


class _DuplicateJsonKey(ValueError):
    pass


class _AuthorizedBlindOcrScoreEvent(dict):
    """Carry non-serialized source authority from aggregation to publication."""

    def __init__(
        self,
        event: dict,
        *,
        authority_snapshot: VerifiedPrivateOcrAuthoritySnapshot,
        append_ticket: str,
    ):
        super().__init__(event)
        self.authority_snapshot = authority_snapshot
        self.append_ticket = append_ticket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--judgment", required=True, action="append", type=Path)
    parser.add_argument("--append-journal", required=True, type=Path)
    args = parser.parse_args()
    if args.append_journal.resolve() != QUALITY_EVENTS_PATH.resolve():
        raise ValueError("verified blind OCR aggregation requires the canonical quality journal")
    event = aggregate_verified_judgments(
        args.mapping,
        args.judgment,
    )
    _append_public_event_once(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def aggregate_verified_judgments(
    mapping_path: Path,
    judgment_paths: list[Path],
) -> dict:
    """Aggregate only against canonical journals and the active registry."""

    return _aggregate_verified_judgments(
        mapping_path,
        judgment_paths,
        quality_events_path=QUALITY_EVENTS_PATH,
        sustained_events_path=SUSTAINED_EVENTS_PATH,
        registry_path=SUSTAINED_REGISTRY_PATH,
    )


def _aggregate_verified_judgments(
    mapping_path: Path,
    judgment_paths: list[Path],
    *,
    quality_events_path: Path,
    sustained_events_path: Path,
    registry_path: Path,
) -> dict:
    resolved_mapping_path, _, mapping = _read_bounded_json_object(
        mapping_path,
        maximum_bytes=_MAX_MAPPING_BYTES,
        description="verified blind OCR mapping",
    )
    mapping = _validated_mapping(mapping)
    mapping_root = resolved_mapping_path.parent
    _, packet_bytes, packet = _read_bounded_json_object(
        mapping_root / "packet.json",
        maximum_bytes=_MAX_PACKET_BYTES,
        description="verified blind OCR packet",
    )
    if not hmac.compare_digest(
        hashlib.sha256(packet_bytes).hexdigest(),
        mapping["packet_fingerprint"],
    ):
        raise ValueError("verified blind OCR packet fingerprint does not match")
    packet_samples = _validated_packet(
        packet,
        preparation_id=mapping["preparation_id"],
        packet_root=mapping_root,
    )
    sample_mapping = _validated_mapping_samples(
        mapping["samples"],
        mapping["variant_ids"],
    )
    if set(packet_samples) != set(sample_mapping):
        raise ValueError("verified blind OCR packet and mapping samples differ")
    authority_snapshot = capture_verified_private_ocr_authority(
        sustained_events_path=sustained_events_path,
        registry_path=registry_path,
    )
    _verify_preparation_producers(
        mapping["preparation_producer_sha256"],
        registry_bytes=authority_snapshot.registry_bytes,
    )
    _verify_private_packet_commitment(mapping, packet)
    preparation_event = _load_verified_preparation_event(
        quality_events_path,
        mapping=mapping,
        packet=packet,
    )
    sources = _load_verified_sources(
        mapping_root,
        mapping=mapping,
        sustained_events_path=sustained_events_path,
        registry_path=registry_path,
        authority_snapshot=authority_snapshot,
    )
    record_status_counts = _verify_packet_reconstruction(
        sample_mapping,
        packet_samples,
        sources,
    )
    judgments = _load_distinct_judgments(
        judgment_paths,
        packet_fingerprint=mapping["packet_fingerprint"],
        sample_ids=set(sample_mapping),
    )
    registered_sources = _public_source_evidence(sources, record_status_counts)
    source_by_variant = {
        source["variant_id"]: source for source in registered_sources
    }
    metrics = _aggregate_metrics(
        sample_mapping,
        judgments,
        source_by_variant=source_by_variant,
        record_status_counts=record_status_counts,
    )
    producer_fingerprint = fingerprint_json(
        {
            "preparation": mapping["preparation_producer_sha256"],
            "aggregation": _aggregation_producer_sha256(),
        }
    )
    verify_private_ocr_authority_is_current(authority_snapshot)
    event = {
        "event": "blind_ocr_quality_scored",
        "protocol": AGGREGATE_PROTOCOL,
        "candidate_id": "private_course_blind_ocr_comparison",
        "workload_class": "private_course",
        "judgment_file_count": len(judgments),
        "preparation_public_event_sha256": preparation_event[
            "public_event_sha256"
        ],
        "source_candidates": [
            {
                key: source[key]
                for key in _PUBLIC_SOURCE_FIELDS
            }
            for source in registered_sources
        ],
        "producer_fingerprint": producer_fingerprint,
        "metrics": validate_public_summary(metrics),
        "interpretation": dict(_INTERPRETATION),
    }
    event["public_event_sha256"] = public_event_sha256(event)
    event["timestamp_utc"] = utc_timestamp_now()
    _validate_public_event(event)
    return _AuthorizedBlindOcrScoreEvent(
        event,
        authority_snapshot=authority_snapshot,
        append_ticket=_score_append_ticket(event, authority_snapshot),
    )


def _validated_mapping(mapping: object) -> dict:
    if (
        not isinstance(mapping, dict)
        or set(mapping) != _MAPPING_FIELDS
        or type(mapping.get("schema_version")) is not int
        or mapping["schema_version"] != 2
        or mapping.get("protocol") != PROTOCOL
    ):
        raise ValueError("unsupported verified blind OCR mapping protocol")
    preparation_id = mapping.get("preparation_id")
    try:
        canonical_preparation_id = str(uuid.UUID(preparation_id))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("verified blind OCR preparation ID is invalid") from error
    variant_ids = mapping.get("variant_ids")
    if (
        canonical_preparation_id != preparation_id
        or not isinstance(variant_ids, list)
        or len(variant_ids) != 2
        or any(
            type(variant_id) is not str
            or _PUBLIC_ID.fullmatch(variant_id) is None
            for variant_id in variant_ids
        )
        or len(set(variant_ids)) != 2
        or variant_ids != sorted(variant_ids)
        or type(mapping.get("packet_fingerprint")) is not str
        or _SHA256.fullmatch(mapping["packet_fingerprint"]) is None
    ):
        raise ValueError("verified blind OCR mapping identity is invalid")
    _validated_mapping_samples(mapping.get("samples"), variant_ids)
    _validated_source_bindings(mapping.get("source_bindings"), variant_ids)
    _validated_preparation_producers(mapping.get("preparation_producer_sha256"))
    commitment = mapping.get("private_packet_commitment")
    if (
        not isinstance(commitment, dict)
        or set(commitment) != {"scheme", "key_hex", "hmac_sha256"}
        or commitment.get("scheme") != COMMITMENT_SCHEME
        or type(commitment.get("key_hex")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", commitment["key_hex"]) is None
        or type(commitment.get("hmac_sha256")) is not str
        or _SHA256.fullmatch(commitment["hmac_sha256"]) is None
    ):
        raise ValueError("verified blind OCR private packet commitment is invalid")
    return mapping


def _validated_mapping_samples(value: object, variant_ids: list[str]) -> dict[str, dict]:
    if not isinstance(value, list) or not value or len(value) > 100:
        raise ValueError("verified blind OCR mapping samples are invalid")
    result = {}
    source_ids = set()
    for sample in value:
        if not isinstance(sample, dict) or set(sample) != _SAMPLE_FIELDS:
            raise ValueError("verified blind OCR mapping sample is invalid")
        sample_id = sample.get("sample_id")
        source_id = sample.get("source_id")
        identities = sample.get("identities")
        statuses = sample.get("record_statuses")
        image_sha256 = sample.get("image_sha256")
        if (
            type(sample_id) is not str
            or _BLIND_ID.fullmatch(sample_id) is None
            or sample_id in result
            or type(source_id) is not str
            or _SOURCE_ID.fullmatch(source_id) is None
            or source_id in source_ids
            or not isinstance(identities, dict)
            or set(identities) != {"A", "B"}
            or any(type(variant_id) is not str for variant_id in identities.values())
            or sorted(identities.values()) != sorted(variant_ids)
            or not isinstance(statuses, dict)
            or set(statuses) != {"A", "B"}
            or any(type(status) is not str for status in statuses.values())
            or not set(statuses.values()) <= _RECORD_STATUSES
            or type(image_sha256) is not str
            or _SHA256.fullmatch(image_sha256) is None
        ):
            raise ValueError("verified blind OCR mapping sample is invalid")
        source_ids.add(source_id)
        result[sample_id] = sample
    return result


def _validated_source_bindings(value: object, variant_ids: list[str]) -> dict[str, dict]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("verified blind OCR source bindings are invalid")
    result = {}
    attempt_ids = set()
    tags = set()
    for source in value:
        if not isinstance(source, dict) or set(source) != _SOURCE_BINDING_FIELDS:
            raise ValueError("verified blind OCR source binding is invalid")
        variant_id = source.get("variant_id")
        attempt_id = source.get("attempt_id")
        commitment = source.get("private_artifact_commitment")
        try:
            canonical_attempt_id = str(uuid.UUID(attempt_id))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("verified blind OCR source binding is invalid") from error
        if (
            variant_id not in variant_ids
            or variant_id in result
            or canonical_attempt_id != attempt_id
            or attempt_id in attempt_ids
            or not isinstance(commitment, dict)
            or set(commitment) != {"scheme", "hmac_sha256"}
            or commitment.get("scheme") != PRIVATE_RECORDS_COMMITMENT_SCHEME
            or type(commitment.get("hmac_sha256")) is not str
            or _SHA256.fullmatch(commitment["hmac_sha256"]) is None
            or commitment["hmac_sha256"] in tags
        ):
            raise ValueError("verified blind OCR source binding is invalid")
        result[variant_id] = source
        attempt_ids.add(attempt_id)
        tags.add(commitment["hmac_sha256"])
    if set(result) != set(variant_ids):
        raise ValueError("verified blind OCR source bindings are incomplete")
    return result


def _validated_packet(
    packet: object,
    *,
    preparation_id: str,
    packet_root: Path,
) -> dict[str, dict]:
    if (
        not isinstance(packet, dict)
        or set(packet) != _PACKET_FIELDS
        or type(packet.get("schema_version")) is not int
        or packet["schema_version"] != 2
        or packet.get("protocol") != PROTOCOL
        or packet.get("preparation_id") != preparation_id
        or packet.get("instructions") != _EXPECTED_INSTRUCTIONS
        or not isinstance(packet.get("samples"), list)
        or not packet["samples"]
        or len(packet["samples"]) > 100
    ):
        raise ValueError("unsupported verified blind OCR packet protocol")
    image_root = (packet_root / "image-snapshots").resolve(strict=True)
    result = {}
    total_characters = 0
    image_paths = set()
    for sample in packet["samples"]:
        if not isinstance(sample, dict) or set(sample) != _PACKET_SAMPLE_FIELDS:
            raise ValueError("verified blind OCR packet sample is invalid")
        sample_id = sample.get("sample_id")
        image_path = sample.get("image_path")
        options = sample.get("options")
        statuses = sample.get("option_statuses")
        try:
            resolved_image_path = Path(image_path).resolve(strict=True)
        except (OSError, TypeError) as error:
            raise ValueError("verified blind OCR packet image is unavailable") from error
        if (
            type(sample_id) is not str
            or _BLIND_ID.fullmatch(sample_id) is None
            or sample_id in result
            or type(image_path) is not str
            or not Path(image_path).is_absolute()
            or not resolved_image_path.is_file()
            or not resolved_image_path.is_relative_to(image_root)
            or resolved_image_path in image_paths
            or not isinstance(options, dict)
            or set(options) != {"A", "B"}
            or not isinstance(statuses, dict)
            or set(statuses) != {"A", "B"}
            or any(type(status) is not str for status in statuses.values())
            or not set(statuses.values()) <= _RECORD_STATUSES
        ):
            raise ValueError("verified blind OCR packet sample is invalid")
        for label, lines in options.items():
            if (
                not isinstance(lines, list)
                or len(lines) > 10_000
                or any(type(line) is not str or len(line) > 10_000 for line in lines)
                or (statuses[label] != "available" and lines)
            ):
                raise ValueError("verified blind OCR packet option is invalid")
            total_characters += sum(len(line) for line in lines)
            if total_characters > 2_000_000:
                raise ValueError("verified blind OCR packet text budget exceeded")
        image_paths.add(resolved_image_path)
        result[sample_id] = {
            **sample,
            "resolved_image_path": resolved_image_path,
        }
    return result


def _verify_private_packet_commitment(mapping: dict, packet: dict) -> None:
    commitment = mapping["private_packet_commitment"]
    expected = hmac.new(
        bytes.fromhex(commitment["key_hex"]),
        b"private-ocr-blind-packet-v10\0" + packet_commitment_payload(mapping, packet),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(commitment["hmac_sha256"], expected):
        raise ValueError("verified blind OCR private packet commitment does not match")


def _load_verified_preparation_event(
    path: Path,
    *,
    mapping: dict,
    packet: dict,
) -> dict:
    events = _read_quality_events(path)
    matches = [
        event
        for event in events
        if event.get("event") == "blind_ocr_packet_prepared"
        and event.get("preparation_id") == mapping["preparation_id"]
    ]
    if len(matches) != 1:
        raise ValueError("verified blind OCR requires exactly one public preparation event")
    event = validate_preparation_event(matches[0])
    expected = _preparation_event(mapping, packet)
    expected_body = {key: value for key, value in expected.items() if key != "timestamp_utc"}
    actual_body = {key: value for key, value in event.items() if key != "timestamp_utc"}
    if not _json_values_equal(actual_body, expected_body):
        raise ValueError("verified blind OCR public preparation anchor does not match")
    return event


def _load_verified_sources(
    mapping_root: Path,
    *,
    mapping: dict,
    sustained_events_path: Path,
    registry_path: Path,
    authority_snapshot: VerifiedPrivateOcrAuthoritySnapshot,
) -> dict[str, dict]:
    bindings = _validated_source_bindings(
        mapping["source_bindings"],
        mapping["variant_ids"],
    )
    sources = {}
    workload_fingerprint = None
    workload_summary = None
    identities = set()
    for variant_id in mapping["variant_ids"]:
        records_path = (
            mapping_root
            / "source-snapshots"
            / variant_id
            / "private-records.jsonl"
        )
        source = load_verified_private_ocr_source(
            records_path,
            expected_workload_fingerprint=workload_fingerprint,
            expected_workload_summary=workload_summary,
            sustained_events_path=sustained_events_path,
            registry_path=registry_path,
            authority_snapshot=authority_snapshot,
        )
        provenance = source["provenance"]
        binding = bindings[variant_id]
        private_commitment = provenance["private_records_commitment"]
        expected_binding = {
            "variant_id": variant_id,
            "attempt_id": provenance["attempt_id"],
            "private_artifact_commitment": {
                "scheme": private_commitment["scheme"],
                "hmac_sha256": private_commitment["hmac_sha256"],
            },
        }
        identity = (
            source["registered_source"]["candidate_id"],
            source["registered_source"]["config_index"],
        )
        if not _json_values_equal(binding, expected_binding) or identity in identities:
            raise ValueError("verified blind OCR source attribution does not match")
        identities.add(identity)
        sources[variant_id] = source
        if workload_fingerprint is None:
            workload_fingerprint = provenance["workload_fingerprint"]
            workload_summary = source["workload_summary"]
    if len(sources) != 2:
        raise ValueError("verified blind OCR requires two distinct sources")
    return sources


def _verify_packet_reconstruction(
    sample_mapping: dict[str, dict],
    packet_samples: dict[str, dict],
    sources: dict[str, dict],
) -> dict[str, dict[str, int]]:
    counts = {
        variant_id: {status: 0 for status in sorted(_RECORD_STATUSES)}
        for variant_id in sources
    }
    for sample_id, mapping_sample in sample_mapping.items():
        packet_sample = packet_samples[sample_id]
        if packet_sample["option_statuses"] != mapping_sample["record_statuses"]:
            raise ValueError("verified blind OCR packet status does not match mapping")
        for label, variant_id in mapping_sample["identities"].items():
            record = sources[variant_id]["records"].get(mapping_sample["source_id"])
            if record is None:
                status = "unavailable"
                lines = []
            elif record["success"]:
                status = "available"
                lines = [line["text"] for line in record["lines"]]
            else:
                status = "failed"
                lines = []
            if (
                mapping_sample["record_statuses"][label] != status
                or not _json_values_equal(packet_sample["options"][label], lines)
            ):
                raise ValueError("verified blind OCR packet does not reconstruct from sources")
            counts[variant_id][status] += 1
        _verify_private_image(
            packet_sample["resolved_image_path"],
            mapping_sample["image_sha256"],
        )
    for variant_id, source in sources.items():
        source_status = source["provenance"]["status"]
        if (
            source_status == "succeeded"
            and (counts[variant_id]["failed"] or counts[variant_id]["unavailable"])
        ) or (
            source_status == "all_failed" and counts[variant_id]["available"]
        ):
            raise ValueError("verified blind OCR selected statuses contradict source status")
    return counts


def _load_distinct_judgments(
    judgment_paths: list[Path],
    *,
    packet_fingerprint: str,
    sample_ids: set[str],
) -> list[dict[str, dict]]:
    resolved_paths = [path.resolve(strict=True) for path in judgment_paths]
    if (
        not 2 <= len(resolved_paths) <= 8
        or len(resolved_paths) != len(set(resolved_paths))
    ):
        raise ValueError("two to eight distinct verified judgment files are required")
    judgments = [
        _load_judgment(
            path,
            packet_fingerprint=packet_fingerprint,
            sample_ids=sample_ids,
        )
        for path in resolved_paths
    ]
    semantic_fingerprints = [
        _judgment_semantic_fingerprint(judgment) for judgment in judgments
    ]
    if len(semantic_fingerprints) != len(set(semantic_fingerprints)):
        raise ValueError("verified judgment files contain duplicate semantic votes")
    return judgments


def _load_judgment(
    path: Path,
    *,
    packet_fingerprint: str,
    sample_ids: set[str],
) -> dict[str, dict]:
    _, _, document = _read_bounded_json_object(
        path,
        maximum_bytes=_MAX_JUDGMENT_BYTES,
        description="verified blind OCR judgment",
    )
    if (
        set(document) != _JUDGMENT_FIELDS
        or type(document.get("schema_version")) is not int
        or document["schema_version"] != 2
        or document.get("protocol") != PROTOCOL
        or document.get("packet_fingerprint") != packet_fingerprint
        or not isinstance(document.get("samples"), list)
        or len(document["samples"]) > 100
    ):
        raise ValueError("unsupported verified blind OCR judgment protocol")
    result = {}
    for item in document["samples"]:
        if not isinstance(item, dict) or set(item) != _JUDGMENT_SAMPLE_FIELDS:
            raise ValueError("verified blind OCR judgment sample is invalid")
        sample_id = item.get("sample_id")
        if sample_id in result or sample_id not in sample_ids:
            raise ValueError("verified blind OCR judgment sample ID is invalid")
        if item.get("winner") not in {"A", "B", "tie"}:
            raise ValueError("verified blind OCR judgment winner is invalid")
        for label in ("a", "b"):
            severity = item.get(f"{label}_severity")
            codes = item.get(f"{label}_error_codes")
            if (
                type(severity) is not int
                or severity not in {0, 1, 2, 3}
                or type(item.get(f"{label}_usable")) is not bool
                or not isinstance(codes, list)
                or any(type(code) is not str for code in codes)
                or len(codes) != len(set(codes))
                or not set(codes) <= _ALLOWED_ERROR_CODES
            ):
                raise ValueError("verified blind OCR judgment rating is invalid")
        result[sample_id] = {
            "sample_id": sample_id,
            "winner": item["winner"],
            "a_severity": item["a_severity"],
            "b_severity": item["b_severity"],
            "a_usable": item["a_usable"],
            "b_usable": item["b_usable"],
            "a_error_codes": sorted(item["a_error_codes"]),
            "b_error_codes": sorted(item["b_error_codes"]),
        }
    if set(result) != sample_ids:
        raise ValueError("verified judgment does not cover every blind sample")
    return result


def _aggregate_metrics(
    sample_mapping: dict[str, dict],
    judgments: list[dict[str, dict]],
    *,
    source_by_variant: dict[str, dict],
    record_status_counts: dict[str, dict[str, int]],
) -> dict:
    variant_stats = {
        variant_id: {"wins": 0, "severity_sum": 0, "usable": 0, "votes": 0}
        for variant_id in source_by_variant
    }
    tie_votes = 0
    strict_majority_samples = 0
    unanimous_samples = 0
    pairwise_agreements = 0
    pairwise_comparisons = 0
    consensus_wins = Counter()
    consensus_tie_samples = 0
    no_strict_majority_samples = 0
    for sample_id, private_sample in sample_mapping.items():
        identities = private_sample["identities"]
        sample_votes = []
        for judgment in judgments:
            item = judgment[sample_id]
            winner = item["winner"]
            if winner == "tie":
                tie_votes += 1
                sample_votes.append("tie")
            else:
                variant_id = identities[winner]
                variant_stats[variant_id]["wins"] += 1
                sample_votes.append(variant_id)
            for label in ("A", "B"):
                variant_id = identities[label]
                lower_label = label.casefold()
                variant_stats[variant_id]["severity_sum"] += item[
                    f"{lower_label}_severity"
                ]
                variant_stats[variant_id]["usable"] += int(
                    item[f"{lower_label}_usable"]
                )
                variant_stats[variant_id]["votes"] += 1
        vote_counts = Counter(sample_votes)
        top_value, top_count = vote_counts.most_common(1)[0]
        pairwise_agreements += sum(
            count * (count - 1) // 2 for count in vote_counts.values()
        )
        pairwise_comparisons += len(judgments) * (len(judgments) - 1) // 2
        if top_count == len(judgments):
            unanimous_samples += 1
        if top_count > len(judgments) / 2:
            strict_majority_samples += 1
            if top_value == "tie":
                consensus_tie_samples += 1
            else:
                consensus_wins[top_value] += 1
        else:
            no_strict_majority_samples += 1
    aggregate_status_counts = {
        status: sum(counts[status] for counts in record_status_counts.values())
        for status in sorted(_RECORD_STATUSES)
    }
    fully_available_count = sum(
        all(status == "available" for status in sample["record_statuses"].values())
        for sample in sample_mapping.values()
    )
    sample_count = len(sample_mapping)
    judgment_count = len(judgments)
    return {
        "sample_count": sample_count,
        "judgment_file_count": judgment_count,
        "vote_count": sample_count * judgment_count,
        "tie_vote_count": tie_votes,
        "consensus_tie_sample_count": consensus_tie_samples,
        "no_strict_majority_sample_count": no_strict_majority_samples,
        "strict_majority_sample_fraction": strict_majority_samples / sample_count,
        "unanimous_sample_fraction": unanimous_samples / sample_count,
        "pairwise_winner_agreement_fraction": (
            pairwise_agreements / pairwise_comparisons
            if pairwise_comparisons
            else 0.0
        ),
        "comparison_sample_denominators": {
            "total_selected_sample_count": sample_count,
            "individual_winner_vote_denominator": sample_count * judgment_count,
            "consensus_winner_sample_denominator": sample_count,
            "strict_majority_sample_denominator": sample_count,
            "unanimous_sample_denominator": sample_count,
            "pairwise_winner_agreement_denominator": pairwise_comparisons,
            "fully_available_comparison_count": fully_available_count,
            "not_fully_available_comparison_count": sample_count - fully_available_count,
        },
        "source_record_availability": {
            "candidate_sample_count": sample_count * len(source_by_variant),
            "available_record_count": aggregate_status_counts["available"],
            "failed_record_count": aggregate_status_counts["failed"],
            "unavailable_record_count": aggregate_status_counts["unavailable"],
        },
        "candidates": [
            {
                "candidate_evidence_id": source_by_variant[variant_id][
                    "candidate_evidence_id"
                ],
                "win_votes": stats["wins"],
                "consensus_wins": consensus_wins[variant_id],
                "mean_error_severity": stats["severity_sum"] / stats["votes"],
                "usable_vote_fraction": stats["usable"] / stats["votes"],
                "mean_error_severity_vote_denominator": stats["votes"],
                "usable_vote_denominator": stats["votes"],
            }
            for variant_id, stats in sorted(
                variant_stats.items(),
                key=lambda item: source_by_variant[item[0]]["candidate_evidence_id"],
            )
        ],
    }


def _public_source_evidence(
    sources: dict[str, dict],
    record_status_counts: dict[str, dict[str, int]],
) -> list[dict]:
    ordered = sorted(
        sources.items(),
        key=lambda item: (
            item[1]["registered_source"]["candidate_id"],
            item[1]["registered_source"]["config_index"],
        ),
    )
    result = []
    for evidence_id, (variant_id, source) in enumerate(ordered, start=1):
        registered = source["registered_source"]
        counts = record_status_counts[variant_id]
        result.append(
            {
                "variant_id": variant_id,
                "candidate_evidence_id": evidence_id,
                "candidate_id": registered["candidate_id"],
                "config_index": registered["config_index"],
                "config_fingerprint": fingerprint_json(registered["config"]),
                "attempt_status": source["provenance"]["status"],
                "selected_available_record_count": counts["available"],
                "selected_failed_record_count": counts["failed"],
                "selected_unavailable_record_count": counts["unavailable"],
            }
        )
    return result


def _validate_public_event(event: object) -> None:
    validate_score_event(event)


def _append_public_event_once(path: Path, event: dict) -> bool:
    if (
        not isinstance(event, _AuthorizedBlindOcrScoreEvent)
        or type(getattr(event, "append_ticket", None)) is not str
        or not hmac.compare_digest(
            event.append_ticket,
            _score_append_ticket(event, event.authority_snapshot),
        )
    ):
        raise ValueError("verified blind OCR score was not authorized by aggregation")
    with locked_file_bytes(event.authority_snapshot.registry_path) as registry_bytes:
        with locked_journal_bytes(
            event.authority_snapshot.sustained_events_path
        ) as sustained_events_bytes:
            _validate_public_event(event)
            preparation_matches = [
                candidate
                for candidate in _read_quality_events(path)
                if candidate.get("event") == "blind_ocr_packet_prepared"
                and candidate.get("public_event_sha256")
                == event["preparation_public_event_sha256"]
            ]
            if len(preparation_matches) != 1:
                raise ValueError("verified blind OCR score has no unique preparation anchor")
            validate_preparation_event(preparation_matches[0])
            validate_public_ocr_score_sources(
                event.authority_snapshot,
                event["source_candidates"],
                sustained_events_bytes=sustained_events_bytes,
                registry_bytes=registry_bytes,
            )
            return append_event_once(
                path,
                event,
                identity_fields=(
                    "event",
                    "protocol",
                    "preparation_public_event_sha256",
                ),
            )


def _score_append_ticket(
    event: dict,
    authority_snapshot: VerifiedPrivateOcrAuthoritySnapshot,
) -> str:
    payload = {
        "event": event,
        "registry_sha256": hashlib.sha256(authority_snapshot.registry_bytes).hexdigest(),
        "sustained_events_sha256": authority_snapshot.sustained_events_sha256,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    return hmac.new(_APPEND_TICKET_KEY, canonical, hashlib.sha256).hexdigest()


def _verify_private_image(path: Path, expected_sha256: str) -> None:
    try:
        with path.open("rb") as handle:
            image_bytes = handle.read(_MAX_IMAGE_BYTES + 1)
    except OSError as error:
        raise ValueError("verified blind OCR image snapshot is unavailable") from error
    if (
        len(image_bytes) > _MAX_IMAGE_BYTES
        or not hmac.compare_digest(
            hashlib.sha256(image_bytes).hexdigest(),
            expected_sha256,
        )
    ):
        raise ValueError("verified blind OCR image snapshot changed")


def _judgment_semantic_fingerprint(judgment: dict[str, dict]) -> str:
    canonical = json.dumps(
        {
            "protocol": PROTOCOL,
            "samples": [judgment[sample_id] for sample_id in sorted(judgment)],
        },
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"private-ocr-blind-v10-judgment\0" + canonical).hexdigest()


def _validated_preparation_producers(value: object) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != set(PREPARATION_PRODUCER_PATHS)
        or any(
            type(digest) is not str or _SHA256.fullmatch(digest) is None
            for digest in value.values()
        )
    ):
        raise ValueError("verified blind OCR preparation producer hashes are invalid")
    return value


def _verify_preparation_producers(
    value: dict[str, str],
    *,
    registry_bytes: bytes | None = None,
) -> None:
    expected = _validated_preparation_producers(value)
    current = {
        relative_path: _sha256(PROJECT_ROOT / relative_path)
        for relative_path in PREPARATION_PRODUCER_PATHS
    }
    if registry_bytes is not None:
        current["registries/sustained_candidates.json"] = hashlib.sha256(
            registry_bytes
        ).hexdigest()
    if not _json_values_equal(expected, current):
        raise ValueError("verified blind OCR preparation producers changed")


def _aggregation_producer_sha256() -> dict[str, str]:
    return {
        relative_path: _sha256(PROJECT_ROOT / relative_path)
        for relative_path in AGGREGATION_PRODUCER_PATHS
    }


def _read_quality_events(path: Path) -> list[dict]:
    try:
        raw = read_journal_bytes(path)
    except OSError as error:
        raise ValueError("verified blind OCR quality journal is unavailable") from error
    if len(raw) > _MAX_QUALITY_JOURNAL_BYTES:
        raise ValueError("verified blind OCR quality journal byte budget exceeded")
    if raw and not raw.endswith(b"\n"):
        raise ValueError("verified blind OCR quality journal is incomplete")
    events = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"invalid quality journal line {line_number}")
        try:
            event = _decode_json_object(line, description="quality journal event")
        except ValueError as error:
            raise ValueError(f"invalid quality journal line {line_number}") from error
        events.append(event)
    return events


def _read_bounded_json_object(
    path: Path,
    *,
    maximum_bytes: int,
    description: str,
) -> tuple[Path, bytes, dict]:
    try:
        resolved = path.resolve(strict=True)
        with resolved.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
    except OSError as error:
        raise ValueError(f"{description} is unavailable") from error
    if len(raw) > maximum_bytes:
        raise ValueError(f"{description} byte budget exceeded")
    return resolved, raw, _decode_json_object(raw, description=description)


def _decode_json_object(raw: bytes, *, description: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{description} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} is invalid")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant is invalid: {value}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is invalid: {value}")
    return parsed


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
