"""Aggregate ignored blinded OCR judgments into privacy-safe evidence."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from local_inference_bench.event_journal import append_event
from local_inference_bench.validate_public_summary import validate_public_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--judgment", required=True, action="append", type=Path)
    parser.add_argument("--append-journal", type=Path)
    args = parser.parse_args()
    event = aggregate_judgments(args.mapping, args.judgment)
    if args.append_journal is not None:
        append_event(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def aggregate_judgments(mapping_path: Path, judgment_paths: list[Path]) -> dict:
    resolved_mapping_path = mapping_path.resolve(strict=True)
    mapping = json.loads(resolved_mapping_path.read_text(encoding="utf-8"))
    if (
        not isinstance(mapping, dict)
        or set(mapping) != _MAPPING_FIELDS
        or mapping.get("schema_version") != 1
        or mapping.get("protocol") != _MAPPING_PROTOCOL
    ):
        raise ValueError("unsupported blind mapping protocol")
    variant_ids = _validated_variant_ids(mapping.get("variant_ids"))
    sample_mapping = _validated_sample_mapping(
        mapping.get("samples"),
        variant_ids,
    )
    source_attempts = _validated_source_attempts(
        mapping.get("source_attempts"),
        variant_ids,
    )
    preparation_producer_sha256 = _validated_producer_sha256(
        mapping.get("preparation_producer_sha256"),
        expected_paths=_PREPARATION_PRODUCER_PATHS,
        label="preparation",
    )
    packet_fingerprint = mapping.get("packet_fingerprint")
    mapping_commitment = mapping.get("mapping_commitment")
    mapping_commitment_nonce = mapping.get("mapping_commitment_nonce")
    if (
        type(packet_fingerprint) is not str
        or re.fullmatch(r"[0-9a-f]{64}", packet_fingerprint) is None
    ):
        raise ValueError("blind mapping packet fingerprint is invalid")
    if (
        type(mapping_commitment) is not str
        or re.fullmatch(r"[0-9a-f]{64}", mapping_commitment) is None
        or type(mapping_commitment_nonce) is not str
        or re.fullmatch(r"[0-9a-f]{64}", mapping_commitment_nonce) is None
    ):
        raise ValueError("blind mapping commitment is invalid")
    expected_commitment = _mapping_commitment(
        variant_ids,
        mapping["samples"],
        source_attempts,
        preparation_producer_sha256,
        mapping_commitment_nonce,
    )
    if not hmac.compare_digest(mapping_commitment, expected_commitment):
        raise ValueError("blind mapping commitment does not match")

    packet_path = resolved_mapping_path.with_name("packet.json").resolve(strict=True)
    if not hmac.compare_digest(_sha256(packet_path), packet_fingerprint):
        raise ValueError("blind packet fingerprint does not match")
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet_samples = _validated_packet(packet, mapping_commitment)
    if set(packet_samples) != set(sample_mapping):
        raise ValueError("blind packet and mapping samples do not match")
    for sample_id, private_sample in sample_mapping.items():
        packet_sample = packet_samples[sample_id]
        if packet_sample["option_statuses"] != private_sample["record_statuses"]:
            raise ValueError("blind packet and mapping record statuses do not match")
        _verify_private_image(
            packet_sample["image_path"],
            private_sample["image_sha256"],
        )

    resolved_judgment_paths = [path.resolve(strict=True) for path in judgment_paths]
    if (
        not 2 <= len(resolved_judgment_paths) <= 8
        or len(resolved_judgment_paths) != len(set(resolved_judgment_paths))
    ):
        raise ValueError("two to eight distinct judgment files are required")
    judgments = [
        _load_judgment(path, packet_fingerprint, set(sample_mapping))
        for path in resolved_judgment_paths
    ]
    semantic_fingerprints = [
        _judgment_semantic_fingerprint(judgment) for judgment in judgments
    ]
    if len(semantic_fingerprints) != len(set(semantic_fingerprints)):
        raise ValueError("judgment files contain duplicate semantic votes")

    variant_stats = {
        variant_id: {"wins": 0, "severity_sum": 0, "usable": 0, "votes": 0}
        for variant_id in variant_ids
    }
    tie_votes = 0
    strict_majority_samples = 0
    unanimous_samples = 0
    pairwise_agreements = 0
    pairwise_comparisons = 0
    consensus_wins = Counter()
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
                variant_stats[variant_id]["severity_sum"] += item[
                    f"{label.casefold()}_severity"
                ]
                variant_stats[variant_id]["usable"] += int(
                    item[f"{label.casefold()}_usable"]
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
            if top_value != "tie":
                consensus_wins[top_value] += 1

    record_status_counts = {
        variant_id: {status: 0 for status in sorted(_RECORD_STATUSES)}
        for variant_id in variant_ids
    }
    for private_sample in sample_mapping.values():
        for label, variant_id in private_sample["identities"].items():
            record_status_counts[variant_id][
                private_sample["record_statuses"][label]
            ] += 1
    _validate_selected_record_status_counts(source_attempts, record_status_counts)
    aggregate_record_status_counts = {
        status: sum(counts[status] for counts in record_status_counts.values())
        for status in sorted(_RECORD_STATUSES)
    }
    fully_available_comparison_count = sum(
        all(status == "available" for status in sample["record_statuses"].values())
        for sample in sample_mapping.values()
    )
    metrics = {
        "sample_count": len(sample_mapping),
        "judge_count": len(judgments),
        "vote_count": len(sample_mapping) * len(judgments),
        "tie_vote_count": tie_votes,
        "strict_majority_sample_fraction": (
            strict_majority_samples / len(sample_mapping)
        ),
        "unanimous_sample_fraction": unanimous_samples / len(sample_mapping),
        "pairwise_winner_agreement_fraction": (
            pairwise_agreements / pairwise_comparisons
            if pairwise_comparisons
            else 0.0
        ),
        "comparison_sample_denominators": {
            "total_selected_sample_count": len(sample_mapping),
            "individual_winner_vote_denominator": (
                len(sample_mapping) * len(judgments)
            ),
            "consensus_winner_sample_denominator": len(sample_mapping),
            "strict_majority_sample_denominator": len(sample_mapping),
            "unanimous_sample_denominator": len(sample_mapping),
            "pairwise_winner_agreement_denominator": pairwise_comparisons,
            "fully_available_comparison_count": fully_available_comparison_count,
            "not_fully_available_comparison_count": (
                len(sample_mapping) - fully_available_comparison_count
            ),
        },
        "source_record_availability": {
            "variant_sample_count": len(sample_mapping) * len(variant_ids),
            "available_record_count": aggregate_record_status_counts["available"],
            "failed_record_count": aggregate_record_status_counts["failed"],
            "unavailable_record_count": aggregate_record_status_counts[
                "unavailable"
            ],
        },
        "variants": {
            variant_id: {
                "win_votes": stats["wins"],
                "consensus_wins": consensus_wins[variant_id],
                "mean_error_severity": stats["severity_sum"] / stats["votes"],
                "usable_vote_fraction": stats["usable"] / stats["votes"],
                "mean_error_severity_vote_denominator": stats["votes"],
                "usable_vote_denominator": stats["votes"],
            }
            for variant_id, stats in variant_stats.items()
        },
    }
    return {
        "event": "blind_ocr_quality_scored",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "private_course_blind_ocr_comparison",
        "protocol": _AGGREGATE_PROTOCOL,
        "workload_class": "private_course",
        "judge_count": len(judgments),
        "source_variants": [
            {
                "variant_id": source["variant_id"],
                "candidate_id": source["candidate_id"],
                "config_index": source["config_index"],
                "config_fingerprint": source["config_fingerprint"],
                "attempt_status": source["attempt_status"],
                "selected_available_record_count": record_status_counts[
                    source["variant_id"]
                ]["available"],
                "selected_failed_record_count": record_status_counts[
                    source["variant_id"]
                ]["failed"],
                "selected_unavailable_record_count": record_status_counts[
                    source["variant_id"]
                ]["unavailable"],
            }
            for source in source_attempts
        ],
        "producer_sha256": _combined_producer_sha256(
            preparation_producer_sha256
        ),
        "metrics": validate_public_summary(metrics),
        "interpretation": {
            "blind_judgment_is_not_ground_truth": True,
            "procedural_blinding_only": True,
            "judge_independence_verified": False,
            "mapping_commitment_verified": True,
            "private_fingerprints_published": False,
            "private_run_identifiers_published": False,
            "raw_text_or_images_published": False,
            "private_image_hashes_published": False,
            "failed_or_unavailable_options_disclosed_to_judges": True,
            "failed_or_unavailable_record_counts_published": True,
            "winner_metrics_filter_to_fully_available_comparisons": False,
            "fully_available_comparisons_reported_separately": True,
            "semantic_duplicate_guard": True,
        },
    }


def _load_judgment(
    path: Path,
    expected_fingerprint: str,
    expected_samples: set[str],
) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or set(document) != _JUDGMENT_FIELDS
        or document.get("schema_version") != 1
        or document.get("protocol") != _MAPPING_PROTOCOL
    ):
        raise ValueError("unsupported judgment protocol")
    fingerprint = document.get("packet_fingerprint")
    if (
        type(fingerprint) is not str
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        or not hmac.compare_digest(fingerprint, expected_fingerprint)
    ):
        raise ValueError("judgment packet fingerprint does not match")
    result = {}
    raw_samples = document.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) > 100:
        raise ValueError("judgment samples are invalid")
    for item in raw_samples:
        if not isinstance(item, dict) or set(item) != _JUDGMENT_SAMPLE_FIELDS:
            raise ValueError("judgment sample is invalid")
        sample_id = item.get("sample_id")
        if sample_id in result or sample_id not in expected_samples:
            raise ValueError("judgment contains an invalid sample ID")
        if item.get("winner") not in {"A", "B", "tie"}:
            raise ValueError("judgment winner must be A, B, or tie")
        for label in ("a", "b"):
            severity = item.get(f"{label}_severity")
            if type(severity) is not int or severity not in {0, 1, 2, 3}:
                raise ValueError("judgment severity must be in [0, 3]")
            if type(item.get(f"{label}_usable")) is not bool:
                raise ValueError("judgment usability must be boolean")
            codes = item.get(f"{label}_error_codes")
            if (
                not isinstance(codes, list)
                or any(type(code) is not str for code in codes)
                or len(codes) != len(set(codes))
                or not set(codes) <= _ALLOWED_ERROR_CODES
            ):
                raise ValueError("judgment contains an unsupported error code")
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
    if set(result) != expected_samples:
        raise ValueError("judgment does not cover every blind sample")
    return result


def _validated_variant_ids(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            type(variant_id) is not str
            or _PUBLIC_ID.fullmatch(variant_id) is None
            for variant_id in value
        )
        or len(set(value)) != 2
    ):
        raise ValueError("blind mapping variant IDs are invalid")
    return list(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_packet(packet: object, expected_commitment: str) -> dict[str, dict]:
    if (
        not isinstance(packet, dict)
        or set(packet) != _PACKET_FIELDS
        or packet.get("schema_version") != 1
        or packet.get("protocol") != _MAPPING_PROTOCOL
        or packet.get("instructions") != _EXPECTED_INSTRUCTIONS
    ):
        raise ValueError("unsupported blind packet protocol")
    commitment = packet.get("mapping_commitment")
    if (
        type(commitment) is not str
        or not hmac.compare_digest(commitment, expected_commitment)
    ):
        raise ValueError("blind packet mapping commitment does not match")
    raw_samples = packet.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples or len(raw_samples) > 100:
        raise ValueError("blind packet samples are invalid")
    samples_by_id = {}
    total_characters = 0
    for sample in raw_samples:
        if not isinstance(sample, dict) or set(sample) != _PACKET_SAMPLE_FIELDS:
            raise ValueError("blind packet sample is invalid")
        sample_id = sample.get("sample_id")
        image_path = sample.get("image_path")
        options = sample.get("options")
        if (
            type(sample_id) is not str
            or _BLIND_SAMPLE_ID.fullmatch(sample_id) is None
            or sample_id in samples_by_id
            or type(image_path) is not str
            or not image_path
            or len(image_path) > 4096
            or not Path(image_path).is_absolute()
            or not isinstance(options, dict)
            or set(options) != {"A", "B"}
            or not isinstance(sample.get("option_statuses"), dict)
            or set(sample["option_statuses"]) != {"A", "B"}
            or any(
                type(status) is not str
                for status in sample["option_statuses"].values()
            )
            or not set(sample["option_statuses"].values()) <= _RECORD_STATUSES
        ):
            raise ValueError("blind packet sample is invalid")
        for lines in options.values():
            if (
                not isinstance(lines, list)
                or len(lines) > 10_000
                or any(type(line) is not str or len(line) > 10_000 for line in lines)
            ):
                raise ValueError("blind packet sample is invalid")
            total_characters += sum(len(line) for line in lines)
            if total_characters > 2_000_000:
                raise ValueError("blind packet text budget exceeded")
        if any(
            sample["option_statuses"][label] != "available" and options[label]
            for label in ("A", "B")
        ):
            raise ValueError("failed or unavailable blind options must be empty")
        samples_by_id[sample_id] = {
            "image_path": image_path,
            "option_statuses": sample["option_statuses"],
        }
    return samples_by_id


def _verify_private_image(image_path: str, expected_sha256: str) -> None:
    path = Path(image_path)
    try:
        resolved_path = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError("blind comparison image is unavailable") from error
    if not resolved_path.is_file() or not hmac.compare_digest(
        _sha256(resolved_path),
        expected_sha256,
    ):
        raise ValueError("blind comparison image changed after packet preparation")


def _mapping_commitment(
    variant_ids: list[str],
    samples: list[dict],
    source_attempts: list[dict],
    preparation_producer_sha256: dict[str, str],
    nonce: str,
) -> str:
    payload = {
        "protocol": _MAPPING_PROTOCOL,
        "variant_ids": sorted(variant_ids),
        "samples": sorted(
            (
                {
                    "sample_id": sample["sample_id"],
                    "source_id": sample["source_id"],
                    "identities": {
                        "A": sample["identities"]["A"],
                        "B": sample["identities"]["B"],
                    },
                    "record_statuses": {
                        "A": sample["record_statuses"]["A"],
                        "B": sample["record_statuses"]["B"],
                    },
                    "image_sha256": sample["image_sha256"],
                }
                for sample in samples
            ),
            key=lambda sample: sample["sample_id"],
        ),
        "source_attempts": sorted(
            (
                {
                    "variant_id": source["variant_id"],
                    "candidate_id": source["candidate_id"],
                    "attempt_id": source["attempt_id"],
                    "attempt_key": source["attempt_key"],
                    "config_index": source["config_index"],
                    "config_fingerprint": source["config_fingerprint"],
                    "trial_index": source["trial_index"],
                    "attempt_status": source["attempt_status"],
                }
                for source in source_attempts
            ),
            key=lambda source: source["variant_id"],
        ),
        "preparation_producer_sha256": {
            key: preparation_producer_sha256[key]
            for key in sorted(preparation_producer_sha256)
        },
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        bytes.fromhex(nonce),
        b"private-ocr-blind-v6-mapping\0" + canonical,
        hashlib.sha256,
    ).hexdigest()


def _judgment_semantic_fingerprint(judgment: dict[str, dict]) -> str:
    canonical = json.dumps(
        {
            "protocol": _MAPPING_PROTOCOL,
            "samples": [judgment[sample_id] for sample_id in sorted(judgment)],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(
        b"private-ocr-blind-v6-judgment\0" + canonical
    ).hexdigest()


def _validated_sample_mapping(value: object, variant_ids: list[str]) -> dict:
    if not isinstance(value, list) or not value or len(value) > 100:
        raise ValueError("blind mapping samples are invalid")
    result = {}
    source_ids = set()
    for sample in value:
        if not isinstance(sample, dict) or set(sample) != {
            "sample_id",
            "source_id",
            "identities",
            "record_statuses",
            "image_sha256",
        }:
            raise ValueError("blind mapping sample is invalid")
        sample_id = sample.get("sample_id")
        source_id = sample.get("source_id")
        identities = sample.get("identities")
        record_statuses = sample.get("record_statuses")
        image_sha256 = sample.get("image_sha256")
        if (
            type(sample_id) is not str
            or _BLIND_SAMPLE_ID.fullmatch(sample_id) is None
            or sample_id in result
            or type(source_id) is not str
            or _SOURCE_SAMPLE_ID.fullmatch(source_id) is None
            or source_id in source_ids
            or not isinstance(identities, dict)
            or set(identities) != {"A", "B"}
            or any(type(variant_id) is not str for variant_id in identities.values())
            or sorted(identities.values()) != sorted(variant_ids)
            or not isinstance(record_statuses, dict)
            or set(record_statuses) != {"A", "B"}
            or any(type(status) is not str for status in record_statuses.values())
            or not set(record_statuses.values()) <= _RECORD_STATUSES
            or type(image_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", image_sha256) is None
        ):
            raise ValueError("blind mapping sample is invalid")
        source_ids.add(source_id)
        result[sample_id] = {
            "identities": identities,
            "record_statuses": record_statuses,
            "image_sha256": image_sha256,
        }
    return result


def _validated_source_attempts(value: object, variant_ids: list[str]) -> list[dict]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("blind mapping source attempts are invalid")
    by_variant = {}
    attempt_ids = set()
    attempt_keys = set()
    for source in value:
        if not isinstance(source, dict) or set(source) != {
            "variant_id",
            "candidate_id",
            "attempt_id",
            "attempt_key",
            "config_index",
            "config_fingerprint",
            "trial_index",
            "attempt_status",
        }:
            raise ValueError("blind mapping source attempts are invalid")
        variant_id = source.get("variant_id")
        candidate_id = source.get("candidate_id")
        attempt_id = source.get("attempt_id")
        attempt_key = source.get("attempt_key")
        try:
            uuid.UUID(attempt_id)
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("blind mapping source attempts are invalid") from error
        if (
            variant_id not in variant_ids
            or variant_id in by_variant
            or type(candidate_id) is not str
            or _PUBLIC_ID.fullmatch(candidate_id) is None
            or type(attempt_key) is not str
            or re.fullmatch(r"[0-9a-f]{16}", attempt_key) is None
            or attempt_id in attempt_ids
            or attempt_key in attempt_keys
            or type(source.get("config_index")) is not int
            or source["config_index"] < 0
            or type(source.get("config_fingerprint")) is not str
            or re.fullmatch(r"[0-9a-f]{16}", source["config_fingerprint"]) is None
            or type(source.get("trial_index")) is not int
            or source["trial_index"] < 0
            or type(source.get("attempt_status")) is not str
            or source["attempt_status"] not in _SOURCE_ATTEMPT_STATUSES
        ):
            raise ValueError("blind mapping source attempts are invalid")
        attempt_ids.add(attempt_id)
        attempt_keys.add(attempt_key)
        by_variant[variant_id] = source
    if set(by_variant) != set(variant_ids):
        raise ValueError("blind mapping source attempts are invalid")
    return [by_variant[variant_id] for variant_id in sorted(by_variant)]


def _validate_selected_record_status_counts(
    source_attempts: list[dict],
    record_status_counts: dict[str, dict[str, int]],
) -> None:
    """Reject committed selected statuses impossible for a source attempt."""

    for source in source_attempts:
        counts = record_status_counts[source["variant_id"]]
        if source["attempt_status"] == "succeeded" and (
            counts["failed"] or counts["unavailable"]
        ):
            raise ValueError(
                "blind mapping selected records contradict attempt status"
            )
        if source["attempt_status"] == "all_failed" and counts["available"]:
            raise ValueError(
                "blind mapping selected records contradict attempt status"
            )


def _validated_producer_sha256(
    value: object,
    *,
    expected_paths: tuple[str, ...],
    label: str,
) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != set(expected_paths)
        or any(
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in value.values()
        )
    ):
        raise ValueError(f"blind {label} producer hashes are invalid")
    return {key: value[key] for key in sorted(value)}


def _aggregation_producer_sha256() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        relative_path: _sha256(project_root / relative_path)
        for relative_path in _AGGREGATION_PRODUCER_PATHS
    }


def _combined_producer_sha256(
    preparation_producer_sha256: dict[str, str],
) -> dict[str, str]:
    aggregation_producer_sha256 = _aggregation_producer_sha256()
    if set(preparation_producer_sha256) & set(aggregation_producer_sha256):
        raise ValueError("blind producer hash roles overlap")
    return {
        **preparation_producer_sha256,
        **aggregation_producer_sha256,
    }


_ALLOWED_ERROR_CODES = {
    "false_positive",
    "formula_or_code",
    "garble",
    "missing_text",
    "reading_order",
    "small_text",
}

_PUBLIC_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_BLIND_SAMPLE_ID = re.compile(r"^blind_[0-9]{3}$")
_SOURCE_SAMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAPPING_PROTOCOL = "private-ocr-blind-v6"
_AGGREGATE_PROTOCOL = "private-ocr-blind-v8"
_PREPARATION_PRODUCER_PATHS = (
    "scripts/prepare_blind_ocr_comparison.py",
    "src/local_inference_bench/fingerprint.py",
    "src/local_inference_bench/load_sustained_workload.py",
)
_AGGREGATION_PRODUCER_PATHS = (
    "scripts/aggregate_blind_ocr_judgments.py",
    "src/local_inference_bench/validate_public_summary.py",
)
_PACKET_FIELDS = {
    "schema_version",
    "protocol",
    "mapping_commitment",
    "instructions",
    "samples",
}
_PACKET_SAMPLE_FIELDS = {
    "sample_id",
    "image_path",
    "options",
    "option_statuses",
}
_MAPPING_FIELDS = {
    "schema_version",
    "protocol",
    "variant_ids",
    "samples",
    "source_attempts",
    "preparation_producer_sha256",
    "mapping_commitment",
    "mapping_commitment_nonce",
    "packet_fingerprint",
}
_JUDGMENT_FIELDS = {
    "schema_version",
    "protocol",
    "packet_fingerprint",
    "samples",
}
_JUDGMENT_SAMPLE_FIELDS = {
    "sample_id",
    "winner",
    "a_severity",
    "b_severity",
    "a_usable",
    "b_usable",
    "a_error_codes",
    "b_error_codes",
}
_EXPECTED_INSTRUCTIONS = {
    "winner_values": ["A", "B", "tie"],
    "severity_scale": [0, 1, 2, 3],
    "allowed_error_codes": sorted(_ALLOWED_ERROR_CODES),
    "option_status_values": ["available", "failed", "unavailable"],
}
_RECORD_STATUSES = {"available", "failed", "unavailable"}
_SOURCE_ATTEMPT_STATUSES = {"succeeded", "partial_failure", "all_failed"}


if __name__ == "__main__":
    main()
