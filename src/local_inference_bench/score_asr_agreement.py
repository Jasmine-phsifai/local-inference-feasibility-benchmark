"""Compare private ASR outputs when no trusted transcript exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

from .event_journal import append_event
from .fingerprint import fingerprint_files, fingerprint_json
from .load_sustained_workload import load_sustained_workload
from .validate_public_summary import validate_public_summary


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_CANDIDATES = 8
_MAX_WORKLOAD_ITEMS = 4_096
_MAX_WORKLOAD_MANIFEST_BYTES = 1_048_576
_MAX_RECORDS_PER_CANDIDATE = 4_096
_MAX_RECORD_FILE_BYTES = 16 * 1_048_576
_MAX_PROVENANCE_BYTES = 65_536
_MAX_PREDICTION_CHARACTERS = 200_000
_MAX_TOTAL_PREDICTION_CHARACTERS = 2_000_000
_MAX_TOTAL_EDIT_CELLS = 5_000_000
_MINIMUM_EXACT_AGGREGATE_DENOMINATOR = 10
_CHARACTER_COUNT_BUCKET_UPPER_BOUNDS = (0, 32, 128, 512, 2_048)
_FRACTION_BUCKET_LOWER_BOUNDS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")
_MIXED_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z]+(?:-[a-z]+)*|[0-9]+(?:\.[0-9]+)?",
    re.IGNORECASE,
)


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
        append_event(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def score_asr_agreement(
    *,
    workload_path: Path,
    candidate_record_paths: dict[str, Path],
) -> dict:
    """Return privacy-bounded agreement evidence without treating it as truth."""

    _validate_candidate_specs(candidate_record_paths)
    workload_document = _read_bounded_json_object(
        workload_path,
        maximum_bytes=_MAX_WORKLOAD_MANIFEST_BYTES,
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

    workload = load_sustained_workload(workload_path, expected_task="asr")
    items = _load_items(workload["items"])
    small_private_cohort = len(items) < _MINIMUM_EXACT_AGGREGATE_DENOMINATOR

    records: dict[int, dict[str, dict]] = {}
    provenances: dict[int, dict] = {}
    resolved_record_paths: set[Path] = set()
    resolved_provenance_paths: set[Path] = set()
    attempt_ids: set[str] = set()
    attempt_keys: set[str] = set()
    sorted_specs = sorted(candidate_record_paths.items())
    for candidate_evidence_id, (alias, path) in enumerate(sorted_specs, start=1):
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
        provenance = _verify_records_provenance(
            resolved_path,
            provenance_path=provenance_path,
            workload_fingerprint=workload["fingerprint"],
        )
        attempt_id = provenance["attempt_id"]
        if attempt_id in attempt_ids:
            raise ValueError("ASR agreement candidates require distinct attempts")
        attempt_ids.add(attempt_id)
        attempt_key = provenance["attempt_key"]
        if attempt_key in attempt_keys:
            raise ValueError("ASR agreement candidates require distinct attempt keys")
        attempt_keys.add(attempt_key)

        candidate_records = _read_records(resolved_path, set(items))
        _validate_source_status(
            provenance["status"],
            expected_ids=set(items),
            records=candidate_records,
        )
        provenances[candidate_evidence_id] = provenance
        records[candidate_evidence_id] = candidate_records

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
    return {
        "event": "asr_agreement_scored",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "private_course_asr_agreement",
        "protocol": "asr-text-agreement-v8",
        "scorer_fingerprint": _scorer_fingerprint(),
        "workload_class": "private_course",
        "source_candidates": [
            {
                "candidate_evidence_id": candidate_evidence_id,
                "candidate_id": provenances[candidate_evidence_id]["candidate_id"],
                "status": provenances[candidate_evidence_id]["status"],
                "config_index": provenances[candidate_evidence_id]["config_index"],
                "config_fingerprint": fingerprint_json(
                    provenances[candidate_evidence_id]["config"]
                ),
            }
            for candidate_evidence_id in sorted(provenances)
        ],
        "privacy": {
            "small_private_cohort_threshold": (
                _MINIMUM_EXACT_AGGREGATE_DENOMINATOR
            ),
            "minimum_exact_aggregate_denominator": (
                _MINIMUM_EXACT_AGGREGATE_DENOMINATOR
            ),
            "exact_character_aggregates_suppressed": small_private_cohort,
            "character_count_bucket_upper_bounds": [
                *_CHARACTER_COUNT_BUCKET_UPPER_BOUNDS,
                None,
            ],
            "fraction_bucket_lower_bounds": list(_FRACTION_BUCKET_LOWER_BOUNDS),
            "private_fingerprints_published": False,
            "private_run_identifiers_published": False,
        },
        "interpretation": {
            "agreement_is_not_ground_truth": True,
            "trusted_gold_used": False,
            "failed_outputs_excluded_from_success_metrics": True,
            "text_only": True,
            "timestamps_compared": False,
        },
        "metrics": metrics,
    }


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


def _read_records(path: Path, expected_ids: set[str]) -> dict[str, dict]:
    if path.stat().st_size > _MAX_RECORD_FILE_BYTES:
        raise ValueError("ASR agreement record file byte budget exceeded")
    records = {}
    total_prediction_characters = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if len(records) >= _MAX_RECORDS_PER_CANDIDATE:
                raise ValueError("ASR agreement record count budget exceeded")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid ASR agreement record at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"invalid ASR agreement record at line {line_number}"
                )
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
    provenance_path: Path | None = None,
    workload_fingerprint: str,
) -> dict:
    if records_path.stat().st_size > _MAX_RECORD_FILE_BYTES:
        raise ValueError("ASR agreement record file byte budget exceeded")
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
        provenance.get("schema_version") != 1
        or provenance.get("protocol") != "sustained-process-v1"
        or provenance.get("status") not in _SOURCE_STATUSES
        or provenance.get("task") != "asr"
        or provenance.get("phase") != "quality"
        or provenance.get("workload_class") != "private_course"
        or provenance.get("workload_fingerprint") != workload_fingerprint
        or provenance.get("records_sha256") != _sha256(records_path)
        or type(candidate_id) is not str
        or _CANDIDATE_ID.fullmatch(candidate_id) is None
        or not isinstance(provenance.get("config"), dict)
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
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"{description} byte budget exceeded")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} is invalid")
    return value


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scorer_fingerprint() -> str:
    """Bind the event to all repository-owned code used to produce it."""

    return fingerprint_files(_scorer_dependency_paths())


def _scorer_dependency_paths(module_path: Path | None = None) -> list[Path]:
    if module_path is None:
        module_path = Path(__file__).resolve()
    return [
        module_path,
        module_path.with_name("event_journal.py"),
        module_path.with_name("fingerprint.py"),
        module_path.with_name("load_sustained_workload.py"),
        module_path.with_name("validate_public_summary.py"),
    ]


if __name__ == "__main__":
    main()
