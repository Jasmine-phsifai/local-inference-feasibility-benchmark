"""Score ignored ASR predictions and append aggregate-only quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .event_journal import append_event
from .fingerprint import fingerprint_files, fingerprint_json
from .load_sustained_workload import load_sustained_workload
from .validate_public_summary import validate_public_summary


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_CATEGORY_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_PREDICTION_CHARACTERS = 200_000
_MAX_TOTAL_PREDICTION_CHARACTERS = 500_000
_MAX_REFERENCE_CHARACTERS = 200_000
_MAX_TOTAL_REFERENCE_CHARACTERS = 500_000
_MAX_REQUIRED_TERMS = 256
_MAX_ALIASES_PER_TERM = 16
_MAX_ALIAS_CHARACTERS = 256
_MAX_TOTAL_ALIAS_CHARACTERS = 100_000
_MAX_SPEECH_INTERVALS = 10_000
_MAX_TOTAL_EDIT_CELLS = 5_000_000
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")
_MIXED_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z]+(?:-[a-z]+)*|[0-9]+(?:\.[0-9]+)?",
    re.IGNORECASE,
)
_REFERENCE_FIELDS = {
    "audio_sha256",
    "category",
    "expected_speech",
    "required_terms",
    "speech_intervals",
    "transcript",
}
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--append-journal", type=Path)
    args = parser.parse_args()

    event = score_asr_quality(
        manifest_path=args.manifest,
        records_path=args.records,
        candidate_id=args.candidate,
    )
    if args.append_journal is not None:
        append_event(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def score_asr_quality(
    *,
    manifest_path: Path,
    records_path: Path,
    candidate_id: str,
) -> dict:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("ASR quality manifest is invalid") from error
    if manifest.get("task") != "asr" or not isinstance(manifest.get("references"), dict):
        raise ValueError("ASR quality manifest requires a references mapping")
    if manifest.get("workload_class") != "generated_quality_control":
        raise ValueError("ASR quality scoring accepts generated controls only")
    if _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("ASR quality candidate ID must be a public identifier")
    workload = load_sustained_workload(manifest_path, expected_task="asr")
    provenance = _verify_records_provenance(
        records_path,
        candidate_id=candidate_id,
        workload_fingerprint=workload["fingerprint"],
    )
    records = _read_records(records_path)
    references = manifest["references"]
    items = {item["id"]: item for item in workload["items"]}
    if set(references) != set(items):
        raise ValueError("ASR quality references must match workload items")
    unknown = set(records) - set(references)
    if unknown:
        raise ValueError("ASR records contain sample IDs outside the manifest")
    _validate_source_status(
        provenance["status"],
        expected_ids=set(references),
        records=records,
    )

    sample_scores = []
    edit_budget = [_MAX_TOTAL_EDIT_CELLS]
    reference_character_budget = [_MAX_TOTAL_REFERENCE_CHARACTERS]
    alias_character_budget = [_MAX_TOTAL_ALIAS_CHARACTERS]
    for sample_id, reference in references.items():
        reference = _validated_reference(
            reference,
            media_path=Path(items[sample_id]["path"]),
            duration_seconds=float(items[sample_id]["duration_seconds"]),
            expected_speech=bool(items[sample_id]["expected_speech"]),
            reference_character_budget=reference_character_budget,
            alias_character_budget=alias_character_budget,
        )
        record = records.get(sample_id)
        missing_record = record is None
        success = record is not None and record["success"]
        explicit_failed_record = record is not None and not record["success"]
        prediction = record["prediction"] if success else ""
        sample_scores.append(
            _score_sample(
                category=reference["category"],
                reference_text=reference["transcript"],
                prediction=prediction,
                required_terms=reference.get("required_terms", []),
                expected_speech=items[sample_id]["expected_speech"],
                duration_seconds=float(items[sample_id]["duration_seconds"]),
                speech_intervals=reference.get("speech_intervals", []),
                segments=record.get("segments") if success else None,
                failed=not success,
                explicit_failed_record=explicit_failed_record,
                missing_record=missing_record,
                edit_budget=edit_budget,
            )
        )

    metrics = validate_public_summary(_aggregate_scores(sample_scores))
    return {
        "event": "asr_quality_scored",
        "scorer_protocol": "asr-quality-v6",
        "scorer_fingerprint": _scorer_fingerprint(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "workload_class": manifest.get(
            "workload_class",
            "generated_quality_control",
        ),
        "dataset_fingerprint": _sha256(manifest_path),
        "workload_fingerprint": workload["fingerprint"],
        "records_fingerprint": _sha256(records_path),
        "source_attempt": {
            "status": provenance["status"],
            "candidate_id": provenance["candidate_id"],
            "attempt_id": provenance["attempt_id"],
            "attempt_key": provenance["attempt_key"],
            "config_fingerprint": fingerprint_json(provenance["config"]),
            "config_index": provenance["config_index"],
            "trial_index": provenance["trial_index"],
            "code_fingerprint": provenance["code_fingerprint"],
            "environment_fingerprint": provenance["environment_fingerprint"],
            "controller_environment_fingerprint": provenance[
                "controller_environment_fingerprint"
            ],
            "execution_policy_fingerprint": provenance[
                "execution_policy_fingerprint"
            ],
        },
        "metrics": metrics,
    }


def _validated_reference(
    reference: object,
    *,
    media_path: Path,
    duration_seconds: float,
    expected_speech: bool,
    reference_character_budget: list[int],
    alias_character_budget: list[int],
) -> dict:
    if not isinstance(reference, dict) or set(reference) != _REFERENCE_FIELDS:
        raise ValueError("ASR quality reference must use the exact schema")
    category = reference.get("category")
    transcript = reference.get("transcript")
    required_terms = reference["required_terms"]
    speech_intervals = reference["speech_intervals"]
    audio_sha256 = reference["audio_sha256"]
    if (
        type(audio_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", audio_sha256) is None
        or _sha256(media_path) != audio_sha256
    ):
        raise ValueError("ASR quality reference media hash is invalid")
    if not isinstance(category, str) or _CATEGORY_ID.fullmatch(category) is None:
        raise ValueError("ASR quality reference category is invalid")
    if not isinstance(transcript, str) or len(transcript) > _MAX_REFERENCE_CHARACTERS:
        raise ValueError("ASR quality reference transcript is invalid")
    reference_character_budget[0] -= len(transcript)
    if reference_character_budget[0] < 0:
        raise ValueError("ASR quality reference transcript budget exceeded")
    declared_speech = reference["expected_speech"]
    if type(declared_speech) is not bool or declared_speech is not expected_speech:
        raise ValueError("ASR quality reference speech expectation is invalid")
    if not isinstance(required_terms, list) or len(required_terms) > _MAX_REQUIRED_TERMS:
        raise ValueError("ASR quality required terms are invalid")
    validated_terms = []
    for term in required_terms:
        if not isinstance(term, dict) or set(term) != {"aliases"}:
            raise ValueError("ASR quality required term is invalid")
        aliases = term["aliases"]
        if (
            not isinstance(aliases, list)
            or not aliases
            or len(aliases) > _MAX_ALIASES_PER_TERM
            or any(
                not isinstance(alias, str)
                or not alias
                or len(alias) > _MAX_ALIAS_CHARACTERS
                for alias in aliases
            )
        ):
            raise ValueError("ASR quality required-term aliases are invalid")
        alias_character_budget[0] -= sum(len(alias) for alias in aliases)
        if alias_character_budget[0] < 0:
            raise ValueError("ASR quality alias budget exceeded")
        validated_terms.append({"aliases": list(aliases)})
    if not isinstance(speech_intervals, list) or len(speech_intervals) > _MAX_SPEECH_INTERVALS:
        raise ValueError("ASR quality speech intervals are invalid")
    validated_intervals = []
    previous_end = 0.0
    for interval in speech_intervals:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in interval
            )
        ):
            raise ValueError("ASR quality speech interval is invalid")
        start, end = map(float, interval)
        if not 0 <= start <= end <= duration_seconds or start < previous_end:
            raise ValueError("ASR quality speech interval is invalid")
        previous_end = end
        validated_intervals.append([start, end])
    return {
        "category": category,
        "transcript": transcript,
        "required_terms": validated_terms,
        "speech_intervals": validated_intervals,
    }


def _read_records(path: Path) -> dict[str, dict]:
    records = {}
    total_prediction_characters = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, RecursionError) as error:
                raise ValueError(
                    f"invalid ASR quality record at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"invalid ASR quality record at line {line_number}")
            sample_id = record.get("sample_id")
            success = record.get("success")
            prediction = record.get("prediction")
            if (
                not isinstance(sample_id, str)
                or sample_id in records
                or type(success) is not bool
                or ("prediction" in record and type(prediction) is not str)
                or (success and type(prediction) is not str)
                or (
                    isinstance(prediction, str)
                    and len(prediction) > _MAX_PREDICTION_CHARACTERS
                )
                or (
                    "segments" in record
                    and not isinstance(record.get("segments"), list)
                )
                or len(record.get("segments", [])) > 10_000
            ):
                raise ValueError(f"invalid ASR quality record at line {line_number}")
            total_prediction_characters += len(prediction or "")
            if total_prediction_characters > _MAX_TOTAL_PREDICTION_CHARACTERS:
                raise ValueError("ASR quality prediction budget exceeded")
            records[sample_id] = record
    return records


def _verify_records_provenance(
    records_path: Path,
    *,
    candidate_id: str,
    workload_fingerprint: str,
) -> dict:
    provenance_path = records_path.with_name("records-provenance.json")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("ASR quality records provenance is invalid") from error
    if (
        provenance.get("schema_version") != 1
        or provenance.get("protocol") != "sustained-process-v1"
        or provenance.get("status") not in _SOURCE_STATUSES
        or provenance.get("candidate_id") != candidate_id
        or provenance.get("task") != "asr"
        or provenance.get("phase") != "quality"
        or provenance.get("workload_class") != "generated_quality_control"
        or provenance.get("workload_fingerprint") != workload_fingerprint
        or provenance.get("records_sha256") != _sha256(records_path)
        or not isinstance(provenance.get("config"), dict)
    ):
        raise ValueError("ASR quality records provenance is invalid")
    try:
        uuid.UUID(provenance.get("attempt_id", ""))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("ASR quality records provenance is invalid") from error
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
            raise ValueError("ASR quality records provenance is invalid")
    for key in ("config_index", "trial_index"):
        if type(provenance.get(key)) is not int or provenance[key] < 0:
            raise ValueError("ASR quality records provenance is invalid")
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
        raise ValueError("ASR quality source status does not match records")


def _scorer_fingerprint() -> str:
    module_path = Path(__file__).resolve()
    return fingerprint_files(
        [
            module_path,
            module_path.with_name("fingerprint.py"),
            module_path.with_name("load_sustained_workload.py"),
            module_path.with_name("validate_public_summary.py"),
        ]
    )


def _score_sample(
    *,
    category: str,
    reference_text: str,
    prediction: str,
    required_terms: list,
    expected_speech: bool,
    duration_seconds: float,
    speech_intervals: list,
    segments: object,
    failed: bool,
    explicit_failed_record: bool,
    missing_record: bool,
    edit_budget: list[int],
) -> dict:
    prediction = _SENSEVOICE_TAG.sub("", prediction)
    reference_tokens = _mixed_tokens(reference_text)
    prediction_tokens = _mixed_tokens(prediction)
    reference_characters = _normalized_characters(reference_text)
    prediction_characters = _normalized_characters(prediction)
    term_hits = 0
    for term in required_terms:
        aliases = term.get("aliases", []) if isinstance(term, dict) else [term]
        term_hits += any(
            _contains_mixed_token_sequence(prediction_tokens, str(alias))
            for alias in aliases
            if _mixed_tokens(str(alias))
        )
    reference_repetition = _repeated_ngram_ratio(reference_tokens, 3)
    prediction_repetition = _repeated_ngram_ratio(prediction_tokens, 3)
    score = {
        "category": category,
        "reference_tokens": len(reference_tokens),
        "token_edit_distance": _bounded_levenshtein(
            reference_tokens,
            prediction_tokens,
            edit_budget,
        ),
        "reference_characters": len(reference_characters),
        "character_edit_distance": _bounded_levenshtein(
            reference_characters,
            prediction_characters,
            edit_budget,
        ),
        "required_terms": len(required_terms),
        "term_hits": term_hits,
        "silence_characters": len(prediction_characters) if not expected_speech else 0,
        "silence_seconds": duration_seconds if not expected_speech else 0.0,
        "reference_repeated_trigram_ratio": reference_repetition,
        "prediction_repeated_trigram_ratio": prediction_repetition,
        "excess_repeated_trigram_ratio": max(
            0.0,
            prediction_repetition - reference_repetition,
        ),
        "failed": failed,
        "explicit_failed_record": explicit_failed_record,
        "missing_record": missing_record,
    }
    score.update(
        _score_timestamps(
            reference_intervals=speech_intervals,
            predicted_segments=segments,
            duration_seconds=duration_seconds,
        )
    )
    return score


def _aggregate_scores(scores: list[dict]) -> dict:
    categories: dict[str, list[dict]] = {}
    for score in scores:
        categories.setdefault(score["category"], []).append(score)
    return {
        "overall": _aggregate_group(scores),
        "categories": {
            category: _aggregate_group(category_scores)
            for category, category_scores in sorted(categories.items())
        },
    }


def _aggregate_group(scores: list[dict]) -> dict:
    reference_tokens = sum(score["reference_tokens"] for score in scores)
    token_edits = sum(score["token_edit_distance"] for score in scores)
    reference_characters = sum(score["reference_characters"] for score in scores)
    character_edits = sum(score["character_edit_distance"] for score in scores)
    required_terms = sum(score["required_terms"] for score in scores)
    term_hits = sum(score["term_hits"] for score in scores)
    silence_minutes = sum(score["silence_seconds"] for score in scores) / 60.0
    timestamp_samples = sum(score["timestamp_available"] for score in scores)
    timestamp_reference_seconds = sum(
        score["timestamp_reference_seconds"]
        for score in scores
        if score["timestamp_available"]
    )
    timestamp_predicted_seconds = sum(
        score["timestamp_predicted_seconds"]
        for score in scores
        if score["timestamp_available"]
    )
    timestamp_overlap_seconds = sum(
        score["timestamp_overlap_seconds"]
        for score in scores
        if score["timestamp_available"]
    )
    failure_count = sum(score["failed"] for score in scores)
    explicit_failed_record_count = sum(
        score["explicit_failed_record"] for score in scores
    )
    missing_record_count = sum(score["missing_record"] for score in scores)
    successful_sample_count = len(scores) - failure_count
    return {
        "sample_count": len(scores),
        "failure_count": failure_count,
        "availability": {
            "sample_denominator": len(scores),
            "successful_sample_count": successful_sample_count,
            "unavailable_sample_count": failure_count,
            "explicit_failed_record_count": explicit_failed_record_count,
            "missing_record_count": missing_record_count,
        },
        "quality_denominators": {
            "mixed_token_error_denominator": reference_tokens,
            "normalized_character_error_denominator": reference_characters,
            "required_term_recall_denominator": required_terms,
            "silence_minutes_denominator": silence_minutes,
            "repetition_sample_denominator": len(scores),
            "timestamp_availability_sample_denominator": len(scores),
            "timestamp_recall_seconds_denominator": timestamp_reference_seconds,
            "timestamp_precision_seconds_denominator": timestamp_predicted_seconds,
        },
        "mixed_token_error_rate": token_edits / reference_tokens if reference_tokens else 0.0,
        "normalized_character_error_rate": (
            character_edits / reference_characters if reference_characters else 0.0
        ),
        "required_term_recall": term_hits / required_terms if required_terms else 1.0,
        "silence_false_positive_characters_per_minute": (
            sum(score["silence_characters"] for score in scores) / silence_minutes
            if silence_minutes
            else 0.0
        ),
        "mean_expected_repeated_trigram_ratio": (
            sum(score["reference_repeated_trigram_ratio"] for score in scores)
            / len(scores)
            if scores
            else 0.0
        ),
        "mean_observed_repeated_trigram_ratio": (
            sum(score["prediction_repeated_trigram_ratio"] for score in scores)
            / len(scores)
            if scores
            else 0.0
        ),
        "mean_excess_repeated_trigram_ratio": (
            sum(score["excess_repeated_trigram_ratio"] for score in scores)
            / len(scores)
            if scores
            else 0.0
        ),
        "timestamp_metrics_available_fraction": (
            timestamp_samples / len(scores) if scores else 0.0
        ),
        "timestamp_speech_recall_when_available": (
            timestamp_overlap_seconds / timestamp_reference_seconds
            if timestamp_reference_seconds
            else 0.0
        ),
        "timestamp_speech_precision_when_available": (
            timestamp_overlap_seconds / timestamp_predicted_seconds
            if timestamp_predicted_seconds
            else 0.0
        ),
        "timestamp_invalid_segment_count": sum(
            score["timestamp_invalid_segments"] for score in scores
        ),
        "timestamp_nonmonotonic_segment_count": sum(
            score["timestamp_nonmonotonic_segments"] for score in scores
        ),
    }


def _score_timestamps(
    *,
    reference_intervals: list,
    predicted_segments: object,
    duration_seconds: float,
) -> dict:
    reference = _normalize_intervals(reference_intervals, duration_seconds)
    if not isinstance(predicted_segments, list):
        return {
            "timestamp_available": 0,
            "timestamp_reference_seconds": _interval_duration(reference),
            "timestamp_predicted_seconds": 0.0,
            "timestamp_overlap_seconds": 0.0,
            "timestamp_invalid_segments": 0,
            "timestamp_nonmonotonic_segments": 0,
        }
    predicted = []
    invalid = 0
    nonmonotonic = 0
    previous_end = 0.0
    for segment in predicted_segments:
        if not isinstance(segment, dict):
            invalid += 1
            continue
        start = segment.get("start")
        end = segment.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or start < 0
            or end < start
            or end > duration_seconds + 0.1
        ):
            invalid += 1
            continue
        if start + 1e-6 < previous_end:
            nonmonotonic += 1
        previous_end = max(previous_end, float(end))
        predicted.append([max(0.0, float(start)), min(duration_seconds, float(end))])
    predicted = _merge_intervals(predicted)
    return {
        "timestamp_available": 1,
        "timestamp_reference_seconds": _interval_duration(reference),
        "timestamp_predicted_seconds": _interval_duration(predicted),
        "timestamp_overlap_seconds": _overlap_duration(reference, predicted),
        "timestamp_invalid_segments": invalid,
        "timestamp_nonmonotonic_segments": nonmonotonic,
    }


def _normalize_intervals(intervals: object, duration_seconds: float) -> list[list[float]]:
    normalized = []
    if not isinstance(intervals, list):
        return normalized
    for interval in intervals:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or not all(
                not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in interval
            )
        ):
            continue
        start, end = map(float, interval)
        if 0 <= start <= end <= duration_seconds:
            normalized.append([start, end])
    return _merge_intervals(normalized)


def _merge_intervals(intervals: list[list[float]]) -> list[list[float]]:
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _interval_duration(intervals: list[list[float]]) -> float:
    return sum(end - start for start, end in intervals)


def _overlap_duration(left: list[list[float]], right: list[list[float]]) -> float:
    overlap = 0.0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _mixed_tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return _MIXED_TOKEN.findall(normalized)


def _contains_mixed_token_sequence(tokens: list[str], alias: str) -> bool:
    """Match a term on token boundaries, including contiguous CJK characters."""
    alias_tokens = _mixed_tokens(alias)
    if not alias_tokens or len(alias_tokens) > len(tokens):
        return False
    width = len(alias_tokens)
    return any(
        tokens[index : index + width] == alias_tokens
        for index in range(len(tokens) - width + 1)
    )


def _normalized_characters(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _repeated_ngram_ratio(tokens: list[str], width: int) -> float:
    if len(tokens) < width:
        return 0.0
    ngrams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return repeated / len(ngrams)


def _levenshtein(left, right) -> int:
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


def _bounded_levenshtein(left, right, edit_budget: list[int]) -> int:
    if left == right or not left or not right:
        return _levenshtein(left, right)
    cells = len(left) * len(right)
    if cells > edit_budget[0]:
        raise ValueError("ASR quality edit-distance budget exceeded")
    edit_budget[0] -= cells
    return _levenshtein(left, right)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
