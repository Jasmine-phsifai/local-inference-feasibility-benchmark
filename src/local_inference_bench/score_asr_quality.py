"""Score ignored ASR predictions and append aggregate-only quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .event_journal import append_event
from .validate_public_summary import validate_public_summary


_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")
_MIXED_TOKEN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z]+(?:-[a-z]+)*|[0-9]+(?:\.[0-9]+)?",
    re.IGNORECASE,
)


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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "asr" or not isinstance(manifest.get("references"), dict):
        raise ValueError("ASR quality manifest requires a references mapping")
    records = _read_records(records_path)
    references = manifest["references"]
    items = {item["id"]: item for item in manifest.get("items", [])}
    unknown = set(records) - set(references)
    if unknown:
        raise ValueError("ASR records contain sample IDs outside the manifest")

    sample_scores = []
    for sample_id, reference in references.items():
        record = records.get(sample_id)
        prediction = "" if record is None else str(record.get("prediction", ""))
        sample_scores.append(
            _score_sample(
                category=reference["category"],
                reference_text=str(reference.get("transcript", "")),
                prediction=prediction,
                required_terms=reference.get("required_terms", []),
                expected_speech=bool(reference.get("expected_speech", True)),
                duration_seconds=float(items[sample_id]["duration_seconds"]),
                speech_intervals=reference.get("speech_intervals", []),
                segments=[] if record is None else record.get("segments"),
                failed=record is None or not record.get("success", False),
            )
        )

    metrics = validate_public_summary(_aggregate_scores(sample_scores))
    return {
        "event": "asr_quality_scored",
        "scorer_protocol": "asr-quality-v3",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "workload_class": manifest.get(
            "workload_class",
            "generated_quality_control",
        ),
        "dataset_fingerprint": _sha256(manifest_path),
        "records_fingerprint": _sha256(records_path),
        "metrics": metrics,
    }


def _read_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in records:
                raise ValueError(f"invalid or duplicate sample ID at line {line_number}")
            records[sample_id] = record
    return records


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
        "token_edit_distance": _levenshtein(reference_tokens, prediction_tokens),
        "reference_characters": len(reference_characters),
        "character_edit_distance": _levenshtein(
            reference_characters,
            prediction_characters,
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
    return {
        "sample_count": len(scores),
        "failure_count": sum(score["failed"] for score in scores),
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
