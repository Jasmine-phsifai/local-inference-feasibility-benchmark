"""Score ignored OCR predictions and append aggregate-only quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .event_journal import append_event
from .validate_public_summary import validate_public_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--append-journal", type=Path)
    args = parser.parse_args()

    event = score_ocr_quality(
        manifest_path=args.manifest,
        records_path=args.records,
        candidate_id=args.candidate,
    )
    if args.append_journal is not None:
        append_event(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def score_ocr_quality(
    *,
    manifest_path: Path,
    records_path: Path,
    candidate_id: str,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("task") != "ocr" or not isinstance(manifest.get("references"), dict):
        raise ValueError("OCR quality manifest requires a references mapping")
    records = _read_records(records_path)
    references = manifest["references"]
    unknown = set(records) - set(references)
    if unknown:
        raise ValueError("OCR records contain sample IDs outside the manifest")

    sample_scores = []
    for sample_id, reference in references.items():
        record = records.get(sample_id)
        lines = [] if record is None else record.get("lines", [])
        predicted_lines = [str(line.get("text", "")) for line in lines]
        reference_lines = reference["lines"]
        sample_scores.append(
            _score_sample(
                category=reference["category"],
                reference_lines=reference_lines,
                required_tokens=reference.get("required_tokens", []),
                predicted_lines=predicted_lines,
                confidences=[
                    line.get("confidence")
                    for line in lines
                    if isinstance(line.get("confidence"), (int, float))
                ],
                failed=record is None or not record.get("success", False),
            )
        )

    metrics = validate_public_summary(_aggregate_scores(sample_scores))
    return {
        "event": "ocr_quality_scored",
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
    reference_lines: list[str],
    required_tokens: list[str],
    predicted_lines: list[str],
    confidences: list[float],
    failed: bool,
) -> dict:
    reference = _normalize("\n".join(reference_lines))
    prediction = _normalize("\n".join(predicted_lines))
    token_hits = sum(
        _normalize(token) in prediction
        for token in required_tokens
        if _normalize(token)
    )
    return {
        "category": category,
        "reference_characters": len(reference),
        "edit_distance": _levenshtein(reference, prediction),
        "required_tokens": len(required_tokens),
        "token_hits": token_hits,
        "line_count_error": abs(len(reference_lines) - len(predicted_lines)),
        "confidence_sum": sum(float(value) for value in confidences),
        "confidence_count": len(confidences),
        "failed": failed,
    }


def _aggregate_scores(sample_scores: list[dict]) -> dict:
    categories: dict[str, list[dict]] = {}
    for score in sample_scores:
        categories.setdefault(score["category"], []).append(score)
    return {
        "overall": _aggregate_group(sample_scores),
        "categories": {
            category: _aggregate_group(scores)
            for category, scores in sorted(categories.items())
        },
    }


def _aggregate_group(scores: list[dict]) -> dict:
    reference_characters = sum(score["reference_characters"] for score in scores)
    edit_distance = sum(score["edit_distance"] for score in scores)
    required_tokens = sum(score["required_tokens"] for score in scores)
    token_hits = sum(score["token_hits"] for score in scores)
    confidence_count = sum(score["confidence_count"] for score in scores)
    confidence_sum = sum(score["confidence_sum"] for score in scores)
    return {
        "sample_count": len(scores),
        "failure_count": sum(score["failed"] for score in scores),
        "false_positive_characters": sum(
            score["edit_distance"]
            for score in scores
            if score["reference_characters"] == 0
        ),
        "normalized_character_error_rate": (
            edit_distance / reference_characters if reference_characters else 0.0
        ),
        "required_token_recall": (
            token_hits / required_tokens if required_tokens else 1.0
        ),
        "mean_absolute_line_count_error": (
            sum(score["line_count_error"] for score in scores) / len(scores)
            if scores
            else 0.0
        ),
        "mean_reported_confidence": (
            confidence_sum / confidence_count if confidence_count else 0.0
        ),
    }


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
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
