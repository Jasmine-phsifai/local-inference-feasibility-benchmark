"""Score ignored OCR predictions and append aggregate-only quality evidence."""

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
_MAX_LINE_CHARACTERS = 10_000
_MAX_REFERENCE_CHARACTERS = 500_000
_MAX_TOTAL_PREDICTION_CHARACTERS = 1_000_000
_MAX_TOTAL_EDIT_CELLS = 5_000_000
_REFERENCE_FIELDS = {
    "category",
    "image_sha256",
    "lines",
    "required_tokens",
}
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})


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
    if manifest.get("workload_class") != "generated_quality_control":
        raise ValueError("OCR quality scoring accepts generated controls only")
    if _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("OCR quality candidate ID must be a public identifier")
    workload = load_sustained_workload(manifest_path, expected_task="ocr")
    provenance = _verify_records_provenance(
        records_path,
        candidate_id=candidate_id,
        workload_fingerprint=workload["fingerprint"],
    )
    records = _read_records(records_path)
    references = manifest["references"]
    items = {item["id"]: item for item in workload["items"]}
    if set(references) != set(items):
        raise ValueError("OCR quality references must match workload items")
    unknown = set(records) - set(references)
    if unknown:
        raise ValueError("OCR records contain sample IDs outside the manifest")
    _validate_source_status(
        provenance["status"],
        expected_ids=set(references),
        records=records,
    )

    sample_scores = []
    edit_budget = [_MAX_TOTAL_EDIT_CELLS]
    reference_characters = 0
    for sample_id, reference in references.items():
        reference_lines = reference.get("lines") if isinstance(reference, dict) else None
        required_tokens = (
            reference.get("required_tokens", [])
            if isinstance(reference, dict)
            else None
        )
        if (
            not isinstance(reference, dict)
            or set(reference) != _REFERENCE_FIELDS
            or _CATEGORY_ID.fullmatch(reference.get("category", "")) is None
            or type(reference.get("image_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", reference["image_sha256"]) is None
            or _sha256(Path(items[sample_id]["path"]))
            != reference["image_sha256"]
            or not isinstance(reference_lines, list)
            or len(reference_lines) > 10_000
            or any(
                type(line) is not str or len(line) > _MAX_LINE_CHARACTERS
                for line in reference_lines
            )
            or not isinstance(required_tokens, list)
            or len(required_tokens) > 10_000
            or any(
                type(token) is not str or len(token) > _MAX_LINE_CHARACTERS
                for token in required_tokens
            )
        ):
            raise ValueError("OCR quality reference is invalid")
        reference_characters += sum(len(line) for line in reference_lines)
        if reference_characters > _MAX_REFERENCE_CHARACTERS:
            raise ValueError("OCR quality reference budget exceeded")
        record = records.get(sample_id)
        missing_record = record is None
        success = record is not None and record["success"]
        explicit_failed_record = record is not None and not record["success"]
        lines = record["lines"] if success else []
        predicted_lines = [line["text"] for line in lines]
        sample_scores.append(
            _score_sample(
                category=reference["category"],
                reference_lines=reference_lines,
                required_tokens=required_tokens,
                predicted_lines=predicted_lines,
                confidences=[
                    line.get("confidence")
                    for line in lines
                    if type(line.get("confidence")) in {int, float}
                ],
                failed=not success,
                explicit_failed_record=explicit_failed_record,
                missing_record=missing_record,
                edit_budget=edit_budget,
            )
        )

    metrics = validate_public_summary(_aggregate_scores(sample_scores))
    return {
        "event": "ocr_quality_scored",
        "scorer_protocol": "ocr-quality-v5",
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


def _read_records(path: Path) -> dict[str, dict]:
    records = {}
    total_characters = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"invalid OCR quality record at line {line_number}")
            sample_id = record.get("sample_id")
            success = record.get("success")
            lines = record.get("lines")
            if (
                not isinstance(sample_id, str)
                or sample_id in records
                or type(success) is not bool
                or (success and not isinstance(lines, list))
                or (lines is not None and not isinstance(lines, list))
                or len(lines or []) > 10_000
            ):
                raise ValueError(f"invalid OCR quality record at line {line_number}")
            for output_line in lines or []:
                if not isinstance(output_line, dict):
                    raise ValueError(f"invalid OCR quality record at line {line_number}")
                text = output_line.get("text")
                confidence = output_line.get("confidence")
                if (
                    type(text) is not str
                    or len(text) > _MAX_LINE_CHARACTERS
                    or (
                        confidence is not None
                        and (
                            type(confidence) not in {int, float}
                            or not math.isfinite(float(confidence))
                            or not 0.0 <= float(confidence) <= 1.0
                        )
                    )
                ):
                    raise ValueError(f"invalid OCR quality record at line {line_number}")
                total_characters += len(text)
                if total_characters > _MAX_TOTAL_PREDICTION_CHARACTERS:
                    raise ValueError("OCR quality prediction budget exceeded")
            records[sample_id] = record
    return records


def _verify_records_provenance(
    records_path: Path,
    *,
    candidate_id: str,
    workload_fingerprint: str,
) -> dict:
    provenance = json.loads(
        records_path.with_name("records-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        provenance.get("schema_version") != 1
        or provenance.get("protocol") != "sustained-process-v1"
        or provenance.get("status") not in _SOURCE_STATUSES
        or provenance.get("candidate_id") != candidate_id
        or provenance.get("task") != "ocr"
        or provenance.get("phase") != "quality"
        or provenance.get("workload_class") != "generated_quality_control"
        or provenance.get("workload_fingerprint") != workload_fingerprint
        or provenance.get("records_sha256") != _sha256(records_path)
        or not isinstance(provenance.get("config"), dict)
    ):
        raise ValueError("OCR quality records provenance is invalid")
    try:
        uuid.UUID(provenance.get("attempt_id", ""))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("OCR quality records provenance is invalid") from error
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
            raise ValueError("OCR quality records provenance is invalid")
    for key in ("config_index", "trial_index"):
        if type(provenance.get(key)) is not int or provenance[key] < 0:
            raise ValueError("OCR quality records provenance is invalid")
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
        raise ValueError("OCR quality source status does not match records")


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
    reference_lines: list[str],
    required_tokens: list[str],
    predicted_lines: list[str],
    confidences: list[float],
    failed: bool,
    explicit_failed_record: bool,
    missing_record: bool,
    edit_budget: list[int],
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
        "edit_distance": _bounded_levenshtein(reference, prediction, edit_budget),
        "required_tokens": len(required_tokens),
        "token_hits": token_hits,
        "line_count_error": abs(len(reference_lines) - len(predicted_lines)),
        "successful_output_line_count": len(predicted_lines),
        "confidence_sum": sum(float(value) for value in confidences),
        "confidence_count": len(confidences),
        "failed": failed,
        "explicit_failed_record": explicit_failed_record,
        "missing_record": missing_record,
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
    successful_output_line_count = sum(
        score["successful_output_line_count"] for score in scores
    )
    if confidence_count > successful_output_line_count:
        raise ValueError("OCR confidence count exceeds successful output lines")
    confidence_sum = sum(score["confidence_sum"] for score in scores)
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
            "normalized_character_error_denominator": reference_characters,
            "required_token_recall_denominator": required_tokens,
            "line_count_error_sample_denominator": len(scores),
            "confidence_denominator": confidence_count,
            "confidence_availability_line_denominator": (
                successful_output_line_count
            ),
        },
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
        "successful_output_line_count": successful_output_line_count,
        "confidence_count": confidence_count,
        "confidence_available": confidence_count > 0,
        "mean_reported_confidence": (
            confidence_sum / confidence_count if confidence_count else None
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


def _bounded_levenshtein(
    left: str,
    right: str,
    edit_budget: list[int],
) -> int:
    if left == right or not left or not right:
        return _levenshtein(left, right)
    cells = len(left) * len(right)
    if cells > edit_budget[0]:
        raise ValueError("OCR quality edit-distance budget exceeded")
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
