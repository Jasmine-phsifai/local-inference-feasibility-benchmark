import hashlib
import json
import uuid
from pathlib import Path

import pytest

import local_inference_bench.score_ocr_quality as score_module
from local_inference_bench.load_sustained_workload import load_sustained_workload
from local_inference_bench.score_ocr_quality import (
    _levenshtein,
    _normalize,
    score_ocr_quality,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_records(
    manifest_path: Path,
    records_path: Path,
    *,
    candidate_id: str = "candidate",
) -> Path:
    workload = load_sustained_workload(manifest_path, expected_task="ocr")
    expected_ids = {item["id"] for item in workload["items"]}
    successful_ids = {
        record["sample_id"]
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
        if isinstance(record, dict)
        and type(record.get("sample_id")) is str
        and record["sample_id"] in expected_ids
        and record.get("success") is True
    }
    status = (
        "succeeded"
        if successful_ids == expected_ids
        else "all_failed" if not successful_ids else "partial_failure"
    )
    provenance_path = records_path.with_name("records-provenance.json")
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "sustained-process-v1",
                "status": status,
                "attempt_id": str(uuid.UUID("00000000-0000-4000-8000-000000000001")),
                "attempt_key": "1" * 16,
                "candidate_id": candidate_id,
                "task": "ocr",
                "config": {"processes": 1},
                "config_index": 2,
                "phase": "quality",
                "trial_index": 0,
                "workload_class": "generated_quality_control",
                "workload_fingerprint": workload["fingerprint"],
                "code_fingerprint": "a" * 16,
                "environment_fingerprint": "b" * 16,
                "controller_environment_fingerprint": "c" * 16,
                "execution_policy_fingerprint": "d" * 16,
                "records_sha256": _sha256(records_path),
            }
        ),
        encoding="utf-8",
    )
    return provenance_path


def _write_minimal_quality_fixture(
    tmp_path: Path,
    record: dict,
) -> tuple[Path, Path]:
    image_path = tmp_path / "sample.png"
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "private-records.jsonl"
    image_path.write_bytes(b"public-control")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "generated_quality_control",
                "items": [{"id": "sample", "path": image_path.name}],
                "references": {
                    "sample": {
                        "category": "control",
                        "image_sha256": _sha256(image_path),
                        "lines": ["reference"],
                        "required_tokens": ["reference"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _bind_records(manifest_path, records_path)
    return manifest_path, records_path


def test_normalization_and_edit_distance_are_unicode_aware() -> None:
    assert _normalize("\uff21 x \n \u4e2d\u6587") == "ax\u4e2d\u6587"
    assert _levenshtein("kitten", "sitting") == 3


def test_quality_event_contains_aggregate_metrics_only(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {
            "sample_id": "sample",
            "success": True,
            "lines": [{"text": "reference", "confidence": 0.9}],
        },
    )

    event = score_ocr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )

    overall = event["metrics"]["overall"]
    assert event["scorer_protocol"] == "ocr-quality-v5"
    assert event["source_attempt"] == {
        "status": "succeeded",
        "candidate_id": "candidate",
        "attempt_id": "00000000-0000-4000-8000-000000000001",
        "attempt_key": "1111111111111111",
        "config_fingerprint": "2266b7bfde15dc37",
        "config_index": 2,
        "trial_index": 0,
        "code_fingerprint": "a" * 16,
        "environment_fingerprint": "b" * 16,
        "controller_environment_fingerprint": "c" * 16,
        "execution_policy_fingerprint": "d" * 16,
    }
    assert event["workload_fingerprint"] == load_sustained_workload(
        manifest_path,
        expected_task="ocr",
    )["fingerprint"]
    assert len(event["scorer_fingerprint"]) == 16
    assert overall["normalized_character_error_rate"] == 0.0
    assert overall["required_token_recall"] == 1.0
    assert overall["confidence_count"] == 1
    assert overall["successful_output_line_count"] == 1
    assert overall["confidence_available"] is True
    assert overall["mean_reported_confidence"] == 0.9
    assert overall["availability"] == {
        "sample_denominator": 1,
        "successful_sample_count": 1,
        "unavailable_sample_count": 0,
        "explicit_failed_record_count": 0,
        "missing_record_count": 0,
    }
    assert overall["quality_denominators"]["confidence_denominator"] == 1
    assert (
        overall["quality_denominators"]
        ["confidence_availability_line_denominator"]
        == 1
    )
    serialized = json.dumps(event)
    assert "reference" not in serialized


def test_failed_record_output_is_excluded_from_quality_metrics(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {
            "sample_id": "sample",
            "success": False,
            "lines": [{"text": "reference", "confidence": 1.0}],
        },
    )

    event = score_ocr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )

    overall = event["metrics"]["overall"]
    assert event["source_attempt"]["status"] == "all_failed"
    assert overall["failure_count"] == 1
    assert overall["required_token_recall"] == 0.0
    assert overall["confidence_count"] == 0
    assert overall["successful_output_line_count"] == 0
    assert overall["confidence_available"] is False
    assert overall["mean_reported_confidence"] is None
    assert overall["availability"] == {
        "sample_denominator": 1,
        "successful_sample_count": 0,
        "unavailable_sample_count": 1,
        "explicit_failed_record_count": 1,
        "missing_record_count": 0,
    }


def test_partial_confidence_uses_only_explicit_successful_values(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    items = []
    references = {}
    for sample_id in ("success", "failed", "missing"):
        image_path = tmp_path / f"{sample_id}.png"
        image_path.write_bytes(sample_id.encode("ascii"))
        items.append({"id": sample_id, "path": image_path.name})
        references[sample_id] = {
            "category": "control",
            "image_sha256": _sha256(image_path),
            "lines": [sample_id],
            "required_tokens": [sample_id],
        }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "generated_quality_control",
                "items": items,
                "references": references,
            }
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {
                    "sample_id": "success",
                    "success": True,
                    "lines": [
                        {"text": "success", "confidence": 0.8},
                        {"text": "without explicit confidence"},
                    ],
                },
                {
                    "sample_id": "failed",
                    "success": False,
                    "lines": [{"text": "ignored", "confidence": 1.0}],
                },
            )
        ),
        encoding="utf-8",
    )
    _bind_records(manifest_path, records_path)

    event = score_ocr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )

    overall = event["metrics"]["overall"]
    assert event["source_attempt"]["status"] == "partial_failure"
    assert overall["availability"] == {
        "sample_denominator": 3,
        "successful_sample_count": 1,
        "unavailable_sample_count": 2,
        "explicit_failed_record_count": 1,
        "missing_record_count": 1,
    }
    assert overall["confidence_count"] == 1
    assert overall["successful_output_line_count"] == 2
    assert overall["quality_denominators"]["confidence_denominator"] == 1
    assert (
        overall["quality_denominators"]
        ["confidence_availability_line_denominator"]
        == 2
    )
    assert overall["confidence_available"] is True
    assert overall["mean_reported_confidence"] == pytest.approx(0.8)
    assert "ignored" not in json.dumps(event)


def test_success_without_explicit_confidence_reports_unavailable(
    tmp_path: Path,
) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {
            "sample_id": "sample",
            "success": True,
            "lines": [{"text": "reference"}],
        },
    )

    overall = score_ocr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )["metrics"]["overall"]

    assert overall["confidence_count"] == 0
    assert overall["successful_output_line_count"] == 1
    assert overall["quality_denominators"]["confidence_denominator"] == 0
    assert (
        overall["quality_denominators"]
        ["confidence_availability_line_denominator"]
        == 1
    )
    assert overall["confidence_available"] is False
    assert overall["mean_reported_confidence"] is None


def test_rejects_confidence_count_above_successful_output_line_count() -> None:
    with pytest.raises(ValueError, match="confidence count exceeds"):
        score_module._aggregate_group(
            [
                {
                    "reference_characters": 1,
                    "edit_distance": 0,
                    "required_tokens": 0,
                    "token_hits": 0,
                    "line_count_error": 0,
                    "successful_output_line_count": 1,
                    "confidence_sum": 1.5,
                    "confidence_count": 2,
                    "failed": False,
                    "explicit_failed_record": False,
                    "missing_record": False,
                }
            ]
        )


def test_rejects_source_status_that_disagrees_with_records(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "lines": []},
    )
    provenance_path = records_path.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["status"] = "partial_failure"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="source status does not match records"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


def test_rejects_records_mutated_after_provenance_binding(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "lines": []},
    )
    records_path.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "success": True,
                "lines": [{"text": "changed"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance is invalid"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


def test_rejects_replaced_image_with_stale_reference_hash(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "lines": []},
    )
    (tmp_path / "sample.png").write_bytes(b"different-public-control")
    _bind_records(manifest_path, records_path)

    with pytest.raises(ValueError, match="reference is invalid"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


def test_public_registry_config_is_fingerprinted_without_copying_raw_config(
    tmp_path: Path,
) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "lines": []},
    )
    provenance_path = records_path.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["config"] = {"processes": 8, "threads_per_process": 2}
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    event = score_ocr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )
    assert "threads_per_process" not in json.dumps(event)


@pytest.mark.parametrize(
    "record",
    [
        {"sample_id": "sample", "success": "true", "lines": []},
        {"sample_id": "sample", "success": True},
        {"sample_id": "sample", "success": True, "lines": None},
        {"sample_id": "sample", "success": True, "lines": ["text"]},
        {
            "sample_id": "sample",
            "success": True,
            "lines": [{"text": "text", "confidence": True}],
        },
    ],
)
def test_rejects_malformed_quality_records(tmp_path: Path, record: dict) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(tmp_path, record)

    with pytest.raises(ValueError, match="invalid OCR quality record"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


def test_rejects_private_workload_and_unbounded_candidate_id(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "lines": []},
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["workload_class"] = "private_course"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="generated controls only"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )

    document["workload_class"] = "generated_quality_control"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="public identifier"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="D:\\private\\course",
        )


def test_rejects_invalid_reference_and_edit_work_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {
            "sample_id": "sample",
            "success": True,
            "lines": [{"text": "prediction"}],
        },
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["references"]["sample"]["required_tokens"] = [False]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    _bind_records(manifest_path, records_path)
    with pytest.raises(ValueError, match="reference is invalid"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )

    document["references"]["sample"]["required_tokens"] = []
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    _bind_records(manifest_path, records_path)
    monkeypatch.setattr(score_module, "_MAX_TOTAL_EDIT_CELLS", 1)
    with pytest.raises(ValueError, match="edit-distance budget exceeded"):
        score_ocr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )
