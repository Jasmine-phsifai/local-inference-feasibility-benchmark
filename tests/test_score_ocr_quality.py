import json

from local_inference_bench.score_ocr_quality import (
    _levenshtein,
    _normalize,
    score_ocr_quality,
)


def test_normalization_and_edit_distance_are_unicode_aware():
    assert _normalize("Ａ x \n 中文") == "ax中文"
    assert _levenshtein("kitten", "sitting") == 3


def test_quality_event_contains_aggregate_metrics_only(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "ocr",
                "workload_class": "generated_quality_control",
                "references": {
                    "sample_001": {
                        "category": "projected_mixed",
                        "lines": ["Ax = b", "中文"],
                        "required_tokens": ["Ax = b", "中文"],
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "success": True,
                "lines": [
                    {"text": "Ax = b", "confidence": 0.9},
                    {"text": "中文", "confidence": 0.8},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    event = score_ocr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )

    overall = event["metrics"]["overall"]
    assert overall["normalized_character_error_rate"] == 0.0
    assert overall["required_token_recall"] == 1.0
    serialized = json.dumps(event)
    assert "Ax = b" not in serialized
    assert "中文" not in serialized
