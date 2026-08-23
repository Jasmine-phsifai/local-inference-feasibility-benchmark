import pytest

from local_inference_bench.validate_public_summary import validate_public_summary


def test_accepts_numeric_metrics_and_bounded_identity_fields():
    summary = validate_public_summary(
        {
            "candidate_id": "rapidocr_cpu",
            "workload_class": "private_course",
            "counts": {"completed": 12, "failed": 0},
            "throughput": {"value": 42.5, "unit": "images_per_second"},
            "stable": True,
        }
    )
    assert summary["counts"]["completed"] == 12


@pytest.mark.parametrize(
    "value",
    [
        {"output_preview": "recognized private words"},
        {"transcript": "recognized private words"},
        {"stderr_tail": "failure with a private path"},
        {"runtime_name": "D:\\private\\model.exe"},
        {"note": "arbitrary prose"},
    ],
)
def test_rejects_text_paths_and_free_form_fields(value):
    with pytest.raises(ValueError):
        validate_public_summary(value)
