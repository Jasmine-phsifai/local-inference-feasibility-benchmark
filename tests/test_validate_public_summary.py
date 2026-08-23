import pytest

from local_inference_bench.validate_public_summary import (
    validate_public_summary,
    validate_sustained_public_summary,
)


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


def test_accepts_bounded_openvino_device_identity() -> None:
    summary = validate_public_summary(
        {
            "device": "GPU",
            "device_name": "Intel(R) Graphics (iGPU)",
            "execution_devices": ["GPU.0"],
        }
    )
    assert summary["execution_devices"] == ["GPU.0"]


def test_accepts_privacy_safe_hf_native_asr_diagnostic() -> None:
    summary = validate_public_summary(
        {"generation": {"hf_native_asr_request": True}}
    )

    assert summary["generation"]["hf_native_asr_request"] is True


def test_accepts_bounded_prompt_version_identity() -> None:
    summary = validate_public_summary({"prompt_version": "source-faithful.v1"})

    assert summary["prompt_version"] == "source-faithful.v1"


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


def test_sustained_schema_binds_identity_counts_and_timing() -> None:
    summary = validate_sustained_public_summary(
        {
            "candidate_id": "candidate",
            "task": "ocr",
            "runtime_name": "runtime",
            "runtime_version": "1",
            "load_semantics": "resident_model",
            "workload_class": "generated_control",
            "counts": {"completed": 2, "failed": 1, "attempted": 3},
            "throughput": {"value": 10.0, "unit": "images_per_hour"},
            "timing": {
                "steady_wall_seconds": 3.0,
                "target_wall_seconds": 60.0,
            },
        },
        candidate_id="candidate",
        task="ocr",
        workload_class="generated_control",
        target_wall_seconds=60.0,
    )

    assert summary["counts"]["attempted"] == 3


def test_sustained_schema_rejects_wrong_identity() -> None:
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_sustained_public_summary(
            {
                "candidate_id": "wrong",
                "task": "ocr",
                "runtime_name": "runtime",
                "runtime_version": "1",
                "load_semantics": "resident_model",
                "workload_class": "generated_control",
                "counts": {"completed": 1, "failed": 0, "attempted": 1},
                "throughput": {"value": 1.0, "unit": "images_per_hour"},
                "timing": {
                    "steady_wall_seconds": 1.0,
                    "target_wall_seconds": 60.0,
                },
            },
            candidate_id="candidate",
            task="ocr",
            workload_class="generated_control",
            target_wall_seconds=60.0,
        )
