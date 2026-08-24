import copy

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


def test_accepts_current_compound_runtime_and_revision_labels() -> None:
    summary = validate_public_summary(
        {
            "runtime_name": "llama.cpp-cpu",
            "runtime_version": (
                "optimum-intel-1.27.0.dev0+openvino-2026.3.0+"
                "transformers-4.56.1"
            ),
            "model_revision": (
                "5eb144179a02acc5e5ba31e748d22b0cf3e303b0:"
                "optimum-intel:f48d93fddff8c91e198389c47a6d5974789b67f4:with-past"
            ),
            "vad_revision": "6840bae4c5c92ee8c04faaf4db23dd0105098d7f",
            "device_name": "Intel(R) Graphics (iGPU)",
        }
    )

    assert summary["runtime_name"] == "llama.cpp-cpu"


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


@pytest.mark.parametrize(
    "value",
    [
        "/home/alice/private/model.gguf",
        "../private/model.gguf",
        "models/private/model.gguf",
        r"..\private\model.gguf",
        "model.gguf",
        "file:///home/alice/model.gguf",
        "C:model.gguf",
    ],
)
def test_rejects_cross_platform_and_relative_paths_in_allowed_fields(value) -> None:
    with pytest.raises(ValueError, match="local path"):
        validate_public_summary({"runtime_name": value})


@pytest.mark.parametrize(
    "key,value",
    [
        ("runtime_name", "alice@example.edu"),
        ("device_name", "Alice <alice@example.edu>"),
        ("runtime_name", "13800138000"),
        ("runtime_name", "+86 138 0013 8000"),
        ("runtime_name", "sk-proj-0123456789abcdef"),
        ("runtime_name", "ghp_0123456789abcdef"),
        ("runtime_name", "hf_0123456789abcdef"),
        ("runtime_name", "Abcdefghijklmnopqrstuvwxyz0123456789"),
        ("device_name", "Authorization: Bearer abcdefghijklmnop"),
        ("device_name", "api_key=abcdefghijklmnop"),
        ("device_name", "eyJabcdefgh.ijklmnop.qrstuvwx"),
        ("device_name", "-----BEGIN PRIVATE KEY-----"),
    ],
)
def test_rejects_pii_and_credential_like_allowed_strings(key, value) -> None:
    with pytest.raises(ValueError):
        validate_public_summary({key: value})


@pytest.mark.parametrize(
    "value",
    [
        "",
        " runtime",
        "runtime ",
        "runtime label",
        "runtime;label",
    ],
)
def test_runtime_name_must_be_a_bounded_machine_label(value) -> None:
    with pytest.raises(ValueError, match="bounded label"):
        validate_public_summary({"runtime_name": value})


@pytest.mark.parametrize(
    "value",
    [
        {"D:/private/name": 1},
        {"private person": 1},
        {"\u79c1\u4eba": 1},
        {"a" * 65: 1},
    ],
)
def test_rejects_path_bearing_or_unbounded_metric_keys(value) -> None:
    with pytest.raises(ValueError, match="public identifiers"):
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
            "status": "partial_failure",
            "counts": {"completed": 2, "failed": 1, "attempted": 3},
            "throughput": {"value": 2400.0, "unit": "images_per_hour"},
            "timing": {
                "steady_wall_seconds": 3.0,
                "target_wall_seconds": 60.0,
            },
        },
        candidate_id="candidate",
        task="ocr",
        workload_class="generated_control",
        target_wall_seconds=60.0,
        phase="sustained",
    )

    assert summary["counts"]["attempted"] == 3


def test_sustained_schema_derives_missing_status_from_counts() -> None:
    summary = validate_sustained_public_summary(
        {
            "candidate_id": "candidate",
            "task": "ocr",
            "runtime_name": "runtime",
            "runtime_version": "1",
            "load_semantics": "resident_model",
            "workload_class": "generated_control",
            "counts": {"completed": 1, "failed": 0, "attempted": 1},
            "throughput": {"value": 60.0, "unit": "images_per_hour"},
            "timing": {
                "steady_wall_seconds": 60.0,
                "target_wall_seconds": 60.0,
            },
        },
        candidate_id="candidate",
        task="ocr",
        workload_class="generated_control",
        target_wall_seconds=60.0,
        phase="sustained",
    )

    assert summary["status"] == "complete"


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
                "status": "complete",
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
            phase="sustained",
        )


@pytest.mark.parametrize(
    ("task", "unit"),
    [
        ("ocr", "images_per_hour"),
        ("asr", "audio_hours_per_wall_hour"),
    ],
)
def test_all_failed_summary_requires_zero_throughput(task: str, unit: str) -> None:
    summary = {
        "candidate_id": "candidate",
        "task": task,
        "runtime_name": "runtime",
        "runtime_version": "1",
        "load_semantics": "resident_model",
        "workload_class": "generated_control",
        "status": "all_failed",
        "counts": {"completed": 0, "failed": 2, "attempted": 2},
        "throughput": {"value": 1.0, "unit": unit},
        "timing": {"steady_wall_seconds": 10.0, "target_wall_seconds": 10.0},
    }

    with pytest.raises(ValueError, match="all-failed"):
        validate_sustained_public_summary(
            summary,
            candidate_id="candidate",
            task=task,
            workload_class="generated_control",
            target_wall_seconds=10.0,
            phase="sustained",
        )

    summary["throughput"]["value"] = 0.0
    validated = validate_sustained_public_summary(
        summary,
        candidate_id="candidate",
        task=task,
        workload_class="generated_control",
        target_wall_seconds=10.0,
        phase="sustained",
    )
    assert validated["throughput"]["value"] == 0.0


def test_ocr_throughput_is_derived_from_completed_images_and_wall_time() -> None:
    summary = {
        "candidate_id": "candidate",
        "task": "ocr",
        "runtime_name": "runtime",
        "runtime_version": "1",
        "load_semantics": "resident_model",
        "workload_class": "generated_control",
        "status": "partial_failure",
        "counts": {"completed": 2, "failed": 1, "attempted": 3},
        "throughput": {"value": 719.0, "unit": "images_per_hour"},
        "timing": {"steady_wall_seconds": 10.0, "target_wall_seconds": 10.0},
    }

    with pytest.raises(ValueError, match="OCR.*throughput mismatch"):
        validate_sustained_public_summary(
            summary,
            candidate_id="candidate",
            task="ocr",
            workload_class="generated_control",
            target_wall_seconds=10.0,
            phase="sustained",
        )

    summary["throughput"]["value"] = 720.0
    assert validate_sustained_public_summary(
        summary,
        candidate_id="candidate",
        task="ocr",
        workload_class="generated_control",
        target_wall_seconds=10.0,
        phase="sustained",
    )["throughput"]["value"] == 720.0


def test_faster_whisper_concurrency_is_bound_to_runner_config() -> None:
    summary = {
        "candidate_id": "faster_whisper_cpu",
        "task": "asr",
        "runtime_name": "faster_whisper",
        "runtime_version": "1.2.1",
        "load_semantics": "resident_model",
        "workload_class": "private_course",
        "status": "complete",
        "counts": {"completed": 6, "failed": 0, "attempted": 6},
        "throughput": {"value": 10.0, "unit": "audio_hours_per_wall_hour"},
        "timing": {"steady_wall_seconds": 10.0, "target_wall_seconds": 10.0},
        "concurrency": {
            "configured_processes": 1,
            "configured_model_workers_per_process": 6,
            "configured_total_model_workers": 6,
            "instrumented_process_count": 1,
            "runtime_model_workers_min": 6,
            "runtime_model_workers_max": 6,
            "python_calls_in_flight_peak_per_process": 6,
            "python_calls_in_flight_peak_min_per_process": 6,
            "ctranslate2_active_batches_peak_per_process": 6,
            "ctranslate2_queued_batches_peak_per_process": 0,
            "ctranslate2_processing_batches_peak_per_process": 6,
            "ctranslate2_processing_batches_peak_min_per_process": 6,
            "ctranslate2_sampler_sample_count": 100,
            "ctranslate2_sampler_sample_count_min_per_process": 100,
            "ctranslate2_busy_sample_count": 80,
            "ctranslate2_busy_sample_count_min_per_process": 80,
            "ctranslate2_fully_busy_sample_count": 60,
            "ctranslate2_fully_busy_fraction_when_busy": 0.75,
            "ctranslate2_sampler_failure_count": 0,
            "ctranslate2_discarded_sample_count": 2,
        },
    }

    validated = validate_sustained_public_summary(
        summary,
        candidate_id="faster_whisper_cpu",
        task="asr",
        workload_class="private_course",
        target_wall_seconds=10.0,
        phase="sustained",
        config={"processes": 1, "model_workers": 6},
    )
    assert validated["concurrency"]["runtime_model_workers_min"] == 6

    summary["concurrency"]["runtime_model_workers_min"] = 5
    with pytest.raises(ValueError, match="concurrency mismatch"):
        validate_sustained_public_summary(
            summary,
            candidate_id="faster_whisper_cpu",
            task="asr",
            workload_class="private_course",
            target_wall_seconds=10.0,
            phase="sustained",
            config={"processes": 1, "model_workers": 6},
        )


def test_faster_whisper_rejects_idle_only_concurrency_evidence() -> None:
    summary = {
        "candidate_id": "faster_whisper_cpu",
        "task": "asr",
        "runtime_name": "faster_whisper",
        "runtime_version": "1.2.1",
        "load_semantics": "resident_model",
        "workload_class": "private_course",
        "status": "complete",
        "counts": {"completed": 1, "failed": 0, "attempted": 1},
        "throughput": {"value": 1.0, "unit": "audio_hours_per_wall_hour"},
        "timing": {"steady_wall_seconds": 1.0, "target_wall_seconds": 1.0},
        "concurrency": {
            "configured_processes": 1,
            "configured_model_workers_per_process": 1,
            "configured_total_model_workers": 1,
            "instrumented_process_count": 1,
            "runtime_model_workers_min": 1,
            "runtime_model_workers_max": 1,
            "python_calls_in_flight_peak_per_process": 0,
            "python_calls_in_flight_peak_min_per_process": 0,
            "ctranslate2_active_batches_peak_per_process": 0,
            "ctranslate2_queued_batches_peak_per_process": 0,
            "ctranslate2_processing_batches_peak_per_process": 0,
            "ctranslate2_processing_batches_peak_min_per_process": 0,
            "ctranslate2_sampler_sample_count": 1,
            "ctranslate2_sampler_sample_count_min_per_process": 1,
            "ctranslate2_busy_sample_count": 0,
            "ctranslate2_busy_sample_count_min_per_process": 0,
            "ctranslate2_fully_busy_sample_count": 0,
            "ctranslate2_fully_busy_fraction_when_busy": 0.0,
            "ctranslate2_sampler_failure_count": 0,
            "ctranslate2_discarded_sample_count": 0,
        },
    }

    with pytest.raises(ValueError, match="active concurrency"):
        validate_sustained_public_summary(
            summary,
            candidate_id="faster_whisper_cpu",
            task="asr",
            workload_class="private_course",
            target_wall_seconds=1.0,
            phase="sustained",
            config={"processes": 1, "model_workers": 1},
        )

    validated = validate_sustained_public_summary(
        summary,
        candidate_id="faster_whisper_cpu",
        task="asr",
        workload_class="private_course",
        target_wall_seconds=1.0,
        phase="quality",
        config={"processes": 1, "model_workers": 1},
    )
    assert validated["counts"]["completed"] == 1


def _valid_two_process_faster_summary() -> dict:
    return {
        "candidate_id": "faster_whisper_cpu",
        "task": "asr",
        "runtime_name": "faster_whisper",
        "runtime_version": "1.2.1",
        "load_semantics": "resident_model",
        "workload_class": "private_course",
        "status": "complete",
        "counts": {"completed": 6, "failed": 0, "attempted": 6},
        "throughput": {"value": 10.0, "unit": "audio_hours_per_wall_hour"},
        "timing": {"steady_wall_seconds": 10.0, "target_wall_seconds": 10.0},
        "concurrency": {
            "configured_processes": 2,
            "configured_model_workers_per_process": 3,
            "configured_total_model_workers": 6,
            "instrumented_process_count": 2,
            "runtime_model_workers_min": 3,
            "runtime_model_workers_max": 3,
            "python_calls_in_flight_peak_per_process": 3,
            "python_calls_in_flight_peak_min_per_process": 3,
            "ctranslate2_active_batches_peak_per_process": 3,
            "ctranslate2_queued_batches_peak_per_process": 0,
            "ctranslate2_processing_batches_peak_per_process": 3,
            "ctranslate2_processing_batches_peak_min_per_process": 3,
            "ctranslate2_sampler_sample_count": 200,
            "ctranslate2_sampler_sample_count_min_per_process": 100,
            "ctranslate2_busy_sample_count": 160,
            "ctranslate2_busy_sample_count_min_per_process": 80,
            "ctranslate2_fully_busy_sample_count": 120,
            "ctranslate2_fully_busy_fraction_when_busy": 0.75,
            "ctranslate2_sampler_failure_count": 0,
            "ctranslate2_discarded_sample_count": 0,
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("python_calls_in_flight_peak_min_per_process", 4),
        ("ctranslate2_processing_batches_peak_min_per_process", 4),
        ("ctranslate2_sampler_sample_count", 199),
        ("ctranslate2_busy_sample_count", 159),
        ("ctranslate2_busy_sample_count_min_per_process", 101),
    ],
)
def test_faster_whisper_rejects_impossible_two_process_aggregates(
    field: str,
    value: int,
) -> None:
    summary = _valid_two_process_faster_summary()
    summary["concurrency"][field] = value

    with pytest.raises(ValueError, match="concurrency"):
        validate_sustained_public_summary(
            summary,
            candidate_id="faster_whisper_cpu",
            task="asr",
            workload_class="private_course",
            target_wall_seconds=10.0,
            phase="sustained",
            config={"processes": 2, "model_workers": 3},
        )


def test_faster_whisper_validation_cannot_be_bypassed_by_runtime_label() -> None:
    summary = _valid_two_process_faster_summary()
    summary["runtime_name"] = "renamed_runtime"
    del summary["concurrency"]

    with pytest.raises(ValueError, match="concurrency evidence is missing"):
        validate_sustained_public_summary(
            summary,
            candidate_id="faster_whisper_cpu",
            task="asr",
            workload_class="private_course",
            target_wall_seconds=10.0,
            phase="sustained",
            config={"processes": 2, "model_workers": 3},
        )


def test_complete_sustained_response_must_reach_requested_duration() -> None:
    summary = {
        "candidate_id": "candidate",
        "task": "ocr",
        "runtime_name": "runtime",
        "runtime_version": "1",
        "load_semantics": "resident_model",
        "workload_class": "generated_control",
        "status": "complete",
        "counts": {"completed": 1, "failed": 0, "attempted": 1},
        "throughput": {"value": 3600.0, "unit": "images_per_hour"},
        "timing": {"steady_wall_seconds": 1.0, "target_wall_seconds": 600.0},
    }

    with pytest.raises(ValueError, match="ended before its target"):
        validate_sustained_public_summary(
            copy.deepcopy(summary),
            candidate_id="candidate",
            task="ocr",
            workload_class="generated_control",
            target_wall_seconds=600.0,
            phase="sustained",
        )

    validate_sustained_public_summary(
        summary,
        candidate_id="candidate",
        task="ocr",
        workload_class="generated_control",
        target_wall_seconds=600.0,
        phase="compatibility",
    )


def test_public_summary_rejects_unbounded_containers_and_depth() -> None:
    with pytest.raises(ValueError, match="mapping exceeds"):
        validate_public_summary({f"metric_{index}": index for index in range(257)})

    with pytest.raises(ValueError, match="sequence exceeds"):
        validate_public_summary({"metrics": list(range(257))})

    nested: object = 1
    for _ in range(13):
        nested = {"metrics": nested}
    with pytest.raises(ValueError, match="nesting exceeds"):
        validate_public_summary(nested)


def test_public_summary_rejects_excessive_total_values() -> None:
    summary = {f"metric_{index}": list(range(256)) for index in range(20)}

    with pytest.raises(ValueError, match="value count exceeds"):
        validate_public_summary(summary)
