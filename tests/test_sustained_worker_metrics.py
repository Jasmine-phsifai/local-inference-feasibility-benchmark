from workers.sustained_worker_metrics import build_public_summary


def test_audio_throughput_uses_aggregate_duration_over_group_wall():
    summary = build_public_summary(
        candidate_id="asr",
        task="asr",
        runtime_name="runtime",
        runtime_version="1",
        workload_class="generated_control",
        records=[
            {
                "success": True,
                "latency_seconds": 4.0,
                "units": 60.0,
                "completed_offset_seconds": 4.0,
            },
            {
                "success": True,
                "latency_seconds": 4.0,
                "units": 60.0,
                "completed_offset_seconds": 8.0,
            },
        ],
        load_seconds=[1.0],
        warmup_seconds=[2.0],
        steady_wall_seconds=10.0,
        target_wall_seconds=10.0,
        load_semantics="resident_model",
    )

    assert summary["throughput"] == {
        "value": 12.0,
        "unit": "audio_hours_per_wall_hour",
    }
    assert summary["counts"]["completed"] == 2


def test_ocr_throughput_counts_failures_without_dropping_attempts():
    summary = build_public_summary(
        candidate_id="ocr",
        task="ocr",
        runtime_name="runtime",
        runtime_version="1",
        workload_class="generated_control",
        records=[
            {
                "success": True,
                "latency_seconds": 1.0,
                "units": 1.0,
                "completed_offset_seconds": 1.0,
            },
            {
                "success": False,
                "latency_seconds": 0.5,
                "units": 0.0,
                "completed_offset_seconds": 1.5,
            },
        ],
        load_seconds=[1.0],
        warmup_seconds=[1.0],
        steady_wall_seconds=2.0,
        target_wall_seconds=2.0,
        load_semantics="resident_model",
    )

    assert summary["throughput"]["value"] == 1800.0
    assert summary["counts"] == {"completed": 1, "failed": 1, "attempted": 2}


def test_long_concurrent_jobs_contribute_across_active_windows():
    summary = build_public_summary(
        candidate_id="asr",
        task="asr",
        runtime_name="runtime",
        runtime_version="1",
        workload_class="private_course",
        records=[
            {
                "success": True,
                "latency_seconds": 60.0,
                "units": 900.0,
                "completed_offset_seconds": 60.0,
            },
            {
                "success": True,
                "latency_seconds": 60.0,
                "units": 900.0,
                "completed_offset_seconds": 60.0,
            },
        ],
        load_seconds=[1.0, 1.0],
        warmup_seconds=[60.0, 60.0],
        steady_wall_seconds=60.0,
        target_wall_seconds=60.0,
        load_semantics="per_file_cli_startup_estimate",
    )

    assert summary["stability"]["throughput_window_cv"] == 0.0
    assert summary["stability"]["last_to_first_window_ratio"] == 1.0
    assert summary["stability"]["stability_status"] == "stable"
