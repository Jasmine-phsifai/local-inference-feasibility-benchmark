from workers.sustained_worker_metrics import (
    _throughput_windows,
    build_public_summary,
)


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
    assert summary["status"] == "complete"


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
    assert summary["status"] == "partial_failure"
    assert summary["timing"]["attempted_latency_seconds_max"] == 1.0


def test_all_failed_attempt_keeps_failure_latency_visible():
    summary = build_public_summary(
        candidate_id="asr",
        task="asr",
        runtime_name="runtime",
        runtime_version="1",
        workload_class="private_course",
        records=[
            {
                "success": False,
                "latency_seconds": 172.0,
                "units": 0.0,
                "completed_offset_seconds": 172.0,
            }
        ],
        load_seconds=[1.0],
        warmup_seconds=[1.0],
        steady_wall_seconds=172.0,
        target_wall_seconds=60.0,
        load_semantics="resident_model",
    )

    assert summary["status"] == "all_failed"
    assert summary["timing"]["latency_seconds_max"] == 0.0
    assert summary["timing"]["attempted_latency_seconds_max"] == 172.0


def test_end_only_completions_are_not_smoothed_across_active_windows():
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

    stability = summary["stability"]
    assert stability["completion_event_attribution"] is True
    assert stability["window_seconds"] == 12.0
    assert stability["window_count"] == 5
    assert stability["zero_completion_window_count"] == 4
    assert stability["throughput_window_cv"] == 2.0
    assert stability["last_to_first_window_ratio"] is None
    assert stability["stability_status"] == "variable"


def test_partial_tail_is_observed_but_not_counted_as_analyzed_window():
    records = [
        {
            "success": True,
            "latency_seconds": 1.0,
            "units": 1.0,
            "completed_offset_seconds": completed,
        }
        for completed in (10.0, 20.0, 21.0)
    ]

    summary = build_public_summary(
        candidate_id="ocr",
        task="ocr",
        runtime_name="runtime",
        runtime_version="1",
        workload_class="generated_control",
        records=records,
        load_seconds=[1.0],
        warmup_seconds=[1.0],
        steady_wall_seconds=27.0,
        target_wall_seconds=50.0,
        load_semantics="resident_model",
    )

    stability = summary["stability"]
    assert stability["minimum_window_coverage"] == 0.8
    assert stability["observed_window_count"] == 3
    assert stability["window_count"] == 2
    assert stability["discarded_partial_window_count"] == 1


def test_exact_eighty_percent_tail_is_included_and_boundary_is_previous_window():
    windows = _throughput_windows(
        [
            {
                "success": True,
                "latency_seconds": 60.0,
                "units": 1.0,
                "completed_offset_seconds": completed,
            }
            for completed in (10.0, 20.0, 28.0)
        ],
        task="ocr",
        steady_wall_seconds=28.0,
        target_wall_seconds=50.0,
    )

    assert [window["completion_count"] for window in windows] == [1, 1, 1]
    assert [window["coverage"] for window in windows] == [1.0, 1.0, 0.8]

    summary = build_public_summary(
        candidate_id="ocr",
        task="ocr",
        runtime_name="runtime",
        runtime_version="1",
        workload_class="generated_control",
        records=[
            {
                "success": True,
                "latency_seconds": 60.0,
                "units": 1.0,
                "completed_offset_seconds": completed,
            }
            for completed in (10.0, 20.0, 28.0)
        ],
        load_seconds=[1.0],
        warmup_seconds=[1.0],
        steady_wall_seconds=28.0,
        target_wall_seconds=50.0,
        load_semantics="resident_model",
    )
    stability = summary["stability"]
    assert stability["observed_window_count"] == 3
    assert stability["window_count"] == 3
    assert stability["discarded_partial_window_count"] == 0


def test_low_variance_upward_drift_is_not_labeled_stable():
    records = [
        {
            "success": True,
            "latency_seconds": 1.0,
            "units": units,
            "completed_offset_seconds": completed,
        }
        for completed, units in zip(
            (1.0, 11.0, 21.0, 31.0, 41.0),
            (1.0, 1.0, 1.02, 1.03, 1.06),
            strict=True,
        )
    ]

    summary = build_public_summary(
        candidate_id="asr",
        task="asr",
        runtime_name="runtime",
        runtime_version="1",
        workload_class="generated_control",
        records=records,
        load_seconds=[1.0],
        warmup_seconds=[1.0],
        steady_wall_seconds=50.0,
        target_wall_seconds=50.0,
        load_semantics="resident_model",
    )

    stability = summary["stability"]
    assert stability["throughput_window_cv"] < 0.05
    assert stability["last_to_first_window_ratio"] == 1.06
    assert stability["minimum_stable_last_to_first_ratio"] == 0.95
    assert stability["maximum_stable_last_to_first_ratio"] == 1.05
    assert stability["stability_status"] == "variable"
