"""Aggregate runtime-worker records without publishing recognized content."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


MINIMUM_STABILITY_WINDOW_COVERAGE = 0.8
MINIMUM_STABLE_LAST_TO_FIRST_RATIO = 0.95
MAXIMUM_STABLE_LAST_TO_FIRST_RATIO = 1.05


def build_public_summary(
    *,
    candidate_id: str,
    task: str,
    runtime_name: str,
    runtime_version: str,
    workload_class: str,
    records: list[dict],
    load_seconds: list[float],
    warmup_seconds: list[float],
    steady_wall_seconds: float,
    target_wall_seconds: float,
    load_semantics: str,
) -> dict:
    successes = [record for record in records if record["success"]]
    failures = len(records) - len(successes)
    successful_latencies = [record["latency_seconds"] for record in successes]
    attempted_latencies = [record["latency_seconds"] for record in records]
    processed_units = sum(record["units"] for record in successes)
    if task == "asr":
        throughput = {
            "value": processed_units / steady_wall_seconds if steady_wall_seconds else 0.0,
            "unit": "audio_hours_per_wall_hour",
        }
    else:
        throughput = {
            "value": processed_units / steady_wall_seconds * 3600
            if steady_wall_seconds
            else 0.0,
            "unit": "images_per_hour",
        }
    windows = _throughput_windows(
        successes,
        task=task,
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=target_wall_seconds,
    )
    analyzed_windows = [
        window
        for window in windows
        if window["coverage"] >= MINIMUM_STABILITY_WINDOW_COVERAGE
    ]
    rates = [window["rate"] for window in analyzed_windows]
    coefficient_of_variation = (
        statistics.pstdev(rates) / statistics.fmean(rates)
        if len(rates) >= 2 and statistics.fmean(rates) > 0
        else 0.0
    )
    first_last_ratio = (
        rates[-1] / rates[0]
        if len(rates) >= 2 and rates[0] > 0
        else None
    )
    if len(rates) < 2:
        stability_status = "insufficient"
    elif (
        failures == 0
        and coefficient_of_variation <= 0.05
        and first_last_ratio is not None
        and MINIMUM_STABLE_LAST_TO_FIRST_RATIO
        <= first_last_ratio
        <= MAXIMUM_STABLE_LAST_TO_FIRST_RATIO
    ):
        stability_status = "stable"
    else:
        stability_status = "variable"
    if not successes:
        benchmark_status = "all_failed"
    elif failures:
        benchmark_status = "partial_failure"
    else:
        benchmark_status = "complete"
    return {
        "candidate_id": candidate_id,
        "task": task,
        "runtime_name": runtime_name,
        "runtime_version": runtime_version,
        "workload_class": workload_class,
        "load_semantics": load_semantics,
        "status": benchmark_status,
        "counts": {
            "completed": len(successes),
            "failed": failures,
            "attempted": len(records),
        },
        "timing": {
            "load_seconds_mean": _mean(load_seconds),
            "load_seconds_p95": _percentile(load_seconds, 0.95),
            "warmup_seconds_mean": _mean(warmup_seconds),
            "steady_wall_seconds": steady_wall_seconds,
            "target_wall_seconds": target_wall_seconds,
            "latency_seconds_p50": _percentile(successful_latencies, 0.50),
            "latency_seconds_p95": _percentile(successful_latencies, 0.95),
            "latency_seconds_max": max(successful_latencies, default=0.0),
            "attempted_latency_seconds_p50": _percentile(
                attempted_latencies,
                0.50,
            ),
            "attempted_latency_seconds_p95": _percentile(
                attempted_latencies,
                0.95,
            ),
            "attempted_latency_seconds_max": max(
                attempted_latencies,
                default=0.0,
            ),
        },
        "throughput": throughput,
        "stability": {
            "stability_status": stability_status,
            "completion_event_attribution": True,
            "window_seconds": (
                windows[0]["nominal_duration_seconds"] if windows else 0.0
            ),
            "minimum_window_coverage": MINIMUM_STABILITY_WINDOW_COVERAGE,
            "minimum_stable_last_to_first_ratio": (
                MINIMUM_STABLE_LAST_TO_FIRST_RATIO
            ),
            "maximum_stable_last_to_first_ratio": (
                MAXIMUM_STABLE_LAST_TO_FIRST_RATIO
            ),
            "observed_window_count": len(windows),
            "window_count": len(analyzed_windows),
            "discarded_partial_window_count": (
                len(windows) - len(analyzed_windows)
            ),
            "zero_completion_window_count": sum(
                window["completion_count"] == 0
                for window in analyzed_windows
            ),
            "throughput_window_cv": coefficient_of_variation,
            "last_to_first_window_ratio": first_last_ratio,
        },
    }


def write_private_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _throughput_windows(
    records: list[dict],
    *,
    task: str,
    steady_wall_seconds: float,
    target_wall_seconds: float,
) -> list[dict]:
    if steady_wall_seconds <= 0:
        return []
    window_seconds = max(10.0, min(60.0, target_wall_seconds / 5.0))
    window_count = max(1, math.ceil(steady_wall_seconds / window_seconds))
    units = [0.0] * window_count
    completion_counts = [0] * window_count
    for record in records:
        completed = min(
            steady_wall_seconds,
            max(0.0, record["completed_offset_seconds"]),
        )
        # Completion windows use (start, end] boundaries. This assigns a
        # completion exactly on the final boundary to the final real window
        # without inventing a zero-duration successor window.
        index = min(
            window_count - 1,
            max(0, math.ceil(completed / window_seconds) - 1),
        )
        units[index] += record["units"]
        completion_counts[index] += 1
    windows = []
    for index, processed_units in enumerate(units):
        started = index * window_seconds
        duration = min(window_seconds, max(0.0, steady_wall_seconds - started))
        if duration <= 0:
            continue
        if task == "asr":
            rate = processed_units / duration
        else:
            rate = processed_units / duration * 3600
        windows.append(
            {
                "rate": rate,
                "coverage": duration / window_seconds,
                "completion_count": completion_counts[index],
                "nominal_duration_seconds": window_seconds,
            }
        )
    return windows


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
