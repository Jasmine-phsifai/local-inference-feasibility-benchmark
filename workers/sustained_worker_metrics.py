"""Aggregate runtime-worker records without publishing recognized content."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


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
    latencies = [record["latency_seconds"] for record in successes]
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
    rates = [window["rate"] for window in windows if window["coverage"] >= 0.8]
    coefficient_of_variation = (
        statistics.pstdev(rates) / statistics.fmean(rates)
        if len(rates) >= 2 and statistics.fmean(rates) > 0
        else 0.0
    )
    first_last_ratio = rates[-1] / rates[0] if len(rates) >= 2 and rates[0] > 0 else 1.0
    if len(rates) < 2:
        stability_status = "insufficient"
    elif failures == 0 and coefficient_of_variation <= 0.05 and first_last_ratio >= 0.95:
        stability_status = "stable"
    else:
        stability_status = "variable"
    return {
        "candidate_id": candidate_id,
        "task": task,
        "runtime_name": runtime_name,
        "runtime_version": runtime_version,
        "workload_class": workload_class,
        "load_semantics": load_semantics,
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
            "latency_seconds_p50": _percentile(latencies, 0.50),
            "latency_seconds_p95": _percentile(latencies, 0.95),
            "latency_seconds_max": max(latencies, default=0.0),
        },
        "throughput": throughput,
        "stability": {
            "stability_status": stability_status,
            "window_count": len(windows),
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
    for record in records:
        completed = min(steady_wall_seconds, record["completed_offset_seconds"])
        latency = max(0.0, record["latency_seconds"])
        started = max(0.0, completed - latency)
        active_seconds = completed - started
        if active_seconds <= 0:
            index = min(
                window_count - 1,
                int(completed // window_seconds),
            )
            units[index] += record["units"]
            continue
        for index in range(window_count):
            window_started = index * window_seconds
            window_ended = min(steady_wall_seconds, window_started + window_seconds)
            overlap = max(
                0.0,
                min(completed, window_ended) - max(started, window_started),
            )
            units[index] += record["units"] * overlap / active_seconds
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
