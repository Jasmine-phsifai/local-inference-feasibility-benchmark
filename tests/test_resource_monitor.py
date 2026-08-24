import subprocess
import sys
import time
import os

import psutil

from local_inference_bench.resource_monitor import (
    ProcessTreeMonitor,
    _midrun_rss_stability,
)


def test_monitor_captures_cpu_and_memory(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "sum(i*i for i in range(20000000))"])
    sample_path = tmp_path / "samples.jsonl"
    monitor = ProcessTreeMonitor(
        process.pid,
        interval_seconds=0.02,
        sample_path=sample_path,
    )
    monitor.start()
    process.wait(timeout=30)
    summary = monitor.stop()
    assert summary["sample_count"] >= 2
    assert summary["peak_rss_bytes"] > 0
    assert summary["peak_cpu_percent_sum"] > 0
    assert summary["p95_cpu_percent_of_host"] > 0
    assert summary["minimum_available_memory_bytes"] > 0
    assert sample_path.read_text(encoding="utf-8").count("\n") >= 2


def test_monitor_surfaces_sample_file_failure(tmp_path):
    sample_path = tmp_path / "samples"
    sample_path.mkdir()
    monitor = ProcessTreeMonitor(
        os.getpid(),
        interval_seconds=0.01,
        sample_path=sample_path,
    )

    monitor.start()
    for _ in range(100):
        if monitor.monitor_error is not None:
            break
        time.sleep(0.01)
    monitor.stop()

    assert monitor.monitor_error in {"IsADirectoryError", "PermissionError"}


def test_monitor_surfaces_access_denied(monkeypatch):
    monitor = ProcessTreeMonitor(os.getpid(), interval_seconds=0.01)

    def deny_process(_pid):
        raise psutil.AccessDenied(pid=os.getpid())

    monkeypatch.setattr(
        "local_inference_bench.resource_monitor.psutil.Process",
        deny_process,
    )

    monitor.start()
    monitor.stop()

    assert monitor.monitor_error == "AccessDenied"


def test_midrun_rss_stability_excludes_startup_and_teardown():
    samples = [
        {
            "time_monotonic": float(index),
            "rss_bytes": (
                10 if index == 0 else 100 if index < 65 else 110 if index <= 80 else 1
            ),
        }
        for index in range(101)
    ]

    stability = _midrun_rss_stability(samples)

    assert stability["midrun_rss_first_median_bytes"] == 100.0
    assert stability["midrun_rss_last_median_bytes"] == 110.0
    assert stability["midrun_rss_last_to_first_ratio"] == 1.1
    assert stability["midrun_rss_growth_bytes_per_hour"] == 800.0
