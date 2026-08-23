import subprocess
import sys
import time

from local_inference_bench.resource_monitor import ProcessTreeMonitor


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
