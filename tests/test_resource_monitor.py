import subprocess
import sys
import time

from local_inference_bench.resource_monitor import ProcessTreeMonitor


def test_monitor_captures_cpu_and_memory():
    process = subprocess.Popen([sys.executable, "-c", "sum(i*i for i in range(20000000))"])
    monitor = ProcessTreeMonitor(process.pid, interval_seconds=0.02)
    monitor.start()
    process.wait(timeout=30)
    summary = monitor.stop()
    assert summary["sample_count"] >= 2
    assert summary["peak_rss_bytes"] > 0
    assert summary["peak_cpu_percent_sum"] > 0
