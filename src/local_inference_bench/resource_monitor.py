import json
import statistics
import threading
import time
from pathlib import Path

import psutil


class ProcessTreeMonitor:
    def __init__(
        self,
        pid: int,
        interval_seconds: float = 0.25,
        sample_path: Path | None = None,
    ):
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.sample_path = sample_path
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._previous_cpu_by_pid: dict[int, float] = {}
        self._previous_sample_time: float | None = None

    def start(self) -> None:
        psutil.cpu_percent(interval=None)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.summary()

    def _sample_until_stopped(self) -> None:
        while not self._stop.is_set():
            try:
                root = psutil.Process(self.pid)
                processes = [root, *root.children(recursive=True)]
                rss = threads = 0
                cpu_by_pid = {}
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                        cpu_times = process.cpu_times()
                        cpu_by_pid[process.pid] = cpu_times.user + cpu_times.system
                        threads += process.num_threads()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                sample_time = time.monotonic()
                cpu_percent = 0.0
                if self._previous_sample_time is not None:
                    elapsed = sample_time - self._previous_sample_time
                    if elapsed > 0:
                        cpu_capacity = float(psutil.cpu_count(logical=True) or 1) * 100
                        cpu_delta = sum(
                            max(0.0, seconds - self._previous_cpu_by_pid.get(pid, seconds))
                            for pid, seconds in cpu_by_pid.items()
                        )
                        cpu_percent = min(cpu_capacity, cpu_delta / elapsed * 100)
                self._previous_cpu_by_pid = cpu_by_pid
                self._previous_sample_time = sample_time
                memory = psutil.virtual_memory()
                sample = {
                    "time_monotonic": sample_time,
                    "time_unix": time.time(),
                    "rss_bytes": rss,
                    "cpu_percent_sum": cpu_percent,
                    "host_cpu_percent": psutil.cpu_percent(interval=None),
                    "host_available_memory_bytes": memory.available,
                    "host_memory_percent": memory.percent,
                    "threads": threads,
                    "processes": len(cpu_by_pid),
                }
                self.samples.append(sample)
                if self.sample_path is not None:
                    self.sample_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.sample_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(sample, sort_keys=True) + "\n")
                        handle.flush()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict:
        peak_cpu = max((s["cpu_percent_sum"] for s in self.samples), default=0)
        logical_cpus = psutil.cpu_count(logical=True) or 1
        host_cpu = [s["host_cpu_percent"] for s in self.samples]
        process_cpu = [s["cpu_percent_sum"] / logical_cpus for s in self.samples]
        rss = [s["rss_bytes"] for s in self.samples]
        duration = (
            self.samples[-1]["time_monotonic"] - self.samples[0]["time_monotonic"]
            if len(self.samples) >= 2
            else 0.0
        )
        return {
            "sample_count": len(self.samples),
            "sampled_seconds": duration,
            "peak_rss_bytes": max(rss, default=0),
            "start_rss_bytes": rss[0] if rss else 0,
            "end_rss_bytes": rss[-1] if rss else 0,
            "rss_growth_bytes_per_hour": (
                (rss[-1] - rss[0]) / duration * 3600 if duration > 0 else 0.0
            ),
            "peak_cpu_percent_sum": peak_cpu,
            "peak_cpu_percent_of_host": peak_cpu / logical_cpus,
            "mean_cpu_percent_of_host": statistics.fmean(process_cpu) if process_cpu else 0.0,
            "p95_cpu_percent_of_host": _percentile(process_cpu, 0.95),
            "mean_host_cpu_percent": statistics.fmean(host_cpu) if host_cpu else 0.0,
            "p95_host_cpu_percent": _percentile(host_cpu, 0.95),
            "minimum_available_memory_bytes": min(
                (s["host_available_memory_bytes"] for s in self.samples),
                default=0,
            ),
            "peak_threads": max((s["threads"] for s in self.samples), default=0),
            "peak_processes": max((s["processes"] for s in self.samples), default=0),
        }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
