import threading
import time

import psutil


class ProcessTreeMonitor:
    def __init__(self, pid: int, interval_seconds: float = 0.25):
        self.pid = pid
        self.interval_seconds = interval_seconds
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)
        self._previous_cpu_seconds: float | None = None
        self._previous_sample_time: float | None = None

    def start(self) -> None:
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
                rss = cpu_seconds = threads = 0
                for process in processes:
                    try:
                        rss += process.memory_info().rss
                        cpu_times = process.cpu_times()
                        cpu_seconds += cpu_times.user + cpu_times.system
                        threads += process.num_threads()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                sample_time = time.time()
                cpu_percent = 0.0
                if self._previous_cpu_seconds is not None and self._previous_sample_time is not None:
                    elapsed = sample_time - self._previous_sample_time
                    if elapsed > 0:
                        cpu_capacity = float(psutil.cpu_count(logical=True) or 1) * 100
                        cpu_percent = min(cpu_capacity, max(0.0, (cpu_seconds - self._previous_cpu_seconds) / elapsed * 100))
                self._previous_cpu_seconds = cpu_seconds
                self._previous_sample_time = sample_time
                self.samples.append({"time": sample_time, "rss_bytes": rss, "cpu_percent_sum": cpu_percent, "threads": threads})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict:
        peak_cpu = max((s["cpu_percent_sum"] for s in self.samples), default=0)
        logical_cpus = psutil.cpu_count(logical=True) or 1
        return {
            "sample_count": len(self.samples),
            "peak_rss_bytes": max((s["rss_bytes"] for s in self.samples), default=0),
            "peak_cpu_percent_sum": peak_cpu,
            "peak_cpu_percent_of_host": peak_cpu / logical_cpus,
            "peak_threads": max((s["threads"] for s in self.samples), default=0),
        }
