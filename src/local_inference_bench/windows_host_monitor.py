"""Stream built-in Windows performance and throttle counters during a run."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


COUNTERS = (
    ("cpu_utility_percent", r"\Processor Information(_Total)\% Processor Utility"),
    ("cpu_actual_frequency_mhz", r"\Processor Information(_Total)\Actual Frequency"),
    ("cpu_performance_limit_percent", r"\Processor Information(_Total)\% Performance Limit"),
    ("cpu_performance_limit_flags", r"\Processor Information(_Total)\Performance Limit Flags"),
    ("available_memory_mib", r"\Memory\Available MBytes"),
    ("committed_memory_percent", r"\Memory\% Committed Bytes In Use"),
    ("pages_input_per_second", r"\Memory\Pages Input/sec"),
    ("processor_queue_length", r"\System\Processor Queue Length"),
    ("acpi_zone_temperature_tenths_kelvin", r"\Thermal Zone Information(\_TZ.TZ00)\High Precision Temperature"),
    ("thermal_throttle_reasons", r"\Thermal Zone Information(\_TZ.TZ00)\Throttle Reasons"),
    ("thermal_passive_limit_percent", r"\Thermal Zone Information(\_TZ.TZ00)\% Passive Limit"),
    ("rapl_package_power_milliwatts", r"\Energy Meter(RAPL_Package0_PKG)\Power"),
)


class WindowsHostMonitor:
    """Poll typeperf and expose privacy-safe telemetry summaries."""

    def __init__(self, sample_path: Path, interval_seconds: float = 2.0):
        self.sample_path = sample_path
        self.interval_seconds = interval_seconds
        self.samples: list[dict] = []
        self.stop_reason: str | None = None
        self.start_error: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._process_lock = threading.Lock()
        self._consecutive_hot = 0
        self._consecutive_passive_limit = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read_samples, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        with self._process_lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=max(10.0, self.interval_seconds + 5.0))
        return self.summary()

    def _read_samples(self) -> None:
        command = [
            "typeperf.exe",
            *(counter for _, counter in COUNTERS),
            "-si",
            str(max(1, round(self.interval_seconds))),
            "-sc",
            "1",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        expected_columns = len(COUNTERS) + 1
        while not self._stop.is_set():
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=creationflags,
                )
            except OSError as error:
                self.start_error = type(error).__name__
                return
            with self._process_lock:
                self._process = process
            stdout, _ = process.communicate()
            with self._process_lock:
                if self._process is process:
                    self._process = None
            if process.returncode != 0:
                if not self._stop.is_set():
                    self.start_error = "counter_process_exit"
                return
            self._parse_output(stdout, expected_columns)

    def _parse_output(self, stdout: str, expected_columns: int) -> None:
        for line in stdout.splitlines():
            if not line.startswith('"'):
                continue
            try:
                row = next(csv.reader([line]))
            except (csv.Error, StopIteration):
                continue
            if not row or row[0].startswith("(PDH-CSV"):
                continue
            if len(row) != expected_columns:
                continue
            try:
                raw_values = [float(value) for value in row[1:]]
            except ValueError:
                continue
            sample = {
                key: value for (key, _), value in zip(COUNTERS, raw_values, strict=True)
            }
            sample["time_unix"] = time.time()
            sample["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            sample["acpi_zone_temperature_celsius"] = (
                sample["acpi_zone_temperature_tenths_kelvin"] / 10.0 - 273.15
            )
            sample["rapl_package_power_watts"] = (
                sample["rapl_package_power_milliwatts"] / 1000.0
            )
            self.samples.append(sample)
            self.sample_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sample_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(sample, sort_keys=True) + "\n")
                handle.flush()
            self._update_stop_reason(sample)

    def _update_stop_reason(self, sample: dict) -> None:
        if self.stop_reason is not None:
            return
        if sample["thermal_throttle_reasons"] > 0:
            self.stop_reason = "thermal_or_electrical_throttle_counter"
            return
        self._consecutive_passive_limit = (
            self._consecutive_passive_limit + 1
            if sample["thermal_passive_limit_percent"] < 100
            else 0
        )
        self._consecutive_hot = (
            self._consecutive_hot + 1
            if sample["acpi_zone_temperature_celsius"] >= 90
            else 0
        )
        if self._consecutive_passive_limit >= 2:
            self.stop_reason = "passive_thermal_limit"
        elif self._consecutive_hot >= 2:
            self.stop_reason = "acpi_thermal_zone_limit"
        elif sample["available_memory_mib"] < 4096:
            self.stop_reason = "available_memory_below_4_gib"
        elif sample["committed_memory_percent"] >= 95:
            self.stop_reason = "committed_memory_at_or_above_95_percent"

    def summary(self) -> dict:
        if self.start_error is not None and not self.samples:
            return {
                "status": "unavailable",
                "sample_count": 0,
                "package_temperature_available": False,
            }
        if not self.samples:
            return {
                "status": "no_samples",
                "sample_count": 0,
                "package_temperature_available": False,
            }
        cpu_utility = [sample["cpu_utility_percent"] for sample in self.samples]
        frequency = [sample["cpu_actual_frequency_mhz"] for sample in self.samples]
        power = [sample["rapl_package_power_watts"] for sample in self.samples]
        return {
            "status": (
                "stopped_by_guard"
                if self.stop_reason
                else "partial" if self.start_error is not None else "observed"
            ),
            "sample_count": len(self.samples),
            "monitor_partial": self.start_error is not None,
            "package_temperature_available": False,
            "mean_cpu_utility_percent": statistics.fmean(cpu_utility),
            "p95_cpu_utility_percent": _percentile(cpu_utility, 0.95),
            "minimum_cpu_frequency_mhz": min(frequency),
            "mean_cpu_frequency_mhz": statistics.fmean(frequency),
            "minimum_performance_limit_percent": min(
                sample["cpu_performance_limit_percent"] for sample in self.samples
            ),
            "maximum_performance_limit_flags": max(
                sample["cpu_performance_limit_flags"] for sample in self.samples
            ),
            "minimum_available_memory_mib": min(
                sample["available_memory_mib"] for sample in self.samples
            ),
            "maximum_committed_memory_percent": max(
                sample["committed_memory_percent"] for sample in self.samples
            ),
            "maximum_pages_input_per_second": max(
                sample["pages_input_per_second"] for sample in self.samples
            ),
            "maximum_acpi_zone_temperature_celsius": max(
                sample["acpi_zone_temperature_celsius"] for sample in self.samples
            ),
            "maximum_thermal_throttle_reasons": max(
                sample["thermal_throttle_reasons"] for sample in self.samples
            ),
            "minimum_thermal_passive_limit_percent": min(
                sample["thermal_passive_limit_percent"] for sample in self.samples
            ),
            "mean_rapl_package_power_watts": statistics.fmean(power),
            "p95_rapl_package_power_watts": _percentile(power, 0.95),
        }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
