from local_inference_bench import windows_host_monitor
from local_inference_bench.windows_host_monitor import WindowsHostMonitor


def _sample(**overrides):
    sample = {
        "cpu_utility_percent": 80.0,
        "cpu_actual_frequency_mhz": 4500.0,
        "cpu_performance_limit_percent": 100.0,
        "cpu_performance_limit_flags": 0.0,
        "available_memory_mib": 32000.0,
        "committed_memory_percent": 25.0,
        "pages_input_per_second": 0.0,
        "processor_queue_length": 1.0,
        "acpi_zone_temperature_tenths_kelvin": 3100.0,
        "thermal_throttle_reasons": 0.0,
        "thermal_passive_limit_percent": 100.0,
        "rapl_package_power_milliwatts": 90000.0,
        "acpi_zone_temperature_celsius": 36.85,
        "rapl_package_power_watts": 90.0,
    }
    sample.update(overrides)
    return sample


def test_summary_marks_package_temperature_unavailable(tmp_path):
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")
    monitor.samples = [_sample(), _sample(cpu_utility_percent=90.0)]

    summary = monitor.summary()

    assert summary["status"] == "observed"
    assert summary["package_temperature_available"] is False
    assert summary["mean_cpu_utility_percent"] == 85.0


def test_ready_requires_a_valid_sample_and_no_monitor_error(tmp_path):
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")

    assert monitor.wait_until_ready(0.001) is False
    monitor.samples.append(_sample())
    monitor._ready.set()
    assert monitor.wait_until_ready(0.001) is True

    monitor.start_error = "counter_process_exit"
    assert monitor.wait_until_ready(0.001) is False


def test_stop_surfaces_counter_termination_failure_and_still_joins(
    tmp_path,
) -> None:
    class FakeProcess:
        @staticmethod
        def poll():
            return None

        @staticmethod
        def terminate():
            raise PermissionError("injected termination failure")

    class FakeThread:
        joined = False

        def join(self, *, timeout):
            assert timeout >= 10.0
            self.joined = True

        @staticmethod
        def is_alive():
            return False

    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")
    fake_thread = FakeThread()
    monitor._process = FakeProcess()
    monitor._thread = fake_thread

    summary = monitor.stop()

    assert fake_thread.joined is True
    assert monitor.start_error == "PermissionError"
    assert summary["status"] == "unavailable"


def test_sample_file_failure_is_exposed_to_preflight(tmp_path, monkeypatch):
    sample_path = tmp_path / "host-samples"
    sample_path.mkdir()
    values = [
        "80",
        "4500",
        "100",
        "0",
        "32000",
        "25",
        "0",
        "1",
        "3100",
        "0",
        "100",
        "90000",
    ]
    stdout = '"timestamp",' + ",".join(f'"{value}"' for value in values)

    class FakeCounterProcess:
        returncode = 0

        @staticmethod
        def communicate():
            return stdout, ""

    monkeypatch.setattr(
        windows_host_monitor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeCounterProcess(),
    )
    monitor = WindowsHostMonitor(sample_path)

    monitor._read_samples()

    assert monitor.wait_until_ready(0.001) is False
    assert monitor.start_error in {"IsADirectoryError", "PermissionError"}


def test_later_counter_pipe_failure_marks_monitor_partial(tmp_path, monkeypatch):
    values = [
        "80",
        "4500",
        "100",
        "0",
        "32000",
        "25",
        "0",
        "1",
        "3100",
        "0",
        "100",
        "90000",
    ]
    stdout = '"timestamp",' + ",".join(f'"{value}"' for value in values)
    counter = {"value": 0}

    class FakeCounterProcess:
        returncode = 0

        def __init__(self):
            counter["value"] += 1
            self.iteration = counter["value"]

        def communicate(self):
            if self.iteration == 2:
                raise OSError("injected counter pipe failure")
            return stdout, ""

    monkeypatch.setattr(
        windows_host_monitor.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeCounterProcess(),
    )
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")

    monitor._read_samples()

    summary = monitor.summary()
    assert monitor.start_error == "OSError"
    assert summary["status"] == "partial"
    assert summary["monitor_partial"] is True
    assert summary["sample_count"] == 1


def test_summary_keeps_samples_if_counter_process_later_fails(tmp_path):
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")
    monitor.samples = [_sample(), _sample(cpu_utility_percent=90.0)]
    monitor.start_error = "counter_process_exit"

    summary = monitor.summary()

    assert summary["status"] == "partial"
    assert summary["monitor_partial"] is True
    assert summary["sample_count"] == 2
    assert summary["mean_cpu_utility_percent"] == 85.0


def test_summary_excludes_frequency_sentinel_values(tmp_path) -> None:
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")
    monitor.samples = [
        _sample(cpu_actual_frequency_mhz=-1.0),
        _sample(cpu_actual_frequency_mhz=4500.0),
    ]

    summary = monitor.summary()

    assert summary["cpu_frequency_available"] is True
    assert summary["minimum_cpu_frequency_mhz"] == 4500.0
    assert summary["mean_cpu_frequency_mhz"] == 4500.0

    monitor.samples = [_sample(cpu_actual_frequency_mhz=-1.0)]
    summary = monitor.summary()
    assert summary["cpu_frequency_available"] is False
    assert "minimum_cpu_frequency_mhz" not in summary
    assert "mean_cpu_frequency_mhz" not in summary


def test_two_passive_limit_samples_trigger_stop(tmp_path):
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")
    sample = _sample(thermal_passive_limit_percent=99.0)

    monitor._update_stop_reason(sample)
    assert monitor.stop_reason is None
    monitor._update_stop_reason(sample)

    assert monitor.stop_reason == "passive_thermal_limit"


def test_low_memory_triggers_stop(tmp_path):
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")

    monitor._update_stop_reason(_sample(available_memory_mib=3000.0))

    assert monitor.stop_reason == "available_memory_below_4_gib"
