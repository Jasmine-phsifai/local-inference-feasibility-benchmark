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


def test_summary_keeps_samples_if_counter_process_later_fails(tmp_path):
    monitor = WindowsHostMonitor(tmp_path / "host.jsonl")
    monitor.samples = [_sample(), _sample(cpu_utility_percent=90.0)]
    monitor.start_error = "counter_process_exit"

    summary = monitor.summary()

    assert summary["status"] == "partial"
    assert summary["monitor_partial"] is True
    assert summary["sample_count"] == 2
    assert summary["mean_cpu_utility_percent"] == 85.0


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
