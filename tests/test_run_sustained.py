import json
import subprocess
from pathlib import Path

import pytest

from local_inference_bench.event_journal import read_events
from local_inference_bench import run_sustained


class _FakeWorkerJob:
    def __init__(self) -> None:
        self.closed = False

    def assign(self, _process) -> None:
        pass

    def resume(self, _process) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def _record_successful_termination(terminated: list[int], pid: int) -> dict:
    terminated.append(pid)
    return {
        "found": 1,
        "terminated": 1,
        "killed": 0,
        "surviving": 0,
        "error_count": 0,
    }


def _common():
    return {
        "protocol": "sustained-process-v1",
        "candidate_id": "candidate",
        "task": "asr",
        "attempt_id": "attempt",
        "attempt_key": "key",
        "code_fingerprint": "code",
        "environment_fingerprint": "environment",
        "controller_environment_fingerprint": "controller",
        "execution_policy_fingerprint": "policy",
        "config": {"processes": 1},
        "config_index": 0,
        "phase": "screen",
        "target_wall_seconds": 10,
        "trial_index": 0,
        "workload": {
            "workload_class": "private_course",
            "item_count": 1,
            "total_duration_seconds": 60,
        },
        "workload_fingerprint": "c" * 64,
    }


def _minimal_attempt_kwargs() -> dict:
    return {
        "registry": {
            "protocol": "sustained-process-v1",
            "resource_sample_interval_seconds": 0.25,
            "host_sample_interval_seconds": 2.0,
            "timeout_overhead_seconds": 10,
        },
        "candidate": {
            "id": "candidate",
            "task": "asr",
            "worker": "workers/fake_worker.py",
        },
        "python": Path("python.exe"),
        "workload": {
            "workload_class": "private_course",
            "items": [{"id": "sample", "path": "ignored.wav"}],
            "warmup_item": {"id": "sample", "path": "ignored.wav"},
            "public_summary": {
                "workload_class": "private_course",
                "item_count": 1,
                "total_duration_seconds": 1,
            },
            "fingerprint": "c" * 64,
        },
        "config": {"processes": 1},
        "config_index": 0,
        "phase": "sustained",
        "target_wall_seconds": 1,
        "trial_index": 0,
        "attempt_key": "a" * 16,
        "code_fingerprint": "d" * 16,
        "environment_fingerprint": "e" * 16,
        "controller_environment_fingerprint": "f" * 16,
        "execution_policy_fingerprint": "1" * 16,
    }


def test_outcome_journal_uses_only_validated_public_summary(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "public_summary": {
                    "candidate_id": "candidate",
                    "task": "asr",
                    "runtime_name": "runtime",
                    "runtime_version": "1",
                    "load_semantics": "resident_model",
                    "workload_class": "private_course",
                    "status": "complete",
                    "counts": {"completed": 1, "failed": 0, "attempted": 1},
                    "throughput": {
                        "value": 1.0,
                        "unit": "audio_hours_per_wall_hour",
                    },
                    "timing": {
                        "steady_wall_seconds": 10.0,
                        "target_wall_seconds": 10.0,
                    },
                },
                "private_records": {
                    "transcript": "must not be copied",
                    "path": "D:\\private\\lecture.wav",
                },
            }
        ),
        encoding="utf-8",
    )
    private_records_path = tmp_path / "private-records.jsonl"
    private_records_path.write_text("{}\n", encoding="utf-8")

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
        private_records_path=private_records_path,
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={"sample_count": 1},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    serialized = json.dumps(event)
    assert event["event"] == "sustained_attempt_succeeded"
    assert "attempt_key" not in event
    assert "transcript" not in serialized
    assert "lecture.wav" not in serialized
    provenance = json.loads(
        (tmp_path / "records-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["attempt_key"] == "key"
    assert provenance["records_sha256"] == run_sustained._sha256(
        private_records_path
    )


def test_invalid_preview_field_becomes_generic_failure(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps({"public_summary": {"output_preview": "private"}}),
        encoding="utf-8",
    )

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
        private_records_path=tmp_path / "missing-records.jsonl",
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    assert event["event"] == "sustained_attempt_failed"
    assert event["failure_kind"] == "invalid_response"
    assert "attempt_key" not in event


def test_empty_summary_cannot_be_journaled_as_success(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    response_path = tmp_path / "response.json"
    response_path.write_text('{"public_summary": {}}', encoding="utf-8")

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
        private_records_path=tmp_path / "missing-records.jsonl",
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    assert event["event"] == "sustained_attempt_failed"
    assert event["failure_kind"] == "invalid_response"


def test_unreadable_response_is_journaled_as_invalid_response(
    tmp_path,
    monkeypatch,
) -> None:
    class UnreadableResponse:
        @staticmethod
        def is_file() -> bool:
            return True

        @staticmethod
        def read_text(*, encoding: str) -> str:
            assert encoding == "utf-8"
            raise PermissionError("injected response read failure")

    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=UnreadableResponse(),
        private_records_path=tmp_path / "missing-records.jsonl",
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={"sample_count": 1},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    assert event["event"] == "sustained_attempt_failed"
    assert event["failure_kind"] == "invalid_response"


def test_all_failed_summary_is_journaled_as_failure_and_not_success_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "public_summary": {
                    "candidate_id": "candidate",
                    "task": "asr",
                    "runtime_name": "runtime",
                    "runtime_version": "1",
                    "load_semantics": "resident_model",
                    "workload_class": "private_course",
                    "status": "all_failed",
                    "counts": {"completed": 0, "failed": 1, "attempted": 1},
                    "throughput": {
                        "value": 0.0,
                        "unit": "audio_hours_per_wall_hour",
                    },
                    "timing": {
                        "steady_wall_seconds": 1.0,
                        "target_wall_seconds": 10.0,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    private_records_path = tmp_path / "private-records.jsonl"
    private_records_path.write_text("{}\n", encoding="utf-8")

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
        private_records_path=private_records_path,
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={"sample_count": 1},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    assert event["event"] == "sustained_attempt_failed"
    assert event["failure_kind"] == "all_items_failed"
    assert event["result"]["status"] == "all_failed"
    provenance = json.loads(
        (tmp_path / "records-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["status"] == "all_failed"


def test_success_cache_excludes_invalidated_and_noncomplete_attempts() -> None:
    events = [
        {
            "event": "sustained_attempt_succeeded",
            "attempt_id": "valid",
            "attempt_key": "valid-key",
            "result": {"status": "complete"},
        },
        {
            "event": "sustained_attempt_succeeded",
            "attempt_id": "invalidated",
            "attempt_key": "invalidated-key",
            "result": {"status": "complete"},
        },
        {
            "event": "sustained_attempt_succeeded",
            "attempt_id": "all-failed",
            "attempt_key": "failed-key",
            "result": {"status": "all_failed"},
        },
        {
            "event": "sustained_attempts_invalidated",
            "invalidated_attempt_ids": ["invalidated"],
        },
        {
            "event": "sustained_attempt_succeeded",
            "attempt_id": "legacy",
            "attempt_key": "legacy-key",
            "result": {"counts": {"completed": 1, "failed": 0}},
        },
    ]

    assert run_sustained._successful_attempt_keys(
        events,
        invalidated_attempt_ids={"invalidated"},
    ) == {
        "valid-key",
        "legacy-key",
    }


def _write_valid_private_resume_artifact(
    candidate_dir: Path,
    *,
    attempt_id: str = "11111111-1111-4111-8111-111111111111",
    attempt_key: str = "a" * 16,
) -> tuple[Path, dict]:
    attempt_dir = candidate_dir / attempt_id
    attempt_dir.mkdir(parents=True)
    records_path = attempt_dir / "private-records.jsonl"
    records_path.write_text("{}\n", encoding="utf-8")
    binding = {
        "protocol": "sustained-process-v1",
        "candidate_id": "candidate",
        "task": "asr",
        "config": {"processes": 1},
        "config_index": 0,
        "phase": "sustained",
        "trial_index": 0,
        "workload_class": "private_course",
        "workload_fingerprint": "c" * 64,
        "code_fingerprint": "d" * 16,
        "environment_fingerprint": "e" * 16,
        "controller_environment_fingerprint": "f" * 16,
        "execution_policy_fingerprint": "1" * 16,
    }
    provenance = {
        "schema_version": 1,
        "status": "succeeded",
        "attempt_id": attempt_id,
        "attempt_key": attempt_key,
        **binding,
        "records_sha256": run_sustained._sha256(records_path),
    }
    (attempt_dir / "records-provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    return attempt_dir, binding


def test_private_resumability_requires_hash_bound_matching_provenance(
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "candidate"
    _, binding = _write_valid_private_resume_artifact(candidate_dir)

    assert run_sustained._successful_private_artifact_attempt_keys(
        candidate_dir,
        invalidated_attempt_ids=set(),
        expected_bindings={"a" * 16: binding},
    ) == {"a" * 16}


@pytest.mark.parametrize(
    "corruption",
    (
        "invalidated",
        "records_changed",
        "candidate_changed",
        "config_changed",
        "attempt_directory_changed",
        "missing_records",
        "partial_status",
        "public_workload",
        "malformed_provenance",
        "scalar_provenance",
        "non_utf8_provenance",
        "nonstr_workload_class",
        "nonstr_records_sha256",
        "controller_environment_changed",
        "execution_policy_changed",
    ),
)
def test_private_resumability_rejects_stale_or_mutated_artifacts(
    tmp_path: Path,
    corruption: str,
) -> None:
    candidate_dir = tmp_path / "candidate"
    attempt_dir, binding = _write_valid_private_resume_artifact(candidate_dir)
    provenance_path = attempt_dir / "records-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    invalidated_attempt_ids: set[str] = set()
    if corruption == "invalidated":
        invalidated_attempt_ids.add(provenance["attempt_id"])
    elif corruption == "records_changed":
        (attempt_dir / "private-records.jsonl").write_text(
            '{"changed": true}\n',
            encoding="utf-8",
        )
    elif corruption == "candidate_changed":
        provenance["candidate_id"] = "other-candidate"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "config_changed":
        provenance["config"] = {"processes": 2}
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "attempt_directory_changed":
        provenance["attempt_id"] = "22222222-2222-4222-8222-222222222222"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "missing_records":
        (attempt_dir / "private-records.jsonl").unlink()
    elif corruption == "partial_status":
        provenance["status"] = "partial_failure"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "public_workload":
        provenance["workload_class"] = "public_course"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "malformed_provenance":
        provenance_path.write_text("{", encoding="utf-8")
    elif corruption == "scalar_provenance":
        provenance_path.write_text("null", encoding="utf-8")
    elif corruption == "non_utf8_provenance":
        provenance_path.write_bytes(b"\xff\xfe")
    elif corruption == "nonstr_workload_class":
        provenance["workload_class"] = ["private_course"]
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "nonstr_records_sha256":
        provenance["records_sha256"] = ["0" * 64]
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "controller_environment_changed":
        provenance["controller_environment_fingerprint"] = "2" * 16
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    elif corruption == "execution_policy_changed":
        provenance["execution_policy_fingerprint"] = "3" * 16
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    assert run_sustained._successful_private_artifact_attempt_keys(
        candidate_dir,
        invalidated_attempt_ids=invalidated_attempt_ids,
        expected_bindings={"a" * 16: binding},
    ) == set()


@pytest.mark.parametrize(
    "failure_point",
    (
        "process_monitor_constructor",
        "process_monitor_start",
        "host_monitor_constructor",
        "host_monitor_start",
        "host_monitor_ready",
    ),
)
def test_monitor_startup_failure_terminates_spawned_worker(
    tmp_path: Path,
    monkeypatch,
    failure_point: str,
) -> None:
    terminated: list[int] = []
    stopped: list[str] = []
    spawned: list[int] = []
    closed_jobs: list[bool] = []

    class FakeWorkerJob(_FakeWorkerJob):
        def close(self) -> None:
            if not self.closed:
                closed_jobs.append(True)
            super().close()

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeProcessMonitor:
        monitor_error = None

        def __init__(self, *_args, **_kwargs):
            if failure_point == "process_monitor_constructor":
                raise RuntimeError("injected process monitor constructor failure")

        def start(self):
            if failure_point == "process_monitor_start":
                raise RuntimeError("injected process monitor start failure")

        @staticmethod
        def stop():
            stopped.append("process")
            return {"sample_count": 0}

    class FakeHostMonitor:
        stop_reason = None
        start_error = None

        def __init__(self, *_args, **_kwargs):
            if failure_point == "host_monitor_constructor":
                raise RuntimeError("injected host monitor constructor failure")

        def start(self):
            if failure_point == "host_monitor_start":
                raise RuntimeError("injected host monitor start failure")

        @staticmethod
        def wait_until_ready(_timeout_seconds):
            return failure_point != "host_monitor_ready"

        @staticmethod
        def stop():
            stopped.append("host")
            return {"status": "no_samples"}

    monkeypatch.setattr(run_sustained, "SUSTAINED_ARTIFACTS_PATH", tmp_path / "runs")
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", tmp_path / "events.jsonl")
    def fake_popen(*_args, **_kwargs):
        spawned.append(FakeProcess.pid)
        return FakeProcess()

    monkeypatch.setattr(run_sustained.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_sustained, "ProcessTreeMonitor", FakeProcessMonitor)
    monkeypatch.setattr(run_sustained, "WindowsHostMonitor", FakeHostMonitor)
    monkeypatch.setattr(run_sustained, "WindowsKillOnCloseJob", FakeWorkerJob)
    monkeypatch.setattr(
        run_sustained,
        "terminate_process_tree",
        lambda pid: _record_successful_termination(terminated, pid),
    )

    with pytest.raises(RuntimeError, match="injected|preflight"):
        run_sustained._run_attempt(
            registry={
                "protocol": "sustained-process-v1",
                "resource_sample_interval_seconds": 0.25,
                "host_sample_interval_seconds": 2.0,
                "timeout_overhead_seconds": 10,
            },
            candidate={
                "id": "candidate",
                "task": "asr",
                "worker": "workers/fake_worker.py",
            },
            python=Path("python.exe"),
            workload={
                "workload_class": "private_course",
                "items": [{"id": "sample", "path": "ignored.wav"}],
                "warmup_item": {"id": "sample", "path": "ignored.wav"},
                "public_summary": {
                    "workload_class": "private_course",
                    "item_count": 1,
                    "total_duration_seconds": 1,
                },
                "fingerprint": "c" * 64,
            },
            config={"processes": 1},
            config_index=0,
            phase="sustained",
            target_wall_seconds=1,
            trial_index=0,
            attempt_key="a" * 16,
            code_fingerprint="d" * 16,
            environment_fingerprint="e" * 16,
            controller_environment_fingerprint="f" * 16,
            execution_policy_fingerprint="1" * 16,
        )

    worker_was_spawned = failure_point in {
        "process_monitor_constructor",
        "process_monitor_start",
    }
    assert spawned == ([4321] if worker_was_spawned else [])
    assert terminated == []
    assert closed_jobs == ([True] if worker_was_spawned else [])
    if failure_point == "process_monitor_start":
        assert "process" in stopped
    if failure_point in {"host_monitor_start", "host_monitor_ready"}:
        assert "host" in stopped
    assert not (tmp_path / "events.jsonl").exists()


def test_assignment_and_fallback_termination_failure_is_fail_closed(
    tmp_path,
    monkeypatch,
) -> None:
    class AssignmentFailingJob(_FakeWorkerJob):
        @staticmethod
        def assign(_process) -> None:
            raise RuntimeError("injected assignment failure")

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeHostMonitor:
        start_error = None
        stop_reason = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def wait_until_ready(_timeout_seconds):
            return True

        @staticmethod
        def stop():
            return {
                "status": "observed",
                "sample_count": 1,
                "monitor_partial": False,
            }

    def fake_popen(*_args, **kwargs):
        assert kwargs["creationflags"] == run_sustained.CREATE_SUSPENDED
        return FakeProcess()

    monkeypatch.setattr(run_sustained, "SUSTAINED_ARTIFACTS_PATH", tmp_path / "runs")
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", tmp_path / "events.jsonl")
    monkeypatch.setattr(run_sustained.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_sustained, "WindowsHostMonitor", FakeHostMonitor)
    monkeypatch.setattr(run_sustained, "WindowsKillOnCloseJob", AssignmentFailingJob)
    monkeypatch.setattr(run_sustained, "_terminate_unassigned_worker", lambda _pid: False)

    with pytest.raises(RuntimeError, match="termination could not be verified") as raised:
        run_sustained._run_attempt(**_minimal_attempt_kwargs())

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "assignment failure" in str(raised.value.__cause__)
    assert not (tmp_path / "events.jsonl").exists()


def test_midrun_host_monitor_failure_terminates_worker(monkeypatch) -> None:
    terminated: list[int] = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeHostMonitor:
        start_error = "counter_process_exit"
        stop_reason = None

    class FakeProcessMonitor:
        monitor_error = None

    monkeypatch.setattr(
        run_sustained,
        "terminate_process_tree",
        lambda pid: _record_successful_termination(terminated, pid),
    )

    assert run_sustained._wait_for_process(
        FakeProcess(),
        host_monitor=FakeHostMonitor(),
        process_monitor=FakeProcessMonitor(),
        timeout_seconds=60,
    ) == (-2, "monitor_failure")
    assert terminated == [4321]


def test_safety_stop_wins_worker_exit_race(monkeypatch) -> None:
    terminated: list[int] = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return 0

    class FakeHostMonitor:
        start_error = None
        stop_reason = "available_memory_below_4_gib"

    class FakeProcessMonitor:
        monitor_error = None

    monkeypatch.setattr(
        run_sustained,
        "terminate_process_tree",
        lambda pid: _record_successful_termination(terminated, pid),
    )

    assert run_sustained._wait_for_process(
        FakeProcess(),
        host_monitor=FakeHostMonitor(),
        process_monitor=FakeProcessMonitor(),
        timeout_seconds=60,
    ) == (0, "safety_stop")
    assert terminated == [4321]


def test_process_monitor_failure_terminates_worker(monkeypatch) -> None:
    terminated: list[int] = []

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeHostMonitor:
        start_error = None
        stop_reason = None

    class FakeProcessMonitor:
        monitor_error = "IsADirectoryError"

    monkeypatch.setattr(
        run_sustained,
        "terminate_process_tree",
        lambda pid: _record_successful_termination(terminated, pid),
    )

    assert run_sustained._wait_for_process(
        FakeProcess(),
        host_monitor=FakeHostMonitor(),
        process_monitor=FakeProcessMonitor(),
        timeout_seconds=60,
    ) == (-2, "monitor_failure")
    assert terminated == [4321]


def test_cleanup_failure_appends_terminal_failure(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return 0

    class FakeProcessMonitor:
        monitor_error = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def stop():
            raise RuntimeError("injected process monitor cleanup failure")

    class FakeHostMonitor:
        start_error = None
        stop_reason = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def wait_until_ready(_timeout_seconds):
            return True

        @staticmethod
        def stop():
            return {
                "status": "observed",
                "sample_count": 1,
                "monitor_partial": False,
                "package_temperature_available": False,
            }

    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_ARTIFACTS_PATH", tmp_path / "runs")
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    monkeypatch.setattr(
        run_sustained.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(run_sustained, "ProcessTreeMonitor", FakeProcessMonitor)
    monkeypatch.setattr(run_sustained, "WindowsHostMonitor", FakeHostMonitor)
    monkeypatch.setattr(run_sustained, "WindowsKillOnCloseJob", _FakeWorkerJob)

    with pytest.raises(RuntimeError, match="injected process monitor cleanup"):
        run_sustained._run_attempt(
            registry={
                "protocol": "sustained-process-v1",
                "resource_sample_interval_seconds": 0.25,
                "host_sample_interval_seconds": 2.0,
                "timeout_overhead_seconds": 10,
            },
            candidate={
                "id": "candidate",
                "task": "asr",
                "worker": "workers/fake_worker.py",
            },
            python=Path("python.exe"),
            workload={
                "workload_class": "private_course",
                "items": [{"id": "sample", "path": "ignored.wav"}],
                "warmup_item": {"id": "sample", "path": "ignored.wav"},
                "public_summary": {
                    "workload_class": "private_course",
                    "item_count": 1,
                    "total_duration_seconds": 1,
                },
                "fingerprint": "c" * 64,
            },
            config={"processes": 1},
            config_index=0,
            phase="sustained",
            target_wall_seconds=1,
            trial_index=0,
            attempt_key="a" * 16,
            code_fingerprint="d" * 16,
            environment_fingerprint="e" * 16,
            controller_environment_fingerprint="f" * 16,
            execution_policy_fingerprint="1" * 16,
        )

    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "sustained_attempt_started",
        "sustained_attempt_failed",
    ]
    assert events[-1]["failure_kind"] == "monitor_failure"


def test_controller_error_and_job_close_failure_are_both_preserved(
    tmp_path,
    monkeypatch,
) -> None:
    class FailingWorkerJob(_FakeWorkerJob):
        def close(self) -> None:
            raise RuntimeError("injected Job Object close failure")

    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    class FakeProcessMonitor:
        monitor_error = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def stop():
            return {"sample_count": 1}

    class FakeHostMonitor:
        start_error = None
        stop_reason = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def wait_until_ready(_timeout_seconds):
            return True

        @staticmethod
        def stop():
            return {
                "status": "observed",
                "sample_count": 1,
                "monitor_partial": False,
                "package_temperature_available": False,
            }

    def fake_popen(*_args, **kwargs):
        assert kwargs["creationflags"] == run_sustained.CREATE_SUSPENDED
        return FakeProcess()

    def injected_controller_failure(*_args, **_kwargs):
        raise RuntimeError("injected controller failure")

    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_ARTIFACTS_PATH", tmp_path / "runs")
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    monkeypatch.setattr(run_sustained.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run_sustained, "ProcessTreeMonitor", FakeProcessMonitor)
    monkeypatch.setattr(run_sustained, "WindowsHostMonitor", FakeHostMonitor)
    monkeypatch.setattr(run_sustained, "WindowsKillOnCloseJob", FailingWorkerJob)
    monkeypatch.setattr(run_sustained, "_wait_for_process", injected_controller_failure)

    with pytest.raises(RuntimeError, match="injected controller failure"):
        run_sustained._run_attempt(**_minimal_attempt_kwargs())

    events = read_events(events_path)
    assert [event["event"] for event in events] == [
        "sustained_attempt_started",
        "sustained_attempt_failed",
    ]
    assert events[-1]["failure_kind"] == "termination_failure"


def test_keyboard_interrupt_is_journaled_and_propagated(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        pid = 4321

        @staticmethod
        def poll():
            return -3

    class FakeProcessMonitor:
        monitor_error = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def stop():
            return {"sample_count": 1}

    class FakeHostMonitor:
        start_error = None
        stop_reason = None

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def start():
            pass

        @staticmethod
        def wait_until_ready(_timeout_seconds):
            return True

        @staticmethod
        def stop():
            return {
                "status": "observed",
                "sample_count": 1,
                "monitor_partial": False,
                "package_temperature_available": False,
            }

    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_ARTIFACTS_PATH", tmp_path / "runs")
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    monkeypatch.setattr(
        run_sustained.subprocess,
        "Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    monkeypatch.setattr(run_sustained, "ProcessTreeMonitor", FakeProcessMonitor)
    monkeypatch.setattr(run_sustained, "WindowsHostMonitor", FakeHostMonitor)
    monkeypatch.setattr(run_sustained, "WindowsKillOnCloseJob", _FakeWorkerJob)
    monkeypatch.setattr(
        run_sustained,
        "_wait_for_process",
        lambda *_args, **_kwargs: (-3, "interrupted"),
    )

    with pytest.raises(KeyboardInterrupt):
        run_sustained._run_attempt(
            registry={
                "protocol": "sustained-process-v1",
                "resource_sample_interval_seconds": 0.25,
                "host_sample_interval_seconds": 2.0,
                "timeout_overhead_seconds": 10,
            },
            candidate={
                "id": "candidate",
                "task": "asr",
                "worker": "workers/fake_worker.py",
            },
            python=Path("python.exe"),
            workload={
                "workload_class": "private_course",
                "items": [{"id": "sample", "path": "ignored.wav"}],
                "warmup_item": {"id": "sample", "path": "ignored.wav"},
                "public_summary": {
                    "workload_class": "private_course",
                    "item_count": 1,
                    "total_duration_seconds": 1,
                },
                "fingerprint": "c" * 64,
            },
            config={"processes": 1},
            config_index=0,
            phase="sustained",
            target_wall_seconds=1,
            trial_index=0,
            attempt_key="a" * 16,
            code_fingerprint="d" * 16,
            environment_fingerprint="e" * 16,
            controller_environment_fingerprint="f" * 16,
            execution_policy_fingerprint="1" * 16,
        )

    events = read_events(events_path)
    assert [event["event"] for event in events] == [
        "sustained_attempt_started",
        "sustained_attempt_failed",
    ]
    assert events[-1]["failure_kind"] == "interrupted"


def test_candidate_sweep_stops_after_keyboard_interrupt(tmp_path, monkeypatch) -> None:
    registry = {
        "protocol": "sustained-process-v1",
        "resource_sample_interval_seconds": 0.5,
        "host_sample_interval_seconds": 2.0,
        "timeout_overhead_seconds": 10,
    }
    candidate = {
        "id": "candidate",
        "task": "asr",
        "environment": "control",
        "environment_manifest": "environment.txt",
        "worker": "worker.py",
        "configs": [{"processes": 1}, {"processes": 2}],
    }
    workload = {
        "workload_class": "generated_quality_control",
        "items": [{"id": "sample", "path": "sample.wav"}],
        "warmup_item": {"id": "sample", "path": "sample.wav"},
        "public_summary": {
            "workload_class": "generated_quality_control",
            "item_count": 1,
            "total_duration_seconds": 1,
        },
        "fingerprint": "c" * 64,
    }
    attempts: list[int] = []
    monkeypatch.setattr(run_sustained, "load_json", lambda _path: registry)
    monkeypatch.setattr(run_sustained, "find_candidate", lambda *_args: candidate)
    monkeypatch.setattr(
        run_sustained,
        "load_sustained_workload",
        lambda *_args, **_kwargs: workload,
    )
    monkeypatch.setattr(run_sustained, "_python_for", lambda _name: Path(__file__))
    monkeypatch.setattr(run_sustained, "_verify_candidate_environment", lambda *_args: None)
    monkeypatch.setattr(run_sustained, "_verify_candidate_artifacts", lambda *_args: None)
    monkeypatch.setattr(run_sustained, "read_events", lambda _path: [])
    monkeypatch.setattr(
        run_sustained,
        "effective_sustained_invalidated_attempt_ids",
        lambda _path: set(),
    )
    monkeypatch.setattr(
        run_sustained,
        "_capture_environment_fingerprint",
        lambda *_args: "e" * 16,
    )
    monkeypatch.setattr(
        run_sustained,
        "_stable_hardware",
        lambda _hardware: {"visible_processors": 24},
    )
    monkeypatch.setattr(run_sustained, "fingerprint_files", lambda _paths: "d" * 16)

    def interrupting_attempt(**kwargs):
        attempts.append(kwargs["config_index"])
        raise KeyboardInterrupt

    monkeypatch.setattr(run_sustained, "_run_attempt", interrupting_attempt)

    with pytest.raises(KeyboardInterrupt):
        run_sustained.run_sustained_candidate(
            "candidate",
            tmp_path / "workload.json",
            phase="screen",
            target_wall_seconds=1,
        )

    assert attempts == [0]


def test_public_attempts_retain_reproducible_attempt_key() -> None:
    common = _common()
    common["workload"] = {
        "workload_class": "generated_quality_control",
        "item_count": 1,
        "total_duration_seconds": 60,
    }

    assert run_sustained._public_attempt_common(common)["attempt_key"] == "key"
    assert (
        run_sustained._public_attempt_common(common)["workload_fingerprint"]
        == "c" * 64
    )
    assert "attempt_key" not in run_sustained._public_attempt_common(_common())
    assert "workload_fingerprint" not in run_sustained._public_attempt_common(
        _common()
    )
    common["workload"]["workload_class"] = "future_unclassified_workload"
    assert "attempt_key" not in run_sustained._public_attempt_common(common)


def test_execution_policy_fingerprint_changes_with_telemetry_or_candidate() -> None:
    registry = {
        "protocol": "sustained-process-v1",
        "resource_sample_interval_seconds": 0.5,
        "host_sample_interval_seconds": 2.0,
        "timeout_overhead_seconds": 1800,
    }
    candidate = {"id": "candidate", "configs": [{"processes": 1}]}
    baseline = run_sustained._execution_policy_fingerprint(registry, candidate)

    changed_registry = dict(registry, host_sample_interval_seconds=3.0)
    changed_candidate = {"id": "candidate", "configs": [{"processes": 2}]}
    assert (
        run_sustained._execution_policy_fingerprint(changed_registry, candidate)
        != baseline
    )
    assert (
        run_sustained._execution_policy_fingerprint(registry, changed_candidate)
        != baseline
    )


def test_phase_scoped_configs_are_selected_without_crossing_safety_gate():
    candidate = {
        "configs": [
            {"phases": ["compatibility"]},
            {"phases": ["quality"]},
            {"processes": 1},
        ]
    }

    assert run_sustained._select_config_indices(
        candidate,
        "compatibility",
        None,
    ) == (0, 2)
    assert run_sustained._select_config_indices(candidate, "quality", None) == (1, 2)


def test_explicit_phase_mismatch_is_rejected():
    candidate = {"configs": [{"phases": ["compatibility"]}]}

    try:
        run_sustained._select_config_indices(candidate, "quality", (0,))
    except ValueError as error:
        assert "does not support" in str(error)
    else:
        raise AssertionError("phase-mismatched config was accepted")


def test_retired_candidate_and_policy_index_cannot_be_selected() -> None:
    with pytest.raises(ValueError, match="candidate is retired"):
        run_sustained._select_config_indices(
            {"status": "retired_legacy", "configs": [{}]},
            "screen",
            None,
        )
    candidate = {
        "retired_config_indices": [0],
        "configs": [{}, {"processes": 1}],
    }
    assert run_sustained._select_config_indices(candidate, "screen", None) == (1,)
    with pytest.raises(ValueError, match="does not support"):
        run_sustained._select_config_indices(candidate, "screen", (0,))


def test_official_qwen_preserves_but_retires_unusable_configs() -> None:
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "registries"
        / "sustained_candidates.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidate = next(
        item
        for item in registry["candidates"]
        if item["id"] == "qwen3_asr_0_6b_openvino_genai_official"
    )

    assert candidate["retired_config_indices"] == [2, 3]
    assert [config["max_new_tokens"] for config in candidate["configs"]] == [
        512,
        512,
        4096,
        4096,
    ]
    assert run_sustained._select_config_indices(candidate, "sustained", None) == (
        0,
        1,
    )
    for retired_index in candidate["retired_config_indices"]:
        with pytest.raises(ValueError, match="does not support"):
            run_sustained._select_config_indices(
                candidate,
                "sustained",
                (retired_index,),
            )


def test_candidate_wide_phase_policy_does_not_mutate_historical_configs() -> None:
    candidate = {
        "allowed_phases": ["quality", "compatibility"],
        "configs": [{"processes": 1}],
    }

    assert run_sustained._select_config_indices(candidate, "quality", None) == (0,)
    with pytest.raises(ValueError, match="no config"):
        run_sustained._select_config_indices(candidate, "screen", None)


def test_duplicate_or_noninteger_config_indices_are_rejected() -> None:
    candidate = {"configs": [{}, {}]}
    with pytest.raises(ValueError, match="distinct"):
        run_sustained._select_config_indices(candidate, "screen", (0, 0))
    for indices in ((True,), ("0",)):
        with pytest.raises(ValueError, match="out of range"):
            run_sustained._select_config_indices(candidate, "screen", indices)


def test_environment_identity_capture_is_bounded(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(run_sustained.subprocess, "run", timeout)

    try:
        run_sustained._capture_environment_fingerprint(
            Path("python.exe"),
            Path("identity.py"),
        )
    except RuntimeError as error:
        assert "timed out" in str(error)
    else:
        raise AssertionError("unbounded environment identity capture was accepted")


def test_trials_alternate_config_order() -> None:
    selected = (15, 16)

    assert run_sustained._ordered_config_indices(selected, 0) == (15, 16)
    assert run_sustained._ordered_config_indices(selected, 1) == (16, 15)
    assert run_sustained._ordered_config_indices(selected, 2) == (15, 16)


def test_artifact_groups_apply_only_to_selected_config() -> None:
    candidate = {
        "artifact_files": ["common.bin"],
        "artifact_groups": {"source": ["source.exe", "provenance.json"]},
    }

    assert run_sustained._artifact_files(candidate, {}) == ["common.bin"]
    assert run_sustained._artifact_files(
        candidate,
        {"artifact_group": "source"},
    ) == ["common.bin", "source.exe", "provenance.json"]
    with pytest.raises(ValueError, match="unknown artifact group"):
        run_sustained._artifact_files(candidate, {"artifact_group": "missing"})

    candidate["artifact_group_by_runtime_variant"] = {"source": "source"}
    assert run_sustained._artifact_files(
        candidate,
        {"runtime_variant": "source"},
    ) == ["common.bin", "source.exe", "provenance.json"]

    candidate["default_artifact_group"] = "source"
    assert run_sustained._artifact_files(candidate, {}) == [
        "common.bin",
        "source.exe",
        "provenance.json",
    ]


def test_artifact_verification_ignores_unselected_group_and_gives_setup_hint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(run_sustained, "PROJECT_ROOT", tmp_path)
    (tmp_path / "common.bin").write_bytes(b"common")
    candidate = {
        "id": "candidate",
        "setup_script": "scripts/setup.ps1",
        "artifact_files": ["common.bin"],
        "artifact_groups": {"source": ["source.exe"]},
        "configs": [{}, {"artifact_group": "source"}],
    }

    run_sustained._verify_candidate_artifacts(candidate, (0,))
    with pytest.raises(FileNotFoundError, match=r"source[.]exe.*setup[.]ps1"):
        run_sustained._verify_candidate_artifacts(candidate, (1,))
