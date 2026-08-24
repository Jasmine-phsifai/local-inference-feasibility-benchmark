import json
import sys
from dataclasses import dataclass

import pytest

from workers import faster_whisper_sustained_worker
from workers import paddleocr_sustained_worker
from workers import rapidocr_sustained_worker


@dataclass(frozen=True)
class WorkerCase:
    module: object
    candidate_id: str
    config: dict
    private_deadline_seconds: float


WORKER_CASES = (
    WorkerCase(
        module=rapidocr_sustained_worker,
        candidate_id="rapidocr_cpu",
        config={
            "processes": 2,
            "threads_per_process": 1,
            "backend": "onnxruntime",
        },
        private_deadline_seconds=1200.0,
    ),
    WorkerCase(
        module=paddleocr_sustained_worker,
        candidate_id="paddleocr_ppocrv6_cpu",
        config={
            "processes": 2,
            "threads_per_process": 1,
            "model_tier": "tiny",
        },
        private_deadline_seconds=1200.0,
    ),
    WorkerCase(
        module=faster_whisper_sustained_worker,
        candidate_id="faster_whisper_cpu",
        config={
            "processes": 2,
            "model_workers": 1,
            "threads_per_worker": 1,
        },
        private_deadline_seconds=1801.0,
    ),
)


class FakeEvent:
    def __init__(self) -> None:
        self.was_set = False

    def set(self) -> None:
        self.was_set = True


class FakeValue:
    def __init__(self, value: float) -> None:
        self.value = value


class FakeProcess:
    def __init__(self, *, fail_on_start: bool = False) -> None:
        self.fail_on_start = fail_on_start
        self.start_attempted = False
        self.started = False
        self.terminated = False
        self.joined = False
        self.exitcode = 0

    def start(self) -> None:
        self.start_attempted = True
        if self.fail_on_start:
            raise OSError("synthetic child start failure")
        self.started = True

    def is_alive(self) -> bool:
        return self.started and not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def join(self, *, timeout: float) -> None:
        assert timeout == 30
        self.joined = True
        self.started = False


class UnusedQueue:
    def get(self, *, timeout: float):
        raise AssertionError(f"queue should not be read with timeout={timeout}")


class FailingQueue:
    def get(self, *, timeout: float):
        assert 0 < timeout <= 5.0
        raise OSError("synthetic queue transport failure")


class ReadyQueue:
    def __init__(self, message: dict) -> None:
        self.message = message

    def get(self, *, timeout: float) -> dict:
        assert 0 < timeout <= 5.0
        return self.message


class DoneQueue:
    def __init__(self, message: dict, *, expected_timeout: float) -> None:
        self.message = message
        self.expected_timeout = expected_timeout
        self.observed_timeouts: list[float] = []

    def get(self, *, timeout: float) -> dict:
        self.observed_timeouts.append(timeout)
        assert timeout == pytest.approx(self.expected_timeout)
        return self.message


class FakeContext:
    def __init__(self, processes: list[FakeProcess], queues: list[object]) -> None:
        self.processes = processes
        self.queues = iter(queues)

    def Event(self) -> FakeEvent:
        return FakeEvent()

    def Value(self, _kind: str, value: float) -> FakeValue:
        return FakeValue(value)

    def Queue(self):
        return next(self.queues)

    def Process(self, *, target, args):
        del target
        return self.processes[args[0]]


def _write_request(tmp_path, case: WorkerCase, *, phase: str) -> str:
    request_path = tmp_path / f"{case.candidate_id}-{phase}.json"
    request = {
        "candidate_id": case.candidate_id,
        "capture_predictions": False,
        "phase": phase,
        "target_wall_seconds": 1.0,
        "config": case.config,
        "workload": {
            "workload_class": "test-private-phase",
            "warmup_item": {"id": "warmup", "path": "unused"},
            "items": [],
        },
        "private_records_path": str(tmp_path / "private.json"),
        "response_path": str(tmp_path / "response.json"),
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return str(request_path)


@pytest.mark.parametrize("case", WORKER_CASES, ids=lambda case: case.candidate_id)
def test_partial_child_start_failure_stops_only_started_children(
    case: WorkerCase,
    monkeypatch,
    tmp_path,
) -> None:
    first = FakeProcess()
    second = FakeProcess(fail_on_start=True)
    context = FakeContext([first, second], [UnusedQueue(), UnusedQueue()])
    request_path = _write_request(tmp_path, case, phase="quality")
    monkeypatch.setattr(case.module.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(sys, "argv", ["worker", "--request", request_path])

    with pytest.raises(OSError, match="synthetic child start failure"):
        case.module.main()

    assert first.start_attempted is True
    assert first.terminated is True
    assert first.joined is True
    assert second.start_attempted is True
    assert second.terminated is False
    assert second.joined is False


def test_rapidocr_queue_transport_failure_stops_started_child(
    monkeypatch,
    tmp_path,
) -> None:
    process = FakeProcess()
    case = WorkerCase(
        module=rapidocr_sustained_worker,
        candidate_id="rapidocr_cpu",
        config={
            "processes": 1,
            "threads_per_process": 1,
            "backend": "onnxruntime",
        },
        private_deadline_seconds=1200.0,
    )
    context = FakeContext([process], [FailingQueue(), UnusedQueue()])
    request_path = _write_request(tmp_path, case, phase="quality")
    monkeypatch.setattr(
        rapidocr_sustained_worker.multiprocessing,
        "get_context",
        lambda _: context,
    )
    monkeypatch.setattr(sys, "argv", ["worker", "--request", request_path])

    with pytest.raises(OSError, match="synthetic queue transport failure"):
        rapidocr_sustained_worker.main()

    assert process.start_attempted is True
    assert process.terminated is True
    assert process.joined is True


@pytest.mark.parametrize("phase", ("quality", "compatibility"))
@pytest.mark.parametrize("case", WORKER_CASES, ids=lambda case: case.candidate_id)
def test_private_phase_retains_bounded_long_result_collection_deadline(
    case: WorkerCase,
    phase: str,
    monkeypatch,
    tmp_path,
) -> None:
    process = FakeProcess()
    process_count_one_config = {**case.config, "processes": 1}
    one_process_case = WorkerCase(
        module=case.module,
        candidate_id=case.candidate_id,
        config=process_count_one_config,
        private_deadline_seconds=case.private_deadline_seconds,
    )
    ready = {
        "process_index": 0,
        "success": True,
        "load_seconds": 0.1,
        "warmup_seconds": (
            [0.2]
            if case.module is faster_whisper_sustained_worker
            else 0.2
        ),
    }
    done = {"kind": "done", "process_index": 0}
    if case.module is faster_whisper_sustained_worker:
        done["concurrency"] = {}
    result_queue = DoneQueue(done, expected_timeout=1.0)
    context = FakeContext([process], [ReadyQueue(ready), result_queue])
    request_path = _write_request(tmp_path, one_process_case, phase=phase)
    monotonic_values = iter(
        (
            0.0,
            0.0,
            0.0,
            one_process_case.private_deadline_seconds - 1.0,
        )
    )
    perf_counter_values = iter((10.0, 11.0))
    monkeypatch.setattr(case.module.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(case.module.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        case.module.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(case.module, "write_private_records", lambda *_: None)
    monkeypatch.setattr(case.module, "build_public_summary", lambda **_: {})
    monkeypatch.setattr(case.module.importlib.metadata, "version", lambda _: "test")
    if case.module is faster_whisper_sustained_worker:
        monkeypatch.setattr(
            case.module,
            "_aggregate_concurrency_diagnostics",
            lambda **_: {},
        )
    monkeypatch.setattr(sys, "argv", ["worker", "--request", request_path])

    case.module.main()

    assert result_queue.observed_timeouts == [pytest.approx(1.0)]
