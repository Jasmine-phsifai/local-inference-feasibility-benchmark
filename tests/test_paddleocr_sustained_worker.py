import json
import sys

import pytest

from workers import paddleocr_sustained_worker
from workers.paddleocr_sustained_worker import (
    _claim_process_index,
    _configure_openmp_environment,
    _recognize,
    _validated_opencv_threads,
)


class _FakeEvent:
    def set(self) -> None:
        pass


class _FakeValue:
    def __init__(self, value: float) -> None:
        self.value = value


class _FakeProcess:
    def __init__(self) -> None:
        self.started = False
        self.terminated = False
        self.joined = False
        self.exitcode = None

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started and not self.terminated

    def terminate(self) -> None:
        self.terminated = True

    def join(self, *, timeout: float) -> None:
        assert timeout == 30
        self.joined = True
        self.started = False


class _QueueResponse:
    def __init__(self, value) -> None:
        self.value = value

    def get(self, *, timeout: float):
        assert 0 < timeout <= 5.0
        if isinstance(self.value, BaseException):
            raise self.value
        return self.value


class _FakeContext:
    def __init__(self, process: _FakeProcess, queues: list[_QueueResponse]) -> None:
        self.process = process
        self.queues = iter(queues)

    def Event(self) -> _FakeEvent:
        return _FakeEvent()

    def Value(self, _kind: str, value: float) -> _FakeValue:
        return _FakeValue(value)

    def Queue(self) -> _QueueResponse:
        return next(self.queues)

    def Process(self, *, target, args) -> _FakeProcess:
        del target, args
        return self.process


def _write_lifecycle_request(tmp_path) -> str:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "candidate_id": "paddleocr_ppocrv6_cpu",
                "capture_predictions": False,
                "phase": "quality",
                "target_wall_seconds": 1.0,
                "config": {
                    "processes": 1,
                    "threads_per_process": 1,
                    "model_tier": "tiny",
                },
                "workload": {
                    "workload_class": "generated-quality-control",
                    "warmup_item": {"id": "warmup", "path": "unused"},
                    "items": [],
                },
                "private_records_path": str(tmp_path / "records.jsonl"),
                "response_path": str(tmp_path / "response.json"),
            }
        ),
        encoding="utf-8",
    )
    return str(request_path)


def _run_with_queue_responses(
    *,
    tmp_path,
    monkeypatch,
    ready_response,
    result_response,
) -> _FakeProcess:
    process = _FakeProcess()
    context = _FakeContext(
        process,
        [_QueueResponse(ready_response), _QueueResponse(result_response)],
    )
    monkeypatch.setattr(
        paddleocr_sustained_worker.multiprocessing,
        "get_context",
        lambda _: context,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["worker", "--request", _write_lifecycle_request(tmp_path)],
    )
    return process


_READY_MESSAGE = {
    "process_index": 0,
    "success": True,
    "load_seconds": 0.1,
    "warmup_seconds": 0.2,
}


@pytest.mark.parametrize("failure", [OSError("transport"), EOFError("closed")])
@pytest.mark.parametrize("stage", ["readiness", "result"])
def test_queue_transport_failure_always_stops_started_child(
    failure,
    stage,
    monkeypatch,
    tmp_path,
) -> None:
    process = _run_with_queue_responses(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        ready_response=failure if stage == "readiness" else _READY_MESSAGE,
        result_response=failure if stage == "result" else None,
    )

    with pytest.raises(type(failure), match=str(failure)):
        paddleocr_sustained_worker.main()

    assert process.started is False
    assert process.terminated is True
    assert process.joined is True


@pytest.mark.parametrize("stage", ["readiness", "result"])
def test_malformed_worker_message_stops_started_child(
    stage,
    monkeypatch,
    tmp_path,
) -> None:
    process = _run_with_queue_responses(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        ready_response=[] if stage == "readiness" else _READY_MESSAGE,
        result_response=[] if stage == "result" else None,
    )

    with pytest.raises(RuntimeError, match=f"{stage} message is invalid"):
        paddleocr_sustained_worker.main()

    assert process.started is False
    assert process.terminated is True
    assert process.joined is True


def test_configures_optional_intel_openmp_blocktime() -> None:
    environment = {}

    _configure_openmp_environment(
        {"kmp_blocktime_ms": 0},
        4,
        environment,
    )

    assert environment == {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "KMP_BLOCKTIME": "0",
    }


def test_default_blocktime_does_not_inherit_parent_override() -> None:
    environment = {"KMP_BLOCKTIME": "0"}

    _configure_openmp_environment({}, 4, environment)

    assert environment == {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}


@pytest.mark.parametrize("invalid", [True, 0.0, "0", -1, 1001])
def test_rejects_unsafe_openmp_blocktime(invalid) -> None:
    with pytest.raises(ValueError, match="kmp_blocktime_ms"):
        _configure_openmp_environment(
            {"kmp_blocktime_ms": invalid},
            4,
            {},
        )


def test_validates_optional_opencv_thread_count() -> None:
    assert _validated_opencv_threads({}) is None
    assert _validated_opencv_threads({"opencv_threads": 1}) == 1

    for invalid in (True, 1.0, "1", 0, 25):
        with pytest.raises(ValueError, match="opencv_threads"):
            _validated_opencv_threads({"opencv_threads": invalid})


def test_process_index_is_bounded_and_unique() -> None:
    assert (
        _claim_process_index(
            {"process_index": 1},
            process_count=2,
            claimed={0},
            stage="completion",
        )
        == 1
    )
    for invalid in (True, -1, 2, "1"):
        with pytest.raises(RuntimeError, match="completion index"):
            _claim_process_index(
                {"process_index": invalid},
                process_count=2,
                claimed=set(),
                stage="completion",
            )
    with pytest.raises(RuntimeError, match="completion index"):
        _claim_process_index(
            {"process_index": 1},
            process_count=2,
            claimed={1},
            stage="completion",
        )


def test_expected_text_rejects_whitespace_only_paddleocr_output() -> None:
    class WhitespaceOnlyOutput:
        json = {
            "res": {
                "rec_texts": ["  ", "\t"],
                "rec_scores": [0.9, 0.8],
                "rec_polys": [],
            }
        }

    class FakeEngine:
        def predict(self, _path):
            return [WhitespaceOnlyOutput()]

    record = _recognize(
        FakeEngine(),
        {"id": "expected-text", "path": "unused.png", "expected_text": True},
        capture_prediction=True,
    )

    assert record["success"] is False
    assert record["failure_kind"] == "empty_output"
    assert record["units"] == 0.0
    assert [line["text"] for line in record["lines"]] == ["  ", "\t"]
