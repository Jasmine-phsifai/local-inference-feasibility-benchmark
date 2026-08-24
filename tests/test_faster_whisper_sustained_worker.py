import json
import sys

import pytest

from workers import faster_whisper_sustained_worker
from workers.faster_whisper_sustained_worker import (
    _PythonCallConcurrencyProbe,
    _aggregate_concurrency_diagnostics,
    _claim_process_index,
    _transcribe,
    _validated_language,
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
                "candidate_id": "faster_whisper_cpu",
                "capture_predictions": False,
                "phase": "quality",
                "target_wall_seconds": 1.0,
                "config": {
                    "processes": 1,
                    "model_workers": 1,
                    "threads_per_worker": 1,
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
        faster_whisper_sustained_worker.multiprocessing,
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
    "warmup_seconds": [0.2],
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
        faster_whisper_sustained_worker.main()

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
        faster_whisper_sustained_worker.main()

    assert process.started is False
    assert process.terminated is True
    assert process.joined is True


def test_tracks_peak_python_calls() -> None:
    probe = _PythonCallConcurrencyProbe()

    probe.enter()
    probe.enter()
    probe.exit()
    probe.exit()

    assert probe.peak == 2


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
        try:
            _claim_process_index(
                {"process_index": invalid},
                process_count=2,
                claimed=set(),
                stage="completion",
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"unexpected accepted process index: {invalid!r}")


def test_aggregates_ctranslate2_saturation_without_free_form_values() -> None:
    summary = _aggregate_concurrency_diagnostics(
        diagnostics=[
            {
                "runtime_num_workers": 6,
                "python_calls_in_flight_peak": 6,
                "sample_count": 100,
                "busy_sample_count": 80,
                "fully_busy_sample_count": 60,
                "active_batches_peak": 6,
                "queued_batches_peak": 1,
                "processing_batches_peak": 6,
                "discarded_sample_count": 0,
                "sampler_failure_count": 0,
            }
        ],
        config={"processes": 1, "model_workers": 6},
    )

    assert summary["configured_total_model_workers"] == 6
    assert summary["runtime_model_workers_min"] == 6
    assert summary["python_calls_in_flight_peak_per_process"] == 6
    assert summary["python_calls_in_flight_peak_min_per_process"] == 6
    assert summary["ctranslate2_processing_batches_peak_per_process"] == 6
    assert summary["ctranslate2_processing_batches_peak_min_per_process"] == 6
    assert summary["ctranslate2_busy_sample_count_min_per_process"] == 80
    assert summary["ctranslate2_fully_busy_fraction_when_busy"] == 0.75


def test_language_hint_defaults_to_automatic_detection() -> None:
    assert _validated_language({}) is None


def test_language_hint_accepts_only_registered_mandarin_route() -> None:
    assert _validated_language({"language": "zh"}) == "zh"

    for invalid in ("en", "zh-CN", "", None, True, 1):
        try:
            _validated_language({"language": invalid})
        except ValueError:
            pass
        else:
            raise AssertionError(f"unexpected accepted language: {invalid!r}")


def test_transcribe_passes_optional_language_without_changing_auto_default() -> None:
    calls = []

    class FakeModel:
        def transcribe(self, path, **options):
            calls.append((path, options))
            info = type("Info", (), {"language": "zh", "language_probability": 1.0})()
            return iter(()), info

    item = {
        "id": "silence-control",
        "path": "public-control.wav",
        "duration_seconds": 1.0,
        "expected_speech": False,
    }
    for language in (None, "zh"):
        result = _transcribe(
            FakeModel(),
            item,
            capture_prediction=False,
            language=language,
        )
        assert result["success"] is True

    auto_options = calls[0][1]
    forced_options = calls[1][1]
    assert "language" not in auto_options
    assert forced_options == {**auto_options, "language": "zh"}
