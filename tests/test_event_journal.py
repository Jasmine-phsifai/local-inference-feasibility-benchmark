import json
import io
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import local_inference_bench.event_journal as journal_module

from local_inference_bench.event_journal import (
    _append_locked,
    _commit_recovery,
    _write_all,
    append_event,
    append_event_once,
    read_events,
    read_journal_bytes,
    successful_attempt_keys,
    unterminated_attempts,
)


def test_read_journal_bytes_returns_one_exact_locked_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    expected = b'{"event":"one"}\n{"event":"two"}\n'
    path.write_bytes(expected)

    assert read_journal_bytes(path) == expected


def _spawn_append_once(path: str, queue) -> None:
    event = {
        "event": "scored",
        "protocol": "test-v1",
        "public_event_sha256": "a" * 64,
    }
    try:
        queue.put(append_event_once(Path(path), event))
    except BaseException as error:
        queue.put((type(error).__name__, str(error)))


def test_journal_tolerates_torn_final_line(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, {"event": "attempt_succeeded", "attempt_key": "abc"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event":')
    events = read_events(path)
    assert successful_attempt_keys(events) == {"abc"}


def test_append_recovers_torn_final_line_without_losing_new_event(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, {"event": "attempt_succeeded", "attempt_key": "first"})
    with path.open("ab") as handle:
        handle.write(b'{"event":')

    append_event(path, {"event": "attempt_succeeded", "attempt_key": "second"})

    events = read_events(path)
    assert successful_attempt_keys(events) == {"first", "second"}
    recovery = next(event for event in events if event["event"] == "journal_recovered")
    assert recovery["reason"] == "invalid_final_fragment"
    assert recovery["discarded_byte_count"] == len(b'{"event":')
    assert recovery["discarded_sha256"]


def test_append_preserves_complete_record_missing_only_newline(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text(
        '{"event":"attempt_succeeded","attempt_key":"first"}',
        encoding="utf-8",
    )

    append_event(path, {"event": "attempt_succeeded", "attempt_key": "second"})

    events = read_events(path)
    assert successful_attempt_keys(events) == {"first", "second"}
    recovery = next(event for event in events if event["event"] == "journal_recovered")
    assert recovery["reason"] == "missing_final_newline"
    assert recovery["discarded_byte_count"] == 0


def test_newline_terminated_invalid_json_fails_before_append(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":"valid"}\n{"event":}\n')

    with pytest.raises(json.JSONDecodeError):
        read_events(path)
    with pytest.raises(json.JSONDecodeError):
        append_event(path, {"event": "must_not_be_appended"})
    assert b"must_not_be_appended" not in path.read_bytes()


def test_failed_append_does_not_repair_tail_after_invalid_retained_record(tmp_path):
    path = tmp_path / "events.jsonl"
    original = b'{"event":"valid"}\n{"event":}\n{"torn":'
    path.write_bytes(original)

    with pytest.raises(json.JSONDecodeError):
        append_event(path, {"event": "must_not_be_appended"})

    assert path.read_bytes() == original


def test_recovery_append_failure_restores_exact_original_tail(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "events.jsonl"
    original = b'{"event":"valid"}\n{"torn":'
    path.write_bytes(original)

    def fail_append(_handle, _serialized):
        raise OSError("injected recovery append failure")

    monkeypatch.setattr(journal_module, "_append_locked", fail_append)
    with pytest.raises(OSError, match="injected recovery append failure"):
        append_event(path, {"event": "new"})
    assert path.read_bytes() == original


def test_append_once_is_atomic_across_threads(tmp_path):
    path = tmp_path / "events.jsonl"
    event = {
        "event": "scored",
        "protocol": "test-v1",
        "public_event_sha256": "a" * 64,
    }

    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(lambda _index: append_event_once(path, event), range(64)))

    assert results.count(True) == 1
    assert results.count(False) == 63
    assert read_events(path) == [event]
    assert not list(tmp_path.glob("*.lock"))


def test_append_once_is_atomic_across_spawned_processes(tmp_path):
    path = tmp_path / "events.jsonl"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_spawn_append_once, args=(str(path), queue))
        for _ in range(8)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    results = [queue.get(timeout=5) for _ in processes]

    assert all(type(result) is bool for result in results), results
    assert results.count(True) == 1
    assert read_events(path) == [
        {
            "event": "scored",
            "protocol": "test-v1",
            "public_event_sha256": "a" * 64,
        }
    ]


def test_concurrent_distinct_appends_preserve_every_event(tmp_path):
    path = tmp_path / "events.jsonl"
    events = [{"event": "distinct", "index": index} for index in range(64)]

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(lambda event: append_event(path, event), events))

    retained = read_events(path)
    assert len(retained) == 64
    assert {event["index"] for event in retained} == set(range(64))


def test_append_once_repairs_missing_newline_without_duplicating(tmp_path):
    path = tmp_path / "events.jsonl"
    event = {
        "event": "scored",
        "protocol": "test-v1",
        "public_event_sha256": "b" * 64,
    }
    path.write_text(json.dumps(event), encoding="utf-8")

    assert append_event_once(path, event) is False

    events = read_events(path)
    assert events[0] == event
    assert events[1]["event"] == "journal_recovered"
    assert events[1]["reason"] == "missing_final_newline"


def test_append_once_rejects_null_identity_and_conflicting_body(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"unrelated":1}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="identity is invalid"):
        append_event_once(
            path,
            {
                "event": None,
                "protocol": None,
                "public_event_sha256": None,
            },
        )

    event = {
        "event": "scored",
        "protocol": "test-v1",
        "public_event_sha256": "c" * 64,
        "metrics": {"count": 1},
    }
    append_event_once(path, event)
    conflicting = {**event, "metrics": {"count": 2}}
    with pytest.raises(ValueError, match="identity conflicts"):
        append_event_once(path, conflicting)
    assert read_events(path)[-1] == event


@pytest.mark.parametrize("payload", [b"42\n", b"[]\n", b"null\n"])
def test_journal_rejects_non_object_records(tmp_path, payload):
    path = tmp_path / "events.jsonl"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="records must be objects"):
        read_events(path)
    with pytest.raises(ValueError, match="records must be objects"):
        append_event(path, {"event": "must_not_append"})
    assert path.read_bytes() == payload


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_journal_rejects_nonfinite_json_constants(tmp_path, constant):
    path = tmp_path / "events.jsonl"
    payload = f'{{"event":"bad","value":{constant}}}\n'.encode("ascii")
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        read_events(path)
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        append_event(path, {"event": "must_not_append"})
    assert path.read_bytes() == payload


def test_unterminated_nonfinite_final_fragment_is_recoverable(tmp_path):
    path = tmp_path / "events.jsonl"
    original_prefix = b'{"event":"valid"}\n'
    path.write_bytes(original_prefix + b'{"event":"torn","value":NaN')

    assert read_events(path) == [{"event": "valid"}]
    append_event(path, {"event": "new"})

    events = read_events(path)
    assert [event["event"] for event in events] == [
        "valid",
        "journal_recovered",
        "new",
    ]
    assert events[1]["reason"] == "invalid_final_fragment"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"event":"first","event":"shadow"}\n', "duplicate JSON key"),
        (b'{"event":"overflow","value":1e999}\n', "non-finite JSON number"),
    ],
)
def test_complete_ambiguous_or_overflowing_record_blocks_all_appends(
    tmp_path,
    payload,
    message,
):
    path = tmp_path / "events.jsonl"
    path.write_bytes(payload)
    event = {
        "event": "new",
        "protocol": "test-v1",
        "public_event_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match=message):
        read_events(path)
    with pytest.raises(ValueError, match=message):
        append_event(path, event)
    with pytest.raises(ValueError, match=message):
        append_event_once(path, event)
    assert path.read_bytes() == payload


def test_complete_excessively_nested_record_blocks_all_appends_atomically(
    tmp_path,
):
    path = tmp_path / "events.jsonl"
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    payload = b'{"event":"deep","value":' + nested + b"}\n"
    path.write_bytes(payload)
    event = {
        "event": "new",
        "protocol": "test-v1",
        "public_event_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="nesting is excessive"):
        read_events(path)
    with pytest.raises(ValueError, match="nesting is excessive"):
        append_event(path, event)
    with pytest.raises(ValueError, match="nesting is excessive"):
        append_event_once(path, event)
    assert path.read_bytes() == payload


def test_unterminated_excessively_nested_final_fragment_is_recoverable(
    tmp_path,
):
    path = tmp_path / "events.jsonl"
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    fragment = b'{"event":"deep","value":' + nested + b"}"
    path.write_bytes(b'{"event":"valid"}\n' + fragment)

    assert read_events(path) == [{"event": "valid"}]
    append_event(path, {"event": "new"})

    events = read_events(path)
    assert [event["event"] for event in events] == [
        "valid",
        "journal_recovered",
        "new",
    ]
    assert events[1]["reason"] == "invalid_final_fragment"
    assert events[1]["discarded_byte_count"] == len(fragment)


def test_new_excessively_nested_event_is_rejected_before_journal_creation(
    tmp_path,
):
    path = tmp_path / "events.jsonl"
    nested: object = 0
    for _ in range(10_000):
        nested = [nested]

    with pytest.raises(ValueError, match="record nesting is excessive"):
        append_event(path, {"event": "deep", "value": nested})

    assert not path.exists()


@pytest.mark.parametrize(
    "fragment",
    [
        b'{"event":"first","event":"shadow"}',
        b'{"event":"overflow","value":1e999}',
    ],
)
def test_unterminated_ambiguous_or_overflowing_final_fragment_is_recoverable(
    tmp_path,
    fragment,
):
    path = tmp_path / "events.jsonl"
    prefix = b'{"event":"valid"}\n'
    path.write_bytes(prefix + fragment)

    assert read_events(path) == [{"event": "valid"}]
    append_event(path, {"event": "new"})

    events = read_events(path)
    assert [event["event"] for event in events] == [
        "valid",
        "journal_recovered",
        "new",
    ]
    assert events[1]["reason"] == "invalid_final_fragment"
    assert events[1]["discarded_byte_count"] == len(fragment)


def test_largest_finite_json_float_remains_valid(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"event":"finite","value":1e308}\n')

    assert read_events(path) == [{"event": "finite", "value": 1e308}]


class _ShortWriter(io.BytesIO):
    def __init__(self, initial: bytes = b"", *, maximum_write: int = 2):
        super().__init__(initial)
        self.maximum_write = maximum_write

    def write(self, value):
        return super().write(value[: self.maximum_write])


@pytest.mark.parametrize("result", [0, None])
def test_write_all_rejects_no_progress(result):
    class NoProgressWriter(io.BytesIO):
        def write(self, value):
            return result

    with pytest.raises(OSError, match="no progress"):
        _write_all(NoProgressWriter(), b"event")


def test_short_writes_are_completed_and_failed_append_rolls_back():
    writer = _ShortWriter()
    _write_all(writer, b"abcdef")
    assert writer.getvalue() == b"abcdef"

    recovery_writer = _ShortWriter(b"valid-prefix plus stale suffix")
    _commit_recovery(
        recovery_writer,
        b"valid-prefix plus stale suffix",
        b"valid-prefix",
    )
    assert recovery_writer.getvalue() == b"valid-prefix"

    class StopsAfterFirstWrite(io.BytesIO):
        call_count = 0

        def write(self, value):
            self.call_count += 1
            if self.call_count == 1:
                return super().write(value[:2])
            return 0

    append_writer = StopsAfterFirstWrite(b"retained\n")
    with pytest.raises(OSError, match="no progress"):
        _append_locked(append_writer, b"new-event\n")
    assert append_writer.getvalue() == b"retained\n"


def test_unserializable_event_fails_without_repairing_tail(tmp_path):
    path = tmp_path / "events.jsonl"
    original = b'{"event":"valid"}\n{"torn":'
    path.write_bytes(original)

    with pytest.raises(TypeError):
        append_event(path, {"event": object()})

    assert path.read_bytes() == original


def test_unterminated_attempt_is_detected():
    events = [
        {"event": "attempt_started", "attempt_id": "one"},
        {"event": "attempt_started", "attempt_id": "two"},
        {"event": "attempt_failed", "attempt_id": "two"},
    ]
    assert [event["attempt_id"] for event in unterminated_attempts(events)] == ["one"]
