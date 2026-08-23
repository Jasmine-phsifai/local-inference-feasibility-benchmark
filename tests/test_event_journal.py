from local_inference_bench.event_journal import (
    append_event,
    read_events,
    successful_attempt_keys,
    unterminated_attempts,
)


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


def test_unterminated_attempt_is_detected():
    events = [
        {"event": "attempt_started", "attempt_id": "one"},
        {"event": "attempt_started", "attempt_id": "two"},
        {"event": "attempt_failed", "attempt_id": "two"},
    ]
    assert [event["attempt_id"] for event in unterminated_attempts(events)] == ["one"]
