from local_inference_bench.event_journal import append_event, read_events, successful_attempt_keys, unterminated_attempts


def test_journal_tolerates_torn_final_line(tmp_path):
    path = tmp_path / "events.jsonl"
    append_event(path, {"event": "attempt_succeeded", "attempt_key": "abc"})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event":')
    events = read_events(path)
    assert successful_attempt_keys(events) == {"abc"}


def test_unterminated_attempt_is_detected():
    events = [
        {"event": "attempt_started", "attempt_id": "one"},
        {"event": "attempt_started", "attempt_id": "two"},
        {"event": "attempt_failed", "attempt_id": "two"},
    ]
    assert [event["attempt_id"] for event in unterminated_attempts(events)] == ["one"]
