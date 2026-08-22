import json
import os
from pathlib import Path


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    events = []
    for index, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return events


def successful_attempt_keys(events: list[dict]) -> set[str]:
    return {event["attempt_key"] for event in events if event.get("event") == "attempt_succeeded"}


def unterminated_attempts(events: list[dict]) -> list[dict]:
    terminal_ids = {
        event["attempt_id"]
        for event in events
        if event.get("event") in {"attempt_succeeded", "attempt_failed", "attempt_interrupted"}
    }
    return [event for event in events if event.get("event") == "attempt_started" and event["attempt_id"] not in terminal_ids]
