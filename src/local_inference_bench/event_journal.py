import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    recovery = _repair_incomplete_final_record(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        if recovery is not None:
            handle.write(json.dumps(recovery, ensure_ascii=False, sort_keys=True) + "\n")
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _repair_incomplete_final_record(path: Path) -> dict | None:
    """Discard only an invalid final fragment before appending valid JSONL."""

    if not path.exists() or path.stat().st_size == 0:
        return None
    contents = path.read_bytes()
    if contents.endswith(b"\n"):
        return None
    final_newline = contents.rfind(b"\n")
    fragment_start = final_newline + 1
    fragment = contents[fragment_start:]
    reason = "missing_final_newline"
    try:
        json.loads(fragment.decode("utf-8"))
        retained = contents + b"\n"
        discarded = b""
    except (UnicodeDecodeError, json.JSONDecodeError):
        retained = contents[:fragment_start]
        discarded = fragment
        reason = "invalid_final_fragment"
    with path.open("r+b") as handle:
        handle.seek(0)
        handle.write(retained)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "event": "journal_recovered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "discarded_byte_count": len(discarded),
        "discarded_sha256": sha256(discarded).hexdigest() if discarded else None,
    }


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
