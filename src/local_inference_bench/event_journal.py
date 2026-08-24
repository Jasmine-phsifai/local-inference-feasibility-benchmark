import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Iterator


_LOCK_TIMEOUT_SECONDS = 30.0
_LOCK_RETRY_SECONDS = 0.025


class _NonFiniteJSONError(ValueError):
    pass


class _DuplicateJSONKeyError(ValueError):
    pass


class _ExcessiveJSONNestingError(ValueError):
    pass


def append_event(path: Path, event: dict) -> None:
    serialized_event = _serialize_event(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked_journal(path, exclusive=True, create=True) as handle:
        contents = _read_locked(handle)
        retained, recovery = _plan_recovery(contents)
        _decode_events(retained)
        try:
            _commit_recovery(handle, contents, retained)
            if recovery is not None:
                _append_locked(handle, _serialize_event(recovery))
            _append_locked(handle, serialized_event)
            _flush_locked(handle)
        except BaseException:
            _restore_locked(handle, contents)
            raise


def append_event_once(
    path: Path,
    event: dict,
    *,
    identity_fields: tuple[str, ...] = (
        "event",
        "protocol",
        "public_event_sha256",
    ),
) -> bool:
    """Atomically append an event only when its public identity is absent."""

    if not identity_fields or any(
        type(field) is not str or not field for field in identity_fields
    ):
        raise ValueError("event journal append-once identity is invalid")
    serialized_event = _serialize_event(event)
    identity = _event_identity(event, identity_fields)
    duplicate_body = _deduplicated_event_body(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked_journal(path, exclusive=True, create=True) as handle:
        contents = _read_locked(handle)
        retained, recovery = _plan_recovery(contents)
        existing_events = _decode_events(retained)
        duplicate = False
        for existing in existing_events:
            if not all(field in existing for field in identity_fields):
                continue
            if _event_identity(existing, identity_fields) != identity:
                continue
            if _deduplicated_event_body(existing) != duplicate_body:
                raise ValueError("event journal append-once identity conflicts")
            duplicate = True
        try:
            _commit_recovery(handle, contents, retained)
            if recovery is not None:
                _append_locked(handle, _serialize_event(recovery))
            if not duplicate:
                _append_locked(handle, serialized_event)
            _flush_locked(handle)
            return not duplicate
        except BaseException:
            _restore_locked(handle, contents)
            raise


def _repair_incomplete_final_record(path: Path) -> dict | None:
    """Discard only an invalid final fragment before appending valid JSONL."""

    if not path.exists():
        return None
    with _locked_journal(path, exclusive=True, create=False) as handle:
        contents = _read_locked(handle)
        retained, recovery = _plan_recovery(contents)
        _decode_events(retained)
        _commit_recovery(handle, contents, retained)
        if retained != contents:
            _flush_locked(handle)
        return recovery


def _plan_recovery(contents: bytes) -> tuple[bytes, dict | None]:
    if not contents or contents.endswith(b"\n"):
        return contents, None
    final_newline = contents.rfind(b"\n")
    fragment_start = final_newline + 1
    fragment = contents[fragment_start:]
    reason = "missing_final_newline"
    try:
        _strict_json_loads(fragment.decode("utf-8"))
        retained = contents + b"\n"
        discarded = b""
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _NonFiniteJSONError,
        _DuplicateJSONKeyError,
        _ExcessiveJSONNestingError,
    ):
        retained = contents[:fragment_start]
        discarded = fragment
        reason = "invalid_final_fragment"
    recovery = {
        "event": "journal_recovered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "discarded_byte_count": len(discarded),
        "discarded_sha256": sha256(discarded).hexdigest() if discarded else None,
    }
    return retained, recovery


def read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with _locked_journal(path, exclusive=False, create=False) as handle:
            return _decode_events(_read_locked(handle))
    except FileNotFoundError:
        return []


def read_journal_bytes(path: Path) -> bytes:
    """Read one writer-coherent journal byte snapshot under the shared lock."""

    with locked_journal_bytes(path) as contents:
        return contents


@contextmanager
def locked_journal_bytes(path: Path) -> Iterator[bytes]:
    """Yield a coherent byte snapshot while retaining the journal's shared lock."""

    with locked_file_bytes(path) as contents:
        yield contents


@contextmanager
def locked_file_bytes(path: Path) -> Iterator[bytes]:
    """Yield an existing file snapshot while retaining its cooperative shared lock."""

    with _locked_journal(path, exclusive=False, create=False) as handle:
        yield _read_locked(handle)


def _serialize_event(event: dict) -> bytes:
    if not isinstance(event, dict):
        raise TypeError("event journal records must be objects")
    try:
        return (
            json.dumps(
                event,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except RecursionError as error:
        raise ValueError("event journal record nesting is excessive") from error


def _event_identity(event: dict, identity_fields: tuple[str, ...]) -> tuple[str, ...]:
    if any(field not in event for field in identity_fields):
        raise ValueError("event journal append-once identity is invalid")
    identity = []
    for field in identity_fields:
        value = event[field]
        if value is None or type(value) not in {str, int, float, bool}:
            raise ValueError("event journal append-once identity is invalid")
        try:
            identity.append(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("event journal append-once identity is invalid") from error
    return tuple(identity)


def _deduplicated_event_body(event: dict) -> str:
    body = {key: value for key, value in event.items() if key != "timestamp_utc"}
    try:
        return json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("event journal append-once body is invalid") from error


def _read_locked(handle: BinaryIO) -> bytes:
    handle.seek(0)
    return handle.read()


def _commit_recovery(
    handle: BinaryIO,
    original: bytes,
    retained: bytes,
) -> None:
    if retained == original:
        return
    handle.seek(0)
    _write_all(handle, retained)
    handle.truncate()


def _append_locked(handle: BinaryIO, serialized: bytes) -> None:
    original_size = handle.seek(0, os.SEEK_END)
    try:
        _write_all(handle, serialized)
    except BaseException:
        handle.seek(original_size)
        handle.truncate(original_size)
        raise


def _restore_locked(handle: BinaryIO, original: bytes) -> None:
    handle.seek(0)
    _write_all(handle, original)
    handle.truncate(len(original))
    _flush_locked(handle)


def _write_all(handle: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = handle.write(view[written:])
        if type(count) is not int or count <= 0:
            raise OSError("event journal write made no progress")
        if count > len(view) - written:
            raise OSError("event journal write count is invalid")
        written += count


def _flush_locked(handle: BinaryIO) -> None:
    handle.flush()
    os.fsync(handle.fileno())


@contextmanager
def _locked_journal(
    path: Path,
    *,
    exclusive: bool,
    create: bool,
) -> Iterator[BinaryIO]:
    if create:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
    else:
        mode = "r+b" if exclusive else "rb"
        handle = path.open(mode, buffering=0)
    try:
        _acquire_file_lock(handle, exclusive=exclusive)
        try:
            yield handle
        finally:
            _release_file_lock(handle)
    finally:
        handle.close()


def _acquire_file_lock(handle: BinaryIO, *, exclusive: bool) -> None:
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), mode, 1)
                return
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError("event journal lock timed out") from error
                time.sleep(_LOCK_RETRY_SECONDS)
    elif os.name == "posix":
        import fcntl

        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(handle.fileno(), mode)
    else:
        raise RuntimeError("event journal locking is unsupported on this platform")


def _release_file_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    elif os.name == "posix":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _decode_events(contents: bytes) -> list[dict]:
    lines = contents.splitlines()
    has_final_newline = contents.endswith(b"\n")
    events = []
    for index, line in enumerate(lines):
        try:
            event = _strict_json_loads(line.decode("utf-8"))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _NonFiniteJSONError,
            _DuplicateJSONKeyError,
            _ExcessiveJSONNestingError,
        ):
            if index != len(lines) - 1 or has_final_newline:
                raise
            continue
        if not isinstance(event, dict):
            raise ValueError("event journal records must be objects")
        events.append(event)
    return events


def _strict_json_loads(value: str):
    def reject_constant(constant: str):
        raise _NonFiniteJSONError(
            f"non-finite JSON constant is invalid: {constant}"
        )

    def parse_finite_float(raw: str) -> float:
        parsed = float(raw)
        if not math.isfinite(parsed):
            raise _NonFiniteJSONError(f"non-finite JSON number is invalid: {raw}")
        return parsed

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, item in pairs:
            if key in result:
                raise _DuplicateJSONKeyError(f"duplicate JSON key is invalid: {key}")
            result[key] = item
        return result

    try:
        return json.loads(
            value,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
        )
    except RecursionError as error:
        raise _ExcessiveJSONNestingError(
            "event journal JSON nesting is excessive"
        ) from error


def successful_attempt_keys(events: list[dict]) -> set[str]:
    return {event["attempt_key"] for event in events if event.get("event") == "attempt_succeeded"}


def unterminated_attempts(events: list[dict]) -> list[dict]:
    terminal_ids = {
        event["attempt_id"]
        for event in events
        if event.get("event") in {"attempt_succeeded", "attempt_failed", "attempt_interrupted"}
    }
    return [event for event in events if event.get("event") == "attempt_started" and event["attempt_id"] not in terminal_ids]
