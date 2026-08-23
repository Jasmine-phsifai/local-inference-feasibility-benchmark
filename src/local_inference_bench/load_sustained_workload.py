"""Load local sustained inputs while exposing only privacy-safe aggregates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


_SAMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_WORKLOAD_CLASSES = {"generated_control", "public_course", "private_course"}


def load_sustained_workload(path: Path, *, expected_task: str) -> dict:
    """Validate a local manifest and return resolved items plus an opaque digest."""

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("sustained workload schema_version must be 1")
    if document.get("task") != expected_task:
        raise ValueError("sustained workload task does not match candidate")
    workload_class = document.get("workload_class")
    if workload_class not in _WORKLOAD_CLASSES:
        raise ValueError("sustained workload_class is invalid")
    raw_items = document.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("sustained workload requires nonempty items")

    items = []
    seen_ids = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise ValueError("sustained workload items must be objects")
        sample_id = raw_item.get("id")
        if (
            type(sample_id) is not str
            or _SAMPLE_ID.fullmatch(sample_id) is None
            or sample_id in seen_ids
        ):
            raise ValueError("sustained workload item IDs must be unique opaque IDs")
        seen_ids.add(sample_id)
        raw_path = raw_item.get("path")
        if type(raw_path) is not str or not raw_path:
            raise ValueError("sustained workload item path is invalid")
        item_path = Path(raw_path)
        if not item_path.is_absolute():
            item_path = (path.parent / item_path).resolve()
        if not item_path.is_file():
            raise FileNotFoundError(f"sustained workload item is missing: {sample_id}")
        item = {"id": sample_id, "path": str(item_path)}
        if expected_task == "asr":
            duration = raw_item.get("duration_seconds")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not 0 < float(duration) <= 7200
            ):
                raise ValueError("ASR workload duration_seconds is invalid")
            item["duration_seconds"] = float(duration)
            expected_speech = raw_item.get("expected_speech", True)
            if type(expected_speech) is not bool:
                raise ValueError("ASR workload expected_speech must be boolean")
            item["expected_speech"] = expected_speech
        else:
            expected_text = raw_item.get("expected_text", True)
            if type(expected_text) is not bool:
                raise ValueError("OCR workload expected_text must be boolean")
            item["expected_text"] = expected_text
        items.append(item)

    warmup_item_id = document.get("warmup_item_id", items[0]["id"])
    if warmup_item_id not in seen_ids:
        raise ValueError("warmup_item_id must name a workload item")
    total_duration = sum(item.get("duration_seconds", 0.0) for item in items)
    return {
        "task": expected_task,
        "workload_class": workload_class,
        "items": items,
        "warmup_item_id": warmup_item_id,
        "fingerprint": _fingerprint_items(items, expected_task),
        "public_summary": {
            "workload_class": workload_class,
            "item_count": len(items),
            "total_duration_seconds": total_duration,
        },
    }


def _fingerprint_items(items: list[dict], task: str) -> str:
    digest = hashlib.sha256()
    digest.update(task.encode("ascii"))
    for item in items:
        path = Path(item["path"])
        digest.update(item["id"].encode("ascii"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        if task == "asr":
            digest.update(repr(item["duration_seconds"]).encode("ascii"))
            digest.update(str(item["expected_speech"]).encode("ascii"))
        else:
            digest.update(str(item["expected_text"]).encode("ascii"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
