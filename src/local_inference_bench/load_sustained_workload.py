"""Load local sustained inputs while exposing only privacy-safe aggregates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


_SAMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_OUTPUT_MARKER = re.compile(
    r"^<!-- meta:(?:page number=[1-9][0-9]*|frame id=[a-z0-9][a-z0-9_-]{0,63}) -->$"
)
_WORKLOAD_FINGERPRINT_PROTOCOL = "sustained-workload-v3"
PUBLIC_WORKLOAD_CLASSES = frozenset(
    {
        "generated_control",
        "generated_quality_control",
        "public_course",
    }
)
PRIVATE_WORKLOAD_CLASSES = frozenset({"private_course"})
_WORKLOAD_CLASSES = PUBLIC_WORKLOAD_CLASSES | PRIVATE_WORKLOAD_CLASSES


def is_private_workload_class(workload_class: object) -> bool:
    """Treat every workload class not explicitly public as private."""

    return workload_class not in PUBLIC_WORKLOAD_CLASSES


def load_sustained_workload(path: Path, *, expected_task: str) -> dict:
    """Validate a local manifest and return resolved items plus an opaque digest."""

    return load_sustained_workload_from_bytes(
        path.read_bytes(),
        manifest_path=path,
        expected_task=expected_task,
    )


def load_sustained_workload_from_bytes(
    manifest_bytes: bytes,
    *,
    manifest_path: Path,
    expected_task: str,
) -> dict:
    """Validate one immutable manifest snapshot and resolve its local inputs."""

    try:
        document = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError("sustained workload manifest is invalid") from error
    if not isinstance(document, dict):
        raise ValueError("sustained workload manifest must be an object")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
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
        items.append(
            _load_item(
                raw_item,
                manifest_path=manifest_path,
                task=expected_task,
                seen_ids=seen_ids,
            )
        )

    raw_warmup = document.get("warmup")
    if raw_warmup is None:
        warmup_item_id = document.get("warmup_item_id", items[0]["id"])
        if warmup_item_id not in seen_ids:
            raise ValueError("warmup_item_id must name a workload item")
        warmup_item = next(item for item in items if item["id"] == warmup_item_id)
        fingerprint_items = items
    else:
        warmup_item = _load_item(
            raw_warmup,
            manifest_path=manifest_path,
            task=expected_task,
            seen_ids=seen_ids,
        )
        fingerprint_items = [*items, warmup_item]
    fingerprint, item_content_bindings = _fingerprint_items(
        fingerprint_items,
        task=expected_task,
        workload_class=workload_class,
        warmup_item_id=warmup_item["id"],
    )
    total_duration = sum(item.get("duration_seconds", 0.0) for item in items)
    return {
        "task": expected_task,
        "workload_class": workload_class,
        "items": items,
        "warmup_item": warmup_item,
        "fingerprint": fingerprint,
        # Internal-only bindings let privacy-safe scorers prove that they decode
        # the exact bytes included in the opaque workload fingerprint.
        "item_content_bindings": item_content_bindings,
        "public_summary": {
            "workload_class": workload_class,
            "item_count": len(items),
            "total_duration_seconds": total_duration,
        },
    }


def _load_item(
    raw_item: object,
    *,
    manifest_path: Path,
    task: str,
    seen_ids: set[str],
) -> dict:
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
        item_path = (manifest_path.parent / item_path).resolve()
    if not item_path.is_file():
        raise FileNotFoundError(f"sustained workload item is missing: {sample_id}")
    item = {"id": sample_id, "path": str(item_path)}
    if task == "asr":
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
        output_marker = raw_item.get("output_marker")
        if output_marker is not None:
            if (
                type(output_marker) is not str
                or _OUTPUT_MARKER.fullmatch(output_marker) is None
            ):
                raise ValueError("OCR workload output_marker is invalid")
            item["output_marker"] = output_marker
    return item


def _fingerprint_items(
    items: list[dict],
    *,
    task: str,
    workload_class: str,
    warmup_item_id: str,
) -> tuple[str, dict[str, dict[str, int | str]]]:
    fingerprint_identity = {
        "protocol": _WORKLOAD_FINGERPRINT_PROTOCOL,
        "task": task,
        "warmup_item_id": warmup_item_id,
        "workload_class": workload_class,
        # Item order is part of the execution contract because it affects
        # process assignment and therefore the measured concurrency pattern.
        "items": [],
    }
    item_content_bindings = {}
    for item in items:
        path = Path(item["path"])
        size_bytes, content_sha256 = _hash_file(path)
        item_content_bindings[item["id"]] = {
            "content_sha256": content_sha256,
            "size_bytes": size_bytes,
        }
        fingerprint_item = {
            "content_sha256": content_sha256,
            "id": item["id"],
            "size_bytes": size_bytes,
        }
        if task == "asr":
            fingerprint_item.update(
                {
                    "duration_seconds": item["duration_seconds"],
                    "expected_speech": item["expected_speech"],
                }
            )
        else:
            fingerprint_item.update(
                {
                    "expected_text": item["expected_text"],
                    "output_marker": item.get("output_marker"),
                }
            )
        fingerprint_identity["items"].append(fingerprint_item)
    canonical = json.dumps(
        fingerprint_identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest(), item_content_bindings


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size_bytes += len(chunk)
            digest.update(chunk)
    return size_bytes, digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate sustained workload JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(constant: str):
    raise ValueError(f"non-finite sustained workload JSON constant: {constant}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite sustained workload JSON number: {value}")
    return parsed
