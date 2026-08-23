"""Reject free-form or path-bearing values before tracked result publication."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence


_BANNED_KEY_PARTS = {
    "path",
    "text",
    "transcript",
    "preview",
    "stdout",
    "stderr",
    "reference",
    "prediction",
    "course",
    "teacher",
    "student",
}
_ALLOWED_STRING_KEYS = {
    "backend",
    "candidate_id",
    "compute_type",
    "device",
    "device_name",
    "execution_devices",
    "failure_kind",
    "load_semantics",
    "mode",
    "model_revision",
    "phase",
    "prompt_version",
    "protocol",
    "runtime_name",
    "runtime_version",
    "stability_status",
    "status",
    "task",
    "unit",
    "workload_class",
}
_WINDOWS_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\)")


def validate_public_summary(summary: object) -> dict:
    """Return a plain validated mapping safe for the tracked event journal."""

    if not isinstance(summary, Mapping):
        raise ValueError("public_summary must be a mapping")
    normalized = _validate_value(summary, key=None)
    assert isinstance(normalized, dict)
    return normalized


def validate_sustained_public_summary(
    summary: object,
    *,
    candidate_id: str,
    task: str,
    workload_class: str,
    target_wall_seconds: float,
) -> dict:
    """Validate benchmark evidence and bind it to the runner-owned request."""

    normalized = validate_public_summary(summary)
    expected_identity = {
        "candidate_id": candidate_id,
        "task": task,
        "workload_class": workload_class,
    }
    for key, expected in expected_identity.items():
        if normalized.get(key) != expected:
            raise ValueError(f"sustained public_summary identity mismatch: {key}")
    for key in ("runtime_name", "runtime_version", "load_semantics"):
        if not isinstance(normalized.get(key), str) or not normalized[key]:
            raise ValueError(f"sustained public_summary is missing {key}")

    counts = normalized.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("sustained public_summary is missing counts")
    completed = _nonnegative_int(counts, "completed")
    failed = _nonnegative_int(counts, "failed")
    attempted = _nonnegative_int(counts, "attempted")
    if attempted == 0 or completed + failed != attempted:
        raise ValueError("sustained public_summary count invariant failed")

    throughput = normalized.get("throughput")
    if not isinstance(throughput, Mapping):
        raise ValueError("sustained public_summary is missing throughput")
    value = throughput.get("value")
    if type(value) not in {int, float} or value < 0:
        raise ValueError("sustained public_summary throughput is invalid")
    expected_unit = {
        "asr": "audio_hours_per_wall_hour",
        "ocr": "images_per_hour",
    }[task]
    if throughput.get("unit") != expected_unit:
        raise ValueError("sustained public_summary throughput unit mismatch")

    timing = normalized.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("sustained public_summary is missing timing")
    steady_wall_seconds = timing.get("steady_wall_seconds")
    if type(steady_wall_seconds) not in {int, float} or steady_wall_seconds <= 0:
        raise ValueError("sustained public_summary steady timing is invalid")
    reported_target = timing.get("target_wall_seconds")
    if (
        type(reported_target) not in {int, float}
        or not math.isclose(
            float(reported_target),
            float(target_wall_seconds),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("sustained public_summary target timing mismatch")
    return normalized


def _nonnegative_int(mapping: Mapping, key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"sustained public_summary count is invalid: {key}")
    return value


def _validate_value(value: object, *, key: str | None) -> object:
    if isinstance(value, Mapping):
        result = {}
        for child_key, child in value.items():
            if type(child_key) is not str or not child_key:
                raise ValueError("public_summary keys must be nonempty strings")
            folded = child_key.casefold()
            if any(part in folded for part in _BANNED_KEY_PARTS):
                raise ValueError(f"private field is forbidden in public_summary: {child_key}")
            result[child_key] = _validate_value(child, key=child_key)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_validate_value(child, key=key) for child in value]
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public_summary numbers must be finite")
        return value
    if isinstance(value, str):
        if key not in _ALLOWED_STRING_KEYS:
            raise ValueError(f"free-form string field is forbidden: {key}")
        if len(value) > 160 or "\n" in value or "\r" in value:
            raise ValueError(f"public_summary string is not bounded: {key}")
        if _WINDOWS_PATH.search(value):
            raise ValueError(f"local path is forbidden in public_summary: {key}")
        return value
    raise ValueError(f"unsupported public_summary value for {key}: {type(value).__name__}")
