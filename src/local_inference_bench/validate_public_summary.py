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
    "failure_kind",
    "load_semantics",
    "mode",
    "model_revision",
    "phase",
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
