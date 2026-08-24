"""Verify an exact Python package inventory declared by a lock file."""

from __future__ import annotations

import re
import sys
from importlib.metadata import distributions
from pathlib import Path
from site import getsitepackages


_LOCK_LINE = re.compile(r"([A-Za-z0-9_.-]+)==([^;\s]+)")


def verify_locked_environment(
    lock_path: Path,
    *,
    expected_python: tuple[int, int, int],
    allowed_extra_packages: dict[str, str] | None = None,
) -> dict:
    expected = _read_exact_lock(lock_path)
    allowed_extra = {
        _canonicalize_name(name): version
        for name, version in (allowed_extra_packages or {}).items()
    }
    actual: dict[str, str] = {}
    for installed in distributions(path=getsitepackages()):
        name = installed.metadata.get("Name")
        if not name:
            raise RuntimeError("installed distribution has no package name")
        canonical_name = _canonicalize_name(name)
        if canonical_name in actual:
            raise RuntimeError(f"duplicate installed distribution: {canonical_name}")
        actual[canonical_name] = installed.version
    if tuple(sys.version_info[:3]) != expected_python:
        raise RuntimeError(
            f"Python version mismatch: {tuple(sys.version_info[:3])}"
        )
    expected_with_allowed_extra = {**expected, **allowed_extra}
    if actual != expected_with_allowed_extra:
        missing = sorted(set(expected_with_allowed_extra) - set(actual))
        unexpected = sorted(set(actual) - set(expected_with_allowed_extra))
        mismatched = sorted(
            name
            for name in set(actual) & set(expected_with_allowed_extra)
            if actual[name] != expected_with_allowed_extra[name]
        )
        raise RuntimeError(
            "locked environment mismatch: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )
    return {
        "python_version": ".".join(map(str, expected_python)),
        "package_count": len(expected_with_allowed_extra),
    }


def _read_exact_lock(path: Path) -> dict[str, str]:
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _LOCK_LINE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid exact lock entry at line {line_number}")
        name = _canonicalize_name(match.group(1))
        if name in expected:
            raise ValueError(f"duplicate lock package at line {line_number}")
        expected[name] = match.group(2)
    if not expected:
        raise ValueError("exact environment lock is empty")
    return expected


def _canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()
