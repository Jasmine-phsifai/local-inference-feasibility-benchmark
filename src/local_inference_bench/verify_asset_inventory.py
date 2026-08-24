"""Verify an exact size-and-SHA-256 inventory under a local asset root."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath


_SHA256 = re.compile(r"[0-9a-f]{64}")


def verify_asset_inventory(
    root: Path,
    required_files: object,
    *,
    scope_roots: tuple[str, ...] = (),
) -> dict:
    if not root.is_dir():
        raise FileNotFoundError("asset root is missing")
    expected = _validate_entries(required_files)
    actual_paths = _inventory_paths(root, scope_roots)
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        unexpected = sorted(actual_paths - set(expected))
        raise RuntimeError(
            f"asset inventory mismatch: missing={missing}, unexpected={unexpected}"
        )
    total_bytes = 0
    for relative_path, identity in expected.items():
        path = root / PurePosixPath(relative_path)
        size_bytes = path.stat().st_size
        if size_bytes != identity["size_bytes"]:
            raise RuntimeError(f"asset size mismatch: {relative_path}")
        if _sha256(path) != identity["sha256"]:
            raise RuntimeError(f"asset SHA-256 mismatch: {relative_path}")
        total_bytes += size_bytes
    return {"file_count": len(expected), "total_bytes": total_bytes}


def _validate_entries(required_files: object) -> dict[str, dict]:
    if not isinstance(required_files, list) or not required_files:
        raise ValueError("required asset file inventory is invalid")
    expected = {}
    for entry in required_files:
        if not isinstance(entry, dict) or set(entry) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("asset identity entry is invalid")
        relative_path = entry["path"]
        if type(relative_path) is not str or not 1 <= len(relative_path) <= 240:
            raise ValueError("asset relative path is invalid")
        path = PurePosixPath(relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in relative_path
            or ":" in relative_path
            or relative_path in expected
        ):
            raise ValueError("asset relative path is unsafe or duplicated")
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] <= 0:
            raise ValueError("asset size is invalid")
        if type(entry["sha256"]) is not str or _SHA256.fullmatch(entry["sha256"]) is None:
            raise ValueError("asset SHA-256 is invalid")
        expected[relative_path] = entry
    return expected


def _inventory_paths(root: Path, scope_roots: tuple[str, ...]) -> set[str]:
    scopes = scope_roots or (".",)
    paths = set()
    for scope in scopes:
        scope_path = PurePosixPath(scope)
        if scope_path.is_absolute() or ".." in scope_path.parts or "\\" in scope:
            raise ValueError("asset inventory scope is unsafe")
        directory = root / scope_path
        if not directory.is_dir():
            raise FileNotFoundError(f"asset inventory scope is missing: {scope}")
        paths.update(
            path.relative_to(root).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
        )
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
