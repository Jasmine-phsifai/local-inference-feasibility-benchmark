import hashlib
import json
import os
from pathlib import Path


def fingerprint_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def fingerprint_files(paths: list[Path]) -> str:
    resolved_paths = [path.resolve(strict=True) for path in paths]
    if not resolved_paths:
        raise ValueError("at least one file is required for a fingerprint")
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("fingerprint file paths must be unique")
    common_root = Path(os.path.commonpath([str(path) for path in resolved_paths]))
    if any(path == common_root for path in resolved_paths):
        common_root = common_root.parent
    digest = hashlib.sha256()
    for path in sorted(
        resolved_paths,
        key=lambda item: item.relative_to(common_root).as_posix().casefold(),
    ):
        relative_path = path.relative_to(common_root).as_posix()
        relative_bytes = relative_path.encode("utf-8")
        content_digest = hashlib.sha256()
        content_bytes = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                content_digest.update(chunk)
                content_bytes += len(chunk)
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(content_bytes.to_bytes(16, "big"))
        digest.update(content_digest.digest())
    return digest.hexdigest()[:16]
