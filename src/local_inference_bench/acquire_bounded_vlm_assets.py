"""Acquire immutable llama.cpp VLM assets and verify every promoted byte."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
import time
from http.client import IncompleteRead
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import quote, urlsplit

from .bounded_vlm_assets import fingerprint_directory, sha256_file


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")
_DOWNLOAD_CHUNK_BYTES = 4 * 1024 * 1024


class IncompleteAssetDownload(OSError):
    """A retryable transfer ended before the declared immutable file size."""


def hugging_face_resolve_url(identity: dict, filename: str) -> str:
    """Build an immutable Hugging Face resolve URL from a pinned tree URL."""

    url = identity.get("url")
    revision = identity.get("revision")
    if type(url) is not str or type(revision) is not str:
        raise ValueError("Hugging Face identity is incomplete")
    parsed = urlsplit(url)
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() != "huggingface.co"
        or len(parts) != 4
        or parts[2] != "tree"
        or parts[3] != revision
    ):
        raise ValueError("Hugging Face identity must be an exact revision tree URL")
    relative = PurePosixPath(filename)
    if (
        not filename
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in filename
    ):
        raise ValueError("Hugging Face filename is invalid")
    repository = f"{quote(parts[0], safe='')}/{quote(parts[1], safe='')}"
    return (
        f"https://huggingface.co/{repository}/resolve/"
        f"{quote(revision, safe='')}/{quote(relative.as_posix(), safe='/')}"
    )


def acquire_verified_file(
    *,
    project_root: Path,
    url: str,
    record: dict,
    open_url: Callable | None = None,
    max_attempts: int = 64,
    timeout_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download one resumable file and promote it only after exact verification."""

    if not url.startswith("https://"):
        raise ValueError("asset download URL must use HTTPS")
    target = _repo_path(project_root, record.get("path"))
    expected_bytes, expected_sha256 = _file_identity(record)
    if target.exists():
        _assert_regular_file_matches(
            target,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            label="existing asset",
        )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")
    partial_bytes = partial.stat().st_size if partial.is_file() else 0
    if partial_bytes > expected_bytes:
        raise RuntimeError(f"partial asset is oversized: {target.name}")
    if partial_bytes == expected_bytes:
        _assert_regular_file_matches(
            partial,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
            label="completed partial asset",
        )
        os.replace(partial, target)
        return target

    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("asset download attempt limit is invalid")
    if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
        raise ValueError("asset download timeout is invalid")
    opener = open_url or urllib.request.urlopen
    for attempt in range(1, max_attempts + 1):
        try:
            _download_once(
                url=url,
                partial=partial,
                expected_bytes=expected_bytes,
                opener=opener,
                timeout_seconds=float(timeout_seconds),
            )
            _assert_completed_download(
                partial,
                expected_bytes=expected_bytes,
                expected_sha256=expected_sha256,
            )
            break
        except urllib.error.HTTPError as error:
            if not _is_transient_http_status(error.code):
                raise
            retry_error = error
        except (IncompleteAssetDownload, IncompleteRead, TimeoutError, OSError) as error:
            retry_error = error
        if attempt == max_attempts:
            raise RuntimeError(
                f"asset download failed after {max_attempts} attempts: {target.name}"
            ) from retry_error
        sleep(min(5.0, float(attempt)))
    os.replace(partial, target)
    return target


def _is_transient_http_status(status: object) -> bool:
    return type(status) is int and (status in (408, 429) or 500 <= status <= 599)


def _download_once(
    *,
    url: str,
    partial: Path,
    expected_bytes: int,
    opener: Callable,
    timeout_seconds: float,
) -> None:
    partial_bytes = partial.stat().st_size if partial.is_file() else 0
    if partial_bytes >= expected_bytes:
        return
    headers = {"User-Agent": "local-inference-feasibility-benchmark/1"}
    if partial_bytes:
        headers["Range"] = f"bytes={partial_bytes}-"
    request = urllib.request.Request(url, headers=headers)
    with opener(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        if type(final_url) is not str or not final_url.startswith("https://"):
            raise RuntimeError("asset download redirected outside HTTPS")
        status = getattr(response, "status", None)
        if status is None:
            status = response.getcode()
        if status not in (200, 206):
            raise RuntimeError(f"asset download returned HTTP {status}")
        append = partial_bytes > 0 and status == 206
        if status == 206:
            content_range = response.headers.get("Content-Range", "")
            match = _CONTENT_RANGE.fullmatch(content_range)
            if (
                match is None
                or int(match.group(1)) != partial_bytes
                or int(match.group(3)) != expected_bytes
                or int(match.group(2)) < partial_bytes
            ):
                raise RuntimeError("asset resume response has the wrong byte range")
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
    if partial.stat().st_size < expected_bytes:
        raise IncompleteAssetDownload("asset transfer ended before the expected size")


def _assert_completed_download(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    actual_bytes = path.stat().st_size if path.is_file() else 0
    if actual_bytes < expected_bytes:
        raise IncompleteAssetDownload("asset transfer ended before the expected size")
    if actual_bytes > expected_bytes:
        raise RuntimeError(f"downloaded asset size mismatch: {path.name}")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"downloaded asset SHA-256 mismatch: {path.name}")


def extract_verified_zip_tree(
    *,
    project_root: Path,
    archive: Path,
    tree_record: dict,
    archive_root: str | None = None,
) -> Path:
    """Safely extract an exact ZIP tree without overwriting an existing tree."""

    target = _repo_path(project_root, tree_record.get("path"))
    expected = _tree_identity(tree_record)
    if target.exists():
        _assert_tree_matches(target, expected)
        return target
    if not archive.is_file():
        raise FileNotFoundError(f"asset archive is missing: {archive.name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".vlm-", dir=target.parent)).resolve()
    promoted = False
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = _archive_member_relative(member, archive_root)
                if relative is None:
                    continue
                destination = (temporary / Path(*relative.parts)).resolve()
                if not destination.is_relative_to(temporary):
                    raise ValueError("ZIP member escapes the extraction directory")
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                unix_mode = member.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise ValueError("ZIP symlinks are not accepted")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, destination.open("wb") as sink:
                    shutil.copyfileobj(source, sink, _DOWNLOAD_CHUNK_BYTES)
        _assert_tree_matches(temporary, expected)
        temporary.replace(target)
        promoted = True
        return target
    finally:
        if not promoted and temporary.is_dir():
            shutil.rmtree(temporary)


def assert_file_record(project_root: Path, record: dict, label: str) -> Path:
    """Fail closed unless one repository-relative file matches its record."""

    path = _repo_path(project_root, record.get("path"))
    expected_bytes, expected_sha256 = _file_identity(record)
    _assert_regular_file_matches(
        path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        label=label,
    )
    return path


def file_record_is_complete(project_root: Path, record: dict) -> bool:
    """Return whether a file is absent or exact; reject an existing mismatch."""

    path = _repo_path(project_root, record.get("path"))
    expected_bytes, expected_sha256 = _file_identity(record)
    if not path.exists():
        return False
    _assert_regular_file_matches(
        path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
        label="existing asset",
    )
    return True


def _archive_member_relative(
    member: zipfile.ZipInfo,
    archive_root: str | None,
) -> PurePosixPath | None:
    name = member.filename
    if "\\" in name or name.startswith("/"):
        raise ValueError("ZIP member path is invalid")
    relative_name = name
    if archive_root is not None:
        prefix = f"{archive_root.rstrip('/')}/"
        if name.rstrip("/") == archive_root.rstrip("/"):
            return None
        if not name.startswith(prefix):
            raise ValueError("ZIP member is outside the declared archive root")
        relative_name = name[len(prefix) :]
    relative = PurePosixPath(relative_name)
    if not relative_name or ".." in relative.parts:
        raise ValueError("ZIP member path is invalid")
    return relative


def _repo_path(project_root: Path, value: object) -> Path:
    root = project_root.resolve()
    if type(value) is not str or not value or Path(value).is_absolute():
        raise ValueError("asset path must be repository-relative")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("asset path escapes the repository")
    return path


def _file_identity(record: object) -> tuple[int, str]:
    if not isinstance(record, dict):
        raise ValueError("asset file record is invalid")
    expected_bytes = record.get("bytes")
    expected_sha256 = record.get("sha256")
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or type(expected_sha256) is not str
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise ValueError("asset file identity is invalid")
    return expected_bytes, expected_sha256


def _tree_identity(record: object) -> dict:
    if not isinstance(record, dict):
        raise ValueError("asset tree record is invalid")
    identity = {
        "file_count": record.get("file_count"),
        "total_bytes": record.get("total_bytes"),
        "sha256": record.get("sha256"),
    }
    if (
        type(identity["file_count"]) is not int
        or identity["file_count"] <= 0
        or type(identity["total_bytes"]) is not int
        or identity["total_bytes"] <= 0
        or type(identity["sha256"]) is not str
        or _SHA256.fullmatch(identity["sha256"]) is None
    ):
        raise ValueError("asset tree identity is invalid")
    return identity


def _assert_regular_file_matches(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path.name}")
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"{label} size mismatch: {path.name}")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError(f"{label} SHA-256 mismatch: {path.name}")


def _assert_tree_matches(path: Path, expected: dict) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"asset tree is missing: {path.name}")
    actual = fingerprint_directory(path)
    if actual != expected:
        raise RuntimeError(f"asset tree identity mismatch: {path.name}")
