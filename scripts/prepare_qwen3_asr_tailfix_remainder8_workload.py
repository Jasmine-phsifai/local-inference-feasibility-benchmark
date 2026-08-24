"""Build one ignored private-course control that exercises remainder-8 geometry."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ITEM_ID = "chunk_001"
SOURCE_SAMPLE_COUNT = 1_920_000
TAIL_SILENCE_SAMPLE_COUNT = 1_280
OUTPUT_SAMPLE_COUNT = SOURCE_SAMPLE_COUNT + TAIL_SILENCE_SAMPLE_COUNT
SAMPLE_RATE_HZ = 16_000
MEL_HOP_SAMPLES = 160
ENCODER_CHUNK_FRAMES = 100
OUTPUT_MEL_FRAME_COUNT = OUTPUT_SAMPLE_COUNT // MEL_HOP_SAMPLES
OUTPUT_REMAINDER_FRAMES = OUTPUT_MEL_FRAME_COUNT % ENCODER_CHUNK_FRAMES
MAX_MANIFEST_BYTES = 1_048_576


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-workload", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    geometry = prepare_remainder8_workload(
        args.source_workload,
        args.output_dir,
    )
    print(json.dumps(geometry, sort_keys=True))


def prepare_remainder8_workload(
    source_workload: Path,
    output_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Create a self-contained ignored workload without disclosing source identity."""

    source_manifest = source_workload.resolve(strict=True)
    output = output_dir.resolve(strict=False)
    _require_ignored_or_external_output(output, project_root=project_root)
    document = _load_source_document(source_manifest)
    source_item = _select_source_item(document)
    warmup_item = _select_warmup_item(document)
    source_audio = _resolve_media(source_manifest, source_item["path"])
    warmup_audio = _resolve_media(source_manifest, warmup_item["path"])

    output_audio = output / "chunk_001.wav"
    output_warmup = output / "warmup.wav"
    output_manifest = output / "manifest.json"
    if os.path.samefile(source_audio, warmup_audio):
        raise ValueError("tail-fix source requires separate warmup media")
    legacy_partials = tuple(
        path.with_suffix(path.suffix + ".part")
        for path in (output_audio, output_warmup, output_manifest)
    )
    _reject_destination_aliases(
        destinations=(
            output_audio,
            output_warmup,
            output_manifest,
            *legacy_partials,
        ),
        sources=(source_manifest, source_audio, warmup_audio),
    )

    source_frames = _read_pcm16_wav(
        source_audio,
        declared_duration_seconds=source_item["duration_seconds"],
        required_sample_count=SOURCE_SAMPLE_COUNT,
    )
    _read_pcm16_wav(
        warmup_audio,
        declared_duration_seconds=warmup_item["duration_seconds"],
    )
    expected_output_frames = source_frames + bytes(TAIL_SILENCE_SAMPLE_COUNT * 2)
    expected_warmup_bytes = warmup_audio.read_bytes()
    manifest = _build_output_manifest(
        expected_speech=source_item["expected_speech"],
        warmup_expected_speech=warmup_item["expected_speech"],
        warmup_duration_seconds=warmup_item["duration_seconds"],
    )
    expected_manifest_bytes = (
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    _validate_existing_outputs(
        output_audio=output_audio,
        expected_output_frames=expected_output_frames,
        output_warmup=output_warmup,
        expected_warmup_bytes=expected_warmup_bytes,
        output_manifest=output_manifest,
        expected_manifest_bytes=expected_manifest_bytes,
    )
    output.mkdir(parents=True, exist_ok=True)
    _reject_legacy_partials(legacy_partials)
    if not output_audio.exists():
        partial, handle = _create_unique_partial(output_audio)
        try:
            with handle:
                _write_pcm16_wav(handle, expected_output_frames)
                handle.flush()
                os.fsync(handle.fileno())
            _validate_output_audio(partial, expected_output_frames)
            _promote_partial(partial, output_audio)
            _validate_output_audio(output_audio, expected_output_frames)
        finally:
            _remove_owned_partial(partial)
    if not output_warmup.exists():
        partial, handle = _create_unique_partial(output_warmup)
        try:
            with handle:
                handle.write(expected_warmup_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            if partial.read_bytes() != expected_warmup_bytes:
                raise RuntimeError("tail-fix warmup copy verification failed")
            _promote_partial(partial, output_warmup)
            if output_warmup.read_bytes() != expected_warmup_bytes:
                raise RuntimeError("tail-fix warmup promotion verification failed")
        finally:
            _remove_owned_partial(partial)
    if not output_manifest.exists():
        partial, handle = _create_unique_partial(output_manifest)
        try:
            with handle:
                handle.write(expected_manifest_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            _promote_partial(partial, output_manifest)
            if output_manifest.read_bytes() != expected_manifest_bytes:
                raise RuntimeError("tail-fix manifest promotion verification failed")
        finally:
            _remove_owned_partial(partial)

    return {
        "duration_seconds": OUTPUT_SAMPLE_COUNT / SAMPLE_RATE_HZ,
        "encoder_remainder_frames": OUTPUT_REMAINDER_FRAMES,
        "item_count": 1,
        "mel_frame_count": OUTPUT_MEL_FRAME_COUNT,
        "sample_count": OUTPUT_SAMPLE_COUNT,
        "sample_rate_hz": SAMPLE_RATE_HZ,
    }


def _load_source_document(path: Path) -> dict:
    if path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("private ASR source manifest exceeds the byte limit")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or document.get("task") != "asr"
        or document.get("workload_class") != "private_course"
        or not isinstance(document.get("items"), list)
    ):
        raise ValueError("tail-fix source must be a private ASR workload")
    return document


def _select_source_item(document: dict) -> dict:
    matching = [
        item
        for item in document["items"]
        if isinstance(item, dict) and item.get("id") == SOURCE_ITEM_ID
    ]
    if len(matching) != 1:
        raise ValueError("tail-fix source requires exactly one generic chunk_001")
    item = matching[0]
    _validate_source_item(item, expected_duration_seconds=120.0)
    return item


def _select_warmup_item(document: dict) -> dict:
    warmup = document.get("warmup")
    if not isinstance(warmup, dict) or warmup.get("id") == SOURCE_ITEM_ID:
        raise ValueError("tail-fix source requires a separate explicit warmup")
    _validate_source_item(warmup)
    return warmup


def _validate_source_item(
    item: dict,
    *,
    expected_duration_seconds: float | None = None,
) -> None:
    duration = item.get("duration_seconds")
    if (
        type(item.get("path")) is not str
        or not item["path"]
        or type(duration) not in {int, float}
        or not math.isfinite(float(duration))
        or not 0 < float(duration) <= 7200
        or type(item.get("expected_speech")) is not bool
    ):
        raise ValueError("tail-fix source item metadata is invalid")
    if expected_duration_seconds is not None and not math.isclose(
        float(duration),
        expected_duration_seconds,
        rel_tol=0.0,
        abs_tol=1 / SAMPLE_RATE_HZ,
    ):
        raise ValueError("tail-fix source chunk duration changed")


def _resolve_media(manifest: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest.parent / path
    if not path.resolve(strict=True).is_file():
        raise ValueError("tail-fix source media is unavailable")
    return path.resolve(strict=True)


def _read_pcm16_wav(
    path: Path,
    *,
    declared_duration_seconds: float,
    required_sample_count: int | None = None,
) -> bytes:
    try:
        with wave.open(str(path), "rb") as reader:
            if (
                reader.getnchannels() != 1
                or reader.getsampwidth() != 2
                or reader.getframerate() != SAMPLE_RATE_HZ
                or reader.getcomptype() != "NONE"
            ):
                raise ValueError("tail-fix source must be PCM16 mono 16 kHz WAV")
            sample_count = reader.getnframes()
            frames = reader.readframes(sample_count)
            if len(frames) != sample_count * 2 or reader.readframes(1):
                raise ValueError("tail-fix source WAV frame data is inconsistent")
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError("tail-fix source must be a valid WAV") from error
    if sample_count <= 0 or (
        required_sample_count is not None and sample_count != required_sample_count
    ):
        raise ValueError("tail-fix source WAV sample count changed")
    if not math.isclose(
        sample_count / SAMPLE_RATE_HZ,
        float(declared_duration_seconds),
        rel_tol=0.0,
        abs_tol=1 / SAMPLE_RATE_HZ,
    ):
        raise ValueError("tail-fix source WAV duration does not match metadata")
    return frames


def _build_output_manifest(
    *,
    expected_speech: bool,
    warmup_expected_speech: bool,
    warmup_duration_seconds: float,
) -> dict:
    return {
        "schema_version": 1,
        "task": "asr",
        "workload_class": "private_course",
        "items": [
            {
                "id": SOURCE_ITEM_ID,
                "path": "chunk_001.wav",
                "duration_seconds": OUTPUT_SAMPLE_COUNT / SAMPLE_RATE_HZ,
                "expected_speech": expected_speech,
            }
        ],
        "warmup": {
            "id": "warmup",
            "path": "warmup.wav",
            "duration_seconds": float(warmup_duration_seconds),
            "expected_speech": warmup_expected_speech,
        },
    }


def _validate_existing_outputs(
    *,
    output_audio: Path,
    expected_output_frames: bytes,
    output_warmup: Path,
    expected_warmup_bytes: bytes,
    output_manifest: Path,
    expected_manifest_bytes: bytes,
) -> None:
    if output_audio.exists():
        _validate_output_audio(output_audio, expected_output_frames)
    if output_warmup.exists() and output_warmup.read_bytes() != expected_warmup_bytes:
        raise ValueError("existing tail-fix warmup is incompatible")
    if output_manifest.exists():
        if (
            output_manifest.read_bytes() != expected_manifest_bytes
            or not output_audio.is_file()
            or not output_warmup.is_file()
        ):
            raise ValueError("existing tail-fix workload commit is incompatible")


def _validate_output_audio(path: Path, expected_frames: bytes) -> None:
    frames = _read_pcm16_wav(
        path,
        declared_duration_seconds=OUTPUT_SAMPLE_COUNT / SAMPLE_RATE_HZ,
        required_sample_count=OUTPUT_SAMPLE_COUNT,
    )
    if frames != expected_frames:
        raise ValueError("existing tail-fix output audio is incompatible")
    if (
        OUTPUT_MEL_FRAME_COUNT != 12_008
        or OUTPUT_REMAINDER_FRAMES != 8
        or frames[-TAIL_SILENCE_SAMPLE_COUNT * 2 :]
        != bytes(TAIL_SILENCE_SAMPLE_COUNT * 2)
    ):
        raise RuntimeError("tail-fix remainder-8 output geometry changed")


def _write_pcm16_wav(handle: BinaryIO, frames: bytes) -> None:
    with wave.open(handle, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE_HZ)
        writer.setcomptype("NONE", "not compressed")
        writer.writeframes(frames)


def _reject_destination_aliases(
    *,
    destinations: tuple[Path, ...],
    sources: tuple[Path, ...],
) -> None:
    resolved_sources = {source.resolve(strict=True) for source in sources}
    for destination in destinations:
        if destination.resolve(strict=False) in resolved_sources:
            raise ValueError("tail-fix output must not alias a source file")
        if not os.path.lexists(destination):
            continue
        metadata = destination.lstat()
        if destination.is_symlink() or (
            getattr(metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError("tail-fix output must not use a reparse point")
        if any(os.path.samefile(destination, source) for source in sources):
            raise ValueError("tail-fix output must not alias a source file")


def _reject_legacy_partials(partials: tuple[Path, ...]) -> None:
    for partial in partials:
        if os.path.lexists(partial):
            raise ValueError("tail-fix legacy partial requires manual review")


def _create_unique_partial(destination: Path) -> tuple[Path, BinaryIO]:
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".part",
    )
    return Path(raw_path), os.fdopen(descriptor, "w+b")


def _promote_partial(partial: Path, destination: Path) -> None:
    try:
        os.link(partial, destination, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(
            "tail-fix destination appeared during preparation"
        ) from error
    metadata = destination.lstat()
    if destination.is_symlink() or (
        getattr(metadata, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    ):
        raise RuntimeError("tail-fix promotion produced a reparse point")
    if not os.path.samefile(partial, destination):
        raise RuntimeError("tail-fix promotion did not preserve file identity")


def _remove_owned_partial(partial: Path) -> None:
    if os.path.lexists(partial) and not partial.is_symlink():
        partial.unlink()


def _require_ignored_or_external_output(output: Path, *, project_root: Path) -> None:
    root = project_root.resolve(strict=True)
    if not output.is_relative_to(root):
        return
    relative = output.relative_to(root)
    if not relative.parts:
        raise ValueError("tail-fix output cannot be the project root")
    completed = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "--quiet", "--", relative.as_posix()],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ValueError("in-repository tail-fix output must be ignored")


if __name__ == "__main__":
    main()
