"""Verify generated quality-control manifests and their declared media."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import wave
from importlib.metadata import version
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FONT_ROOT = Path("C:/Windows/Fonts")
SUITE_MANIFESTS = {
    "ocr": Path("data/inputs/generated/ocr_quality/manifest.json"),
    "document-fidelity": Path(
        "data/inputs/generated/document_fidelity/manifest.json"
    ),
    "asr": Path("data/inputs/generated/asr_quality/manifest.json"),
    "hunyuan-vlm-fixture": Path(
        "data/inputs/generated/ocr_quality/hunyuan_doc_quality.json"
    ),
    "ovis-vlm-fixture": Path(
        "data/inputs/generated/document_fidelity/ovisocr2_page_quality.json"
    ),
}

_SAMPLE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_ITEMS = 64
_MAX_AUDIO_ITEM_SECONDS = 600
_MAX_AUDIO_SUITE_SECONDS = 3_600
_MAX_MEDIA_BYTES = 512 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless a generated-control manifest and every declared "
            "media file match."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--suite", choices=sorted(SUITE_MANIFESTS))
    source.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--verify-regeneration-host",
        action="store_true",
        help=(
            "Also verify declared local font hashes and the declared Pillow "
            "version. This cannot prove SAPI or ffmpeg equivalence."
        ),
    )
    args = parser.parse_args()
    manifest_path = (
        PROJECT_ROOT / SUITE_MANIFESTS[args.suite]
        if args.suite
        else args.manifest
    )
    summary = verify_generated_quality_controls(
        manifest_path,
        verify_regeneration_host=args.verify_regeneration_host,
    )
    print(json.dumps(summary, sort_keys=True))


def verify_generated_quality_controls(
    manifest_path: Path,
    *,
    verify_regeneration_host: bool = False,
    font_root: Path = FONT_ROOT,
) -> dict:
    """Return a privacy-safe summary after verifying one generated suite."""

    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError("generated-control manifest is missing")
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("generated-control manifest is unexpectedly large")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("generated-control manifest is not valid UTF-8 JSON") from exc

    if not isinstance(document, dict):
        raise ValueError("generated-control manifest root must be an object")
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
    ):
        raise ValueError("generated-control schema version is unsupported")
    if document.get("workload_class") != "generated_quality_control":
        raise ValueError("manifest is not a generated quality-control workload")
    task = document.get("task")
    if task not in {"ocr", "asr"}:
        raise ValueError("generated-control task must be ocr or asr")

    items = document.get("items")
    references = document.get("references")
    if not isinstance(items, list) or not 1 <= len(items) <= _MAX_ITEMS:
        raise ValueError("generated-control items must be a bounded nonempty list")
    if not isinstance(references, dict):
        raise ValueError("generated-control references must be an object")

    item_ids: list[str] = []
    item_paths: set[str] = set()
    verified_media: list[dict] = []
    total_duration_seconds = 0.0
    for item in items:
        sample_id, media_path = _validate_media_entry(
            item,
            root=manifest_path.parent,
            seen_paths=item_paths,
        )
        if sample_id in item_ids:
            raise ValueError("generated-control item IDs must be unique")
        item_ids.append(sample_id)
        reference = references.get(sample_id)
        if not isinstance(reference, dict):
            raise ValueError("every generated-control item needs one reference")
        if task == "ocr":
            details = _verify_ocr_item(
                item=item,
                reference=reference,
                media_path=media_path,
                generator=document.get("generator"),
            )
        else:
            details = _verify_asr_item(
                item=item,
                reference=reference,
                media_path=media_path,
                generator=document.get("generator"),
            )
            total_duration_seconds += details["duration_seconds"]
        verified_media.append({"id": sample_id, **details})

    if set(references) != set(item_ids):
        raise ValueError("generated-control references must match item IDs exactly")
    if total_duration_seconds > _MAX_AUDIO_SUITE_SECONDS:
        raise ValueError("generated ASR suite exceeds the bounded duration limit")

    warmup = document.get("warmup")
    warmup_summary = None
    if warmup is not None:
        warmup_id, warmup_path = _validate_media_entry(
            warmup,
            root=manifest_path.parent,
            seen_paths=item_paths,
        )
        if task == "ocr":
            warmup_summary = _inspect_png(warmup_path)
        else:
            warmup_summary = _inspect_wav(
                warmup_path,
                expected_duration=warmup.get("duration_seconds"),
                generator=document.get("generator"),
            )
        warmup_summary = {
            "id": warmup_id,
            "bytes": warmup_path.stat().st_size,
            "sha256": _sha256(warmup_path),
            **warmup_summary,
        }
    elif "warmup_item_id" in document:
        warmup_item_id = document.get("warmup_item_id")
        if warmup_item_id not in item_ids:
            raise ValueError("warmup_item_id must name a declared item")
    else:
        raise ValueError("generated-control manifest must declare a warmup")

    regeneration_checks: list[str] = []
    if verify_regeneration_host:
        regeneration_checks = _verify_declared_regeneration_dependencies(
            document.get("generator"),
            font_root=font_root,
        )

    return {
        "protocol": "generated-quality-control-verification-v1",
        "task": task,
        "item_count": len(items),
        "media_bytes": sum(item["bytes"] for item in verified_media),
        "manifest_sha256": _sha256(manifest_path),
        "warmup": warmup_summary,
        "regeneration_dependency_checks": regeneration_checks,
    }


def _validate_media_entry(
    entry: object,
    *,
    root: Path,
    seen_paths: set[str],
) -> tuple[str, Path]:
    if not isinstance(entry, dict):
        raise ValueError("generated-control media entries must be objects")
    sample_id = entry.get("id")
    if not isinstance(sample_id, str) or _SAMPLE_ID.fullmatch(sample_id) is None:
        raise ValueError("generated-control sample ID is invalid")
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("generated-control media path is invalid")
    relative_path = Path(relative)
    if relative_path.is_absolute() or len(relative_path.parts) != 1:
        raise ValueError("generated-control media path must be one relative filename")
    if relative in seen_paths:
        raise ValueError("generated-control media paths must be unique")
    seen_paths.add(relative)
    media_path = (root / relative_path).resolve()
    if media_path.parent != root.resolve() or not media_path.is_file():
        raise ValueError("generated-control media file is missing or escapes its root")
    if media_path.stat().st_size > _MAX_MEDIA_BYTES:
        raise ValueError("generated-control media file exceeds the bounded size limit")
    return sample_id, media_path


def _verify_ocr_item(
    *,
    item: dict,
    reference: dict,
    media_path: Path,
    generator: object,
) -> dict:
    if type(item.get("expected_text")) is not bool:
        raise ValueError("OCR expected_text must be a boolean")
    expected_sha256 = _expected_hash(reference, "image_sha256")
    actual_sha256 = _sha256(media_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("generated OCR image hash does not match its reference")
    inspected = _inspect_png(media_path)
    if isinstance(generator, dict):
        width = generator.get("canvas_width")
        height = generator.get("canvas_height")
        if isinstance(width, int) and isinstance(height, int):
            if inspected["width"] != width or inspected["height"] != height:
                raise ValueError("generated OCR image dimensions are inconsistent")
    return {
        "bytes": media_path.stat().st_size,
        "sha256": actual_sha256,
        **inspected,
    }


def _verify_asr_item(
    *,
    item: dict,
    reference: dict,
    media_path: Path,
    generator: object,
) -> dict:
    if type(item.get("expected_speech")) is not bool:
        raise ValueError("ASR expected_speech must be a boolean")
    if reference.get("expected_speech") is not item["expected_speech"]:
        raise ValueError("ASR speech expectation differs between item and reference")
    expected_sha256 = _expected_hash(reference, "audio_sha256")
    actual_sha256 = _sha256(media_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("generated ASR audio hash does not match its reference")
    inspected = _inspect_wav(
        media_path,
        expected_duration=item.get("duration_seconds"),
        generator=generator,
    )
    if inspected["duration_seconds"] > _MAX_AUDIO_ITEM_SECONDS:
        raise ValueError("generated ASR item exceeds the bounded duration limit")
    return {
        "bytes": media_path.stat().st_size,
        "sha256": actual_sha256,
        **inspected,
    }


def _inspect_png(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError("generated OCR media must be RGB PNG")
            width, height = image.size
    except (OSError, SyntaxError) as exc:
        raise ValueError("generated OCR media is not a valid PNG") from exc
    if not (1 <= width <= 8_192 and 1 <= height <= 8_192):
        raise ValueError("generated OCR image dimensions are outside bounds")
    return {"width": width, "height": height}


def _inspect_wav(
    path: Path,
    *,
    expected_duration: object,
    generator: object,
) -> dict:
    if not isinstance(expected_duration, (int, float)) or isinstance(
        expected_duration, bool
    ):
        raise ValueError("generated ASR duration must be numeric")
    if not math.isfinite(float(expected_duration)) or expected_duration <= 0:
        raise ValueError("generated ASR duration must be finite and positive")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            compression = handle.getcomptype()
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError("generated ASR media is not a valid WAV") from exc
    if compression != "NONE" or frame_count <= 0 or sample_rate <= 0:
        raise ValueError("generated ASR WAV must contain uncompressed PCM")
    if not (
        1 <= channels <= 8
        and 1 <= sample_width <= 4
        and 8_000 <= sample_rate <= 384_000
    ):
        raise ValueError("generated ASR WAV format is outside bounded limits")
    expected_frames = round(float(expected_duration) * sample_rate)
    if frame_count != expected_frames:
        raise ValueError("generated ASR WAV duration is inconsistent")
    if isinstance(generator, dict):
        declared = (
            generator.get("channels"),
            generator.get("sample_width_bytes"),
            generator.get("sample_rate_hz"),
        )
        if all(isinstance(value, int) for value in declared):
            if (channels, sample_width, sample_rate) != declared:
                raise ValueError("generated ASR WAV format is inconsistent")
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "duration_seconds": frame_count / sample_rate,
    }


def _expected_hash(reference: dict, field: str) -> str:
    expected = reference.get(field)
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ValueError(f"generated-control reference {field} is invalid")
    return expected


def _verify_declared_regeneration_dependencies(
    generator: object,
    *,
    font_root: Path,
) -> list[str]:
    if not isinstance(generator, dict):
        raise ValueError("generator dependency declaration is missing")
    checks: list[str] = []
    font_files = generator.get("font_files")
    if font_files is not None:
        if not isinstance(font_files, dict) or not font_files:
            raise ValueError("declared generator font inventory is invalid")
        for filename, expected_sha256 in font_files.items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(expected_sha256, str)
                or _SHA256.fullmatch(expected_sha256) is None
            ):
                raise ValueError("declared generator font identity is invalid")
            font_path = font_root / filename
            if not font_path.is_file() or _sha256(font_path) != expected_sha256:
                raise ValueError("local font does not match generated-control manifest")
        checks.append("font_file_sha256")
    declared_pillow = generator.get("pillow_version")
    if declared_pillow is not None:
        if not isinstance(declared_pillow, str) or declared_pillow != version("pillow"):
            raise ValueError("local Pillow version does not match the manifest")
        checks.append("pillow_version")
    if not checks:
        checks.append("media_only_no_host_identity_declared")
    return checks


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
