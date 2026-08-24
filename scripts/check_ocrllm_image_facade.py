"""Record provenance-bound aggregate evidence for OCRLLM's image facade."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.event_journal import append_event
from local_inference_bench.validate_public_summary import validate_public_summary

if __package__:
    from scripts.ocrllm_compatibility_provenance import (
        EXPECTED_REVISION,
        SNAPSHOT_RELATIVE_PATH,
        verify_ocrllm_installation,
    )
else:
    from ocrllm_compatibility_provenance import (  # type: ignore[no-redef]
        EXPECTED_REVISION,
        SNAPSHOT_RELATIVE_PATH,
        verify_ocrllm_installation,
    )


_LATEX_MARKER = re.compile(r"(?:\$\$|\\(?:frac|sum|int|sqrt|begin)\b)")
_CANDIDATE_ID = "ocrllm_active_image_facade"
_EVENT_PROTOCOL = "ocrllm-image-compatibility-v3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=PROJECT_ROOT / SNAPSHOT_RELATIVE_PATH,
    )
    parser.add_argument("--append-journal", type=Path)
    parser.add_argument("--private-record", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError("generated OCRLLM compatibility image is missing")
    if args.private_record is not None:
        _require_safe_private_record_path(args.private_record)
    reference = _load_code_formula_reference(args.manifest, args.image)
    provenance = verify_ocrllm_installation(args.snapshot)

    import requests
    import ocrllm
    from ocrllm import (
        Config,
        LocalOCRSettings,
        RecognitionResult,
        recognize,
    )

    distribution = importlib.metadata.distribution("ocrllm")

    network_attempts = 0
    original_request = requests.sessions.Session.request

    def reject_network(*request_args: object, **request_kwargs: object) -> object:
        del request_args, request_kwargs
        nonlocal network_attempts
        network_attempts += 1
        raise AssertionError("OCRLLM local mode attempted a network request")

    requests.sessions.Session.request = reject_network
    try:
        config = Config(
            image_mode="ocr",
            local_ocr=LocalOCRSettings(minimum_confidence=0.5),
        )
        started = time.perf_counter()
        result = recognize(args.image, config=config)
        elapsed_seconds = time.perf_counter() - started
    finally:
        requests.sessions.Session.request = original_request

    result_fields = {item.name for item in fields(RecognitionResult)}
    public_summary = _build_public_summary(
        runtime_version=distribution.version,
        result=result,
        result_fields=result_fields,
        required_tokens=reference["required_tokens"],
        elapsed_seconds=elapsed_seconds,
        network_attempt_count=network_attempts,
        provenance=provenance,
    )
    event = _build_compatibility_event(
        reference=reference,
        provenance=provenance,
        public_summary=public_summary,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    if args.private_record is not None:
        args.private_record.parent.mkdir(parents=True, exist_ok=True)
        args.private_record.write_text(
            json.dumps(
                _json_safe(
                    {
                        "recognize_signature": str(inspect.signature(recognize)),
                        "module_file": str(Path(ocrllm.__file__).resolve()),
                        "snapshot_root": str(args.snapshot.resolve()),
                        "manifest": str(args.manifest.resolve()),
                        "image": str(args.image.resolve()),
                        "facade_markdown": result.markdown,
                        "facade_metadata": result.metadata,
                        "required_tokens": reference["required_tokens"],
                        "provenance": provenance,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    if args.append_journal is not None:
        append_event(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def _build_compatibility_event(
    *,
    reference: dict,
    provenance: dict,
    public_summary: dict,
    timestamp_utc: str,
) -> dict:
    if public_summary.get("candidate_id") != _CANDIDATE_ID:
        raise ValueError("OCRLLM compatibility summary candidate identity is invalid")
    return {
        "event": "ocrllm_compatibility_checked",
        "protocol": _EVENT_PROTOCOL,
        "timestamp_utc": timestamp_utc,
        "candidate_id": _CANDIDATE_ID,
        "dataset_fingerprint": reference["image_sha256"],
        "workload_manifest_fingerprint": reference["manifest_sha256"],
        "exact_reference_fingerprint": reference["required_tokens_sha256"],
        "source_revision": provenance["revision"],
        "source_tree_fingerprint": provenance["source_tree_fingerprint"],
        "installed_python_source_fingerprint": provenance[
            "python_source_fingerprint"
        ],
        "runtime_environment_fingerprint": provenance[
            "runtime_environment_fingerprint"
        ],
        "rapidocr_model_fingerprint": provenance["rapidocr_model_fingerprint"],
        "producer_sha256": _producer_sha256(),
        "result": public_summary,
    }


def _producer_sha256() -> dict[str, str]:
    relative_paths = (
        Path("scripts/check_ocrllm_image_facade.py"),
        Path("scripts/ocrllm_compatibility_provenance.py"),
        Path("src/local_inference_bench/validate_public_summary.py"),
        Path("src/local_inference_bench/event_journal.py"),
    )
    return {
        path.as_posix(): _sha256(PROJECT_ROOT / path)
        for path in relative_paths
    }


def _load_code_formula_reference(manifest_path: Path, image_path: Path) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError("OCR quality-control manifest is missing")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        type(document) is not dict
        or document.get("schema_version") != 1
        or document.get("task") != "ocr"
        or document.get("workload_class") != "generated_quality_control"
    ):
        raise ValueError("OCR quality-control manifest identity is invalid")
    items = document.get("items")
    matches = (
        [item for item in items if type(item) is dict and item.get("id") == "code_formula"]
        if type(items) is list
        else []
    )
    if len(matches) != 1 or matches[0].get("path") != image_path.name:
        raise ValueError("code_formula image is not bound to the manifest")
    declared_image = (manifest_path.parent / matches[0]["path"]).resolve()
    if declared_image != image_path.resolve():
        raise ValueError("code_formula image location does not match the manifest")
    references = document.get("references")
    reference = references.get("code_formula") if type(references) is dict else None
    required_tokens = reference.get("required_tokens") if type(reference) is dict else None
    if (
        type(reference) is not dict
        or reference.get("category") != "code_formula"
        or type(required_tokens) is not list
        or not required_tokens
        or len(required_tokens) > 256
        or any(type(token) is not str or not token for token in required_tokens)
    ):
        raise ValueError("code_formula exact reference is invalid")
    image_sha256 = _sha256(image_path)
    if reference.get("image_sha256") != image_sha256:
        raise ValueError("code_formula image fingerprint does not match the manifest")
    return {
        "required_tokens": required_tokens,
        "image_sha256": image_sha256,
        "manifest_sha256": _sha256(manifest_path),
        "required_tokens_sha256": _sha256_json(required_tokens),
    }


def _require_safe_private_record_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(relative)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError("private OCRLLM record inside the repository must be ignored")


def _build_public_summary(
    *,
    runtime_version: str,
    result: object,
    result_fields: set[str],
    required_tokens: list[str],
    elapsed_seconds: float,
    network_attempt_count: int,
    provenance: dict,
) -> dict:
    markdown = result.markdown
    normalized_markdown = _normalize_visible(markdown)
    exact_hits = sum(token in markdown for token in required_tokens)
    normalized_hits = sum(
        _normalize_visible(token) in normalized_markdown for token in required_tokens
    )
    confidence = result.metadata.get("mean_confidence")
    if (
        type(confidence) not in {int, float}
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("OCRLLM facade did not report a valid mean confidence")
    reported_network_call_count = result.metadata.get("network_call_count")
    if (
        type(reported_network_call_count) is not int
        or reported_network_call_count < 0
    ):
        raise ValueError(
            "OCRLLM facade did not report a nonnegative integer network call count"
        )
    summary = {
        "candidate_id": _CANDIDATE_ID,
        "task": "ocr",
        "runtime_name": "ocrllm",
        "runtime_version": runtime_version,
        "model_revision": EXPECTED_REVISION,
        "backend": "local_rapidocr",
        "mode": "local_ocr",
        "workload_class": "generated_quality_control",
        "provenance": {
            "reviewed_baseline_ancestor": provenance["reviewed_baseline_ancestor"],
            "snapshot_clean": provenance["snapshot_clean"],
            "installed_noneditable": provenance["installed_noneditable"],
            "installed_from_pinned_snapshot": provenance[
                "installed_from_pinned_snapshot"
            ],
            "installed_source_matches_snapshot": provenance[
                "installed_source_matches_snapshot"
            ],
            "installed_python_file_count": provenance["python_file_count"],
            "runtime_component_count": provenance["runtime_component_count"],
            "rapidocr_model_file_count": provenance["rapidocr_model_file_count"],
        },
        "compatibility": {
            "recognition_succeeded": result.status == "complete",
            "elapsed_seconds": elapsed_seconds,
            "network_attempt_count": network_attempt_count,
            "reported_network_call_count": reported_network_call_count,
            "memory_only_output": result.output_path is None,
            "warning_count": len(result.warnings),
            "detected_line_count": int(result.metadata.get("detected_line_count", 0)),
            "facade_line_count": int(result.metadata.get("retained_line_count", 0)),
            "output_character_count": len(markdown),
            "required_token_count": len(required_tokens),
            "exact_required_token_hit_count": exact_hits,
            "exact_required_token_recall": exact_hits / len(required_tokens),
            "whitespace_insensitive_required_token_hit_count": normalized_hits,
            "whitespace_insensitive_required_token_recall": (
                normalized_hits / len(required_tokens)
            ),
            "mean_reported_confidence": float(confidence),
            "latex_marker_count": len(_LATEX_MARKER.findall(markdown)),
            "facade_exposes_boxes": (
                "boxes" in result_fields or "boxes" in result.metadata
            ),
            "facade_exposes_line_confidences": (
                "line_confidences" in result_fields
                or "line_confidences" in result.metadata
            ),
        },
        "authority_boundary": {
            key: provenance[key]
            for key in (
                "experimental_direct_short_mp3_public_facade",
                "local_asr_public_symbol_count",
                "local_asr_facade_available",
                "filetrans_public_symbol_count",
                "filetrans_facade_available",
                "configured_direct_audio_limit_seconds",
                "long_audio_facade_available",
                "direct_short_inmemory_options_accepted",
                "registered_audio_available_capability_count",
                "audio_nonmemory_option_rejection_count",
                "audio_persistence_available",
                "audio_resume_available",
                "audio_worker_command_count",
                "audio_worker_support_available",
                "benchmark_owned_asr_adapters_required",
            )
        },
    }
    return validate_public_summary(summary)


def _normalize_visible(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


def _json_safe(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
