"""Record aggregate-only compatibility evidence for the active OCRLLM image facade."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import re
import time
import unicodedata
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from local_inference_bench.event_journal import append_event
from local_inference_bench.validate_public_summary import validate_public_summary


OCRLLM_REVISION = "379726281e3c374bda65c1bd4a6bdf5c32cde0b3"
_FORMULA_MARKER = re.compile(r"(?:[=+\-*/^]|\\(?:frac|sum|int|sqrt|begin)\b)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--append-journal", type=Path)
    parser.add_argument("--private-record", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError("generated OCRLLM compatibility image is missing")

    import requests
    import ocrllm
    from ocrllm import (
        Config,
        LocalOCRSettings,
        RecognitionResult,
        get_capabilities,
        recognize,
    )
    from rapidocr import RapidOCR

    distribution = importlib.metadata.distribution("ocrllm")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    if bool(direct_url.get("dir_info", {}).get("editable")):
        raise RuntimeError("OCRLLM compatibility requires an independent install")

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
        capabilities = get_capabilities(config)
        started = time.perf_counter()
        result = recognize(args.image, config=config)
        elapsed_seconds = time.perf_counter() - started
    finally:
        requests.sessions.Session.request = original_request

    raw_engine = RapidOCR(
        params={
            "Global.log_level": "critical",
            "Global.text_score": 0.5,
        }
    )
    raw_result = raw_engine(args.image)
    raw_texts = [] if raw_result.txts is None else [str(value) for value in raw_result.txts]
    raw_scores = [] if raw_result.scores is None else list(raw_result.scores)
    raw_boxes = [] if raw_result.boxes is None else list(raw_result.boxes)
    normalized_markdown = _normalize_visible(result.markdown)
    retained_lines = sum(
        bool(normalized := _normalize_visible(line))
        and normalized in normalized_markdown
        for line in raw_texts
    )
    formula_lines = [line for line in raw_texts if _looks_formula_like(line)]
    retained_formula_lines = sum(
        _normalize_visible(line) in normalized_markdown
        for line in formula_lines
        if _normalize_visible(line)
    )
    result_fields = {item.name for item in fields(RecognitionResult)}
    common_asr_symbols = {
        "recognize_audio",
        "transcribe",
        "ASRSettings",
        "AudioSettings",
        "FileTransSettings",
    }
    public_asr_symbols = common_asr_symbols.intersection(dir(ocrllm))
    audio_capabilities = [
        report
        for report in capabilities
        if "audio" in report.name.casefold() or "filetrans" in report.name.casefold()
    ]
    public_summary = validate_public_summary(
        {
            "candidate_id": "ocrllm_active_image_facade",
            "task": "ocr",
            "runtime_name": "ocrllm",
            "runtime_version": distribution.version,
            "model_revision": OCRLLM_REVISION,
            "backend": "local_rapidocr",
            "workload_class": "generated_quality_control",
            "compatibility": {
                "recognition_succeeded": result.status == "complete",
                "elapsed_seconds": elapsed_seconds,
                "network_attempt_count": network_attempts,
                "memory_only_output": result.output_path is None,
                "warning_count": len(result.warnings),
                "raw_line_count": len(raw_texts),
                "facade_line_count": int(
                    result.metadata.get("retained_line_count", 0)
                ),
                "exact_raw_lines_retained_count": retained_lines,
                "raw_formula_like_line_count": len(formula_lines),
                "exact_formula_like_lines_retained_count": retained_formula_lines,
                "raw_box_count": len(raw_boxes),
                "raw_confidence_count": len(raw_scores),
                "facade_exposes_boxes": (
                    "boxes" in result_fields or "boxes" in result.metadata
                ),
                "facade_exposes_line_confidences": (
                    "line_confidences" in result_fields
                    or "line_confidences" in result.metadata
                ),
                "public_asr_symbol_count": len(public_asr_symbols),
                "audio_capability_count": len(audio_capabilities),
                "audio_capabilities_deferred": bool(audio_capabilities)
                and all(report.status == "deferred" for report in audio_capabilities),
            },
        }
    )
    event = {
        "event": "ocrllm_compatibility_checked",
        "protocol": "ocrllm-image-compatibility-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": _sha256(args.image),
        "result": public_summary,
    }
    if args.private_record is not None:
        args.private_record.parent.mkdir(parents=True, exist_ok=True)
        args.private_record.write_text(
            json.dumps(
                _json_safe(
                    {
                        "recognize_signature": str(inspect.signature(recognize)),
                        "module_file": str(Path(ocrllm.__file__).resolve()),
                        "direct_url": direct_url,
                        "facade_markdown": result.markdown,
                        "facade_metadata": result.metadata,
                        "raw_texts": raw_texts,
                        "raw_scores": raw_scores,
                        "raw_boxes": raw_boxes,
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


def _normalize_visible(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if not character.isspace())


def _looks_formula_like(value: str) -> bool:
    return bool(_FORMULA_MARKER.search(value))


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


if __name__ == "__main__":
    main()
