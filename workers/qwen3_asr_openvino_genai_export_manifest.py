"""Verify the official stateful Qwen3-ASR OpenVINO GenAI export."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCE_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
EXPORTER_REVISION = "f48d93fddff8c91e198389c47a6d5974789b67f4"
SOURCE_WEIGHT_BYTES = 1_876_091_704
SOURCE_WEIGHT_SHA256 = (
    "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
)
EXPORT_TASK = "automatic-speech-recognition-with-past"
MARKER_FILENAME = "export-complete.json"
PROVENANCE_FILENAME = "official-export-provenance.json"
SOURCE_LOCAL_AUXILIARY_FILES = {
    "download-attempts.jsonl",
    "openvino-export-attempts.jsonl",
    "openvino-genai-export-attempts.jsonl",
}
EXPECTED_SOURCE_FILES = {
    ".gitattributes": (1519, "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361"),
    "README.md": (57456, "5058416891bc47a2051557765997e8c42f8eb78a0e33c3e775bd17d4b0ba4d50"),
    "chat_template.json": (1161, "75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff"),
    "config.json": (6193, "76d3ae4601ce939830b2517f4a6cadb86cc51316c3900af6b020b051c21a478c"),
    "generation_config.json": (142, "1da527824d81e07118facff437e03f2e24a23311e3bdeb2368973fe77e5f275c"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model.safetensors": (1876091704, "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"),
    "preprocessor_config.json": (330, "45e120a4eda2c20c5d7f2ea9354e63536bf35e27aa573fb7cdf78017b378770d"),
    "tokenizer_config.json": (12487, "4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
EXPECTED_IR_FILES = {
    f"openvino_{component}.{suffix}"
    for component in ("decoder_model", "detokenizer", "encoder_model", "tokenizer")
    for suffix in ("bin", "xml")
}
EXPECTED_DECODER_INTERFACE = {
    "inputs": ["beam_idx", "encoder_hidden_states", "input_ids"],
    "outputs": ["logits"],
}
EXPECTED_ENCODER_INTERFACE = {
    "inputs": ["input_features"],
    "outputs": ["last_hidden_state"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    verify_source_parser = subparsers.add_parser("verify-source")
    verify_source_parser.add_argument("--source-model", required=True, type=Path)
    write_parser = subparsers.add_parser("write-marker")
    write_parser.add_argument("--source-model", required=True, type=Path)
    write_parser.add_argument("--export-model", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify-export")
    verify_parser.add_argument("--source-model", required=True, type=Path)
    verify_parser.add_argument("--export-model", required=True, type=Path)
    args = parser.parse_args()

    if args.action == "verify-source":
        print(json.dumps(verify_source(args.source_model), sort_keys=True))
        return
    if args.action == "write-marker":
        marker = write_export_marker(args.source_model, args.export_model)
        print(json.dumps({"marker": marker.name, "written": True}, sort_keys=True))
        return
    summary = verify_export(args.source_model, args.export_model)
    print(json.dumps(summary, sort_keys=True))


def verify_source(source_model: Path) -> dict:
    source_model = source_model.resolve(strict=True)
    actual_names = {
        path.relative_to(source_model).as_posix()
        for path in source_model.rglob("*")
        if path.is_file() and path.name not in SOURCE_LOCAL_AUXILIARY_FILES
    }
    if actual_names != set(EXPECTED_SOURCE_FILES):
        raise RuntimeError("pinned Qwen3-ASR source inventory changed")
    for name, (expected_size, expected_sha256) in EXPECTED_SOURCE_FILES.items():
        path = source_model / name
        if (
            path.stat().st_size != expected_size
            or _sha256(path) != expected_sha256
        ):
            raise RuntimeError(f"pinned Qwen3-ASR source file changed: {name}")
    config = json.loads((source_model / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "qwen3_asr"
        or config.get("architectures")
        != ["Qwen3ASRForConditionalGeneration"]
    ):
        raise RuntimeError("pinned Qwen3-ASR source configuration changed")
    return {
        "revision": SOURCE_REVISION,
        "weight_bytes": SOURCE_WEIGHT_BYTES,
        "weight_sha256": SOURCE_WEIGHT_SHA256,
    }


def write_export_marker(source_model: Path, export_model: Path) -> Path:
    export_model = export_model.resolve(strict=True)
    marker_path = export_model / MARKER_FILENAME
    if marker_path.exists():
        raise FileExistsError("official OpenVINO GenAI marker already exists")
    marker = _build_marker(source_model, export_model)
    temporary_path = marker_path.with_name(f".{MARKER_FILENAME}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(marker_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return marker_path


def write_export_provenance(source_model: Path, export_model: Path) -> Path:
    """Create the hash-bound provenance required before publishing a marker."""

    export_model = export_model.resolve(strict=True)
    provenance_path = export_model / PROVENANCE_FILENAME
    if provenance_path.exists():
        raise FileExistsError(
            "official OpenVINO GenAI export provenance already exists"
        )
    verify_source(source_model)
    structure = _inspect_export(export_model)
    files = _collect_export_files(export_model)
    source_files = [
        {"name": name, "size_bytes": size, "sha256": sha256}
        for name, (size, sha256) in sorted(EXPECTED_SOURCE_FILES.items())
    ]
    provenance = {
        "schema": "qwen3-asr-official-with-past-export-provenance-v1",
        "preflight": {
            "source": {
                "revision": SOURCE_REVISION,
                "files": source_files,
            },
            "exporter": {"revision": EXPORTER_REVISION},
            "contract": {
                "task": EXPORT_TASK,
                "weight_format": "fp16",
                "stateful_export": True,
                "offline_source_only": True,
            },
        },
        "export": {
            "model_type": "qwen3_asr",
            "beam_idx_present": (
                "beam_idx" in structure["decoder"]["interface"]["inputs"]
            ),
            "stateful_decoder": (
                structure["decoder"]["read_value_count"] > 0
                and structure["decoder"]["read_value_count"]
                == structure["decoder"]["assign_count"]
            ),
            "decoder": structure["decoder"],
            "encoder": structure["encoder"],
            "files": files,
        },
    }
    temporary_path = provenance_path.with_name(
        f".{PROVENANCE_FILENAME}.{os.getpid()}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if provenance_path.exists():
            raise FileExistsError(
                "official OpenVINO GenAI export provenance appeared"
            )
        temporary_path.replace(provenance_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    _verify_export_provenance(export_model, files, structure=structure)
    return provenance_path


def verify_export(source_model: Path, export_model: Path) -> dict:
    export_model = export_model.resolve(strict=True)
    marker_path = export_model / MARKER_FILENAME
    if not marker_path.is_file():
        raise RuntimeError("official OpenVINO GenAI export marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = _build_marker(source_model, export_model)
    if marker != expected:
        raise RuntimeError("official OpenVINO GenAI export marker does not match")
    return {
        "source_revision": SOURCE_REVISION,
        "exporter_revision": EXPORTER_REVISION,
        "task": EXPORT_TASK,
        "beam_idx_present": True,
        "stateful_decoder": True,
        "file_count": len(marker["files"]),
        "total_bytes": sum(item["size_bytes"] for item in marker["files"]),
    }


def _build_marker(source_model: Path, export_model: Path) -> dict:
    source = verify_source(source_model)
    structure = _inspect_export(export_model)
    files = _collect_export_files(export_model)
    provenance = _verify_export_provenance(
        export_model,
        files,
        structure=structure,
    )
    return {
        "schema_version": 2,
        "source": source,
        "exporter_revision": EXPORTER_REVISION,
        "task": EXPORT_TASK,
        "weight_format": "fp16",
        "provenance": provenance,
        "structure": structure,
        "files": files,
    }


def _collect_export_files(export_model: Path) -> list[dict]:
    files = []
    for path in sorted(
        (
            path
            for path in export_model.rglob("*")
            if path.is_file()
            and path.name != MARKER_FILENAME
            and path.name != PROVENANCE_FILENAME
        ),
        key=lambda path: path.relative_to(export_model).as_posix(),
    ):
        files.append(
            {
                "name": path.relative_to(export_model).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return files


def _verify_export_provenance(
    export_model: Path,
    files: list[dict],
    *,
    structure: dict | None = None,
) -> dict:
    provenance_path = export_model / PROVENANCE_FILENAME
    if not provenance_path.is_file():
        raise RuntimeError("official OpenVINO GenAI export provenance is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    preflight = provenance.get("preflight", {})
    source = preflight.get("source", {})
    exporter = preflight.get("exporter", {})
    contract = preflight.get("contract", {})
    export = provenance.get("export", {})
    if structure is None:
        structure = _inspect_export(export_model)
    expected_source_files = [
        {"name": name, "size_bytes": size, "sha256": sha256}
        for name, (size, sha256) in sorted(EXPECTED_SOURCE_FILES.items())
    ]
    if (
        provenance.get("schema")
        != "qwen3-asr-official-with-past-export-provenance-v1"
        or source.get("revision") != SOURCE_REVISION
        or source.get("files") != expected_source_files
        or exporter.get("revision") != EXPORTER_REVISION
        or contract.get("task") != EXPORT_TASK
        or contract.get("weight_format") != "fp16"
        or contract.get("stateful_export") is not True
        or contract.get("offline_source_only") is not True
        or export.get("model_type") != "qwen3_asr"
        or export.get("beam_idx_present") is not True
        or export.get("stateful_decoder") is not True
        or export.get("decoder") != structure["decoder"]
        or export.get("encoder") != structure["encoder"]
        or export.get("files") != files
    ):
        raise RuntimeError("official OpenVINO GenAI export provenance changed")
    return {
        "filename": PROVENANCE_FILENAME,
        "sha256": _sha256(provenance_path),
    }


def _inspect_export(export_model: Path) -> dict:
    if any(path.suffix.casefold() == ".part" for path in export_model.rglob("*")):
        raise RuntimeError("official OpenVINO GenAI export contains a partial file")
    ir_files = {
        path.relative_to(export_model).as_posix()
        for path in export_model.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".xml", ".bin"}
    }
    if ir_files != EXPECTED_IR_FILES:
        raise RuntimeError("official OpenVINO GenAI IR inventory changed")
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json"):
        if not (export_model / name).is_file():
            raise RuntimeError(f"official OpenVINO GenAI export is missing {name}")
    config = json.loads((export_model / "config.json").read_text(encoding="utf-8"))
    if (
        config.get("model_type") != "qwen3_asr"
        or config.get("is_encoder_decoder") is not True
    ):
        raise RuntimeError("official OpenVINO GenAI export configuration changed")

    decoder = _read_ir_interface(export_model / "openvino_decoder_model.xml")
    encoder = _read_ir_interface(export_model / "openvino_encoder_model.xml")
    if decoder["interface"] != EXPECTED_DECODER_INTERFACE:
        raise RuntimeError("official decoder lacks the beam_idx interface")
    if encoder["interface"] != EXPECTED_ENCODER_INTERFACE:
        raise RuntimeError("official encoder interface changed")
    if (
        decoder["read_value_count"] <= 0
        or decoder["read_value_count"] != decoder["assign_count"]
    ):
        raise RuntimeError("official decoder is not a balanced stateful export")
    return {"decoder": decoder, "encoder": encoder}


def _read_ir_interface(path: Path) -> dict:
    root = ET.parse(path).getroot()
    inputs: set[str] = set()
    outputs: set[str] = set()
    read_value_count = 0
    assign_count = 0
    for layer in root.findall("./layers/layer"):
        layer_type = layer.attrib.get("type")
        if layer_type == "Parameter":
            for port in layer.findall("./output/port"):
                inputs.update(_split_names(port.attrib.get("names")))
        elif layer_type == "Result":
            outputs.update(_split_names(layer.attrib.get("output_names")))
        elif layer_type == "ReadValue":
            read_value_count += 1
        elif layer_type == "Assign":
            assign_count += 1
    return {
        "ir_version": root.attrib.get("version"),
        "interface": {"inputs": sorted(inputs), "outputs": sorted(outputs)},
        "read_value_count": read_value_count,
        "assign_count": assign_count,
    }


def _split_names(value: str | None) -> set[str]:
    if not value:
        return set()
    return {name.strip() for name in value.split(",") if name.strip()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
