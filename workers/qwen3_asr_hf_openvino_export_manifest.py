"""Create and verify immutable Qwen3-ASR HF-native OpenVINO exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


MODEL_REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
OPTIMUM_INTEL_REVISION = "4ca1144eafc3ef7d3d805a99c7b92953441437e5"
EXPECTED_WEIGHT_SHA256 = (
    "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35"
)
CONVERSION_TASK = "automatic-speech-recognition-with-past"
CONVERSION_CONFIG = "official_notebook_export_true_equivalent_with_past"
REQUIRED_IR_STEMS = {"openvino_encoder_model", "openvino_decoder_model"}
EXPECTED_DECODER_INPUTS = ("beam_idx", "encoder_hidden_states", "input_ids")
EXPECTED_DECODER_OUTPUTS = ("logits",)
MARKER_NAME = "export-complete.json"


def verify_source_weight(
    source_model_dir: Path,
    *,
    expected_weight_sha256: str = EXPECTED_WEIGHT_SHA256,
) -> str:
    weight_path = source_model_dir / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("pinned HF-native Qwen3-ASR checkpoint is missing")
    actual = _sha256(weight_path)
    if actual != expected_weight_sha256.casefold():
        raise RuntimeError("HF-native Qwen3-ASR checkpoint hash mismatch")
    return actual


def build_ir_manifest(export_model_dir: Path) -> list[dict]:
    files = sorted(
        (
            path
            for path in export_model_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".xml", ".bin"}
        ),
        key=lambda path: path.relative_to(export_model_dir).as_posix(),
    )
    relative_names = {path.relative_to(export_model_dir).as_posix() for path in files}
    for stem in REQUIRED_IR_STEMS:
        for suffix in (".xml", ".bin"):
            if f"{stem}{suffix}" not in relative_names:
                raise RuntimeError("HF-native OpenVINO export is missing a required IR pair")
    for relative_name in relative_names:
        path = Path(relative_name)
        paired_name = path.with_suffix(".bin" if path.suffix.casefold() == ".xml" else ".xml").as_posix()
        if paired_name not in relative_names:
            raise RuntimeError("HF-native OpenVINO export contains an unpaired IR file")
    return [
        {
            "name": path.relative_to(export_model_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]


def decoder_interface(export_model_dir: Path) -> dict[str, list[str]]:
    """Read only the public top-level decoder ports from an OpenVINO IR."""

    decoder_xml = export_model_dir / "openvino_decoder_model.xml"
    if not decoder_xml.is_file():
        raise FileNotFoundError("HF-native OpenVINO decoder IR is missing")
    root = ET.parse(decoder_xml).getroot()
    input_names: set[str] = set()
    output_names: set[str] = set()
    for layer in root.findall("./layers/layer"):
        layer_type = layer.attrib.get("type")
        if layer_type == "Parameter":
            for port in layer.findall("./output/port"):
                input_names.update(_split_port_names(port.attrib.get("names")))
        elif layer_type == "Result":
            output_names.update(_split_port_names(layer.attrib.get("output_names")))
    interface = {
        "inputs": sorted(input_names),
        "outputs": sorted(output_names),
    }
    expected = {
        "inputs": sorted(EXPECTED_DECODER_INPUTS),
        "outputs": sorted(EXPECTED_DECODER_OUTPUTS),
    }
    if interface != expected:
        raise RuntimeError("HF-native OpenVINO decoder cache interface mismatch")
    return interface


def _split_port_names(value: str | None) -> set[str]:
    if not value:
        return set()
    return {name.strip() for name in value.split(",") if name.strip()}


def write_export_marker(
    source_model_dir: Path,
    export_model_dir: Path,
    *,
    expected_weight_sha256: str = EXPECTED_WEIGHT_SHA256,
) -> dict:
    source_weight_sha256 = verify_source_weight(
        source_model_dir,
        expected_weight_sha256=expected_weight_sha256,
    )
    marker = {
        "source_revision": MODEL_REVISION,
        "source_weight_sha256": source_weight_sha256,
        "optimum_intel_revision": OPTIMUM_INTEL_REVISION,
        "conversion_task": CONVERSION_TASK,
        "conversion_config": CONVERSION_CONFIG,
        "decoder_interface": decoder_interface(export_model_dir),
        "ir_manifest": build_ir_manifest(export_model_dir),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (export_model_dir / MARKER_NAME).write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return marker


def verify_export(
    source_model_dir: Path,
    export_model_dir: Path,
    *,
    expected_weight_sha256: str = EXPECTED_WEIGHT_SHA256,
) -> dict:
    source_weight_sha256 = verify_source_weight(
        source_model_dir,
        expected_weight_sha256=expected_weight_sha256,
    )
    marker_path = export_model_dir / MARKER_NAME
    if not marker_path.is_file():
        raise FileNotFoundError("verified HF-native OpenVINO export marker is missing")
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    expected_identity = {
        "source_revision": MODEL_REVISION,
        "source_weight_sha256": source_weight_sha256,
        "optimum_intel_revision": OPTIMUM_INTEL_REVISION,
        "conversion_task": CONVERSION_TASK,
        "conversion_config": CONVERSION_CONFIG,
    }
    for key, expected in expected_identity.items():
        if marker.get(key) != expected:
            raise RuntimeError(f"HF-native OpenVINO export provenance mismatch: {key}")
    if marker.get("decoder_interface") != decoder_interface(export_model_dir):
        raise RuntimeError("HF-native OpenVINO decoder interface manifest mismatch")
    if marker.get("ir_manifest") != build_ir_manifest(export_model_dir):
        raise RuntimeError("HF-native OpenVINO IR manifest mismatch")
    return marker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify-source", "write-marker", "verify-export"))
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--export-model", type=Path)
    args = parser.parse_args()
    if args.action == "verify-source":
        verify_source_weight(args.source_model)
        return
    if args.export_model is None:
        raise ValueError("export-model is required for export marker actions")
    if args.action == "write-marker":
        write_export_marker(args.source_model, args.export_model)
    else:
        verify_export(args.source_model, args.export_model)


if __name__ == "__main__":
    main()
