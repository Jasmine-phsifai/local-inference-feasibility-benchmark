"""Verify the isolated RapidOCR ORT/OpenVINO comparison environment."""

import hashlib

from importlib.metadata import version
from pathlib import Path

import onnxruntime
import openvino
import rapidocr
from rapidocr import EngineType, RapidOCR


EXPECTED_PACKAGE_VERSIONS = {
    "numpy": "2.2.6",
    "onnxruntime": "1.29.0",
    "opencv-python": "4.13.0.92",
    "openvino": "2026.3.0",
    "pillow": "12.3.0",
    "psutil": "7.2.2",
    "rapidocr": "3.9.2",
}
EXPECTED_MODEL_HASHES = {
    "PP-OCRv6_det_small.onnx": (
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"
    ),
    "PP-OCRv6_rec_small.onnx": (
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"
    ),
    "ch_ppocr_mobile_v2.0_cls_mobile.onnx": (
        "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"
    ),
}


def main() -> None:
    actual_versions = {
        package: version(package) for package in EXPECTED_PACKAGE_VERSIONS
    }
    if actual_versions != EXPECTED_PACKAGE_VERSIONS:
        raise RuntimeError(
            f"RapidOCR OpenVINO environment version mismatch: {actual_versions}"
        )
    if not RapidOCR or {
        EngineType.ONNXRUNTIME.value,
        EngineType.OPENVINO.value,
    } != {"onnxruntime", "openvino"}:
        raise RuntimeError("RapidOCR backend surface is incomplete")
    if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("ONNX Runtime CPU execution provider is unavailable")

    model_root = Path(rapidocr.__file__).resolve().parent / "models"
    actual_model_hashes = {
        name: _sha256(model_root / name) for name in EXPECTED_MODEL_HASHES
    }
    if actual_model_hashes != EXPECTED_MODEL_HASHES:
        raise RuntimeError(f"RapidOCR model hash mismatch: {actual_model_hashes}")

    core = openvino.Core()
    if "CPU" not in core.available_devices:
        raise RuntimeError("OpenVINO CPU device is unavailable")
    print(
        {
            "versions": actual_versions,
            "model_hashes": actual_model_hashes,
            "onnxruntime_providers": onnxruntime.get_available_providers(),
            "openvino_cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
