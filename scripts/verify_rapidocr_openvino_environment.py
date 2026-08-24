"""Verify the isolated RapidOCR ORT/OpenVINO comparison environment."""

import hashlib
import json
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_locked_environment import verify_locked_environment


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
    environment = verify_locked_environment(
        PROJECT_ROOT
        / "environments"
        / "rapidocr_openvino"
        / "requirements.lock.txt",
        expected_python=(3, 11, 15),
    )

    import onnxruntime
    import openvino
    import rapidocr
    from rapidocr import EngineType, RapidOCR

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
        json.dumps(
            {
                "status": "verified",
                "environment": environment,
                "model_hashes": actual_model_hashes,
                "onnxruntime_providers": onnxruntime.get_available_providers(),
                "openvino_cpu": core.get_property("CPU", "FULL_DEVICE_NAME"),
            },
            sort_keys=True,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
