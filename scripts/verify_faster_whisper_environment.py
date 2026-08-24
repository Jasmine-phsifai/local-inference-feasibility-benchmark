"""Verify the exact faster-whisper runtime and pinned Small-model snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_asset_inventory import verify_asset_inventory
from local_inference_bench.verify_locked_environment import verify_locked_environment


MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def main() -> None:
    environment = verify_locked_environment(
        PROJECT_ROOT
        / "environments"
        / "faster_whisper_cpu"
        / "requirements.lock.txt",
        expected_python=(3, 11, 15),
    )
    manifest = json.loads(
        (
            PROJECT_ROOT / "registries" / "faster_whisper_small_assets.json"
        ).read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("candidate_id") != "faster_whisper_cpu"
        or manifest.get("repository") != "Systran/faster-whisper-small"
        or manifest.get("revision") != MODEL_REVISION
    ):
        raise RuntimeError("faster-whisper asset manifest identity is invalid")
    model_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--Systran--faster-whisper-small"
        / "snapshots"
        / MODEL_REVISION
    )
    assets = verify_asset_inventory(model_root, manifest.get("required_files"))

    import ctranslate2
    import faster_whisper
    import onnxruntime

    if "int8" not in ctranslate2.get_supported_compute_types("cpu"):
        raise RuntimeError("CTranslate2 CPU int8 support is unavailable")
    if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("ONNX Runtime CPU execution provider is unavailable")
    if not faster_whisper.WhisperModel:
        raise RuntimeError("faster-whisper model surface is unavailable")
    print(
        json.dumps(
            {
                "status": "verified",
                "environment": environment,
                "assets": assets,
                "model_revision": MODEL_REVISION,
                "compute_type": "int8",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
