"""Verify the immutable HF-native Qwen3-ASR OpenVINO environment."""

from __future__ import annotations

import json
import sys
from importlib.metadata import PackageNotFoundError, distribution, version

import openvino


EXPECTED_OPTIMUM_INTEL_REVISION = (
    "4ca1144eafc3ef7d3d805a99c7b92953441437e5"
)
EXPECTED_PACKAGE_VERSIONS = {
    "huggingface-hub": "1.28.0",
    "librosa": "0.11.0",
    "nncf": "3.3.0",
    "numpy": "2.4.6",
    "openvino": "2026.3.0",
    "openvino-tokenizers": "2026.3.0.0",
    "optimum": "2.2.0",
    "requests": "2.34.2",
    "safetensors": "0.8.0",
    "scipy": "1.17.1",
    "soundfile": "0.14.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0+cpu",
    "torchaudio": "2.11.0+cpu",
    "transformers": "5.13.1",
    "psutil": "7.2.2",
}


def main() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("HF-native OpenVINO environment requires Python 3.11")
    actual_versions = {
        package: version(package) for package in EXPECTED_PACKAGE_VERSIONS
    }
    if actual_versions != EXPECTED_PACKAGE_VERSIONS:
        raise RuntimeError(
            f"HF-native OpenVINO environment version mismatch: {actual_versions}"
        )

    try:
        version("qwen-asr")
    except PackageNotFoundError:
        pass
    else:
        raise RuntimeError("legacy qwen-asr must not enter the HF-native environment")

    optimum_intel = distribution("optimum-intel")
    direct_url_text = optimum_intel.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    commit_id = direct_url.get("vcs_info", {}).get("commit_id")
    if commit_id != EXPECTED_OPTIMUM_INTEL_REVISION:
        raise RuntimeError(
            f"Optimum Intel revision mismatch: expected "
            f"{EXPECTED_OPTIMUM_INTEL_REVISION}, got {commit_id}"
        )
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("Optimum Intel remained editable")

    from optimum.intel import (  # noqa: PLC0415
        OVModelForQwen3ASRForcedAligner,
        OVModelForSpeechSeq2Seq,
    )
    from transformers import (  # noqa: PLC0415
        Qwen3ASRForConditionalGeneration,
        Qwen3ASRForTokenClassification,
    )

    if not all(
        (
            OVModelForSpeechSeq2Seq,
            OVModelForQwen3ASRForcedAligner,
            Qwen3ASRForConditionalGeneration,
            Qwen3ASRForTokenClassification,
        )
    ):
        raise RuntimeError("HF-native Qwen3-ASR import surface is incomplete")

    core = openvino.Core()
    devices = list(core.available_devices)
    if "CPU" not in devices or "GPU" not in devices:
        raise RuntimeError(f"required OpenVINO CPU/GPU devices missing: {devices}")
    details = {
        device: core.get_property(device, "FULL_DEVICE_NAME")
        for device in devices
    }
    print(
        {
            "versions": actual_versions,
            "optimum_intel_version": optimum_intel.version,
            "optimum_intel_revision": commit_id,
            "devices": details,
        }
    )


if __name__ == "__main__":
    main()
