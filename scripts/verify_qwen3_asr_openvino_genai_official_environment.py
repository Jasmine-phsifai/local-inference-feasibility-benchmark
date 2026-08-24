"""Verify the pinned official OpenVINO GenAI Qwen3-ASR environment."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_locked_environment import verify_locked_environment


EXPECTED_ENVIRONMENT = "local-bench-qwen3-asr-openvino-genai-official"
EXPECTED_OPTIMUM_INTEL_REPOSITORY = (
    "https://github.com/huggingface/optimum-intel.git"
)
EXPECTED_OPTIMUM_INTEL_REVISION = (
    "f48d93fddff8c91e198389c47a6d5974789b67f4"
)
OPTIMUM_INTEL_VERSION = "2.1.0.dev0+f48d93f"


def main() -> None:
    if Path(sys.prefix).name != EXPECTED_ENVIRONMENT:
        raise RuntimeError("official OpenVINO GenAI environment name changed")
    if sys.version_info[:3] != (3, 11, 15):
        raise RuntimeError("official OpenVINO GenAI requires Python 3.11.15")

    environment = verify_locked_environment(
        PROJECT_ROOT
        / "environments"
        / "qwen3_asr_openvino_genai_official"
        / "requirements.lock.txt",
        expected_python=(3, 11, 15),
        allowed_extra_packages={"optimum-intel": OPTIMUM_INTEL_VERSION},
    )
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        diagnostics = (pip_check.stdout + pip_check.stderr).strip()
        raise RuntimeError(
            f"official OpenVINO GenAI environment failed pip check: {diagnostics}"
        )

    import openvino
    import openvino_genai

    optimum_intel = distribution("optimum-intel")
    direct_url_text = optimum_intel.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    vcs_info = direct_url.get("vcs_info", {})
    if (
        direct_url.get("url") != EXPECTED_OPTIMUM_INTEL_REPOSITORY
        or vcs_info.get("commit_id") != EXPECTED_OPTIMUM_INTEL_REVISION
        or direct_url.get("dir_info", {}).get("editable")
    ):
        raise RuntimeError("official Optimum Intel provenance changed")
    if not hasattr(openvino_genai, "ASRPipeline"):
        raise RuntimeError("OpenVINO GenAI ASRPipeline is unavailable")
    optimum_cli = Path(sys.prefix) / "Scripts/optimum-cli.exe"
    if not optimum_cli.is_file():
        raise RuntimeError("official Optimum CLI is unavailable")

    core = openvino.Core()
    devices = list(core.available_devices)
    if "CPU" not in devices:
        raise RuntimeError("OpenVINO CPU device is unavailable")
    details = {
        device: core.get_property(device, "FULL_DEVICE_NAME")
        for device in devices
    }
    print(
        {
            "environment": environment,
            "optimum_intel_revision": EXPECTED_OPTIMUM_INTEL_REVISION,
            "devices": details,
        }
    )


if __name__ == "__main__":
    main()
