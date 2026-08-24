"""Verify the exact Windows nightly that contains the Qwen3-ASR tail fix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from importlib.metadata import distribution, version
from pathlib import Path
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_locked_environment import verify_locked_environment
from local_inference_bench.qwen3_asr_tailfix_profile import (
    TAILFIX_PACKAGE_VERSIONS,
    load_qwen3_asr_tailfix_profile,
)


PROFILE_PATH = (
    PROJECT_ROOT
    / "environments/qwen3_asr_openvino_genai_tailfix_20260821/runtime-provenance.json"
)
LOCK_PATH = PROFILE_PATH.with_name("requirements.lock.txt")
_DIRECT_WHEEL_PACKAGES = {
    "openvino",
    "openvino-genai",
    "openvino-tokenizers",
}


def main() -> None:
    profile = load_runtime_profile(PROFILE_PATH)
    if Path(sys.prefix).name != profile["environment_name"]:
        raise RuntimeError("tail-fix OpenVINO GenAI environment name changed")
    if ".".join(str(value) for value in sys.version_info[:3]) != profile[
        "python_version"
    ]:
        raise RuntimeError("tail-fix OpenVINO GenAI Python version changed")

    environment = verify_locked_environment(
        LOCK_PATH,
        expected_python=(3, 11, 15),
    )
    if environment["package_count"] != len(profile["package_versions"]):
        raise RuntimeError("tail-fix OpenVINO GenAI package count changed")
    _verify_pip_dependencies()

    for package, expected_version in profile["package_versions"].items():
        if version(package) != expected_version:
            raise RuntimeError("tail-fix OpenVINO GenAI package version changed")
    for package in sorted(_DIRECT_WHEEL_PACKAGES):
        _verify_direct_wheel(package, profile["wheel_artifacts"][package])

    import openvino
    import openvino_genai

    product_version = profile["openvino_genai_product_version"]
    if (
        openvino_genai.__version__ != product_version
        or openvino_genai.get_version() != product_version
    ):
        raise RuntimeError("tail-fix OpenVINO GenAI DLL ProductVersion changed")
    package_root = Path(openvino_genai.__file__).resolve().parent
    dlls = list(package_root.glob("openvino_genai.dll"))
    if len(dlls) != 1 or not dlls[0].is_file():
        raise RuntimeError("tail-fix OpenVINO GenAI DLL is unavailable")
    if not hasattr(openvino_genai, "ASRPipeline"):
        raise RuntimeError("tail-fix OpenVINO GenAI ASRPipeline is unavailable")

    source_contains_fix = _verify_source_ancestry(profile)
    core = openvino.Core()
    if "CPU" not in core.available_devices:
        raise RuntimeError("OpenVINO CPU device is unavailable")
    print(
        json.dumps(
            {
                "environment": profile["environment_name"],
                "package_count": environment["package_count"],
                "openvino_genai_product_version": product_version,
                "source_contains_required_fix": source_contains_fix,
                "cpu_device": core.get_property("CPU", "FULL_DEVICE_NAME"),
            },
            sort_keys=True,
        )
    )


def load_runtime_profile(path: Path) -> dict:
    profile = load_qwen3_asr_tailfix_profile(path)
    if profile["package_versions"] != TAILFIX_PACKAGE_VERSIONS:
        raise RuntimeError("tail-fix package inventory changed")
    return profile


def _verify_pip_dependencies() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    if completed.returncode != 0:
        raise RuntimeError("tail-fix OpenVINO GenAI environment failed pip check")


def _verify_direct_wheel(package: str, artifact: dict) -> None:
    direct_url_text = distribution(package).read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    parsed = urlparse(direct_url.get("url", ""))
    installed_filename = Path(unquote(parsed.path)).name
    installed_sha256 = (
        direct_url.get("archive_info", {}).get("hashes", {}).get("sha256")
    )
    official_url = artifact["url"]
    if (
        installed_filename != artifact["filename"]
        or installed_sha256 != artifact["sha256"]
        or parsed.scheme not in {"file", "https"}
        or (parsed.scheme == "https" and direct_url["url"] != official_url)
    ):
        raise RuntimeError("tail-fix OpenVINO wheel provenance changed")


def _verify_source_ancestry(profile: dict) -> bool:
    relative_checkout = Path(profile["source_checkout"])
    if relative_checkout.is_absolute() or ".." in relative_checkout.parts:
        raise RuntimeError("tail-fix source checkout identity is invalid")
    checkout = (PROJECT_ROOT / relative_checkout).resolve(strict=True)
    vendor_root = (PROJECT_ROOT / "data/vendor").resolve(strict=True)
    if not checkout.is_relative_to(vendor_root):
        raise RuntimeError("tail-fix source checkout escaped vendor directory")
    if _run_git(checkout, "remote", "get-url", "origin") != profile[
        "source_repository"
    ]:
        raise RuntimeError("tail-fix source repository changed")
    if _run_git(checkout, "rev-parse", "--is-shallow-repository") != "false":
        raise RuntimeError("tail-fix source ancestry requires full history")
    for revision in (
        profile["stable_source_revision"],
        profile["required_fix_revision"],
        profile["associated_source_revision"],
    ):
        if _run_git(checkout, "rev-parse", f"{revision}^{{commit}}") != revision:
            raise RuntimeError("tail-fix source revision is unavailable")
    if not _git_is_ancestor(
        checkout,
        profile["required_fix_revision"],
        profile["associated_source_revision"],
    ):
        raise RuntimeError("tail-fix source revision does not contain the fix")
    return True


def _run_git(checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError("tail-fix source verification failed")
    return completed.stdout.strip()


def _git_is_ancestor(checkout: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError("tail-fix source ancestry verification failed")
    return completed.returncode == 0


if __name__ == "__main__":
    main()
