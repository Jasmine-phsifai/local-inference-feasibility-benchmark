"""Verify the pinned Qwen3-ASR OpenVINO environment and visible devices."""

from importlib.metadata import version

import openvino
import qwen_asr  # noqa: F401 - importing proves the package is usable
import transformers


EXPECTED_PACKAGE_VERSIONS = {
    "openvino": "2026.3.0",
    "optimum-intel": "2.1.0",
    "qwen-asr": "0.0.6",
    "transformers": "4.57.6",
}


def main() -> None:
    actual_versions = {
        package: version(package) for package in EXPECTED_PACKAGE_VERSIONS
    }
    if actual_versions != EXPECTED_PACKAGE_VERSIONS:
        raise RuntimeError(
            f"OpenVINO environment version mismatch: {actual_versions}"
        )
    openvino_package_version = EXPECTED_PACKAGE_VERSIONS["openvino"]
    if not (
        openvino.__version__ == openvino_package_version
        or openvino.__version__.startswith(f"{openvino_package_version}-")
    ):
        raise RuntimeError("OpenVINO import metadata does not match its package")
    if transformers.__version__ != EXPECTED_PACKAGE_VERSIONS["transformers"]:
        raise RuntimeError("Transformers import metadata does not match its package")

    core = openvino.Core()
    devices = list(core.available_devices)
    if "CPU" not in devices:
        raise RuntimeError("OpenVINO CPU device is unavailable")
    details = {
        device: core.get_property(device, "FULL_DEVICE_NAME") for device in devices
    }
    print({"versions": actual_versions, "devices": details})


if __name__ == "__main__":
    main()
