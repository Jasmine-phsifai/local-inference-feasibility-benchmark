"""Verify that OCRLLM is an independent, non-editable installation."""

import json
from importlib.metadata import distribution
from pathlib import Path

import ocrllm


EXPECTED_VERSION = "0.1.0"


def main() -> None:
    installed_distribution = distribution("ocrllm")
    if installed_distribution.version != EXPECTED_VERSION:
        raise RuntimeError(
            "OCRLLM version mismatch: "
            f"expected {EXPECTED_VERSION}, got {installed_distribution.version}"
        )

    direct_url_text = installed_distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("OCRLLM remained editable")

    module_path = Path(ocrllm.__file__).resolve()
    if "site-packages" not in {part.casefold() for part in module_path.parts}:
        raise RuntimeError(
            f"OCRLLM did not install into site-packages: {module_path}"
        )

    print(
        {
            "version": installed_distribution.version,
            "module_path": str(module_path),
            "editable": False,
        }
    )


if __name__ == "__main__":
    main()
