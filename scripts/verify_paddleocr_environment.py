"""Verify the exact PaddleOCR runtime and PP-OCRv6 cached model inventories."""

from __future__ import annotations

import json
import hashlib
import ctypes
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_asset_inventory import verify_asset_inventory
from local_inference_bench.verify_locked_environment import verify_locked_environment


MODEL_REVISIONS = {
    "PP-OCRv6_tiny_det": "d3177d4e5551463292a61e27cfca2b53e7c3fe9d",
    "PP-OCRv6_tiny_rec": "0736086f72f666350ebcdc0c3a504eeac89cdfad",
    "PP-OCRv6_medium_det": "8e0f56fb2ef86b461d99cfc7ac5c137738985f61",
    "PP-OCRv6_medium_rec": "e5a92bcbc5cc1b494628e458d267778f0704fd7c",
}
VCOMP_IDENTITY = {
    "conda_package": "vcomp14",
    "version": "14.44.35208",
    "build": "h4927774_12",
    "dll_relative_path": "vcomp140.dll",
    "size_bytes": 193184,
    "sha256": "e7e50211906ab1c6226a3133fda8d8c8e35a23a8b6c6133c9606601604680b85",
}


def main() -> None:
    environment = verify_locked_environment(
        PROJECT_ROOT / "environments" / "paddleocr_cpu" / "requirements.lock.txt",
        expected_python=(3, 11, 15),
    )
    manifest = json.loads(
        (
            PROJECT_ROOT / "registries" / "ppocrv6_cached_assets.json"
        ).read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("candidate_ids")
        != ["ppocrv6_tiny_cpu", "ppocrv6_medium_cpu"]
        or {
            model.get("name"): model.get("revision")
            for model in manifest.get("models", [])
            if isinstance(model, dict)
        }
        != MODEL_REVISIONS
        or manifest.get("native_runtime") != VCOMP_IDENTITY
    ):
        raise RuntimeError("PP-OCRv6 asset manifest identity is invalid")
    assets = verify_asset_inventory(
        Path.home() / ".paddlex" / "official_models",
        manifest.get("required_files"),
        scope_roots=tuple(MODEL_REVISIONS),
    )

    native_runtime = _verify_vcomp_runtime()

    expected_dll = (Path(sys.prefix) / VCOMP_IDENTITY["dll_relative_path"]).resolve()
    vcomp_handle = ctypes.WinDLL(str(expected_dll))

    import cv2
    import paddle
    import paddleocr
    import psutil

    paddle.set_device("cpu")
    if not paddleocr.PaddleOCR or cv2.getNumThreads() < 1:
        raise RuntimeError("PaddleOCR CPU surface is unavailable")
    loaded_vcomp = {
        Path(mapping.path).resolve()
        for mapping in psutil.Process().memory_maps()
        if Path(mapping.path).name.casefold() == "vcomp140.dll"
    }
    if expected_dll not in loaded_vcomp:
        raise RuntimeError("PaddleOCR did not load the verified environment VCOMP runtime")
    if vcomp_handle._handle == 0:
        raise RuntimeError("verified VCOMP runtime handle is invalid")
    print(
        json.dumps(
            {
                "status": "verified",
                "environment": environment,
                "assets": assets,
                "native_runtime": native_runtime,
                "device": str(paddle.device.get_device()),
            },
            sort_keys=True,
        )
    )


def _verify_vcomp_runtime() -> dict:
    dll_path = Path(sys.prefix) / VCOMP_IDENTITY["dll_relative_path"]
    metadata_path = (
        Path(sys.prefix)
        / "conda-meta"
        / f"vcomp14-{VCOMP_IDENTITY['version']}-{VCOMP_IDENTITY['build']}.json"
    )
    if (
        not dll_path.is_file()
        or dll_path.stat().st_size != VCOMP_IDENTITY["size_bytes"]
        or _sha256(dll_path) != VCOMP_IDENTITY["sha256"]
        or not metadata_path.is_file()
    ):
        raise RuntimeError("pinned Paddle/OpenMP native runtime is missing or changed")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("name") != VCOMP_IDENTITY["conda_package"]
        or metadata.get("version") != VCOMP_IDENTITY["version"]
        or metadata.get("build") != VCOMP_IDENTITY["build"]
    ):
        raise RuntimeError("Paddle/OpenMP Conda metadata is invalid")
    return {
        "package": metadata["name"],
        "version": metadata["version"],
        "build": metadata["build"],
        "dll_sha256": VCOMP_IDENTITY["sha256"],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
