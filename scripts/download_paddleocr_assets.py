"""Acquire immutable PP-OCRv6 snapshots into PaddleX's model cache."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_asset_inventory import verify_asset_inventory


def main() -> None:
    manifest = json.loads(
        (PROJECT_ROOT / "registries" / "ppocrv6_cached_assets.json").read_text(
            encoding="utf-8"
        )
    )
    models = manifest.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise RuntimeError("PP-OCRv6 model manifest is invalid")
    cache_root = Path.home() / ".paddlex" / "official_models"
    required_files = manifest.get("required_files")
    for model in models:
        name = model["name"]
        prefix = f"{name}/"
        entries = [
            entry
            for entry in required_files
            if entry.get("path", "").startswith(prefix)
        ]
        if not entries:
            raise RuntimeError(f"PP-OCRv6 manifest has no files for {name}")
        for entry in entries:
            relative_name = entry["path"][len(prefix) :]
            cached_path = Path(
                hf_hub_download(
                    repo_id=model["repository"],
                    filename=relative_name,
                    revision=model["revision"],
                )
            )
            target_path = cache_root / entry["path"]
            if not _matches_entry(target_path, entry):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cached_path, target_path)
            if not _matches_entry(target_path, entry):
                raise RuntimeError(f"downloaded PP-OCRv6 asset failed verification: {entry['path']}")
    assets = verify_asset_inventory(
        cache_root,
        required_files,
        scope_roots=tuple(model["name"] for model in models),
    )
    print(json.dumps({"status": "downloaded", "assets": assets}, sort_keys=True))


def _matches_entry(path: Path, entry: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size_bytes"]
        and _sha256(path) == entry["sha256"]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
