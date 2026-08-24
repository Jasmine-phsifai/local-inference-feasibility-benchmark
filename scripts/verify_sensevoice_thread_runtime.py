"""Verify pinned official and thread-controlled SenseVoice runtime assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_ROOT = PROJECT_ROOT / "data/models/sensevoice"
SOURCE_VARIANTS = {
    "v0.1.9": {
        "bin_root": PROJECT_ROOT
        / "data/models/sensevoice-runtime-v0.1.9-thread-build/build-pinned/bin",
        "provenance": PROJECT_ROOT
        / "results/artifacts/sensevoice-thread-build/provenance.json",
        "patch": PROJECT_ROOT
        / "patches/sensevoice-runtime-v0.1.9-thread-option.patch",
        "source_revision": "73ccdd3577db37e92dbf22a4a9fc323b038cf13b",
        "tag_object": "8f07e2f3624a1340bbfde5f5ddd5022ea37862d2",
        "llama_revision": "8086439a4cea94c71a5dfb8fe4ad1546aebd640f",
        "patch_sha256": "151d1e42490b1a0bff7103308b3a7f424699ac38ce2a6918891f1d6df25d52f8",
    },
    "v0.2.0": {
        "bin_root": PROJECT_ROOT
        / "data/models/sensevoice-runtime-v0.2.0-thread-build/build-pinned/bin",
        "provenance": PROJECT_ROOT
        / "results/artifacts/sensevoice-v020-thread-build/provenance.json",
        "patch": PROJECT_ROOT
        / "patches/sensevoice-runtime-v0.2.0-thread-option.patch",
        "source_revision": "500956bc331bb7edbaac58d8f84a84f28bd3d29f",
        "tag_object": "ef27da6d5332d801ede62dcba9811151d1b936ce",
        "llama_revision": "803b7fcae893e9caaee3921779628fef83ac0965",
        "patch_sha256": "3a6831762c5379cd37b5cf64435396e6ab78fa91f17fa306ad64f635a0cdd194",
    },
}
SHARED_MODEL_FILES = {
    "sensevoice-small-q8.gguf": (
        254_208_320,
        "4ae45c94422de949b387e2e0fb10d7e14e4c42c69db30c3444ecc7d4b844b7c5",
    ),
    "fsmn-vad.gguf": (
        1_720_512,
        "1270f2559c495f4e7b6e739541151027d360761a3fda43fc147034f5719f5479",
    ),
}
OFFICIAL_RUNTIME_FILES = {
    "llama-funasr-sensevoice.exe": (
        1_651_712,
        "a1b40105cd0e0956ec0f34e8787cf2fa4eb39383aaa72b82e5f97a7118f24f3c",
    ),
    "vcomp140.dll": (
        192_112,
        "e36a5c5e329bc7af35d4faa610a29aeee826a7810e06712f0f54e9b2cfe6a728",
    ),
}
SOURCE_BUNDLE_NAMES = {
    "llama-funasr-sensevoice.exe",
    "libc++.dll",
    "libomp.dll",
    "libunwind.dll",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument(
        "--source-version",
        choices=sorted(SOURCE_VARIANTS),
        default="v0.1.9",
    )
    args = parser.parse_args()

    if not args.source_only:
        _verify_official_assets()
    versions = [args.source_version] if args.source_only else list(SOURCE_VARIANTS)
    verified_versions = []
    for version_name in versions:
        settings = SOURCE_VARIANTS[version_name]
        source_paths = [settings["provenance"]] + [
            settings["bin_root"] / name for name in SOURCE_BUNDLE_NAMES
        ]
        present = [path.is_file() for path in source_paths]
        if args.source_only or any(present):
            if not all(present):
                raise RuntimeError("thread-controlled SenseVoice bundle is incomplete")
            _verify_source_bundle(settings)
            verified_versions.append(version_name)
    print(
        json.dumps(
            {
                "official_assets_verified": not args.source_only,
                "source_bundles_verified": verified_versions,
            },
            sort_keys=True,
        )
    )


def _verify_official_assets() -> None:
    for name, (size, sha256) in {
        **SHARED_MODEL_FILES,
        **OFFICIAL_RUNTIME_FILES,
    }.items():
        _verify_file(OFFICIAL_ROOT / name, size=size, sha256=sha256)


def _verify_source_bundle(settings: dict) -> None:
    if _sha256(settings["patch"]) != settings["patch_sha256"]:
        raise RuntimeError("tracked SenseVoice thread patch changed")
    provenance = json.loads(settings["provenance"].read_text(encoding="utf-8"))
    source = provenance.get("source", {})
    dependency = provenance.get("pinned_dependency", {})
    patch = provenance.get("patch", {})
    toolchain = provenance.get("toolchain", {})
    configuration = provenance.get("configuration", {})
    bundle = provenance.get("bundle")
    if (
        provenance.get("schema_version") != 1
        or source.get("commit") != settings["source_revision"]
        or source.get("annotated_tag_object") != settings["tag_object"]
        or dependency.get("commit") != settings["llama_revision"]
        or patch.get("sha256") != settings["patch_sha256"]
        or toolchain.get("release") != "20260616"
        or toolchain.get("archive_sha256")
        != "b9b68a4d276e16fa25802aaba458e4638f64b3884c290aaccdc2d87083b6ca35"
        or toolchain.get("compiler") != "clang 22.1.8"
        or toolchain.get("cmake") != "4.2.3"
        or toolchain.get("ninja") != "1.13.1"
        or configuration.get("build_type") != "Release"
        or configuration.get("ggml_native") is not False
        or configuration.get("ggml_avx") is not True
        or configuration.get("ggml_avx2") is not True
        or configuration.get("ggml_f16c") is not True
        or configuration.get("ggml_fma") is not True
        or configuration.get("target_only") != "llama-funasr-sensevoice"
        or not isinstance(bundle, list)
    ):
        raise RuntimeError("thread-controlled SenseVoice provenance changed")
    by_name = {
        item.get("name"): item
        for item in bundle
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if set(by_name) != SOURCE_BUNDLE_NAMES:
        raise RuntimeError("thread-controlled SenseVoice bundle inventory changed")
    for name in sorted(SOURCE_BUNDLE_NAMES):
        item = by_name[name]
        size = item.get("size_bytes")
        sha256 = item.get("sha256")
        if type(size) is not int or not isinstance(sha256, str):
            raise RuntimeError("thread-controlled SenseVoice bundle record is invalid")
        _verify_file(settings["bin_root"] / name, size=size, sha256=sha256)
    binary = settings["bin_root"] / "llama-funasr-sensevoice.exe"
    completed = subprocess.run(
        [str(binary), "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0 or "--threads N (default: 8)" not in output:
        raise RuntimeError("thread-controlled SenseVoice help probe failed")


def _verify_file(path: Path, *, size: int, sha256: str) -> None:
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != sha256:
        raise RuntimeError(f"pinned SenseVoice asset changed: {path.name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
