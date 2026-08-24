import hashlib
import sys
from pathlib import Path

import pytest

import local_inference_bench.verify_locked_environment as locked_environment
from local_inference_bench.verify_asset_inventory import verify_asset_inventory


class _Distribution:
    def __init__(self, name: str, version: str):
        self.metadata = {"Name": name}
        self.version = version


def test_exact_environment_lock_rejects_unexpected_distribution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text("Package_A==1.2.3\npackage-b==4.5.6\n", encoding="utf-8")
    installed = [
        _Distribution("package-a", "1.2.3"),
        _Distribution("Package_B", "4.5.6"),
    ]
    monkeypatch.setattr(
        locked_environment,
        "distributions",
        lambda **_kwargs: iter(installed),
    )

    result = locked_environment.verify_locked_environment(
        lock,
        expected_python=tuple(sys.version_info[:3]),
    )
    assert result["package_count"] == 2

    installed.append(_Distribution("unexpected", "1"))
    with pytest.raises(RuntimeError, match="unexpected"):
        locked_environment.verify_locked_environment(
            lock,
            expected_python=tuple(sys.version_info[:3]),
        )


def test_asset_inventory_rejects_mutated_and_unexpected_files(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    model = root / "model.bin"
    model.write_bytes(b"pinned model bytes")
    required = [
        {
            "path": "model.bin",
            "size_bytes": model.stat().st_size,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
    ]

    assert verify_asset_inventory(root, required)["file_count"] == 1
    model.write_bytes(b"mutated model bytes")
    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_asset_inventory(root, required)

    model.write_bytes(b"pinned model bytes")
    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory mismatch"):
        verify_asset_inventory(root, required)


def test_sensevoice_official_runtime_preserves_measured_vcomp_identity() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = (
        project_root
        / "environments/sensevoice_official_runtime/environment.yml"
    ).read_text(encoding="utf-8")
    build_environment = (
        project_root / "environments/sensevoice_build/environment.yml"
    ).read_text(encoding="utf-8")
    creator = (
        project_root / "scripts/create_sensevoice_official_runtime_environment.ps1"
    ).read_text(encoding="utf-8")
    downloader = (
        project_root / "scripts/download_sensevoice_assets.ps1"
    ).read_text(encoding="utf-8")
    verifier = (
        project_root / "scripts/verify_sensevoice_thread_runtime.py"
    ).read_text(encoding="utf-8")

    assert "vs2015_runtime=14.42.34433=hbfb602d_5" in environment
    assert "vcomp14=" not in build_environment
    for source in (creator, downloader, verifier):
        expected_sha256 = (
            "E36A5C5E329BC7AF35D4FAA610A29AEEE826A7810E06712F0F54E9B2CFE6A728"
        )
        assert expected_sha256.casefold() in source.casefold()
        assert "192112" in source.replace("_", "")
    assert "sensevoice-official-runtime\\Library\\bin\\vcomp140.dll" in downloader
