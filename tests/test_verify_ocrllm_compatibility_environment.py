from pathlib import Path

import pytest

from local_inference_bench.verify_locked_environment import _read_exact_lock
from scripts.ocrllm_compatibility_provenance import (
    EXPECTED_RAPIDOCR_MODEL_HASHES,
    EXPECTED_RUNTIME_VERSIONS,
    _verify_rapidocr_model_hashes,
)


def test_ocrllm_lock_is_full_and_contains_the_declared_runtime() -> None:
    lock = _read_exact_lock(
        Path("environments/ocrllm_compatibility/requirements.lock.txt")
    )

    assert len(lock) == 34
    assert "ocrllm" not in lock
    assert lock["pip"] == "26.1.2"
    assert lock["hatchling"] == "1.31.0"
    assert lock["pathspec"] == "1.1.1"
    assert lock["pluggy"] == "1.6.0"
    assert lock["trove-classifiers"] == "2026.6.1.19"
    for name, version in EXPECTED_RUNTIME_VERSIONS.items():
        assert lock[name] == version


def test_ocrllm_source_install_disables_mutable_build_isolation() -> None:
    creator = Path(
        "scripts/create_ocrllm_compatibility_environment.ps1"
    ).read_text(encoding="utf-8")

    assert "--no-build-isolation $snapshotRoot" in creator


def test_rapidocr_model_allowlist_is_exactly_three_onnx_files() -> None:
    assert len(EXPECTED_RAPIDOCR_MODEL_HASHES) == 3
    assert all(
        path.startswith("rapidocr/models/") and path.endswith(".onnx")
        for path in EXPECTED_RAPIDOCR_MODEL_HASHES
    )
    assert all(
        len(digest) == 64 for digest in EXPECTED_RAPIDOCR_MODEL_HASHES.values()
    )
    _verify_rapidocr_model_hashes(dict(EXPECTED_RAPIDOCR_MODEL_HASHES))


def test_rapidocr_model_allowlist_rejects_missing_extra_and_changed_models() -> None:
    expected = dict(EXPECTED_RAPIDOCR_MODEL_HASHES)
    missing = dict(expected)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="missing"):
        _verify_rapidocr_model_hashes(missing)

    extra = {**expected, "rapidocr/models/unpinned.onnx": "0" * 64}
    with pytest.raises(RuntimeError, match="unexpected"):
        _verify_rapidocr_model_hashes(extra)

    changed = dict(expected)
    changed[next(iter(changed))] = "0" * 64
    with pytest.raises(RuntimeError, match="mismatched"):
        _verify_rapidocr_model_hashes(changed)
