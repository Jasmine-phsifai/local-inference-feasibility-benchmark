import hashlib
import json
from pathlib import Path

import pytest

from workers import qwen3_asr_openvino_genai_export_manifest as manifest


def _structure() -> dict:
    return {
        "decoder": {
            "interface": {
                "inputs": ["beam_idx", "encoder_hidden_states", "input_ids"],
                "outputs": ["logits"],
            },
            "read_value_count": 57,
            "assign_count": 57,
        },
        "encoder": {
            "interface": {
                "inputs": ["input_features"],
                "outputs": ["last_hidden_state"],
            },
            "read_value_count": 0,
            "assign_count": 0,
        },
    }


def _provenance(files: list[dict], structure: dict | None = None) -> dict:
    source_files = [
        {"name": name, "size_bytes": size, "sha256": sha256}
        for name, (size, sha256) in sorted(manifest.EXPECTED_SOURCE_FILES.items())
    ]
    return {
        "schema": "qwen3-asr-official-with-past-export-provenance-v1",
        "preflight": {
            "source": {
                "revision": manifest.SOURCE_REVISION,
                "files": source_files,
            },
            "exporter": {"revision": manifest.EXPORTER_REVISION},
            "contract": {
                "task": manifest.EXPORT_TASK,
                "weight_format": "fp16",
                "stateful_export": True,
                "offline_source_only": True,
            },
        },
        "export": {
            "model_type": "qwen3_asr",
            "beam_idx_present": True,
            "stateful_decoder": True,
            "decoder": (structure or _structure())["decoder"],
            "encoder": (structure or _structure())["encoder"],
            "files": files,
        },
    }


def test_export_provenance_binds_file_inventory(tmp_path: Path) -> None:
    files = [{"name": "model.bin", "size_bytes": 3, "sha256": "a" * 64}]
    path = tmp_path / manifest.PROVENANCE_FILENAME
    path.write_text(json.dumps(_provenance(files)), encoding="utf-8")

    result = manifest._verify_export_provenance(
        tmp_path,
        files,
        structure=_structure(),
    )

    assert result["filename"] == manifest.PROVENANCE_FILENAME
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    changed = _provenance(files)
    changed["export"]["stateful_decoder"] = False
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance changed"):
        manifest._verify_export_provenance(
            tmp_path,
            files,
            structure=_structure(),
        )


def test_export_provenance_rejects_relabelled_files(tmp_path: Path) -> None:
    files = [{"name": "model.bin", "size_bytes": 3, "sha256": "a" * 64}]
    path = tmp_path / manifest.PROVENANCE_FILENAME
    path.write_text(json.dumps(_provenance(files)), encoding="utf-8")

    with pytest.raises(RuntimeError, match="provenance changed"):
        manifest._verify_export_provenance(
            tmp_path,
            [{"name": "other.bin", "size_bytes": 3, "sha256": "a" * 64}],
            structure=_structure(),
        )


def test_export_provenance_binds_ir_structure(tmp_path: Path) -> None:
    files = [{"name": "model.bin", "size_bytes": 3, "sha256": "a" * 64}]
    path = tmp_path / manifest.PROVENANCE_FILENAME
    document = _provenance(files)
    path.write_text(json.dumps(document), encoding="utf-8")

    changed_structure = _structure()
    changed_structure["decoder"]["assign_count"] = 56
    with pytest.raises(RuntimeError, match="provenance changed"):
        manifest._verify_export_provenance(
            tmp_path,
            files,
            structure=changed_structure,
        )


def test_clean_export_writes_its_own_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    (export / "model.bin").write_bytes(b"abc")
    structure = _structure()
    monkeypatch.setattr(manifest, "verify_source", lambda _: {})
    monkeypatch.setattr(manifest, "_inspect_export", lambda _: structure)

    path = manifest.write_export_provenance(source, export)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["export"]["beam_idx_present"] is True
    assert document["export"]["stateful_decoder"] is True
    assert document["export"]["files"][0]["name"] == "model.bin"
    manifest._verify_export_provenance(
        export,
        document["export"]["files"],
        structure=structure,
    )
    with pytest.raises(FileExistsError, match="already exists"):
        manifest.write_export_provenance(source, export)
