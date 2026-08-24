import hashlib
import json
from pathlib import Path

import pytest

from local_inference_bench.bounded_vlm_assets import (
    _validate_candidate_contract,
    _validate_runtime_contract,
    fingerprint_directory,
    load_and_verify_candidate_assets,
)


OVIS = "ovisocr2_q8_cpu"


def _file_record(root: Path, relative_path: str, content: bytes) -> dict:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": relative_path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _registry_fixture(tmp_path: Path) -> tuple[Path, Path]:
    archive = _file_record(tmp_path, "downloads/runtime.zip", b"archive")
    mtmd = _file_record(tmp_path, "runtime/mtmd.exe", b"mtmd")
    server = _file_record(tmp_path, "runtime/server.exe", b"server")
    model = _file_record(tmp_path, "models/model.gguf", b"model")
    projector = _file_record(tmp_path, "models/projector.gguf", b"projector")
    fixture_manifest = _file_record(tmp_path, "fixtures/manifest.json", b"{}")
    image = _file_record(tmp_path, "fixtures/control.png", b"png")
    generator = _file_record(tmp_path, "scripts/generate.py", b"pass\n")
    tree = fingerprint_directory(tmp_path / "runtime")
    registry = {
        "schema_version": 1,
        "protocol": "bounded-vlm-b10598-assets-v1",
        "runtime": {
            "url": (
                "https://github.com/ggml-org/llama.cpp/releases/download/"
                "b10598/llama-b10598-bin-win-cpu-x64.zip"
            ),
            "revision": "b10598",
            "source_revision": "56db501e73cfb10c8fcce61be708f5c3ee749271",
            "archive": archive,
            "extracted_tree": {"path": "runtime", **tree},
            "entrypoints": {"llama_mtmd_cli": mtmd, "llama_server": server},
        },
        "candidates": {
            OVIS: {
                "upstream": {
                    "url": (
                        "https://huggingface.co/ATH-MaaS/OvisOCR2/tree/"
                        "65c619d374b55d4152e85150fc1b003700bc1f0c"
                    ),
                    "revision": "65c619d374b55d4152e85150fc1b003700bc1f0c",
                },
                "artifact_repository": {
                    "url": (
                        "https://huggingface.co/Abiray/OvisOCR2-GGUF/tree/"
                        "8a22290a0c42dbcf84739d6d4c2763f877494ae0"
                    ),
                    "revision": "8a22290a0c42dbcf84739d6d4c2763f877494ae0",
                },
                "artifacts": {"model": model, "projector": projector},
                "lineage_files": {},
                "compute_type": "q8_0_text_bf16_projector",
                "sample_ids": ["page_008_table_columns"],
                "fixtures": {
                    "source_kind": "tracked_deterministic_generated_public",
                    "contains_private_course_data": False,
                    "manifest": fixture_manifest,
                    "images": {"page_008_table_columns": image},
                    "generator_files": {"document_fidelity": generator},
                },
                "quality_gate": {
                    "maximum_failure_count": 0,
                    "maximum_token_cap_hit_count": 0,
                    "minimum_reading_order_pair_accuracy": 1.0,
                    "minimum_protected_span_recall": 1.0,
                    "minimum_lexical_recall": 0.9,
                    "minimum_lexical_precision": 0.9,
                    "minimum_table_cell_exact_fraction": 0.95,
                },
                "failed_outcome_kind": "experimental_quality_gate_failed",
            }
        },
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path, tmp_path / model["path"]


def test_asset_registry_verifies_complete_runtime_model_and_public_fixture(tmp_path):
    registry_path, _ = _registry_fixture(tmp_path)

    verified = load_and_verify_candidate_assets(
        project_root=tmp_path,
        candidate_id=OVIS,
        registry_path=registry_path,
    )

    assert verified["runtime"]["tree_fingerprint"]["file_count"] == 2
    assert verified["artifacts"]["model"]["sha256"] == hashlib.sha256(
        b"model"
    ).hexdigest()
    assert verified["fixtures"]["images"]["page_008_table_columns"]["bytes"] == 3


def test_asset_registry_rejects_model_mutation(tmp_path):
    registry_path, model_path = _registry_fixture(tmp_path)
    model_path.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="candidate artifact identity mismatch"):
        load_and_verify_candidate_assets(
            project_root=tmp_path,
            candidate_id=OVIS,
            registry_path=registry_path,
        )


def test_directory_fingerprint_binds_relative_name_and_content(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "a.dll").write_bytes(b"same")
    first = fingerprint_directory(runtime)
    (runtime / "a.dll").rename(runtime / "b.dll")
    second = fingerprint_directory(runtime)

    assert first["total_bytes"] == second["total_bytes"]
    assert first["sha256"] != second["sha256"]


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    (
        ("compute_type", "q4", "compute type"),
        ("sample_ids", ["wrong"], "sample identity"),
        ("failed_outcome_kind", "implementation_failed", "outcome kind"),
    ),
)
def test_candidate_contract_rejects_malformed_published_identity(
    tmp_path,
    field,
    invalid,
    message,
):
    registry_path, _ = _registry_fixture(tmp_path)
    candidate = json.loads(registry_path.read_text(encoding="utf-8"))["candidates"][
        OVIS
    ]
    candidate[field] = invalid

    with pytest.raises(ValueError, match=message):
        _validate_candidate_contract(OVIS, candidate)


def test_candidate_contract_rejects_boolean_and_extra_quality_threshold(tmp_path):
    registry_path, _ = _registry_fixture(tmp_path)
    candidate = json.loads(registry_path.read_text(encoding="utf-8"))["candidates"][
        OVIS
    ]
    candidate["quality_gate"]["minimum_lexical_recall"] = True
    with pytest.raises(ValueError, match="minimum_lexical_recall"):
        _validate_candidate_contract(OVIS, candidate)

    candidate["quality_gate"]["minimum_lexical_recall"] = 0.9
    candidate["quality_gate"]["unpublished_threshold"] = 0
    with pytest.raises(ValueError, match="quality gate names"):
        _validate_candidate_contract(OVIS, candidate)


def test_hunyuan_contract_requires_complete_lineage_and_exact_conversion():
    registry = json.loads(
        Path("registries/bounded_vlm_b10598_assets.json").read_text(encoding="utf-8")
    )
    candidate = registry["candidates"]["hunyuanocr_1_5_gguf_cpu"]
    _validate_candidate_contract("hunyuanocr_1_5_gguf_cpu", candidate)

    generation = candidate["lineage_files"].pop("generation_config")
    with pytest.raises(ValueError, match="lineage"):
        _validate_candidate_contract("hunyuanocr_1_5_gguf_cpu", candidate)
    candidate["lineage_files"]["generation_config"] = generation

    candidate["conversion"]["projector_arguments"] = ["--outtype", "f16"]
    with pytest.raises(ValueError, match="projector conversion arguments"):
        _validate_candidate_contract("hunyuanocr_1_5_gguf_cpu", candidate)


def test_candidate_contract_rejects_mutable_or_substituted_source_identity(tmp_path):
    registry_path, _ = _registry_fixture(tmp_path)
    candidate = json.loads(registry_path.read_text(encoding="utf-8"))["candidates"][
        OVIS
    ]
    candidate["upstream"] = {
        "url": "https://huggingface.co/ATH-MaaS/OvisOCR2/tree/main",
        "revision": "main",
    }
    with pytest.raises(ValueError, match="upstream identity"):
        _validate_candidate_contract(OVIS, candidate)


def test_runtime_contract_pins_release_source_and_entrypoints(tmp_path):
    registry_path, _ = _registry_fixture(tmp_path)
    runtime = json.loads(registry_path.read_text(encoding="utf-8"))["runtime"]
    _validate_runtime_contract(runtime)

    runtime["source_revision"] = "0" * 40
    with pytest.raises(ValueError, match="runtime identity"):
        _validate_runtime_contract(runtime)
    runtime["source_revision"] = "56db501e73cfb10c8fcce61be708f5c3ee749271"

    runtime["entrypoints"]["unexpected"] = runtime["entrypoints"][
        "llama_server"
    ]
    with pytest.raises(ValueError, match="runtime entrypoint names"):
        _validate_runtime_contract(runtime)
