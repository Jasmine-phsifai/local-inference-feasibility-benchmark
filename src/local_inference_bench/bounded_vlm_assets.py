"""Verify the pinned assets used by the bounded llama.cpp VLM quality gates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


REGISTRY_PROTOCOL = "bounded-vlm-b10598-assets-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_HUNYUAN_CONVERSION_REVISION = "70adb1b4cea5ee39f867792c78dc59320921eda7"
_HUNYUAN_CONVERSION_URL = (
    "https://github.com/ggml-org/llama.cpp/blob/"
    f"{_HUNYUAN_CONVERSION_REVISION}/convert_hf_to_gguf.py"
)
_HUNYUAN_CONVERSION_ARCHIVE_URL = (
    "https://github.com/ggml-org/llama.cpp/archive/"
    f"{_HUNYUAN_CONVERSION_REVISION}.zip"
)
_RUNTIME_REVISION = "b10598"
_RUNTIME_SOURCE_REVISION = "56db501e73cfb10c8fcce61be708f5c3ee749271"
_RUNTIME_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    "b10598/llama-b10598-bin-win-cpu-x64.zip"
)
_CANDIDATE_CONTRACTS = {
    "ovisocr2_q8_cpu": {
        "upstream_url": (
            "https://huggingface.co/ATH-MaaS/OvisOCR2/tree/"
            "65c619d374b55d4152e85150fc1b003700bc1f0c"
        ),
        "upstream_revision": "65c619d374b55d4152e85150fc1b003700bc1f0c",
        "artifact_repository_url": (
            "https://huggingface.co/Abiray/OvisOCR2-GGUF/tree/"
            "8a22290a0c42dbcf84739d6d4c2763f877494ae0"
        ),
        "artifact_repository_revision": "8a22290a0c42dbcf84739d6d4c2763f877494ae0",
        "compute_type": "q8_0_text_bf16_projector",
        "sample_ids": ["page_008_table_columns"],
        "artifact_names": {"model", "projector"},
        "lineage_names": set(),
        "fixture_image_names": {"page_008_table_columns"},
        "generator_names": {"document_fidelity"},
        "failed_outcome_kind": "experimental_quality_gate_failed",
        "quality_gate": {
            "maximum_failure_count": "nonnegative_integer",
            "maximum_token_cap_hit_count": "nonnegative_integer",
            "minimum_reading_order_pair_accuracy": "unit_interval",
            "minimum_protected_span_recall": "unit_interval",
            "minimum_structure_visible_lexical_recall": "unit_interval",
            "minimum_structure_visible_lexical_precision": "unit_interval",
            "minimum_semantic_table_cell_exact_fraction": "unit_interval",
        },
        "requires_artifact_repository": True,
        "requires_conversion": False,
    },
    "hunyuanocr_1_5_gguf_cpu": {
        "upstream_url": (
            "https://huggingface.co/tencent/HunyuanOCR/tree/"
            "449e7d471a8a1ef5bd5d652e4881183d7252cbc7"
        ),
        "upstream_revision": "449e7d471a8a1ef5bd5d652e4881183d7252cbc7",
        "artifact_repository_url": None,
        "artifact_repository_revision": None,
        "compute_type": "f16",
        "sample_ids": ["code_formula", "dense_table", "negative_diagram"],
        "artifact_names": {"model", "projector"},
        "lineage_names": {
            "model_safetensors",
            "config",
            "generation_config",
            "preprocessor_config",
            "tokenizer",
            "tokenizer_config",
            "chat_template",
        },
        "fixture_image_names": {
            "warmup",
            "code_formula",
            "dense_table",
            "negative_diagram",
        },
        "generator_names": {"ocr_quality", "quality_subset"},
        "failed_outcome_kind": "quality_blocker",
        "quality_gate": {
            "maximum_failure_count": "nonnegative_integer",
            "maximum_token_cap_hit_count": "nonnegative_integer",
            "maximum_structure_visible_normalized_character_error_rate": "unit_interval",
            "minimum_structure_visible_required_token_recall": "unit_interval",
            "maximum_structure_visible_negative_false_positive_characters": "nonnegative_integer",
            "maximum_raw_negative_false_positive_characters": "nonnegative_integer",
        },
        "requires_artifact_repository": False,
        "requires_conversion": True,
    },
}


def sha256_file(path: Path) -> str:
    """Return the full lowercase SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_directory(path: Path) -> dict:
    """Hash every regular file in a directory using a stable tree encoding."""

    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for item in files:
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        item_sha256 = sha256_file(item)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(item_sha256.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def load_and_verify_candidate_assets(
    *,
    project_root: Path,
    candidate_id: str,
    registry_path: Path | None = None,
) -> dict:
    """Fail closed unless the complete pinned runtime, model, and fixtures match."""

    root = project_root.resolve()
    path = (
        registry_path.resolve()
        if registry_path is not None
        else root / "registries" / "bounded_vlm_b10598_assets.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 1
        or document.get("protocol") != REGISTRY_PROTOCOL
    ):
        raise ValueError("bounded VLM asset registry identity is invalid")
    if _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("bounded VLM candidate ID is invalid")
    candidates = document.get("candidates")
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise ValueError("bounded VLM candidate is not declared")

    runtime = document.get("runtime")
    candidate = candidates[candidate_id]
    if not isinstance(runtime, dict) or not isinstance(candidate, dict):
        raise ValueError("bounded VLM asset registry is incomplete")
    _validate_runtime_contract(runtime)
    _validate_candidate_contract(candidate_id, candidate)
    _verify_https_identity(runtime, require_revision=True)
    archive = _verify_file_record(root, runtime.get("archive"), "runtime archive")
    entrypoints = _verify_named_file_records(
        root,
        runtime.get("entrypoints"),
        "runtime entrypoint",
    )
    tree_record = runtime.get("extracted_tree")
    if not isinstance(tree_record, dict):
        raise ValueError("bounded VLM runtime tree record is invalid")
    tree_path = _resolve_repo_path(root, tree_record.get("path"), kind="directory")
    actual_tree = fingerprint_directory(tree_path)
    _verify_fingerprint_fields(tree_record, actual_tree, "runtime tree")

    _verify_https_identity(candidate.get("upstream"), require_revision=True)
    artifact_repository = candidate.get("artifact_repository")
    if artifact_repository is not None:
        _verify_https_identity(artifact_repository, require_revision=True)
    conversion = candidate.get("conversion")
    if conversion is not None:
        _verify_https_identity(conversion, require_revision=True)
    artifacts = _verify_named_file_records(
        root,
        candidate.get("artifacts"),
        "candidate artifact",
    )
    lineage = _verify_named_file_records(
        root,
        candidate.get("lineage_files", {}),
        "candidate lineage file",
    )
    fixtures = candidate.get("fixtures")
    if not isinstance(fixtures, dict):
        raise ValueError("bounded VLM fixture contract is invalid")
    if fixtures.get("source_kind") != "tracked_deterministic_generated_public":
        raise ValueError("bounded VLM fixtures must be generated public controls")
    if fixtures.get("contains_private_course_data") is not False:
        raise ValueError("bounded VLM fixtures must explicitly exclude private data")
    fixture_manifest = _verify_file_record(
        root,
        fixtures.get("manifest"),
        "fixture manifest",
    )
    fixture_images = _verify_named_file_records(
        root,
        fixtures.get("images"),
        "fixture image",
    )
    generator_files = _verify_named_file_records(
        root,
        fixtures.get("generator_files"),
        "fixture generator",
    )

    quality_gate = candidate.get("quality_gate")
    if not isinstance(quality_gate, dict) or not quality_gate:
        raise ValueError("bounded VLM quality gate is missing")
    return {
        "registry": document,
        "registry_path": path,
        "runtime": {
            "archive": archive,
            "entrypoints": entrypoints,
            "tree_path": tree_path,
            "tree_fingerprint": actual_tree,
        },
        "candidate": candidate,
        "artifacts": artifacts,
        "lineage_files": lineage,
        "fixtures": {
            "manifest": fixture_manifest,
            "images": fixture_images,
            "generator_files": generator_files,
        },
    }


def _validate_candidate_contract(candidate_id: str, candidate: dict) -> None:
    contract = _CANDIDATE_CONTRACTS.get(candidate_id)
    if contract is None:
        raise ValueError("bounded VLM candidate has no exact schema")
    if candidate.get("compute_type") != contract["compute_type"]:
        raise ValueError("bounded VLM compute type is invalid")
    upstream = candidate.get("upstream")
    if not isinstance(upstream, dict) or (
        upstream.get("url") != contract["upstream_url"]
        or upstream.get("revision") != contract["upstream_revision"]
    ):
        raise ValueError("bounded VLM upstream identity is invalid")
    if candidate.get("sample_ids") != contract["sample_ids"]:
        raise ValueError("bounded VLM sample identity contract is invalid")
    if candidate.get("failed_outcome_kind") != contract["failed_outcome_kind"]:
        raise ValueError("bounded VLM failed outcome kind is invalid")

    _require_exact_names(
        candidate.get("artifacts"),
        contract["artifact_names"],
        "candidate artifact",
    )
    _require_exact_names(
        candidate.get("lineage_files"),
        contract["lineage_names"],
        "candidate lineage",
    )
    fixtures = candidate.get("fixtures")
    if not isinstance(fixtures, dict):
        raise ValueError("bounded VLM fixture contract is invalid")
    _require_exact_names(
        fixtures.get("images"),
        contract["fixture_image_names"],
        "fixture image",
    )
    _require_exact_names(
        fixtures.get("generator_files"),
        contract["generator_names"],
        "fixture generator",
    )

    artifact_repository = candidate.get("artifact_repository")
    if contract["requires_artifact_repository"] != (
        artifact_repository is not None
    ):
        raise ValueError("bounded VLM artifact repository contract is invalid")
    if artifact_repository is not None:
        if not isinstance(artifact_repository, dict) or (
            artifact_repository.get("url") != contract["artifact_repository_url"]
            or artifact_repository.get("revision")
            != contract["artifact_repository_revision"]
        ):
            raise ValueError("bounded VLM artifact repository identity is invalid")
    conversion = candidate.get("conversion")
    if contract["requires_conversion"] != (conversion is not None):
        raise ValueError("bounded VLM conversion contract is invalid")
    if conversion is not None:
        _validate_hunyuan_conversion_contract(conversion)
    _validate_quality_gate(candidate.get("quality_gate"), contract["quality_gate"])


def _validate_runtime_contract(runtime: dict) -> None:
    if (
        runtime.get("url") != _RUNTIME_URL
        or runtime.get("revision") != _RUNTIME_REVISION
        or runtime.get("source_revision") != _RUNTIME_SOURCE_REVISION
    ):
        raise ValueError("bounded VLM runtime identity is invalid")
    _require_exact_names(
        runtime.get("entrypoints"),
        {"llama_mtmd_cli", "llama_server"},
        "runtime entrypoint",
    )


def _validate_hunyuan_conversion_contract(conversion: object) -> None:
    if not isinstance(conversion, dict):
        raise ValueError("bounded VLM conversion contract is invalid")
    _require_exact_names(
        conversion,
        {
            "url",
            "revision",
            "source_archive",
            "extracted_tree",
            "converter",
            "base_arguments",
            "projector_arguments",
        },
        "conversion field",
    )
    _verify_https_identity(conversion, require_revision=True)
    if (
        conversion.get("revision") != _HUNYUAN_CONVERSION_REVISION
        or conversion.get("url") != _HUNYUAN_CONVERSION_URL
    ):
        raise ValueError("bounded VLM conversion revision is invalid")
    source_archive = conversion.get("source_archive")
    if not isinstance(source_archive, dict):
        raise ValueError("bounded VLM conversion source archive is invalid")
    _verify_https_identity(
        {
            "url": source_archive.get("url"),
            "revision": _HUNYUAN_CONVERSION_REVISION,
        },
        require_revision=True,
    )
    if source_archive.get("url") != _HUNYUAN_CONVERSION_ARCHIVE_URL:
        raise ValueError("bounded VLM conversion source archive URL is invalid")
    _validate_file_record(source_archive, "conversion source archive")
    _validate_file_record(conversion.get("converter"), "conversion program")
    tree = conversion.get("extracted_tree")
    if not isinstance(tree, dict):
        raise ValueError("bounded VLM conversion tree is invalid")
    _verify_fingerprint_schema(tree, "conversion tree")
    if tree.get("archive_root") != f"llama.cpp-{_HUNYUAN_CONVERSION_REVISION}":
        raise ValueError("bounded VLM conversion archive root is invalid")
    if conversion.get("base_arguments") != ["--outtype", "f16"]:
        raise ValueError("bounded VLM base conversion arguments are invalid")
    if conversion.get("projector_arguments") != [
        "--outtype",
        "f16",
        "--mmproj",
    ]:
        raise ValueError("bounded VLM projector conversion arguments are invalid")


def _validate_quality_gate(value: object, schema: dict[str, str]) -> None:
    _require_exact_names(value, set(schema), "quality gate")
    assert isinstance(value, dict)
    for name, kind in schema.items():
        item = value[name]
        if kind == "nonnegative_integer":
            if type(item) is not int or item < 0:
                raise ValueError(f"bounded VLM quality gate {name} is invalid")
        elif (
            kind == "unit_interval"
            and (type(item) not in (int, float) or not 0.0 <= item <= 1.0)
        ):
            raise ValueError(f"bounded VLM quality gate {name} is invalid")


def _require_exact_names(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"bounded VLM {label} names are invalid")


def _validate_file_record(value: object, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"bounded VLM {label} record is invalid")
    expected_bytes = value.get("bytes")
    expected_sha256 = value.get("sha256")
    path = value.get("path")
    if (
        type(path) is not str
        or not path
        or Path(path).is_absolute()
        or type(expected_bytes) is not int
        or expected_bytes <= 0
        or type(expected_sha256) is not str
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise ValueError(f"bounded VLM {label} identity is invalid")


def _verify_fingerprint_schema(value: dict, label: str) -> None:
    for key in ("file_count", "total_bytes"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError(f"bounded VLM {label} {key} is invalid")
    if (
        type(value.get("sha256")) is not str
        or _SHA256.fullmatch(value["sha256"]) is None
        or type(value.get("path")) is not str
        or not value["path"]
        or Path(value["path"]).is_absolute()
    ):
        raise ValueError(f"bounded VLM {label} identity is invalid")


def _verify_https_identity(value: object, *, require_revision: bool) -> None:
    if not isinstance(value, dict):
        raise ValueError("bounded VLM source identity is invalid")
    url = value.get("url")
    revision = value.get("revision")
    if type(url) is not str or not url.startswith("https://"):
        raise ValueError("bounded VLM source URL must use HTTPS")
    if require_revision and (
        type(revision) is not str
        or not revision
        or revision not in url
    ):
        raise ValueError("bounded VLM source URL must embed its revision")


def _verify_named_file_records(
    root: Path,
    value: object,
    label: str,
) -> dict[str, dict]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} mapping is invalid")
    verified = {}
    for name, record in value.items():
        if type(name) is not str or not name:
            raise ValueError(f"{label} name is invalid")
        verified[name] = _verify_file_record(root, record, label)
    return verified


def _verify_file_record(root: Path, value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} record is invalid")
    expected_bytes = value.get("bytes")
    expected_sha256 = value.get("sha256")
    if (
        type(expected_bytes) is not int
        or expected_bytes <= 0
        or type(expected_sha256) is not str
        or _SHA256.fullmatch(expected_sha256) is None
    ):
        raise ValueError(f"{label} identity is invalid")
    path = _resolve_repo_path(root, value.get("path"), kind="file")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise RuntimeError(f"{label} identity mismatch: {path.name}")
    return {
        "path": path,
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def _resolve_repo_path(root: Path, value: object, *, kind: str) -> Path:
    if type(value) is not str or not value or Path(value).is_absolute():
        raise ValueError("bounded VLM asset paths must be repository-relative")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bounded VLM asset path escapes the repository")
    if (kind == "file" and not path.is_file()) or (
        kind == "directory" and not path.is_dir()
    ):
        raise FileNotFoundError(f"bounded VLM {kind} is missing: {value}")
    return path


def _verify_fingerprint_fields(
    expected: dict,
    actual: dict,
    label: str,
) -> None:
    for key in ("file_count", "total_bytes"):
        if type(expected.get(key)) is not int or expected[key] <= 0:
            raise ValueError(f"{label} {key} is invalid")
    if (
        type(expected.get("sha256")) is not str
        or _SHA256.fullmatch(expected["sha256"]) is None
    ):
        raise ValueError(f"{label} sha256 is invalid")
    if any(expected[key] != actual[key] for key in actual):
        raise RuntimeError(f"{label} identity mismatch")
