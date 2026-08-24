"""Prepare the exact runtime and model assets for bounded b10598 VLM gates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.acquire_bounded_vlm_assets import (
    acquire_verified_file,
    assert_file_record,
    extract_verified_zip_tree,
    file_record_is_complete,
    hugging_face_resolve_url,
)
from local_inference_bench.bounded_vlm_assets import (
    REGISTRY_PROTOCOL,
    load_and_verify_candidate_assets,
)


REGISTRY_PATH = PROJECT_ROOT / "registries" / "bounded_vlm_b10598_assets.json"
OVIS = "ovisocr2_q8_cpu"
HUNYUAN = "hunyuanocr_1_5_gguf_cpu"
CANDIDATE_CHOICES = (OVIS, HUNYUAN, "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATE_CHOICES, required=True)
    parser.add_argument(
        "--conversion-python",
        type=Path,
        help="Exact bounded-VLM conversion Python; required if Hunyuan GGUFs are absent.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify already-prepared runtime, model, and fixture assets without network use.",
    )
    return parser.parse_args()


def load_registry() -> dict:
    document = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != 1
        or document.get("protocol") != REGISTRY_PROTOCOL
    ):
        raise RuntimeError("bounded VLM asset registry identity is invalid")
    return document


def prepare_runtime(registry: dict) -> None:
    runtime = registry["runtime"]
    archive = acquire_verified_file(
        project_root=PROJECT_ROOT,
        url=runtime["url"],
        record=runtime["archive"],
    )
    extract_verified_zip_tree(
        project_root=PROJECT_ROOT,
        archive=archive,
        tree_record=runtime["extracted_tree"],
    )


def prepare_ovis(candidate: dict) -> None:
    _acquire_hugging_face_records(
        identity=candidate["artifact_repository"],
        records=candidate["artifacts"],
    )


def prepare_hunyuan(candidate: dict, conversion_python: Path | None) -> None:
    _acquire_hugging_face_records(
        identity=candidate["upstream"],
        records=candidate["lineage_files"],
    )
    missing_artifacts = [
        (name, record)
        for name, record in candidate["artifacts"].items()
        if not file_record_is_complete(PROJECT_ROOT, record)
    ]
    if not missing_artifacts:
        return
    if conversion_python is None or not conversion_python.is_file():
        raise RuntimeError(
            "Hunyuan GGUF conversion requires --conversion-python from the exact "
            "bounded VLM conversion environment"
        )

    conversion = candidate["conversion"]
    source_archive_record = conversion["source_archive"]
    source_archive = acquire_verified_file(
        project_root=PROJECT_ROOT,
        url=source_archive_record["url"],
        record=source_archive_record,
    )
    source_tree_record = conversion["extracted_tree"]
    source_tree = extract_verified_zip_tree(
        project_root=PROJECT_ROOT,
        archive=source_archive,
        tree_record=source_tree_record,
        archive_root=source_tree_record["archive_root"],
    )
    converter = assert_file_record(
        PROJECT_ROOT,
        conversion["converter"],
        "pinned Hunyuan converter",
    )
    _verify_conversion_environment(conversion_python)

    model_root = assert_file_record(
        PROJECT_ROOT,
        candidate["lineage_files"]["model_safetensors"],
        "Hunyuan source checkpoint",
    ).parent
    arguments = {
        "model": conversion["base_arguments"],
        "projector": conversion["projector_arguments"],
    }
    for artifact_name, artifact_record in missing_artifacts:
        _convert_hunyuan_artifact(
            conversion_python=conversion_python,
            converter=converter,
            source_tree=source_tree,
            model_root=model_root,
            artifact_name=artifact_name,
            artifact_record=artifact_record,
            arguments=arguments[artifact_name],
        )


def verify_candidate(candidate_id: str) -> None:
    load_and_verify_candidate_assets(
        project_root=PROJECT_ROOT,
        candidate_id=candidate_id,
        registry_path=REGISTRY_PATH,
    )


def _acquire_hugging_face_records(*, identity: dict, records: dict) -> None:
    for record in records.values():
        filename = Path(record["path"]).name
        acquire_verified_file(
            project_root=PROJECT_ROOT,
            url=hugging_face_resolve_url(identity, filename),
            record=record,
        )


def _verify_conversion_environment(conversion_python: Path) -> None:
    subprocess.run(
        [
            str(conversion_python),
            str(PROJECT_ROOT / "scripts" / "verify_bounded_vlm_conversion_environment.py"),
        ],
        cwd=PROJECT_ROOT,
        env=_offline_conversion_environment(),
        timeout=120.0,
        check=True,
    )


def _convert_hunyuan_artifact(
    *,
    conversion_python: Path,
    converter: Path,
    source_tree: Path,
    model_root: Path,
    artifact_name: str,
    artifact_record: dict,
    arguments: list[str],
) -> None:
    final_path = (PROJECT_ROOT / artifact_record["path"]).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f"{final_path.stem}.building-", dir=final_path.parent)
    ).resolve()
    try:
        temporary_output = temporary_root / final_path.name
        subprocess.run(
            [
                str(conversion_python),
                str(converter),
                str(model_root),
                *arguments,
                "--outfile",
                str(temporary_output),
            ],
            cwd=source_tree,
            env=_offline_conversion_environment(),
            timeout=1800.0,
            check=True,
        )
        temporary_record = {
            **artifact_record,
            "path": temporary_output.relative_to(PROJECT_ROOT).as_posix(),
        }
        assert_file_record(
            PROJECT_ROOT,
            temporary_record,
            f"converted Hunyuan {artifact_name}",
        )
        os.replace(temporary_output, final_path)
    finally:
        if temporary_root.is_dir():
            shutil.rmtree(temporary_root)


def _offline_conversion_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return environment


def main() -> None:
    args = parse_args()
    registry = load_registry()
    selected = (
        (OVIS, HUNYUAN) if args.candidate == "all" else (args.candidate,)
    )
    if not args.verify_only:
        prepare_runtime(registry)
        for candidate_id in selected:
            candidate = registry["candidates"][candidate_id]
            if candidate_id == OVIS:
                prepare_ovis(candidate)
            else:
                prepare_hunyuan(candidate, args.conversion_python)
    for candidate_id in selected:
        verify_candidate(candidate_id)
    print(
        json.dumps(
            {"status": "verified", "candidates": list(selected)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
