"""Prepare one explicitly selected generated quality-control suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_generated_quality_controls import (  # noqa: E402
    SUITE_MANIFESTS,
    verify_generated_quality_controls,
)


PREPARATION_SCRIPTS = {
    "ocr": (
        Path("scripts/generate_ocr_quality_controls.py"),
        Path("scripts/prepare_hard_ocr_quality_subsets.py"),
    ),
    "document-fidelity": (Path("scripts/generate_document_fidelity_controls.py"),),
    "asr": (Path("scripts/prepare_asr_quality_controls.py"),),
}
_STAGED_SUITES = {"ocr", "document-fidelity"}
_OUTPUT_DIRECTORIES = {
    suite: manifest.parent for suite, manifest in SUITE_MANIFESTS.items()
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare only the named public/generated suite, then verify all "
            "declared media. Repeat --suite to opt into more than one suite."
        )
    )
    parser.add_argument(
        "--suite",
        action="append",
        required=True,
        choices=sorted(PREPARATION_SCRIPTS),
    )
    args = parser.parse_args()
    suites = _unique_suites(args.suite)
    _verify_control_environment(PROJECT_ROOT, Path(sys.executable))
    summaries = [
        prepare_generated_quality_suite(
            suite,
            project_root=PROJECT_ROOT,
            python_executable=Path(sys.executable),
        )
        for suite in suites
    ]
    print(json.dumps({"prepared": summaries}, sort_keys=True))


def prepare_generated_quality_suite(
    suite: str,
    *,
    project_root: Path,
    python_executable: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> dict:
    """Prepare one allowlisted suite and return its verification summary."""

    if suite not in PREPARATION_SCRIPTS:
        raise ValueError("generated-control preparation suite is unsupported")
    root = project_root.resolve()
    python = python_executable.resolve()
    if not python.is_file():
        raise FileNotFoundError("generated-control preparation Python is missing")

    if suite in _STAGED_SUITES:
        with tempfile.TemporaryDirectory(prefix=f"local-bench-{suite}-") as raw_temp:
            staging_root = Path(raw_temp).resolve()
            _copy_generator_scripts(
                root=root,
                staging_root=staging_root,
                scripts=PREPARATION_SCRIPTS[suite],
            )
            _run_generators(
                root=staging_root,
                python=python,
                scripts=PREPARATION_SCRIPTS[suite],
                runner=runner,
            )
            staging_manifest = staging_root / SUITE_MANIFESTS[suite]
            summary = verify_generated_quality_controls(
                staging_manifest,
                verify_regeneration_host=True,
            )
            staging_output = staging_root / _OUTPUT_DIRECTORIES[suite]
            _verify_staged_pinned_outputs(
                suite=suite,
                staging_output=staging_output,
                project_root=root,
            )
            _promote_staged_output(
                staging_output=staging_output,
                destination=root / _OUTPUT_DIRECTORIES[suite],
            )
    else:
        _run_generators(
            root=root,
            python=python,
            scripts=PREPARATION_SCRIPTS[suite],
            runner=runner,
        )
        summary = verify_generated_quality_controls(
            root / SUITE_MANIFESTS[suite],
            verify_regeneration_host=True,
        )
    return {"suite": suite, **summary}


def _verify_control_environment(project_root: Path, python_executable: Path) -> None:
    verifier = project_root / "scripts" / "verify_control_environment.py"
    subprocess.run(
        [str(python_executable), str(verifier)],
        cwd=project_root,
        env=_isolated_subprocess_environment(),
        check=True,
    )


def _copy_generator_scripts(
    *,
    root: Path,
    staging_root: Path,
    scripts: tuple[Path, ...],
) -> None:
    for relative in scripts:
        source = (root / relative).resolve()
        if not source.is_file() or source.parent != (root / "scripts").resolve():
            raise FileNotFoundError("generated-control producer script is missing")
        destination = staging_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _run_generators(
    *,
    root: Path,
    python: Path,
    scripts: tuple[Path, ...],
    runner: Callable[..., subprocess.CompletedProcess],
) -> None:
    for relative in scripts:
        runner(
            [str(python), str(root / relative)],
            cwd=root,
            env=_isolated_subprocess_environment(),
            check=True,
        )


def _isolated_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CI"] = "true"
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _verify_staged_pinned_outputs(
    *,
    suite: str,
    staging_output: Path,
    project_root: Path,
) -> None:
    """Protect tracked bounded-VLM fixtures from host-dependent replacement."""

    registry_path = project_root / "registries" / "bounded_vlm_b10598_assets.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    candidates = registry.get("candidates")
    if not isinstance(candidates, dict):
        raise ValueError("bounded VLM asset registry is invalid")
    destination_relative = _OUTPUT_DIRECTORIES[suite]
    checked = 0
    for candidate in candidates.values():
        if not isinstance(candidate, dict):
            continue
        fixtures = candidate.get("fixtures")
        if not isinstance(fixtures, dict):
            continue
        records: list[object] = [fixtures.get("manifest")]
        images = fixtures.get("images")
        if isinstance(images, dict):
            records.extend(images.values())
        for record in records:
            if not isinstance(record, dict):
                continue
            relative = record.get("path")
            if not isinstance(relative, str):
                continue
            relative_path = Path(relative)
            if relative_path.parent != destination_relative:
                continue
            staged = staging_output / relative_path.name
            if not staged.is_file():
                continue
            if (
                staged.stat().st_size != record.get("bytes")
                or _sha256(staged) != record.get("sha256")
            ):
                raise ValueError(
                    "generated output differs from a pinned tracked public fixture"
                )
            checked += 1
    if checked == 0:
        raise ValueError("no pinned tracked fixture was checked before promotion")


def _promote_staged_output(*, staging_output: Path, destination: Path) -> None:
    if not staging_output.is_dir():
        raise FileNotFoundError("staged generated-control output is missing")
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(staging_output.iterdir(), key=lambda path: path.name):
        if not source.is_file():
            raise ValueError("staged generated-control output must be flat")
        shutil.copy2(source, destination / source.name)


def _unique_suites(suites: list[str]) -> list[str]:
    unique: list[str] = []
    for suite in suites:
        if suite in unique:
            raise ValueError("each generated-control suite may be selected once")
        unique.append(suite)
    return unique


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
