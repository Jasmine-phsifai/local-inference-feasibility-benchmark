"""Create or verify the official stateful OpenVINO GenAI Qwen3-ASR export."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from workers.qwen3_asr_openvino_genai_export_manifest import (  # noqa: E402
    EXPORT_TASK,
    EXPORTER_REVISION,
    MARKER_FILENAME,
    SOURCE_REVISION,
    verify_export,
    verify_source,
    write_export_marker,
    write_export_provenance,
)


ENVIRONMENT_ROOT = Path(
    "D:/Anaconda/envs/local-bench-qwen3-asr-openvino-genai-official"
)
PYTHON = ENVIRONMENT_ROOT / "python.exe"
OPTIMUM_CLI = ENVIRONMENT_ROOT / "Scripts/optimum-cli.exe"
SOURCE_MODEL = PROJECT_ROOT / "data/models/qwen3-asr-0.6b-original"
FINAL_MODEL = (
    PROJECT_ROOT / "data/models/qwen3-asr-0.6b-openvino-genai-official-f48d93f"
)
PARTIAL_MODEL = FINAL_MODEL.with_name(f"{FINAL_MODEL.name}.partial")
ATTEMPT_LOG = (
    PROJECT_ROOT
    / "results/artifacts/setup/qwen3-asr-openvino-genai-official/export-attempts.jsonl"
)
EXPORTED_ENVIRONMENT_KEYS = {
    "APPDATA",
    "COMSPEC",
    "LOCALAPPDATA",
    "NUMBER_OF_PROCESSORS",
    "PATH",
    "PATHEXT",
    "PROGRAMDATA",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "WINDIR",
}


def main() -> None:
    _verify_environment()
    verify_source(SOURCE_MODEL)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    status = "failed"
    try:
        if FINAL_MODEL.exists():
            marker = FINAL_MODEL / MARKER_FILENAME
            if not marker.exists():
                raise FileExistsError(
                    "markerless final export requires explicit provenance adoption"
                )
            summary = verify_export(SOURCE_MODEL, FINAL_MODEL)
            status = "existing_verified"
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        if PARTIAL_MODEL.exists():
            raise FileExistsError(
                "official OpenVINO GenAI partial export is preserved for inspection"
            )

        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in EXPORTED_ENVIRONMENT_KEYS
        }
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            str(OPTIMUM_CLI),
            "export",
            "openvino",
            "--model",
            str(SOURCE_MODEL),
            "--task",
            EXPORT_TASK,
            "--trust-remote-code",
            "--weight-format",
            "fp16",
            str(PARTIAL_MODEL),
        ]
        completed = subprocess.run(command, check=False, env=environment)
        if completed.returncode != 0:
            raise RuntimeError(
                f"official OpenVINO GenAI export failed: {completed.returncode}"
            )
        write_export_provenance(SOURCE_MODEL, PARTIAL_MODEL)
        write_export_marker(SOURCE_MODEL, PARTIAL_MODEL)
        summary = verify_export(SOURCE_MODEL, PARTIAL_MODEL)
        if FINAL_MODEL.exists():
            raise FileExistsError("official OpenVINO GenAI final target appeared")
        PARTIAL_MODEL.rename(FINAL_MODEL)
        status = "succeeded"
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        attempt = {
            "timestamp_utc": started_utc,
            "status": status,
            "source_revision": SOURCE_REVISION,
            "exporter_revision": EXPORTER_REVISION,
            "task": EXPORT_TASK,
            "elapsed_seconds": time.perf_counter() - started,
        }
        ATTEMPT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ATTEMPT_LOG.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(attempt, sort_keys=True) + "\n")


def _verify_environment() -> None:
    verifier = (
        PROJECT_ROOT
        / "scripts/verify_qwen3_asr_openvino_genai_official_environment.py"
    )
    for executable in (PYTHON, OPTIMUM_CLI):
        if not executable.is_file():
            raise FileNotFoundError(
                f"official OpenVINO GenAI executable is missing: {executable.name}"
            )
    completed = subprocess.run([str(PYTHON), str(verifier)], check=False)
    if completed.returncode != 0:
        raise RuntimeError("official OpenVINO GenAI environment verification failed")


if __name__ == "__main__":
    main()
