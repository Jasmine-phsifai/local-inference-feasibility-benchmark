"""Run trial-aware sustained candidate configurations with sanitized journals."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .event_journal import append_event, read_events
from .fingerprint import fingerprint_files, fingerprint_json
from .load_registry import find_candidate, load_json
from .load_sustained_workload import load_sustained_workload
from .project_paths import (
    HARDWARE_PATH,
    PROJECT_ROOT,
    SUSTAINED_ARTIFACTS_PATH,
    SUSTAINED_EVENTS_PATH,
    SUSTAINED_REGISTRY_PATH,
)
from .resource_monitor import ProcessTreeMonitor
from .terminate_process_tree import terminate_process_tree
from .validate_public_summary import validate_sustained_public_summary
from .windows_host_monitor import WindowsHostMonitor


_PHASES = {"screen", "sustained", "quality", "compatibility"}


def run_sustained_candidate(
    candidate_id: str,
    workload_path: Path,
    *,
    phase: str,
    target_wall_seconds: float,
    trial_count: int = 1,
    config_indices: tuple[int, ...] | None = None,
) -> None:
    if phase not in _PHASES:
        raise ValueError(f"unknown sustained benchmark phase: {phase}")
    if not 1 <= target_wall_seconds <= 7200:
        raise ValueError("target_wall_seconds must be in [1, 7200]")
    if not 1 <= trial_count <= 10:
        raise ValueError("trial_count must be in [1, 10]")
    registry = load_json(SUSTAINED_REGISTRY_PATH)
    candidate = find_candidate(registry, candidate_id)
    workload = load_sustained_workload(
        workload_path,
        expected_task=candidate["task"],
    )
    python = _python_for(candidate["environment"])
    if not python.is_file():
        raise FileNotFoundError(f"candidate Python is missing for {candidate_id}")
    if candidate.get("verify_script"):
        _verify_candidate_environment(
            python,
            PROJECT_ROOT / candidate["verify_script"],
        )
    selected_indices = _select_config_indices(candidate, phase, config_indices)
    successful_keys = {
        event["attempt_key"]
        for event in read_events(SUSTAINED_EVENTS_PATH)
        if event.get("event") == "sustained_attempt_succeeded"
    }
    environment_identity_script = (
        PROJECT_ROOT / "scripts" / "capture_environment_identity.py"
    )
    fingerprint_paths = [
        Path(__file__),
        PROJECT_ROOT / candidate["worker"],
        PROJECT_ROOT / candidate["environment_manifest"],
        environment_identity_script,
        PROJECT_ROOT / "workers" / "sustained_worker_metrics.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "load_sustained_workload.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "resource_monitor.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "validate_public_summary.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "windows_host_monitor.py",
    ]
    if candidate.get("verify_script"):
        fingerprint_paths.append(PROJECT_ROOT / candidate["verify_script"])
    fingerprint_paths.extend(
        PROJECT_ROOT / relative_path
        for relative_path in candidate.get("fingerprint_files", [])
    )
    fingerprint_paths.extend(
        PROJECT_ROOT / relative_path
        for relative_path in candidate.get("artifact_files", [])
    )
    code_fingerprint = fingerprint_files(fingerprint_paths)
    environment_fingerprint = _capture_environment_fingerprint(
        python,
        environment_identity_script,
    )
    stable_hardware = _stable_hardware(load_json(HARDWARE_PATH))
    for trial_index in range(trial_count):
        for config_index in _ordered_config_indices(selected_indices, trial_index):
            config = candidate["configs"][config_index]
            attempt_key = fingerprint_json(
                {
                    "protocol": registry["protocol"],
                    "candidate": candidate_id,
                    "config": config,
                    "workload": workload["fingerprint"],
                    "hardware": stable_hardware,
                    "code": code_fingerprint,
                    "environment": {
                        "name": candidate["environment"],
                        "fingerprint": environment_fingerprint,
                    },
                    "phase": phase,
                    "target_wall_seconds": target_wall_seconds,
                    "trial_index": trial_index,
                }
            )
            if attempt_key in successful_keys:
                continue
            _run_attempt(
                registry=registry,
                candidate=candidate,
                python=python,
                workload=workload,
                config=config,
                config_index=config_index,
                phase=phase,
                target_wall_seconds=target_wall_seconds,
                trial_index=trial_index,
                attempt_key=attempt_key,
                code_fingerprint=code_fingerprint,
                environment_fingerprint=environment_fingerprint,
            )


def _python_for(environment: str) -> Path:
    return Path("D:/Anaconda/envs") / environment / "python.exe"


def _capture_environment_fingerprint(python: Path, script: Path) -> str:
    try:
        completed = subprocess.run(
            [str(python), str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("candidate environment identity capture timed out") from error
    if completed.returncode != 0:
        raise RuntimeError("candidate environment identity capture failed")
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("candidate environment identity was invalid") from error
    return fingerprint_json(identity)


def _verify_candidate_environment(python: Path, script: Path) -> None:
    environment = os.environ.copy()
    environment["CI"] = "true"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    try:
        completed = subprocess.run(
            [str(python), str(script)],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("candidate environment verification timed out") from error
    if completed.returncode != 0:
        raise RuntimeError("candidate environment verification failed")


def _ordered_config_indices(
    selected_indices: tuple[int, ...],
    trial_index: int,
) -> tuple[int, ...]:
    """Alternate A/B order across trials to reduce thermal and cache bias."""

    return selected_indices if trial_index % 2 == 0 else tuple(reversed(selected_indices))


def _select_config_indices(
    candidate: dict,
    phase: str,
    requested_indices: tuple[int, ...] | None,
) -> tuple[int, ...]:
    configs = candidate["configs"]
    if requested_indices is not None and any(
        index < 0 or index >= len(configs) for index in requested_indices
    ):
        raise ValueError("config index is out of range")
    indices = (
        tuple(range(len(configs)))
        if requested_indices is None
        else requested_indices
    )
    selected = tuple(
        index
        for index in indices
        if phase in configs[index].get("phases", _PHASES)
    )
    if requested_indices is not None and selected != requested_indices:
        raise ValueError("requested config does not support this phase")
    if not selected:
        raise ValueError("candidate has no config for this phase")
    return selected


def _stable_hardware(hardware: dict) -> dict:
    return {
        "os": {
            "caption": hardware["os"]["caption"],
            "version": hardware["os"]["version"],
            "build": hardware["os"]["build"],
        },
        "cpu": hardware["cpu"],
        "memory_total_bytes": hardware["memory"]["total_bytes"],
        "gpu": [
            {
                "name": adapter["name"],
                "driver_version": adapter["driver_version"],
            }
            for adapter in hardware["gpu"]
        ],
    }


def _run_attempt(
    *,
    registry: dict,
    candidate: dict,
    python: Path,
    workload: dict,
    config: dict,
    config_index: int,
    phase: str,
    target_wall_seconds: float,
    trial_index: int,
    attempt_key: str,
    code_fingerprint: str,
    environment_fingerprint: str,
) -> None:
    attempt_id = str(uuid.uuid4())
    artifact_dir = SUSTAINED_ARTIFACTS_PATH / candidate["id"] / attempt_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    request_path = artifact_dir / "request.json"
    response_path = artifact_dir / "response.json"
    private_records_path = artifact_dir / "private-records.jsonl"
    request = {
        "protocol": registry["protocol"],
        "candidate_id": candidate["id"],
        "task": candidate["task"],
        "config": config,
        "workload": {
            "workload_class": workload["workload_class"],
            "items": workload["items"],
            "warmup_item": workload["warmup_item"],
        },
        "phase": phase,
        "target_wall_seconds": target_wall_seconds,
        "capture_predictions": phase in {"quality", "compatibility"},
        "response_path": str(response_path),
        "private_records_path": str(private_records_path),
    }
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    common = {
        "protocol": registry["protocol"],
        "candidate_id": candidate["id"],
        "task": candidate["task"],
        "attempt_id": attempt_id,
        "attempt_key": attempt_key,
        "code_fingerprint": code_fingerprint,
        "environment_fingerprint": environment_fingerprint,
        "config": config,
        "config_index": config_index,
        "phase": phase,
        "target_wall_seconds": target_wall_seconds,
        "trial_index": trial_index,
        "workload": workload["public_summary"],
    }
    append_event(
        SUSTAINED_EVENTS_PATH,
        {
            **common,
            "event": "sustained_attempt_started",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    environment = os.environ.copy()
    environment["CI"] = "true"
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        str(python),
        str(PROJECT_ROOT / candidate["worker"]),
        "--request",
        str(request_path),
    ]
    started = time.perf_counter()
    with (
        (artifact_dir / "stdout.txt").open("w", encoding="utf-8") as stdout,
        (artifact_dir / "stderr.txt").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
        process_monitor = ProcessTreeMonitor(
            process.pid,
            registry["resource_sample_interval_seconds"],
            sample_path=artifact_dir / "process-resources.jsonl",
        )
        host_monitor = WindowsHostMonitor(
            artifact_dir / "host-telemetry.jsonl",
            registry["host_sample_interval_seconds"],
        )
        process_monitor.start()
        host_monitor.start()
        try:
            exit_code, failure_kind = _wait_for_process(
                process,
                host_monitor=host_monitor,
                timeout_seconds=(
                    target_wall_seconds + registry["timeout_overhead_seconds"]
                ),
            )
        except BaseException:
            if process.poll() is None:
                terminate_process_tree(process.pid)
            raise
        finally:
            process_resources = process_monitor.stop()
            host_telemetry = host_monitor.stop()
    wall_seconds = time.perf_counter() - started
    _record_attempt_outcome(
        common=common,
        response_path=response_path,
        private_records_path=private_records_path,
        workload_fingerprint=workload["fingerprint"],
        exit_code=exit_code,
        failure_kind=failure_kind,
        wall_seconds=wall_seconds,
        process_resources=process_resources,
        host_telemetry=host_telemetry,
    )


def _wait_for_process(
    process: subprocess.Popen,
    *,
    host_monitor: WindowsHostMonitor,
    timeout_seconds: float,
) -> tuple[int, str | None]:
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                return exit_code, None if exit_code == 0 else "worker_exit"
            if host_monitor.stop_reason is not None:
                terminate_process_tree(process.pid)
                return process.poll() if process.poll() is not None else -2, "safety_stop"
            if time.monotonic() >= deadline:
                terminate_process_tree(process.pid)
                return process.poll() if process.poll() is not None else -1, "timeout"
            time.sleep(0.5)
    except KeyboardInterrupt:
        terminate_process_tree(process.pid)
        return process.poll() if process.poll() is not None else -3, "interrupted"


def _record_attempt_outcome(
    *,
    common: dict,
    response_path: Path,
    private_records_path: Path,
    workload_fingerprint: str,
    exit_code: int,
    failure_kind: str | None,
    wall_seconds: float,
    process_resources: dict,
    host_telemetry: dict,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    public_summary = None
    if exit_code == 0 and response_path.is_file():
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
            public_summary = validate_sustained_public_summary(
                response["public_summary"],
                candidate_id=common["candidate_id"],
                task=common["task"],
                workload_class=common["workload"]["workload_class"],
                target_wall_seconds=common["target_wall_seconds"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            failure_kind = "invalid_response"
    elif exit_code == 0:
        failure_kind = "missing_response"
    if public_summary is not None and failure_kind is None:
        try:
            _write_records_provenance(
                private_records_path,
                common,
                workload_fingerprint=workload_fingerprint,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            failure_kind = "invalid_private_records"
    if public_summary is not None and failure_kind is None:
        append_event(
            SUSTAINED_EVENTS_PATH,
            {
                **common,
                "event": "sustained_attempt_succeeded",
                "timestamp_utc": timestamp,
                "wall_seconds": wall_seconds,
                "resources": process_resources,
                "host_telemetry": host_telemetry,
                "result": public_summary,
            },
        )
        return
    append_event(
        SUSTAINED_EVENTS_PATH,
        {
            **common,
            "event": "sustained_attempt_failed",
            "timestamp_utc": timestamp,
            "wall_seconds": wall_seconds,
            "exit_code": exit_code,
            "failure_kind": failure_kind or "worker_exit",
            "resources": process_resources,
            "host_telemetry": host_telemetry,
        },
    )


def _write_records_provenance(
    private_records_path: Path,
    common: dict,
    *,
    workload_fingerprint: str,
) -> None:
    if not private_records_path.is_file():
        raise FileNotFoundError("worker private records are missing")
    provenance = {
        "schema_version": 1,
        "protocol": common["protocol"],
        "status": "succeeded",
        "attempt_id": common["attempt_id"],
        "attempt_key": common["attempt_key"],
        "candidate_id": common["candidate_id"],
        "task": common["task"],
        "config": common["config"],
        "config_index": common["config_index"],
        "phase": common["phase"],
        "trial_index": common["trial_index"],
        "workload_class": common["workload"]["workload_class"],
        "workload_fingerprint": workload_fingerprint,
        "code_fingerprint": common["code_fingerprint"],
        "environment_fingerprint": common["environment_fingerprint"],
        "records_sha256": _sha256(private_records_path),
    }
    private_records_path.with_name("records-provenance.json").write_text(
        json.dumps(provenance, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
