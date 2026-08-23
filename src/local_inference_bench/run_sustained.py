"""Run trial-aware sustained candidate configurations with sanitized journals."""

from __future__ import annotations

import json
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
from .validate_public_summary import validate_public_summary
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
    selected_indices = (
        tuple(range(len(candidate["configs"])))
        if config_indices is None
        else config_indices
    )
    if any(index < 0 or index >= len(candidate["configs"]) for index in selected_indices):
        raise ValueError("config index is out of range")
    successful_keys = {
        event["attempt_key"]
        for event in read_events(SUSTAINED_EVENTS_PATH)
        if event.get("event") == "sustained_attempt_succeeded"
    }
    code_fingerprint = fingerprint_files(
        [
            Path(__file__),
            PROJECT_ROOT / candidate["worker"],
            PROJECT_ROOT / "workers" / "sustained_worker_metrics.py",
            PROJECT_ROOT / "src" / "local_inference_bench" / "resource_monitor.py",
            PROJECT_ROOT / "src" / "local_inference_bench" / "windows_host_monitor.py",
        ]
    )
    stable_hardware = _stable_hardware(load_json(HARDWARE_PATH))
    for config_index in selected_indices:
        config = candidate["configs"][config_index]
        for trial_index in range(trial_count):
            attempt_key = fingerprint_json(
                {
                    "protocol": registry["protocol"],
                    "candidate": candidate_id,
                    "config": config,
                    "workload": workload["fingerprint"],
                    "hardware": stable_hardware,
                    "code": code_fingerprint,
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
            )


def _python_for(environment: str) -> Path:
    return Path("D:/Anaconda/envs") / environment / "python.exe"


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
            "warmup_item_id": workload["warmup_item_id"],
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
        "attempt_id": attempt_id,
        "attempt_key": attempt_key,
        "code_fingerprint": code_fingerprint,
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
            public_summary = validate_public_summary(response["public_summary"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            failure_kind = "invalid_response"
    elif exit_code == 0:
        failure_kind = "missing_response"
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
