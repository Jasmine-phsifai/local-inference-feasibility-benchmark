"""Run trial-aware sustained candidate configurations with sanitized journals."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .event_journal import append_event
from .fingerprint import (
    fingerprint_files,
    fingerprint_json,
    unique_fingerprint_paths,
)
from .journal_integrity import (
    SUSTAINED_START_EVENT,
    SUSTAINED_TERMINAL_EVENTS,
    read_sustained_journal_snapshot,
)
from .load_registry import find_candidate, load_json
from .load_sustained_workload import (
    is_private_workload_class,
    load_sustained_workload,
)
from .project_paths import (
    HARDWARE_PATH,
    PROJECT_ROOT,
    SUSTAINED_ARTIFACTS_PATH,
    SUSTAINED_EVENTS_PATH,
    SUSTAINED_REGISTRY_PATH,
)
from .private_records_commitment import (
    PRIVATE_RECORDS_COMMITMENT_SCHEME,
    PRIVATE_RECORDS_PROVENANCE_FIELDS,
    create_private_records_commitment,
    create_private_records_bytes_commitment,
    decode_private_records_provenance_bytes,
    verify_private_records_bytes_commitment,
    verify_private_records_commitment,
)
from .resource_monitor import ProcessTreeMonitor
from .terminate_process_tree import terminate_process_tree
from .validate_public_summary import validate_sustained_public_summary
from .windows_host_monitor import WindowsHostMonitor
from .windows_kill_on_close_job import CREATE_SUSPENDED, WindowsKillOnCloseJob


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
    execution_policy_fingerprint = _execution_policy_fingerprint(
        registry,
        candidate,
    )
    workload = load_sustained_workload(
        workload_path,
        expected_task=candidate["task"],
    )
    python = _python_for(candidate["environment"])
    if not python.is_file():
        setup_script = candidate.get("setup_script")
        setup_hint = f" Run {setup_script}." if setup_script else ""
        raise FileNotFoundError(
            f"candidate Python is missing for {candidate_id}.{setup_hint}"
        )
    controller_python = Path(sys.executable).resolve()
    control_verify_script = PROJECT_ROOT / "scripts" / "verify_control_environment.py"
    _verify_candidate_environment(controller_python, control_verify_script)
    selected_indices = _select_config_indices(candidate, phase, config_indices)
    _verify_candidate_artifacts(candidate, selected_indices)
    if candidate.get("verify_script"):
        _verify_candidate_environment(
            python,
            PROJECT_ROOT / candidate["verify_script"],
        )
    events, invalidated_attempt_ids, corrected_attempt_ids = (
        read_sustained_journal_snapshot(SUSTAINED_EVENTS_PATH)
    )
    excluded_attempt_ids = invalidated_attempt_ids | corrected_attempt_ids
    successful_keys = _successful_attempt_keys(
        events,
        invalidated_attempt_ids=excluded_attempt_ids,
    )
    private_candidate_artifact_dir = (
        SUSTAINED_ARTIFACTS_PATH / candidate_id
        if is_private_workload_class(workload["workload_class"])
        else None
    )
    environment_identity_script = (
        PROJECT_ROOT / "scripts" / "capture_environment_identity.py"
    )
    fingerprint_paths = [
        Path(__file__),
        PROJECT_ROOT / candidate["worker"],
        PROJECT_ROOT / candidate["environment_manifest"],
        environment_identity_script,
        PROJECT_ROOT / "workers" / "sustained_worker_metrics.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "event_journal.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "fingerprint.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "journal_integrity.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "load_registry.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "load_sustained_workload.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "project_paths.py",
        PROJECT_ROOT
        / "src"
        / "local_inference_bench"
        / "private_records_commitment.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "resource_monitor.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "terminate_process_tree.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "validate_public_summary.py",
        PROJECT_ROOT / "src" / "local_inference_bench" / "windows_host_monitor.py",
        PROJECT_ROOT
        / "src"
        / "local_inference_bench"
        / "windows_kill_on_close_job.py",
        PROJECT_ROOT / "environments" / "control" / "environment.yml",
        PROJECT_ROOT / "environments" / "control" / "requirements.lock.txt",
        control_verify_script,
        PROJECT_ROOT / "src" / "local_inference_bench" / "verify_locked_environment.py",
    ]
    if candidate.get("verify_script"):
        fingerprint_paths.append(PROJECT_ROOT / candidate["verify_script"])
    fingerprint_paths.extend(
        PROJECT_ROOT / relative_path
        for relative_path in candidate.get("fingerprint_files", [])
    )
    environment_fingerprint = _capture_environment_fingerprint(
        python,
        environment_identity_script,
    )
    controller_environment_fingerprint = _capture_environment_fingerprint(
        controller_python,
        environment_identity_script,
    )
    stable_hardware = _stable_hardware(load_json(HARDWARE_PATH))
    code_fingerprints: dict[tuple[str, ...], str] = {}
    for trial_index in range(trial_count):
        for config_index in _ordered_config_indices(selected_indices, trial_index):
            config = candidate["configs"][config_index]
            artifact_files = tuple(_artifact_files(candidate, config))
            if artifact_files not in code_fingerprints:
                code_fingerprints[artifact_files] = fingerprint_files(
                    unique_fingerprint_paths(
                        fingerprint_paths
                        + [PROJECT_ROOT / path for path in artifact_files]
                    )
                )
            code_fingerprint = code_fingerprints[artifact_files]
            attempt_key = fingerprint_json(
                {
                    "protocol": registry["protocol"],
                    "candidate": candidate_id,
                    "config": config,
                    "workload": workload["fingerprint"],
                    "hardware": stable_hardware,
                    "code": code_fingerprint,
                    "execution_policy": execution_policy_fingerprint,
                    "environment": {
                        "name": candidate["environment"],
                        "fingerprint": environment_fingerprint,
                    },
                    "controller_environment": controller_environment_fingerprint,
                    "phase": phase,
                    "target_wall_seconds": target_wall_seconds,
                    "trial_index": trial_index,
                }
            )
            _verify_workload_content_bindings(workload)
            if private_candidate_artifact_dir is not None:
                successful_keys.update(
                    _successful_private_artifact_attempt_keys(
                        private_candidate_artifact_dir,
                        sustained_events=events,
                        invalidated_attempt_ids=excluded_attempt_ids,
                        expected_bindings={
                            attempt_key: {
                                "protocol": registry["protocol"],
                                "candidate_id": candidate_id,
                                "task": candidate["task"],
                                "config": config,
                                "config_index": config_index,
                                "phase": phase,
                                "target_wall_seconds": target_wall_seconds,
                                "trial_index": trial_index,
                                "workload_class": workload["workload_class"],
                                "workload_fingerprint": workload["fingerprint"],
                                "code_fingerprint": code_fingerprint,
                                "environment_fingerprint": environment_fingerprint,
                                "controller_environment_fingerprint": (
                                    controller_environment_fingerprint
                                ),
                                "execution_policy_fingerprint": (
                                    execution_policy_fingerprint
                                ),
                                "journal_workload": workload["public_summary"],
                                "expected_sample_ids": [
                                    item["id"] for item in workload["items"]
                                ],
                            }
                        },
                    )
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
                controller_environment_fingerprint=(
                    controller_environment_fingerprint
                ),
                execution_policy_fingerprint=execution_policy_fingerprint,
            )


def _python_for(environment: str) -> Path:
    return Path("D:/Anaconda/envs") / environment / "python.exe"


def _execution_policy_fingerprint(registry: dict, candidate: dict) -> str:
    """Bind runner telemetry policy and the selected candidate registration."""

    return fingerprint_json(
        {
            "protocol": registry["protocol"],
            "resource_sample_interval_seconds": registry[
                "resource_sample_interval_seconds"
            ],
            "host_sample_interval_seconds": registry[
                "host_sample_interval_seconds"
            ],
            "timeout_overhead_seconds": registry["timeout_overhead_seconds"],
            "candidate": candidate,
        }
    )


def _successful_attempt_keys(
    events: list[dict],
    *,
    invalidated_attempt_ids: set[str],
) -> set[str]:
    successful_keys = set()
    for event in events:
        if (
            event.get("event") != "sustained_attempt_succeeded"
            or event.get("attempt_id") in invalidated_attempt_ids
            or not isinstance(event.get("attempt_key"), str)
            or not isinstance(event.get("result"), dict)
        ):
            continue
        result = event["result"]
        status = result.get("status")
        if status is None:
            counts = result.get("counts", {})
            is_complete = (
                isinstance(counts, dict)
                and type(counts.get("completed")) is int
                and counts["completed"] > 0
                and counts.get("failed") == 0
            )
        else:
            is_complete = status == "complete"
        if is_complete:
            successful_keys.add(event["attempt_key"])
    return successful_keys


def _successful_private_artifact_attempt_keys(
    candidate_artifact_dir: Path,
    *,
    sustained_events: list[dict],
    invalidated_attempt_ids: set[str],
    expected_bindings: dict[str, dict],
) -> set[str]:
    """Return only journal-committed private artifacts matching current inputs."""

    successful_keys = set()
    if not candidate_artifact_dir.is_dir() or not expected_bindings:
        return successful_keys
    for provenance_path in candidate_artifact_dir.glob(
        "*/records-provenance.json"
    ):
        try:
            provenance = decode_private_records_provenance_bytes(
                provenance_path.read_bytes()
            )
        except (OSError, ValueError):
            continue
        attempt_key = provenance.get("attempt_key")
        if type(attempt_key) is not str:
            continue
        expected = expected_bindings.get(attempt_key)
        if expected is None:
            continue
        raw_expected_sample_ids = expected.get("expected_sample_ids")
        if (
            not isinstance(raw_expected_sample_ids, list)
            or not raw_expected_sample_ids
            or any(type(sample_id) is not str for sample_id in raw_expected_sample_ids)
            or len(raw_expected_sample_ids) != len(set(raw_expected_sample_ids))
        ):
            continue
        records_path = provenance_path.with_name("private-records.jsonl")
        attempt_id = provenance.get("attempt_id")
        workload_class = provenance.get("workload_class")
        declared_records_sha256 = provenance.get("records_sha256")
        start_events = [
            (position, event)
            for position, event in enumerate(sustained_events)
            if event.get("attempt_id") == attempt_id
            and event.get("event") == SUSTAINED_START_EVENT
        ]
        terminal_events = [
            (position, event)
            for position, event in enumerate(sustained_events)
            if event.get("attempt_id") == attempt_id
            and event.get("event") in SUSTAINED_TERMINAL_EVENTS
        ]
        try:
            canonical_attempt_id = str(uuid.UUID(attempt_id))
        except (ValueError, TypeError, AttributeError):
            continue
        journal_workload = expected.get("journal_workload")
        expected_fields_match = all(
            _json_values_equal(provenance.get(key), expected_value)
            for key, expected_value in expected.items()
            if key not in {"journal_workload", "expected_sample_ids"}
        )
        if (
            type(provenance.get("schema_version")) is not int
            or provenance["schema_version"] != 1
            or provenance.get("status") != "succeeded"
            or canonical_attempt_id != attempt_id
            or provenance_path.parent.name != attempt_id
            or attempt_id in invalidated_attempt_ids
            or re.fullmatch(r"[0-9a-f]{16}", attempt_key) is None
            or type(workload_class) is not str
            or not is_private_workload_class(workload_class)
            or not expected_fields_match
            or not records_path.is_file()
            or type(declared_records_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", declared_records_sha256) is None
            or len(start_events) != 1
            or len(terminal_events) != 1
            or start_events[0][0] >= terminal_events[0][0]
        ):
            continue
        try:
            with records_path.open("rb") as handle:
                records_bytes = handle.read(64 * 1024 * 1024 + 1)
            if len(records_bytes) > 64 * 1024 * 1024:
                continue
            records_sha256 = hashlib.sha256(records_bytes).hexdigest()
            record_counts = _validate_private_records(
                records_bytes,
                task=provenance["task"],
                phase=provenance["phase"],
                expected_sample_ids=set(raw_expected_sample_ids),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            continue
        start = start_events[0][1]
        terminal = terminal_events[0][1]
        result = terminal.get("result")
        journal_identity_fields = (
            "protocol",
            "candidate_id",
            "task",
            "config",
            "config_index",
            "phase",
            "target_wall_seconds",
            "trial_index",
            "code_fingerprint",
            "environment_fingerprint",
            "controller_environment_fingerprint",
            "execution_policy_fingerprint",
        )
        lifecycle_identity_matches = all(
            event.get("attempt_id") == attempt_id
            and all(
                _json_values_equal(event.get(key), provenance.get(key))
                for key in journal_identity_fields
            )
            and _json_values_equal(event.get("workload"), journal_workload)
            and "attempt_key" not in event
            and "workload_fingerprint" not in event
            and event.get("private_records_commitment_scheme")
            == PRIVATE_RECORDS_COMMITMENT_SCHEME
            for event in (start, terminal)
        )
        counts = result.get("counts") if isinstance(result, dict) else None
        if (
            records_sha256 != declared_records_sha256
            or record_counts["attempted"] <= 0
            or not lifecycle_identity_matches
            or terminal.get("event") != "sustained_attempt_succeeded"
            or terminal.get("private_records_commitment_scheme")
            != PRIVATE_RECORDS_COMMITMENT_SCHEME
            or not isinstance(result, dict)
            or result.get("status") != "complete"
            or result.get("candidate_id") != provenance.get("candidate_id")
            or result.get("task") != provenance.get("task")
            or result.get("workload_class") != workload_class
            or not isinstance(counts, dict)
            or type(counts.get("attempted")) is not int
            or type(counts.get("completed")) is not int
            or type(counts.get("failed")) is not int
            or counts.get("attempted") != record_counts["attempted"]
            or counts.get("completed") != record_counts["completed"]
            or counts.get("failed") != record_counts["failed"]
            or record_counts["failed"] != 0
        ):
            continue
        try:
            verify_private_records_bytes_commitment(
                records_bytes,
                provenance,
                records_sha256=declared_records_sha256,
                private_commitment=provenance.get("private_records_commitment"),
                public_commitment=terminal.get("private_artifact_commitment"),
            )
        except (OSError, ValueError):
            continue
        successful_keys.add(attempt_key)
    return successful_keys


def _validate_private_records(
    records_bytes: bytes,
    *,
    task: str,
    phase: str,
    expected_sample_ids: set[str],
) -> dict[str, int]:
    """Validate the exact worker record snapshot and return public counts."""

    if (
        type(records_bytes) is not bytes
        or task not in {"asr", "ocr"}
        or phase not in {"screen", "sustained", "quality", "compatibility"}
        or not isinstance(expected_sample_ids, set)
        or not expected_sample_ids
        or any(type(sample_id) is not str or not sample_id for sample_id in expected_sample_ids)
    ):
        raise ValueError("private records are invalid")
    try:
        lines = records_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("private records are invalid") from error

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate private record key")
            result[key] = value
        return result

    def reject_constant(_constant: str):
        raise ValueError("non-finite private record number")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("non-finite private record number")
        return parsed

    attempted = 0
    completed = 0
    seen_quality_ids: set[str] = set()
    total_output_characters = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        attempted += 1
        if attempted > 100_000:
            raise ValueError("private record count budget exceeded")
        try:
            record = json.loads(
                line,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_constant,
                parse_float=parse_finite_float,
            )
        except (json.JSONDecodeError, RecursionError, ValueError) as error:
            raise ValueError(f"private record line {line_number} is invalid") from error
        if not isinstance(record, dict):
            raise ValueError(f"private record line {line_number} is invalid")
        sample_id = record.get("sample_id")
        success = record.get("success")
        if (
            type(sample_id) is not str
            or sample_id not in expected_sample_ids
            or type(success) is not bool
        ):
            raise ValueError(f"private record line {line_number} is invalid")
        if phase in {"quality", "compatibility"}:
            if sample_id in seen_quality_ids:
                raise ValueError("private quality records repeat a sample id")
            seen_quality_ids.add(sample_id)
        captures_predictions = phase in {"quality", "compatibility"}
        if task == "asr" and (
            (captures_predictions and success) or "prediction" in record
        ):
            prediction = record.get("prediction")
            if type(prediction) is not str or len(prediction) > 200_000:
                raise ValueError(f"private record line {line_number} is invalid")
            total_output_characters += len(prediction)
            if total_output_characters > 2_000_000:
                raise ValueError("private ASR record text budget exceeded")
        if task == "ocr" and (
            (captures_predictions and success) or "lines" in record
        ):
            output_lines = record.get("lines")
            if not isinstance(output_lines, list) or len(output_lines) > 10_000:
                raise ValueError(f"private record line {line_number} is invalid")
            for output_line in output_lines:
                text = output_line.get("text") if isinstance(output_line, dict) else None
                if type(text) is not str or len(text) > 10_000:
                    raise ValueError(f"private record line {line_number} is invalid")
                total_output_characters += len(text)
                if total_output_characters > 1_000_000:
                    raise ValueError("private OCR record text budget exceeded")
        completed += success

    if attempted == 0:
        raise ValueError("private records are empty")
    if (
        phase in {"quality", "compatibility"}
        and seen_quality_ids != expected_sample_ids
    ):
        raise ValueError("private quality records do not cover the workload")
    return {
        "attempted": attempted,
        "completed": completed,
        "failed": attempted - completed,
    }


def _json_values_equal(left: object, right: object) -> bool:
    try:
        return json.dumps(
            left,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ) == json.dumps(
            right,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return False


def _artifact_files(candidate: dict, config: dict) -> list[str]:
    artifact_group = config.get("artifact_group")
    if artifact_group is None:
        artifact_group = candidate.get("artifact_group_by_runtime_variant", {}).get(
            config.get("runtime_variant"),
            candidate.get("default_artifact_group"),
        )
    grouped_files = []
    if artifact_group is not None:
        groups = candidate.get("artifact_groups", {})
        if not isinstance(artifact_group, str) or artifact_group not in groups:
            raise ValueError("candidate configuration has an unknown artifact group")
        grouped_files = groups[artifact_group]
    return list(
        dict.fromkeys(
            [
                *candidate.get("artifact_files", []),
                *grouped_files,
            ]
        )
    )


def _verify_candidate_artifacts(
    candidate: dict,
    selected_indices: tuple[int, ...],
) -> None:
    required = []
    for config_index in selected_indices:
        required.extend(_artifact_files(candidate, candidate["configs"][config_index]))
    missing = [
        relative_path
        for relative_path in dict.fromkeys(required)
        if not (PROJECT_ROOT / relative_path).is_file()
    ]
    if not missing:
        return
    setup_script = candidate.get("setup_script")
    setup_hint = f" Run {setup_script}." if setup_script else ""
    raise FileNotFoundError(
        f"candidate {candidate['id']} is missing required local artifacts: "
        f"{', '.join(missing)}.{setup_hint}"
    )


def _capture_environment_fingerprint(python: Path, script: Path) -> str:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(
            [str(python), str(script)],
            cwd=PROJECT_ROOT,
            env=environment,
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
    except (json.JSONDecodeError, RecursionError) as error:
        raise RuntimeError("candidate environment identity was invalid") from error
    return fingerprint_json(identity)


def _verify_candidate_environment(python: Path, script: Path) -> None:
    environment = os.environ.copy()
    environment["CI"] = "true"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
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
    if str(candidate.get("status", "")).startswith("retired"):
        raise ValueError("candidate is retired from sustained execution")
    configs = candidate["configs"]
    retired_indices = set(candidate.get("retired_config_indices", ()))
    candidate_phases = candidate.get("allowed_phases", _PHASES)
    if requested_indices is not None:
        if len(requested_indices) != len(set(requested_indices)):
            raise ValueError("config indices must be distinct")
        if any(
            type(index) is not int or index < 0 or index >= len(configs)
            for index in requested_indices
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
        if index not in retired_indices
        if phase in candidate_phases
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
    controller_environment_fingerprint: str,
    execution_policy_fingerprint: str,
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
        "controller_environment_fingerprint": controller_environment_fingerprint,
        "execution_policy_fingerprint": execution_policy_fingerprint,
        "config": config,
        "config_index": config_index,
        "phase": phase,
        "target_wall_seconds": target_wall_seconds,
        "trial_index": trial_index,
        "workload": workload["public_summary"],
        "workload_fingerprint": workload["fingerprint"],
    }
    environment = os.environ.copy()
    environment["CI"] = "true"
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    command = [
        str(python),
        str(PROJECT_ROOT / candidate["worker"]),
        "--request",
        str(request_path),
    ]
    process = None
    process_monitor = None
    host_monitor = None
    process_job = None
    job_assigned = False
    process_resources = {"sample_count": 0}
    host_telemetry = {
        "status": "unavailable",
        "sample_count": 0,
        "package_temperature_available": False,
    }
    exit_code = -4
    failure_kind: str | None = "controller_failure"
    worker_started: float | None = None
    attempt_started = False
    interrupted = False
    controller_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    termination_failed = False
    with (
        (artifact_dir / "stdout.txt").open("w", encoding="utf-8") as stdout,
        (artifact_dir / "stderr.txt").open("w", encoding="utf-8") as stderr,
    ):
        try:
            host_monitor = WindowsHostMonitor(
                artifact_dir / "host-telemetry.jsonl",
                registry["host_sample_interval_seconds"],
            )
            host_monitor.start()
            ready_timeout_seconds = max(
                10.0,
                registry["host_sample_interval_seconds"] * 3.0,
            )
            if not host_monitor.wait_until_ready(ready_timeout_seconds):
                reason = host_monitor.start_error or "first_sample_timeout"
                raise RuntimeError(f"host monitor preflight failed: {reason}")
            if host_monitor.stop_reason is not None:
                raise RuntimeError(
                    "host monitor preflight safety stop: "
                    f"{host_monitor.stop_reason}"
                )
            worker_started = time.perf_counter()
            process_job = WindowsKillOnCloseJob()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                creationflags=CREATE_SUSPENDED,
            )
            process_job.assign(process)
            job_assigned = True
            process_job.resume(process)
            process_monitor = ProcessTreeMonitor(
                process.pid,
                registry["resource_sample_interval_seconds"],
                sample_path=artifact_dir / "process-resources.jsonl",
            )
            process_monitor.start()
            append_event(
                SUSTAINED_EVENTS_PATH,
                {
                    **_public_attempt_common(common),
                    "event": "sustained_attempt_started",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            attempt_started = True
            exit_code, failure_kind = _wait_for_process(
                process,
                host_monitor=host_monitor,
                process_monitor=process_monitor,
                process_job=process_job,
                timeout_seconds=(
                    target_wall_seconds + registry["timeout_overhead_seconds"]
                ),
            )
            interrupted = failure_kind == "interrupted"
            termination_failed = failure_kind == "termination_failure"
        except BaseException as error:
            controller_error = error
            if process is not None and process.poll() is None:
                if job_assigned and process_job is not None:
                    termination_failed = not _close_worker_job(process_job)
                else:
                    termination_failed = not _terminate_unassigned_worker(
                        process.pid
                    )
        finally:
            if process_job is not None:
                termination_failed = (
                    not _close_worker_job(process_job) or termination_failed
                )
            if process_monitor is not None:
                try:
                    process_resources = process_monitor.stop()
                except BaseException as error:
                    cleanup_error = error
            if host_monitor is not None:
                try:
                    host_telemetry = host_monitor.stop()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
    if not attempt_started:
        if termination_failed:
            raise RuntimeError(
                "benchmark worker termination could not be verified before "
                "the attempt started"
            ) from (controller_error or cleanup_error)
        raise controller_error or cleanup_error or RuntimeError(
            "sustained attempt did not start"
        )
    assert worker_started is not None
    assert process is not None
    assert process_monitor is not None
    assert host_monitor is not None
    if controller_error is not None:
        failure_kind = "controller_failure"
        exit_code = process.poll() if process.poll() is not None else -4
    if (
        cleanup_error is not None
        or process_monitor.monitor_error is not None
        or host_monitor.start_error is not None
        or host_telemetry.get("status") != "observed"
        or host_telemetry.get("monitor_partial") is True
        or host_telemetry.get("sample_count", 0) < 1
        or process_resources.get("sample_count", 0) < 1
    ):
        failure_kind = "monitor_failure"
    if host_monitor.stop_reason is not None:
        failure_kind = "safety_stop"
    if termination_failed:
        failure_kind = "termination_failure"
    wall_seconds = time.perf_counter() - worker_started
    _record_attempt_outcome(
        common=common,
        response_path=response_path,
        private_records_path=private_records_path,
        workload=workload,
        workload_fingerprint=workload["fingerprint"],
        expected_sample_ids={item["id"] for item in workload["items"]},
        exit_code=exit_code,
        failure_kind=failure_kind,
        wall_seconds=wall_seconds,
        process_resources=process_resources,
        host_telemetry=host_telemetry,
    )
    if interrupted:
        raise KeyboardInterrupt
    if controller_error is not None:
        raise controller_error
    if termination_failed:
        raise RuntimeError("benchmark worker termination could not be verified")
    if cleanup_error is not None:
        raise cleanup_error


def _wait_for_process(
    process: subprocess.Popen,
    *,
    host_monitor: WindowsHostMonitor,
    process_monitor: ProcessTreeMonitor,
    process_job: WindowsKillOnCloseJob | None = None,
    timeout_seconds: float,
) -> tuple[int, str | None]:
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            if (
                host_monitor.start_error is not None
                or process_monitor.monitor_error is not None
            ):
                termination_ok = _request_worker_stop(process, process_job)
                return (
                    process.poll() if process.poll() is not None else -2,
                    "monitor_failure" if termination_ok else "termination_failure",
                )
            if host_monitor.stop_reason is not None:
                termination_ok = _request_worker_stop(process, process_job)
                return (
                    process.poll() if process.poll() is not None else -2,
                    "safety_stop" if termination_ok else "termination_failure",
                )
            exit_code = process.poll()
            if exit_code is not None:
                return exit_code, None if exit_code == 0 else "worker_exit"
            if time.monotonic() >= deadline:
                termination_ok = _request_worker_stop(process, process_job)
                return (
                    process.poll() if process.poll() is not None else -1,
                    "timeout" if termination_ok else "termination_failure",
                )
            time.sleep(0.5)
    except KeyboardInterrupt:
        termination_ok = _request_worker_stop(process, process_job)
        return (
            process.poll() if process.poll() is not None else -3,
            "interrupted" if termination_ok else "termination_failure",
        )


def _request_worker_stop(
    process: subprocess.Popen,
    process_job: WindowsKillOnCloseJob | None,
) -> bool:
    if process_job is not None:
        return _close_worker_job(process_job)
    return _terminate_unassigned_worker(process.pid)


def _close_worker_job(process_job: WindowsKillOnCloseJob) -> bool:
    try:
        process_job.close()
    except BaseException:
        return False
    return True


def _terminate_unassigned_worker(pid: int) -> bool:
    try:
        outcome = terminate_process_tree(pid)
    except BaseException:
        return False
    if not isinstance(outcome, dict):
        return False
    surviving = outcome.get("surviving")
    error_count = outcome.get("error_count")
    if type(surviving) is not int or type(error_count) is not int:
        return False
    return surviving == 0 and error_count == 0


def _record_attempt_outcome(
    *,
    common: dict,
    response_path: Path,
    private_records_path: Path,
    workload: dict,
    workload_fingerprint: str,
    expected_sample_ids: set[str],
    exit_code: int,
    failure_kind: str | None,
    wall_seconds: float,
    process_resources: dict,
    host_telemetry: dict,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    public_summary = None
    private_artifact_commitment = None
    try:
        _verify_workload_content_bindings(workload)
    except ValueError:
        if failure_kind is None:
            failure_kind = "workload_content_changed"
    if failure_kind is None and exit_code == 0 and response_path.is_file():
        try:
            response = json.loads(response_path.read_text(encoding="utf-8"))
            public_summary = validate_sustained_public_summary(
                response["public_summary"],
                candidate_id=common["candidate_id"],
                task=common["task"],
                workload_class=common["workload"]["workload_class"],
                target_wall_seconds=common["target_wall_seconds"],
                phase=common["phase"],
                config=common["config"],
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
            RecursionError,
        ):
            failure_kind = "invalid_response"
    elif failure_kind is None and exit_code == 0:
        failure_kind = "missing_response"
    if public_summary is not None and failure_kind is None:
        try:
            private_artifact_commitment = _write_records_provenance(
                private_records_path,
                common,
                workload_fingerprint=workload_fingerprint,
                result_status=public_summary["status"],
                public_counts=public_summary["counts"],
                expected_sample_ids=expected_sample_ids,
            )
        except (FileNotFoundError, OSError, TypeError, ValueError):
            failure_kind = "invalid_private_records"
    if public_summary is not None and failure_kind is None:
        result_status = public_summary["status"]
        event_name = {
            "complete": "sustained_attempt_succeeded",
            "partial_failure": "sustained_attempt_partial",
            "all_failed": "sustained_attempt_failed",
        }[result_status]
        event = {
            **_public_attempt_common(common),
            "event": event_name,
            "timestamp_utc": timestamp,
            "wall_seconds": wall_seconds,
            "resources": process_resources,
            "host_telemetry": host_telemetry,
            "result": public_summary,
        }
        if result_status != "complete":
            event["failure_kind"] = (
                "partial_item_failures"
                if result_status == "partial_failure"
                else "all_items_failed"
            )
        if is_private_workload_class(common["workload"]["workload_class"]):
            event["private_artifact_commitment"] = private_artifact_commitment
        append_event(
            SUSTAINED_EVENTS_PATH,
            event,
        )
        return
    append_event(
        SUSTAINED_EVENTS_PATH,
        {
            **_public_attempt_common(common),
            "event": "sustained_attempt_failed",
            "timestamp_utc": timestamp,
            "wall_seconds": wall_seconds,
            "exit_code": exit_code,
            "failure_kind": failure_kind or "worker_exit",
            "resources": process_resources,
            "host_telemetry": host_telemetry,
        },
    )


def _public_attempt_common(common: dict) -> dict:
    """Project internal attempt metadata into a safe public journal record."""

    public_common = dict(common)
    if is_private_workload_class(common["workload"]["workload_class"]):
        public_common.pop("attempt_key", None)
        public_common.pop("workload_fingerprint", None)
        public_common["private_records_commitment_scheme"] = (
            PRIVATE_RECORDS_COMMITMENT_SCHEME
        )
    return public_common


def _write_records_provenance(
    private_records_path: Path,
    common: dict,
    *,
    workload_fingerprint: str,
    result_status: str,
    public_counts: dict,
    expected_sample_ids: set[str],
) -> dict:
    if not private_records_path.is_file():
        raise FileNotFoundError("worker private records are missing")
    provenance_status = {
        "complete": "succeeded",
        "partial_failure": "partial_failure",
        "all_failed": "all_failed",
    }.get(result_status)
    if provenance_status is None:
        raise ValueError("worker result status is invalid")
    with private_records_path.open("rb") as handle:
        records_bytes = handle.read(64 * 1024 * 1024 + 1)
    if len(records_bytes) > 64 * 1024 * 1024:
        raise ValueError("worker private records exceed the byte budget")
    record_counts = _validate_private_records(
        records_bytes,
        task=common["task"],
        phase=common["phase"],
        expected_sample_ids=expected_sample_ids,
    )
    if (
        not isinstance(public_counts, dict)
        or set(public_counts) != {"attempted", "completed", "failed"}
        or any(type(public_counts.get(key)) is not int for key in record_counts)
        or any(public_counts[key] != value for key, value in record_counts.items())
    ):
        raise ValueError("worker private record counts do not match the public summary")
    provenance = {
        "schema_version": 1,
        "protocol": common["protocol"],
        "status": provenance_status,
        "attempt_id": common["attempt_id"],
        "attempt_key": common["attempt_key"],
        "candidate_id": common["candidate_id"],
        "task": common["task"],
        "config": common["config"],
        "config_index": common["config_index"],
        "phase": common["phase"],
        "target_wall_seconds": common["target_wall_seconds"],
        "trial_index": common["trial_index"],
        "workload_class": common["workload"]["workload_class"],
        "workload_fingerprint": workload_fingerprint,
        "code_fingerprint": common["code_fingerprint"],
        "environment_fingerprint": common["environment_fingerprint"],
        "controller_environment_fingerprint": common[
            "controller_environment_fingerprint"
        ],
        "execution_policy_fingerprint": common["execution_policy_fingerprint"],
    }
    commitment = create_private_records_bytes_commitment(
        records_bytes,
        provenance,
    )
    provenance["records_sha256"] = commitment["records_sha256"]
    provenance["private_records_commitment"] = commitment["private"]
    if set(provenance) != PRIVATE_RECORDS_PROVENANCE_FIELDS:
        raise ValueError("worker private records provenance is invalid")
    provenance_path = private_records_path.with_name("records-provenance.json")
    with provenance_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(provenance, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return commitment["public"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_workload_content_bindings(workload: dict) -> None:
    """Fail if a worker input changed after the manifest snapshot was loaded."""

    items = workload.get("items")
    warmup = workload.get("warmup_item")
    bindings = workload.get("item_content_bindings")
    if (
        not isinstance(items, list)
        or not items
        or not isinstance(warmup, dict)
        or not isinstance(bindings, dict)
    ):
        raise ValueError("workload content bindings are invalid")
    items_by_id = {}
    for item in [*items, warmup]:
        sample_id = item.get("id") if isinstance(item, dict) else None
        path_value = item.get("path") if isinstance(item, dict) else None
        if (
            type(sample_id) is not str
            or type(path_value) is not str
            or (sample_id in items_by_id and items_by_id[sample_id] != path_value)
        ):
            raise ValueError("workload content bindings are invalid")
        items_by_id[sample_id] = path_value
    if set(bindings) != set(items_by_id):
        raise ValueError("workload content bindings are invalid")
    for sample_id, path_value in items_by_id.items():
        binding = bindings.get(sample_id)
        if (
            not isinstance(binding, dict)
            or set(binding) != {"content_sha256", "size_bytes"}
            or type(binding.get("content_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", binding["content_sha256"]) is None
            or type(binding.get("size_bytes")) is not int
            or binding["size_bytes"] < 0
        ):
            raise ValueError("workload content bindings are invalid")
        path = Path(path_value)
        try:
            initial = path.stat()
            size_bytes = 0
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    size_bytes += len(chunk)
                    digest.update(chunk)
            final = path.stat()
        except OSError as error:
            raise ValueError("workload content changed after manifest load") from error
        if (
            initial.st_size != final.st_size
            or initial.st_mtime_ns != final.st_mtime_ns
            or size_bytes != binding["size_bytes"]
            or digest.hexdigest() != binding["content_sha256"]
        ):
            raise ValueError("workload content changed after manifest load")
