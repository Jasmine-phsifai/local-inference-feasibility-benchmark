"""Run one monitored, immutable b10598 VLM quality-gate artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.bounded_vlm_assets import (  # noqa: E402
    load_and_verify_candidate_assets,
    sha256_file,
)
from local_inference_bench.load_sustained_workload import (  # noqa: E402
    load_sustained_workload,
)
from local_inference_bench.fingerprint import fingerprint_json  # noqa: E402
from local_inference_bench.resource_monitor import ProcessTreeMonitor  # noqa: E402
from local_inference_bench.terminate_process_tree import (  # noqa: E402
    terminate_process_tree,
)
from local_inference_bench.windows_host_monitor import (  # noqa: E402
    WindowsHostMonitor,
)
from local_inference_bench.windows_kill_on_close_job import (  # noqa: E402
    CREATE_SUSPENDED,
    WindowsKillOnCloseJob,
)


PROTOCOL = "bounded-vlm-b10598-run-v2"
CANDIDATES = {
    "ovisocr2_q8_cpu": {
        "worker": "workers/ovisocr2_b10598_quality_worker.py",
        "config": {
            "processes": 1,
            "threads_per_process": 24,
            "max_new_tokens": 4096,
            "mode": "source_faithful",
        },
        "worker_timeout_seconds": 900.0,
        "overall_timeout_seconds": 1200.0,
    },
    "hunyuanocr_1_5_gguf_cpu": {
        "worker": "workers/hunyuanocr_1_5_b10598_quality_worker.py",
        "config": {
            "processes": 1,
            "threads_per_process": 24,
            "max_new_tokens": 4096,
            "mode": "doc_parse",
        },
        "worker_timeout_seconds": 900.0,
        "overall_timeout_seconds": 1200.0,
    },
}
COMMON_EVIDENCE_FILES = {
    "host-telemetry.jsonl",
    "monitor-summary.json",
    "private-records.jsonl",
    "process-resources.jsonl",
    "request.json",
    "response.json",
    "worker.stderr.txt",
    "worker.stdout.txt",
}
CANDIDATE_EVIDENCE_FILES = {
    "ovisocr2_q8_cpu": {
        "ovis.command.json",
        "ovis.llama.log",
        "ovis.stderr.txt",
        "ovis.stdout.txt",
    },
    "hunyuanocr_1_5_gguf_cpu": {
        "hunyuan-server.stderr.txt",
        "hunyuan-server.stdout.txt",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=tuple(CANDIDATES))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_bounded_quality_gate(
        candidate_id=args.candidate,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def run_bounded_quality_gate(*, candidate_id: str, output_dir: Path) -> dict:
    """Verify assets, run the fixed worker, and preserve fail-closed monitors."""

    spec = CANDIDATES[candidate_id]
    artifact_root = output_dir.resolve()
    allowed_root = (PROJECT_ROOT / "results" / "artifacts").resolve()
    if not artifact_root.is_relative_to(allowed_root):
        raise ValueError("bounded VLM output must stay under results/artifacts")
    if artifact_root.exists():
        raise FileExistsError("bounded VLM runner refuses to reuse an output directory")

    assets = load_and_verify_candidate_assets(
        project_root=PROJECT_ROOT,
        candidate_id=candidate_id,
    )
    _verify_controller_environment()
    controller_environment_fingerprint = _controller_environment_fingerprint()
    request_path = artifact_root / "request.json"
    response_path = artifact_root / "response.json"
    records_path = artifact_root / "private-records.jsonl"
    monitor_path = artifact_root / "monitor-summary.json"
    provenance_path = artifact_root / "run-provenance.json"
    request = build_request(
        candidate_id=candidate_id,
        assets=assets,
        response_path=response_path,
        records_path=records_path,
    )
    producer_hashes = _producer_hashes(candidate_id)
    artifact_root.mkdir(parents=True)
    request_path.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    worker_path = PROJECT_ROOT / spec["worker"]
    command = [str(Path(sys.executable)), str(worker_path), "--request", str(request_path)]
    environment = _worker_environment()

    host_monitor = WindowsHostMonitor(
        artifact_root / "host-telemetry.jsonl",
        interval_seconds=2.0,
    )
    process: subprocess.Popen | None = None
    process_monitor: ProcessTreeMonitor | None = None
    process_job: WindowsKillOnCloseJob | None = None
    job_assigned = False
    process_summary: dict = {}
    host_summary: dict = {}
    exit_code = -1
    failure_kind: str | None = "worker_not_started"
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    primary_error: BaseException | None = None
    termination_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        host_monitor.start()
        _wait_for_host_monitor(host_monitor, timeout_seconds=15.0)
        with (
            (artifact_root / "worker.stdout.txt").open("x", encoding="utf-8") as stdout,
            (artifact_root / "worker.stderr.txt").open("x", encoding="utf-8") as stderr,
        ):
            process_job = WindowsKillOnCloseJob()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                creationflags=(
                    CREATE_SUSPENDED
                    | getattr(subprocess, "CREATE_NO_WINDOW", 0)
                ),
            )
            process_job.assign(process)
            job_assigned = True
            process_job.resume(process)
            process_monitor = ProcessTreeMonitor(
                process.pid,
                interval_seconds=0.25,
                sample_path=artifact_root / "process-resources.jsonl",
            )
            process_monitor.start()
            exit_code, failure_kind = _wait_for_worker(
                process,
                host_monitor,
                process_monitor,
                process_job,
                timeout_seconds=float(spec["overall_timeout_seconds"]),
            )
    except BaseException as error:
        primary_error = error
    finally:
        try:
            termination_error = _close_worker_containment(
                process=process,
                process_job=process_job,
                job_assigned=job_assigned,
            )
        except BaseException as error:
            termination_error = RuntimeError(
                "bounded VLM worker termination could not be verified"
            )
            termination_error.add_note(
                "termination controller error: " + type(error).__name__
            )
            termination_error.__cause__ = error
        if process_monitor is not None:
            try:
                process_summary = process_monitor.stop()
            except BaseException as error:
                cleanup_error = error
        try:
            host_summary = host_monitor.stop()
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error

    if primary_error is not None:
        if termination_error is not None:
            raise primary_error from termination_error
        if cleanup_error is not None:
            raise primary_error from cleanup_error
        raise primary_error
    if termination_error is not None:
        raise termination_error
    if cleanup_error is not None:
        raise cleanup_error
    if failure_kind != "termination_failure":
        if process_monitor is not None and process_monitor.monitor_error is not None:
            failure_kind = "monitor_failure"
        if host_monitor.start_error is not None:
            failure_kind = "monitor_failure"
        if host_monitor.stop_reason is not None:
            failure_kind = "safety_stop"

    monitor_summary = {
        "exit_code": exit_code,
        "failure_kind": failure_kind,
        "wall_seconds": time.perf_counter() - started,
        "process_resources": process_summary,
        "host_telemetry": host_summary,
    }
    monitor_path.write_text(
        json.dumps(monitor_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    status = _validate_completed_run(
        exit_code=exit_code,
        failure_kind=failure_kind,
        process_summary=process_summary,
        host_summary=host_summary,
        response_path=response_path,
        records_path=records_path,
    )
    provenance = build_run_provenance(
        candidate_id=candidate_id,
        status=status,
        started_utc=started_utc,
        assets=assets,
        producer_hashes=producer_hashes,
        controller_environment_fingerprint=controller_environment_fingerprint,
        artifact_root=artifact_root,
    )
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "candidate_id": candidate_id,
        "status": status,
        "protocol": PROTOCOL,
        "artifact_directory_created": True,
    }


def build_request(
    *,
    candidate_id: str,
    assets: dict,
    response_path: Path,
    records_path: Path,
) -> dict:
    """Build one exact candidate request from its verified public fixture manifest."""

    spec = CANDIDATES[candidate_id]
    fixture_manifest = assets["fixtures"]["manifest"]["path"]
    workload = load_sustained_workload(fixture_manifest, expected_task="ocr")
    expected_ids = assets["candidate"]["sample_ids"]
    if [item["id"] for item in workload["items"]] != expected_ids:
        raise ValueError("bounded VLM fixture manifest item order changed")
    return {
        "protocol": PROTOCOL,
        "candidate_id": candidate_id,
        "task": "ocr",
        "phase": "quality",
        "config": spec["config"],
        "workload": {
            "workload_class": workload["workload_class"],
            "items": workload["items"],
            "warmup_item": workload["warmup_item"],
            "fingerprint": workload["fingerprint"],
        },
        "capture_predictions": True,
        "target_wall_seconds": 600.0,
        "timeout_seconds": spec["worker_timeout_seconds"],
        "response_path": str(response_path),
        "private_records_path": str(records_path),
    }


def build_run_provenance(
    *,
    candidate_id: str,
    status: str,
    started_utc: str,
    assets: dict,
    producer_hashes: dict[str, str],
    controller_environment_fingerprint: str,
    artifact_root: Path,
) -> dict:
    """Bind the private run to full hashes without copying paths or raw output."""

    return {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "candidate_id": candidate_id,
        "status": status,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "asset_registry_sha256": sha256_file(assets["registry_path"]),
        "runtime": {
            "archive_sha256": assets["runtime"]["archive"]["sha256"],
            "tree_sha256": assets["runtime"]["tree_fingerprint"]["sha256"],
            "entrypoint_sha256": {
                name: value["sha256"]
                for name, value in sorted(assets["runtime"]["entrypoints"].items())
            },
        },
        "candidate_artifact_sha256": {
            name: value["sha256"]
            for name, value in sorted(assets["artifacts"].items())
        },
        "input": {
            "manifest_sha256": assets["fixtures"]["manifest"]["sha256"],
            "image_sha256": {
                name: value["sha256"]
                for name, value in sorted(assets["fixtures"]["images"].items())
            },
        },
        "producer_sha256": producer_hashes,
        "controller_environment_fingerprint": controller_environment_fingerprint,
        "run_artifact_sha256": _run_evidence_hashes(
            candidate_id=candidate_id,
            artifact_root=artifact_root,
        ),
    }


def _producer_hashes(candidate_id: str) -> dict[str, str]:
    relative_paths = [
        "scripts/run_bounded_vlm_b10598_quality.py",
        "src/local_inference_bench/bounded_vlm_assets.py",
        "src/local_inference_bench/html_output_projection.py",
        "src/local_inference_bench/load_sustained_workload.py",
        "src/local_inference_bench/resource_monitor.py",
        "src/local_inference_bench/score_document_fidelity.py",
        "src/local_inference_bench/score_ocr_quality.py",
        "src/local_inference_bench/terminate_process_tree.py",
        "src/local_inference_bench/validate_public_summary.py",
        "src/local_inference_bench/windows_host_monitor.py",
        "src/local_inference_bench/windows_kill_on_close_job.py",
        "src/local_inference_bench/fingerprint.py",
        "src/local_inference_bench/verify_locked_environment.py",
        "environments/control/environment.yml",
        "environments/control/requirements.lock.txt",
        "scripts/capture_environment_identity.py",
        "scripts/verify_control_environment.py",
        CANDIDATES[candidate_id]["worker"],
    ]
    if candidate_id == "hunyuanocr_1_5_gguf_cpu":
        relative_paths.extend(
            [
                "workers/hunyuanocr_1_5_server_worker.py",
                "workers/sustained_worker_metrics.py",
            ]
        )
    return {
        path: sha256_file(PROJECT_ROOT / path)
        for path in sorted(relative_paths)
    }


def _verify_controller_environment() -> None:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [str(Path(sys.executable)), str(PROJECT_ROOT / "scripts/verify_control_environment.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=120,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError("bounded VLM controller environment verification failed")


def _worker_environment() -> dict[str, str]:
    """Build the isolated environment inherited by the actual VLM worker."""

    environment = os.environ.copy()
    environment["CI"] = "true"
    environment["HF_HUB_DISABLE_XET"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONNOUSERSITE"] = "1"
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT),
            inherited_pythonpath,
        )
        if value
    )
    return environment


def _controller_environment_fingerprint() -> str:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [str(Path(sys.executable)), str(PROJECT_ROOT / "scripts/capture_environment_identity.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError("bounded VLM controller identity capture failed")
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("bounded VLM controller identity is invalid") from error
    return fingerprint_json(identity)


def _run_evidence_hashes(*, candidate_id: str, artifact_root: Path) -> dict[str, str]:
    expected_names = COMMON_EVIDENCE_FILES | CANDIDATE_EVIDENCE_FILES[candidate_id]
    actual_names = {
        path.name
        for path in artifact_root.iterdir()
        if path.is_file() and path.name != "run-provenance.json"
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"bounded VLM run artifact set changed: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return {
        name: sha256_file(artifact_root / name)
        for name in sorted(expected_names)
    }


def _wait_for_host_monitor(
    monitor: WindowsHostMonitor,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if monitor.start_error is not None:
            raise RuntimeError("host safety monitor failed before inference")
        if monitor.stop_reason is not None:
            raise RuntimeError("host safety guard active before inference")
        if monitor.samples:
            return
        time.sleep(0.1)
    raise TimeoutError("host safety monitor produced no preflight sample")


def _wait_for_worker(
    process: subprocess.Popen,
    host_monitor: WindowsHostMonitor,
    process_monitor: ProcessTreeMonitor,
    process_job: WindowsKillOnCloseJob,
    *,
    timeout_seconds: float,
) -> tuple[int, str | None]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if (
            host_monitor.start_error is not None
            or process_monitor.monitor_error is not None
        ):
            termination_ok = _close_worker_job(process_job)
            return (
                process.poll() if process.poll() is not None else -2,
                "monitor_failure" if termination_ok else "termination_failure",
            )
        if host_monitor.stop_reason is not None:
            termination_ok = _close_worker_job(process_job)
            return (
                process.poll() if process.poll() is not None else -2,
                "safety_stop" if termination_ok else "termination_failure",
            )
        exit_code = process.poll()
        if exit_code is not None:
            return exit_code, None if exit_code == 0 else "worker_exit"
        if time.monotonic() >= deadline:
            termination_ok = _close_worker_job(process_job)
            return (
                process.poll() if process.poll() is not None else -1,
                "timeout" if termination_ok else "termination_failure",
            )
        time.sleep(0.25)


def _close_worker_job(process_job: WindowsKillOnCloseJob) -> bool:
    try:
        process_job.close()
    except BaseException:
        return False
    return True


def _close_worker_containment(
    *,
    process: subprocess.Popen | None,
    process_job: WindowsKillOnCloseJob | None,
    job_assigned: bool,
) -> BaseException | None:
    """Close containment and verify fallback termination when assignment failed."""

    job_close_error: BaseException | None = None
    if process_job is not None:
        try:
            process_job.close()
        except BaseException as error:
            job_close_error = error

    needs_fallback = (
        process is not None
        and process.poll() is None
        and (not job_assigned or job_close_error is not None)
    )
    fallback_error: BaseException | None = None
    if needs_fallback:
        try:
            outcome = terminate_process_tree(process.pid)
            if not isinstance(outcome, dict):
                raise RuntimeError("process-tree termination returned no verification")
            surviving = outcome.get("surviving")
            error_count = outcome.get("error_count")
            if type(surviving) is not int or type(error_count) is not int:
                raise RuntimeError("process-tree termination verification is malformed")
            if surviving != 0 or error_count != 0:
                raise RuntimeError("process-tree termination left an unverified survivor")
        except BaseException as error:
            fallback_error = error

    if job_close_error is None and fallback_error is None:
        return None
    termination_error = RuntimeError(
        "bounded VLM worker termination could not be verified"
    )
    details = [
        type(error).__name__
        for error in (job_close_error, fallback_error)
        if error is not None
    ]
    if details:
        termination_error.add_note("termination errors: " + ", ".join(details))
    termination_error.__cause__ = job_close_error or fallback_error
    return termination_error


def _validate_completed_run(
    *,
    exit_code: int,
    failure_kind: str | None,
    process_summary: dict,
    host_summary: dict,
    response_path: Path,
    records_path: Path,
) -> str:
    if exit_code != 0 or failure_kind is not None:
        raise RuntimeError(f"bounded VLM worker failed: {failure_kind or exit_code}")
    if host_summary.get("status") != "observed" or host_summary.get("monitor_partial"):
        raise RuntimeError("bounded VLM host monitoring was incomplete")
    if host_summary.get("sample_count", 0) < 2:
        raise RuntimeError("bounded VLM host monitor produced too few samples")
    if process_summary.get("sample_count", 0) < 2:
        raise RuntimeError("bounded VLM process monitor produced too few samples")
    if not response_path.is_file() or not records_path.is_file():
        raise RuntimeError("bounded VLM worker omitted required artifacts")
    return "succeeded"


if __name__ == "__main__":
    main()
