import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .event_journal import append_event, read_events, successful_attempt_keys, unterminated_attempts
from .fingerprint import fingerprint_files, fingerprint_json
from .load_registry import find_candidate, load_json
from .project_paths import ARTIFACTS_PATH, EVENTS_PATH, HARDWARE_PATH, INPUT_MANIFEST_PATH, PLAN_PATH, PROJECT_ROOT, REGISTRY_PATH
from .resource_monitor import ProcessTreeMonitor
from .projections import audio_projection, ocr_projection


def _python_for(candidate: dict) -> Path:
    env = candidate["environment"]
    if env == "OCRLLM":
        return Path("D:/Anaconda/envs/OCRLLM/python.exe")
    return Path("D:/Anaconda/envs") / env / "python.exe"


def run_candidate(candidate_id: str) -> None:
    registry = load_json(REGISTRY_PATH)
    candidate = find_candidate(registry, candidate_id)
    if candidate["status"] not in {"enabled", "planned"}:
        raise RuntimeError(f"Candidate {candidate_id} is {candidate['status']}")
    python = _python_for(candidate)
    if not python.exists():
        raise FileNotFoundError(f"Candidate Python is missing: {python}")
    hardware = load_json(HARDWARE_PATH)
    inputs = load_json(INPUT_MANIFEST_PATH)
    plan = load_json(PLAN_PATH)
    existing_events = read_events(EVENTS_PATH)
    for interrupted in unterminated_attempts(existing_events):
        append_event(EVENTS_PATH, {**interrupted, "event": "attempt_interrupted", "detected_on_resume": True, "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    successes = successful_attempt_keys(read_events(EVENTS_PATH))
    code_fingerprint = fingerprint_files([
        PROJECT_ROOT / candidate["worker"],
        Path(__file__),
        PROJECT_ROOT / "src" / "local_inference_bench" / "resource_monitor.py",
    ])
    for config in candidate["configs"]:
        attempt_key = fingerprint_json({"candidate": candidate_id, "config": config, "hardware": hardware, "inputs": inputs, "code": code_fingerprint})
        if attempt_key in successes:
            continue
        attempt_id = str(uuid.uuid4())
        artifact_dir = ARTIFACTS_PATH / candidate_id / attempt_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        request_path = artifact_dir / "request.json"
        response_path = artifact_dir / "response.json"
        request = {
            "candidate": candidate,
            "config": config,
            "inputs": inputs,
            "response_path": str(response_path),
            "generation_stop_tokens_per_second": plan["generation_stop_tokens_per_second"],
        }
        request_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
        common = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "candidate_id": candidate_id, "attempt_id": attempt_id, "attempt_key": attempt_key, "code_fingerprint": code_fingerprint, "config": config}
        append_event(EVENTS_PATH, {**common, "event": "attempt_started"})
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(config.get("threads", 1))
        env["HF_HUB_DISABLE_XET"] = "1"
        command = [str(python), str(PROJECT_ROOT / candidate["worker"]), "--request", str(request_path)]
        started = time.perf_counter()
        with (artifact_dir / "stdout.txt").open("w", encoding="utf-8") as stdout, (artifact_dir / "stderr.txt").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=stdout, stderr=stderr)
            monitor = ProcessTreeMonitor(process.pid, plan["resource_sample_interval_seconds"])
            monitor.start()
            try:
                exit_code = process.wait(timeout=plan["candidate_timeout_seconds"])
            except subprocess.TimeoutExpired:
                process.kill()
                exit_code = -1
            resources = monitor.stop()
        wall_seconds = time.perf_counter() - started
        (artifact_dir / "resource_samples.json").write_text(json.dumps(monitor.samples), encoding="utf-8")
        if exit_code == 0 and response_path.exists():
            response = load_json(response_path)
            if candidate["task"].startswith("asr") and response.get("audio_seconds") and response.get("inference_seconds"):
                response["projections"] = audio_projection(response["audio_seconds"], response["inference_seconds"])
            elif candidate["task"].startswith("ocr") and response.get("input_count") and response.get("inference_seconds"):
                response["projections"] = ocr_projection(response["input_count"], response["inference_seconds"])
            append_event(EVENTS_PATH, {**common, "event": "attempt_succeeded", "wall_seconds": wall_seconds, "resources": resources, "result": response})
            if candidate["task"] in {"asr_generative", "ocr_vlm"} and response.get("below_generation_cutoff"):
                append_event(EVENTS_PATH, {
                    **common,
                    "event": "candidate_stopped_below_generation_cutoff",
                    "minimum_tokens_per_second": plan["generation_stop_tokens_per_second"],
                    "measured_tokens_per_second": response.get("tokens_per_second", response.get("throughput", {}).get("estimated_output_tokens_per_second")),
                })
                break
        else:
            stderr_tail = (artifact_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")[-4000:]
            append_event(EVENTS_PATH, {**common, "event": "attempt_failed", "wall_seconds": wall_seconds, "resources": resources, "exit_code": exit_code, "stderr_tail": stderr_tail})
