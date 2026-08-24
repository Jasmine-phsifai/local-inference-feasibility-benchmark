"""Measure official and thread-controlled SenseVoice CLI concurrency."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


_RUNTIME_SECONDS = re.compile(r"\[sensevoice\].*?done\s+([0-9.]+)s", re.IGNORECASE)
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    processes = int(config["processes"])
    threads = int(config["effective_threads_per_process"])
    capture_predictions = request["capture_predictions"]
    if type(capture_predictions) is not bool:
        raise ValueError("capture_predictions must be boolean")
    if capture_predictions and request["phase"] not in {"quality", "compatibility"}:
        raise ValueError("predictions may be captured only in a private quality phase")
    items = request["workload"]["items"]
    warmup = request["workload"]["warmup_item"]
    project_root = Path(__file__).resolve().parents[1]
    runtime_root = project_root / "data" / "models" / "sensevoice"
    runtime_variant = config.get("runtime_variant", "official_fixed8")
    (
        binary,
        runtime_version,
        thread_arguments,
        explicit_cpu_backend,
    ) = _select_runtime(project_root, runtime_variant, threads)

    model = runtime_root / "sensevoice-small-q8.gguf"
    vad_model = runtime_root / "fsmn-vad.gguf"
    if not binary.is_file() or not model.is_file():
        raise FileNotFoundError("SenseVoice runtime assets are incomplete")
    environment = os.environ.copy()
    environment["PATH"] = str(binary.parent) + os.pathsep + environment.get(
        "PATH",
        "",
    )

    def invoke(item: dict, capture_prediction: bool) -> dict:
        command = _build_command(
            binary=binary,
            model=model,
            audio=Path(item["path"]),
            vad_model=vad_model if vad_model.is_file() else None,
            thread_arguments=thread_arguments,
            explicit_cpu_backend=explicit_cpu_backend,
        )
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return {
                "sample_id": item["id"],
                "success": False,
                "failure_kind": type(error).__name__,
                "latency_seconds": time.perf_counter() - started,
                "units": 0.0,
            }
        latency = time.perf_counter() - started
        transcript = completed.stdout.strip()
        match = _RUNTIME_SECONDS.search(completed.stderr)
        success, failure_kind = _invocation_outcome(
            returncode=completed.returncode,
            stderr=completed.stderr,
            transcript=transcript,
            expected_speech=item.get("expected_speech", True),
        )
        record = {
            "sample_id": item["id"],
            "success": success,
            "failure_kind": failure_kind,
            "latency_seconds": latency,
            "runtime_seconds": float(match.group(1)) if match else None,
            "startup_load_estimate_seconds": (
                max(0.0, latency - float(match.group(1))) if match else None
            ),
            "units": (
                float(item["duration_seconds"])
                if success
                else 0.0
            ),
            "output_character_count": len(transcript),
        }
        if capture_prediction:
            record["prediction"] = transcript
        return record

    with ThreadPoolExecutor(max_workers=processes) as executor:
        warmup_records = list(
            executor.map(
                lambda _: invoke(warmup, False),
                range(processes),
            )
        )
        if not all(record["success"] for record in warmup_records):
            raise RuntimeError("SenseVoice warmup failed")
        started = time.perf_counter()
        deadline = started + float(request["target_wall_seconds"])

        def run_lane(lane_index: int) -> list[dict]:
            lane_records = []
            if request["phase"] in {"quality", "compatibility"}:
                for item in items[lane_index::processes]:
                    record = invoke(item, capture_predictions)
                    record["completed_offset_seconds"] = time.perf_counter() - started
                    lane_records.append(record)
                return lane_records
            item_index = lane_index
            consecutive_failures = 0
            while time.perf_counter() < deadline:
                item = items[item_index % len(items)]
                record = invoke(item, capture_predictions)
                record["completed_offset_seconds"] = time.perf_counter() - started
                lane_records.append(record)
                consecutive_failures = 0 if record["success"] else consecutive_failures + 1
                if consecutive_failures >= 3:
                    break
                item_index += processes
            return lane_records

        partitions = list(executor.map(run_lane, range(processes)))
    steady_wall_seconds = time.perf_counter() - started
    records = [record for partition in partitions for record in partition]
    load_seconds = [
        record["startup_load_estimate_seconds"]
        for record in records
        if record.get("startup_load_estimate_seconds") is not None
    ]
    write_private_records(Path(request["private_records_path"]), records)
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="asr",
        runtime_name="funasr_llamacpp_sensevoice",
        runtime_version=runtime_version,
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=load_seconds,
        warmup_seconds=[
            record["latency_seconds"] for record in warmup_records
        ],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="per_file_cli_startup_estimate_after_integrity_hashing",
    )
    public_summary["model"] = {
        "mode": runtime_variant,
        "model_revision": "90c1c61912018b70ada0fcc024ea24aca62f2e63",
        "vad_revision": "6840bae4c5c92ee8c04faaf4db23dd0105098d7f",
        "processes": processes,
        "threads_per_process": threads,
        "configured_total_threads": processes * threads,
    }
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _select_runtime(
    project_root: Path,
    runtime_variant: str,
    threads: int,
) -> tuple[Path, str, list[str], bool]:
    if runtime_variant == "official_fixed8":
        if threads != 8:
            raise ValueError("official SenseVoice v0.1.9 is fixed at eight threads")
        return (
            project_root / "data/models/sensevoice/llama-funasr-sensevoice.exe",
            "runtime-llamacpp-v0.1.9-official-binary",
            [],
            True,
        )
    if runtime_variant == "source_thread_control":
        if not 1 <= threads <= 24:
            raise ValueError("thread-controlled SenseVoice requires 1 to 24 threads")
        return (
            project_root
            / "data/models/sensevoice-runtime-v0.1.9-thread-build"
            / "build-pinned/bin/llama-funasr-sensevoice.exe",
            "runtime-llamacpp-v0.1.9-legacy-qwenaudio-73ccdd3577db-llama-8086439a4cea-thread-option-v1",
            ["--threads", str(threads)],
            False,
        )
    if runtime_variant == "source_thread_control_v020":
        if not 1 <= threads <= 24:
            raise ValueError("thread-controlled SenseVoice requires 1 to 24 threads")
        return (
            project_root
            / "data/models/sensevoice-runtime-v0.2.0-thread-build"
            / "build-pinned/bin/llama-funasr-sensevoice.exe",
            "runtime-llamacpp-v0.2.0-modelscope-500956bc331b-llama-803b7fcae893-thread-option-v1",
            ["--threads", str(threads)],
            True,
        )
    raise ValueError("unknown SenseVoice runtime variant")


def _invocation_outcome(
    *,
    returncode: int,
    stderr: str,
    transcript: str,
    expected_speech: bool,
) -> tuple[bool, str | None]:
    if returncode != 0:
        return False, "runtime_exit"
    if "compute failed" in stderr.casefold():
        return False, "compute_failed"
    if expected_speech and not _SENSEVOICE_TAG.sub("", transcript).strip():
        return False, "empty_output"
    return True, None


def _build_command(
    *,
    binary: Path,
    model: Path,
    audio: Path,
    vad_model: Path | None,
    thread_arguments: list[str],
    explicit_cpu_backend: bool,
) -> list[str]:
    command = [str(binary), "-m", str(model), "-a", str(audio)]
    if explicit_cpu_backend:
        command.extend(["--backend", "cpu"])
    command.extend(["--keep-tags", *thread_arguments])
    if vad_model is not None:
        command.extend(["--vad", str(vad_model), "--vad-maxseg", "30000"])
    return command


if __name__ == "__main__":
    main()
