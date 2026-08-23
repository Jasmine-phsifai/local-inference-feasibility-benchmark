"""Measure fixed-eight-thread SenseVoice CLI process concurrency."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sustained_worker_metrics import build_public_summary, write_private_records


_RUNTIME_SECONDS = re.compile(r"\[sensevoice\].*?done\s+([0-9.]+)s", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    processes = int(config["processes"])
    if int(config["effective_threads_per_process"]) != 8:
        raise ValueError("SenseVoice v0.1.9 has a fixed eight-thread CPU backend")
    items = request["workload"]["items"]
    warmup = next(
        item
        for item in items
        if item["id"] == request["workload"]["warmup_item_id"]
    )
    project_root = Path(__file__).resolve().parents[1]
    runtime_root = project_root / "data" / "models" / "sensevoice"
    binary = runtime_root / "llama-funasr-sensevoice.exe"
    model = runtime_root / "sensevoice-small-q8.gguf"
    vad_model = runtime_root / "fsmn-vad.gguf"
    if not binary.is_file() or not model.is_file():
        raise FileNotFoundError("SenseVoice runtime assets are incomplete")
    environment = os.environ.copy()
    environment["PATH"] = (
        str(Path("D:/Anaconda/Library/bin"))
        + os.pathsep
        + environment.get("PATH", "")
    )

    def invoke(item: dict, capture_prediction: bool) -> dict:
        command = [
            str(binary),
            "-m",
            str(model),
            "-a",
            item["path"],
            "--backend",
            "cpu",
            "--keep-tags",
        ]
        if vad_model.is_file():
            command.extend(["--vad", str(vad_model), "--vad-maxseg", "30000"])
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
        output_is_valid = bool(transcript) or not item.get("expected_speech", True)
        record = {
            "sample_id": item["id"],
            "success": completed.returncode == 0 and output_is_valid,
            "failure_kind": (
                None
                if completed.returncode == 0 and output_is_valid
                else "empty_output" if completed.returncode == 0 else "runtime_exit"
            ),
            "latency_seconds": latency,
            "runtime_seconds": float(match.group(1)) if match else None,
            "startup_load_estimate_seconds": (
                max(0.0, latency - float(match.group(1))) if match else None
            ),
            "units": (
                float(item["duration_seconds"])
                if completed.returncode == 0 and output_is_valid
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
        started = time.perf_counter()
        deadline = started + float(request["target_wall_seconds"])

        def run_lane(lane_index: int) -> list[dict]:
            lane_records = []
            if request["phase"] in {"quality", "compatibility"}:
                for item in items[lane_index::processes]:
                    record = invoke(item, bool(request["capture_predictions"]))
                    record["completed_offset_seconds"] = time.perf_counter() - started
                    lane_records.append(record)
                return lane_records
            item_index = lane_index
            consecutive_failures = 0
            while time.perf_counter() < deadline:
                item = items[item_index % len(items)]
                record = invoke(item, bool(request["capture_predictions"]))
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
        runtime_version="runtime-llamacpp-v0.1.9",
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=load_seconds,
        warmup_seconds=[
            record["latency_seconds"] for record in warmup_records
        ],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="per_file_cli_startup_estimate",
    )
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
