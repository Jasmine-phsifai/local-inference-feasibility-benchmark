"""Measure CTranslate2 shared workers and independent process shapes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sustained_worker_metrics import build_public_summary, write_private_records


_MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    process_count = int(request["config"]["processes"])
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    start_time = context.Value("d", 0.0)
    ready_queue = context.Queue()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker_process,
            args=(
                process_index,
                request,
                start_event,
                start_time,
                ready_queue,
                result_queue,
            ),
        )
        for process_index in range(process_count)
    ]
    for process in processes:
        process.start()
    ready = []
    try:
        for _ in processes:
            ready.append(ready_queue.get(timeout=1200))
    except queue.Empty:
        for process in processes:
            process.terminate()
        raise RuntimeError("faster-whisper worker readiness timed out") from None
    if any(not item["success"] for item in ready):
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=30)
        raise RuntimeError("faster-whisper worker initialization failed")

    start_time.value = time.perf_counter()
    start_event.set()
    records = []
    done_count = 0
    while done_count < process_count:
        try:
            message = result_queue.get(timeout=1800)
        except queue.Empty:
            for process in processes:
                process.terminate()
            raise RuntimeError("faster-whisper result collection timed out") from None
        if message["kind"] == "done":
            done_count += 1
        else:
            records.append(message["record"])
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    steady_wall_seconds = time.perf_counter() - start_time.value
    write_private_records(Path(request["private_records_path"]), records)
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="asr",
        runtime_name="faster_whisper",
        runtime_version=importlib.metadata.version("faster-whisper"),
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[item["load_seconds"] for item in ready],
        warmup_seconds=[
            value
            for item in ready
            for value in item["warmup_seconds"]
        ],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="resident_model",
    )
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _worker_process(
    process_index: int,
    request: dict,
    start_event,
    start_time,
    ready_queue,
    result_queue,
) -> None:
    config = request["config"]
    model_workers = int(config["model_workers"])
    threads_per_worker = int(config["threads_per_worker"])
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    try:
        from faster_whisper import WhisperModel

        model_root = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--Systran--faster-whisper-small"
            / "snapshots"
            / _MODEL_REVISION
        )
        if not model_root.is_dir():
            raise FileNotFoundError
        loaded_at = time.perf_counter()
        model = WhisperModel(
            str(model_root),
            device="cpu",
            compute_type="int8",
            cpu_threads=threads_per_worker,
            num_workers=model_workers,
            local_files_only=True,
        )
        load_seconds = time.perf_counter() - loaded_at
        items = request["workload"]["items"]
        warmup = next(
            item
            for item in items
            if item["id"] == request["workload"]["warmup_item_id"]
        )
        with ThreadPoolExecutor(max_workers=model_workers) as executor:
            warmup_records = list(
                executor.map(
                    lambda _: _transcribe(model, warmup, capture_prediction=False),
                    range(model_workers),
                )
            )
        ready_queue.put(
            {
                "success": all(record["success"] for record in warmup_records),
                "load_seconds": load_seconds,
                "warmup_seconds": [
                    record["latency_seconds"] for record in warmup_records
                ],
            }
        )
    except Exception as error:
        ready_queue.put(
            {
                "success": False,
                "failure_kind": type(error).__name__,
                "load_seconds": 0.0,
                "warmup_seconds": [],
            }
        )
        return

    start_event.wait()
    capture = bool(request["capture_predictions"])
    if request["phase"] in {"quality", "compatibility"}:
        assigned = items[process_index::int(config["processes"])]
        with ThreadPoolExecutor(max_workers=model_workers) as executor:
            process_records = list(
                executor.map(
                    lambda item: _transcribe(model, item, capture_prediction=capture),
                    assigned,
                )
            )
        for record in process_records:
            record["completed_offset_seconds"] = time.perf_counter() - start_time.value
            result_queue.put({"kind": "record", "record": record})
    else:
        deadline = start_time.value + float(request["target_wall_seconds"])
        item_index = process_index * model_workers
        with ThreadPoolExecutor(max_workers=model_workers) as executor:
            while time.perf_counter() < deadline:
                batch = [
                    items[(item_index + offset) % len(items)]
                    for offset in range(model_workers)
                ]
                process_records = list(
                    executor.map(
                        lambda item: _transcribe(
                            model,
                            item,
                            capture_prediction=capture,
                        ),
                        batch,
                    )
                )
                for record in process_records:
                    record["completed_offset_seconds"] = (
                        time.perf_counter() - start_time.value
                    )
                    result_queue.put({"kind": "record", "record": record})
                if sum(not record["success"] for record in process_records) == len(
                    process_records
                ):
                    break
                item_index += model_workers * int(config["processes"])
    result_queue.put({"kind": "done"})


def _transcribe(model, item: dict, *, capture_prediction: bool) -> dict:
    started = time.perf_counter()
    try:
        segments_iterator, info = model.transcribe(
            item["path"],
            beam_size=1,
            temperature=0.0,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        segments = list(segments_iterator)
    except Exception as error:
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
    latency = time.perf_counter() - started
    text = " ".join(segment.text for segment in segments).strip()
    output_is_valid = bool(text) or not item.get("expected_speech", True)
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": None if output_is_valid else "empty_output",
        "latency_seconds": latency,
        "units": float(item["duration_seconds"]) if output_is_valid else 0.0,
        "output_character_count": len(text),
    }
    if capture_prediction:
        record["prediction"] = text
        record["language"] = info.language
        record["language_probability"] = info.language_probability
        record["segments"] = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "average_log_probability": segment.avg_logprob,
                "no_speech_probability": segment.no_speech_prob,
                "compression_ratio": segment.compression_ratio,
                "words": [
                    {
                        "start": word.start,
                        "end": word.end,
                        "word": word.word,
                        "probability": word.probability,
                    }
                    for word in (segment.words or [])
                ],
            }
            for segment in segments
        ]
    return record


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
