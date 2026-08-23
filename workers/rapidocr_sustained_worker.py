"""Measure one reusable RapidOCR engine per independent process."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import queue
import time
from pathlib import Path

from sustained_worker_metrics import build_public_summary, write_private_records


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
            ready.append(ready_queue.get(timeout=600))
    except queue.Empty:
        for process in processes:
            process.terminate()
        raise RuntimeError("RapidOCR worker readiness timed out") from None
    if any(not item["success"] for item in ready):
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=30)
        raise RuntimeError("RapidOCR worker initialization failed")
    start_time.value = time.perf_counter()
    start_event.set()
    records = []
    done_count = 0
    while done_count < process_count:
        try:
            message = result_queue.get(timeout=1200)
        except queue.Empty:
            for process in processes:
                process.terminate()
            raise RuntimeError("RapidOCR result collection timed out") from None
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
        task="ocr",
        runtime_name="rapidocr",
        runtime_version=importlib.metadata.version("rapidocr"),
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[item["load_seconds"] for item in ready],
        warmup_seconds=[item["warmup_seconds"] for item in ready],
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
    process_count = int(config["processes"])
    threads = int(config["threads_per_process"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    try:
        from rapidocr import RapidOCR

        loaded_at = time.perf_counter()
        engine = RapidOCR(
            params={
                "Global.log_level": "critical",
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
        load_seconds = time.perf_counter() - loaded_at
        items = request["workload"]["items"]
        warmup = request["workload"]["warmup_item"]
        warmup_record = _recognize(engine, warmup, capture_prediction=False)
        ready_queue.put(
            {
                "success": warmup_record["success"],
                "load_seconds": load_seconds,
                "warmup_seconds": warmup_record["latency_seconds"],
            }
        )
    except Exception as error:
        ready_queue.put(
            {
                "success": False,
                "failure_kind": type(error).__name__,
                "load_seconds": 0.0,
                "warmup_seconds": 0.0,
            }
        )
        return

    start_event.wait()
    capture = bool(request["capture_predictions"])
    if request["phase"] in {"quality", "compatibility"}:
        assigned = items[process_index::process_count]
        for item in assigned:
            record = _recognize(engine, item, capture_prediction=capture)
            record["completed_offset_seconds"] = time.perf_counter() - start_time.value
            result_queue.put({"kind": "record", "record": record})
    else:
        deadline = start_time.value + float(request["target_wall_seconds"])
        item_index = process_index
        consecutive_failures = 0
        while time.perf_counter() < deadline:
            item = items[item_index % len(items)]
            record = _recognize(engine, item, capture_prediction=capture)
            record["completed_offset_seconds"] = time.perf_counter() - start_time.value
            result_queue.put({"kind": "record", "record": record})
            consecutive_failures = 0 if record["success"] else consecutive_failures + 1
            if consecutive_failures >= 3:
                break
            item_index += process_count
    result_queue.put({"kind": "done"})


def _recognize(engine, item: dict, *, capture_prediction: bool) -> dict:
    started = time.perf_counter()
    try:
        output = engine(item["path"])
        texts = [] if output.txts is None else list(output.txts)
        scores = [] if output.scores is None else list(output.scores)
        boxes = [] if output.boxes is None else list(output.boxes)
    except Exception as error:
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
    latency = time.perf_counter() - started
    output_is_valid = bool(texts) or not item.get("expected_text", True)
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": None if output_is_valid else "empty_output",
        "latency_seconds": latency,
        "units": 1.0 if output_is_valid else 0.0,
        "detected_line_count": len(texts),
        "output_character_count": sum(len(str(text)) for text in texts),
    }
    if capture_prediction:
        record["lines"] = [
            {
                "text": str(text),
                "confidence": float(scores[index]) if index < len(scores) else None,
                "polygon": (
                    boxes[index].tolist()
                    if index < len(boxes) and hasattr(boxes[index], "tolist")
                    else boxes[index] if index < len(boxes) else None
                ),
            }
            for index, text in enumerate(texts)
        ]
    return record


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
