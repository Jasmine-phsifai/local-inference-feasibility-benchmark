"""Measure one reusable PP-OCRv6 tiny pipeline per independent process."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import multiprocessing
import os
import queue
import sys
import time
from pathlib import Path

try:
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    capture_predictions = request["capture_predictions"]
    if type(capture_predictions) is not bool:
        raise ValueError("capture_predictions must be boolean")
    if capture_predictions and request["phase"] not in {"quality", "compatibility"}:
        raise ValueError("predictions may be captured only in a private quality phase")
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
    started_processes = []
    try:
        for process in processes:
            process.start()
            started_processes.append(process)
        _coordinate_started_processes(
            request=request,
            process_count=process_count,
            processes=processes,
            start_event=start_event,
            start_time=start_time,
            ready_queue=ready_queue,
            result_queue=result_queue,
        )
    finally:
        _stop_processes(started_processes)


def _coordinate_started_processes(
    *,
    request: dict,
    process_count: int,
    processes,
    start_event,
    start_time,
    ready_queue,
    result_queue,
) -> None:
    ready_by_index = {}
    readiness_deadline = time.monotonic() + 900.0
    while len(ready_by_index) < process_count:
        remaining = readiness_deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("PaddleOCR worker readiness timed out")
        try:
            message = ready_queue.get(timeout=min(5.0, remaining))
        except queue.Empty:
            if any(
                index not in ready_by_index and process.exitcode is not None
                for index, process in enumerate(processes)
            ):
                raise RuntimeError("PaddleOCR worker exited before readiness") from None
            continue
        process_index = _claim_process_index(
            message,
            process_count=process_count,
            claimed=ready_by_index,
            stage="readiness",
        )
        if type(message.get("success")) is not bool:
            raise RuntimeError("PaddleOCR worker readiness message is invalid")
        ready_by_index[process_index] = message
    ready = [ready_by_index[index] for index in range(process_count)]
    if any(not item["success"] for item in ready):
        raise RuntimeError("PaddleOCR worker initialization failed")
    start_time.value = time.perf_counter()
    start_event.set()
    records = []
    done_process_indices = set()
    result_deadline = time.monotonic() + max(
        float(request["target_wall_seconds"]) + 600.0,
        1200.0,
    )
    while len(done_process_indices) < process_count:
        remaining = result_deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("PaddleOCR result collection timed out")
        try:
            message = result_queue.get(timeout=min(5.0, remaining))
        except queue.Empty:
            if any(
                index not in done_process_indices and process.exitcode is not None
                for index, process in enumerate(processes)
            ):
                raise RuntimeError("PaddleOCR worker exited before completion") from None
            continue
        if not isinstance(message, dict):
            raise RuntimeError("PaddleOCR worker result message is invalid")
        if message.get("kind") == "done":
            process_index = _claim_process_index(
                message,
                process_count=process_count,
                claimed=done_process_indices,
                stage="completion",
            )
            done_process_indices.add(process_index)
        elif message.get("kind") == "record":
            records.append(message["record"])
        else:
            raise RuntimeError("PaddleOCR worker failed during inference")
    steady_wall_seconds = time.perf_counter() - start_time.value
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            raise RuntimeError("PaddleOCR worker did not exit after completion")
        if process.exitcode != 0:
            raise RuntimeError("PaddleOCR worker exit code was nonzero")
    write_private_records(Path(request["private_records_path"]), records)
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="ocr",
        runtime_name=f"paddleocr_ppocrv6_{request['config'].get('model_tier', 'tiny')}",
        runtime_version=importlib.metadata.version("paddleocr"),
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
    model_tier = config.get("model_tier", "tiny")
    if model_tier not in {"tiny", "medium"}:
        raise ValueError("unsupported PP-OCRv6 model tier")
    _configure_openmp_environment(config, threads, os.environ)
    opencv_threads = _validated_opencv_threads(config)
    try:
        if os.name == "nt":
            vcomp_path = Path(sys.prefix) / "vcomp140.dll"
            if not vcomp_path.is_file():
                raise FileNotFoundError("pinned VCOMP runtime is missing")
            vcomp_runtime = ctypes.WinDLL(str(vcomp_path.resolve()))
        if opencv_threads is not None:
            import cv2

            cv2.setNumThreads(opencv_threads)
        from paddleocr import PaddleOCR

        loaded_at = time.perf_counter()
        engine = PaddleOCR(
            device="cpu",
            text_detection_model_name=f"PP-OCRv6_{model_tier}_det",
            text_recognition_model_name=f"PP-OCRv6_{model_tier}_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            cpu_threads=threads,
            enable_mkldnn=True,
        )
        if os.name == "nt" and vcomp_runtime._handle == 0:
            raise RuntimeError("pinned VCOMP runtime handle is invalid")
        load_seconds = time.perf_counter() - loaded_at
        items = request["workload"]["items"]
        warmup = request["workload"]["warmup_item"]
        warmup_record = _recognize(engine, warmup, capture_prediction=False)
        ready_queue.put(
            {
                "process_index": process_index,
                "success": warmup_record["success"],
                "load_seconds": load_seconds,
                "warmup_seconds": warmup_record["latency_seconds"],
            }
        )
    except Exception as error:
        ready_queue.put(
            {
                "process_index": process_index,
                "success": False,
                "failure_kind": type(error).__name__,
                "load_seconds": 0.0,
                "warmup_seconds": 0.0,
            }
        )
        return

    start_event.wait()
    capture = request["capture_predictions"]
    try:
        if request["phase"] in {"quality", "compatibility"}:
            assigned = items[process_index::process_count]
            for item in assigned:
                record = _recognize(engine, item, capture_prediction=capture)
                record["completed_offset_seconds"] = (
                    time.perf_counter() - start_time.value
                )
                result_queue.put({"kind": "record", "record": record})
        else:
            deadline = start_time.value + float(request["target_wall_seconds"])
            item_index = process_index
            consecutive_failures = 0
            while time.perf_counter() < deadline:
                item = items[item_index % len(items)]
                record = _recognize(engine, item, capture_prediction=capture)
                record["completed_offset_seconds"] = (
                    time.perf_counter() - start_time.value
                )
                result_queue.put({"kind": "record", "record": record})
                consecutive_failures = (
                    0 if record["success"] else consecutive_failures + 1
                )
                if consecutive_failures >= 3:
                    break
                item_index += process_count
    except BaseException as error:
        result_queue.put(
            {
                "kind": "worker_failed",
                "process_index": process_index,
                "failure_kind": type(error).__name__,
            }
        )
        return
    result_queue.put({"kind": "done", "process_index": process_index})


def _stop_processes(processes) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=30)


def _claim_process_index(
    message: dict,
    *,
    process_count: int,
    claimed,
    stage: str,
) -> int:
    if not isinstance(message, dict):
        raise RuntimeError(f"PaddleOCR worker {stage} message is invalid")
    process_index = message.get("process_index")
    if (
        type(process_index) is not int
        or not 0 <= process_index < process_count
        or process_index in claimed
    ):
        raise RuntimeError(f"PaddleOCR worker {stage} index is invalid")
    return process_index


def _configure_openmp_environment(
    config: dict,
    threads: int,
    environment: dict[str, str],
) -> None:
    """Apply the explicit per-process OpenMP settings before Paddle imports."""

    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    if "kmp_blocktime_ms" not in config:
        environment.pop("KMP_BLOCKTIME", None)
        return
    blocktime = config["kmp_blocktime_ms"]
    if type(blocktime) is not int:
        raise ValueError("kmp_blocktime_ms must be an integer in [0, 1000]")
    if not 0 <= blocktime <= 1000:
        raise ValueError("kmp_blocktime_ms must be in [0, 1000]")
    environment["KMP_BLOCKTIME"] = str(blocktime)


def _validated_opencv_threads(config: dict) -> int | None:
    if "opencv_threads" not in config:
        return None
    threads = config["opencv_threads"]
    if type(threads) is not int or not 1 <= threads <= 24:
        raise ValueError("opencv_threads must be an integer in [1, 24]")
    return threads


def _recognize(engine, item: dict, *, capture_prediction: bool) -> dict:
    started = time.perf_counter()
    try:
        outputs = list(engine.predict(item["path"]))
        lines = _extract_lines(outputs)
    except Exception as error:
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
    latency = time.perf_counter() - started
    output_is_valid = any(line["text"].strip() for line in lines) or not item.get(
        "expected_text",
        True,
    )
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": None if output_is_valid else "empty_output",
        "latency_seconds": latency,
        "units": 1.0 if output_is_valid else 0.0,
        "detected_line_count": len(lines),
        "output_character_count": sum(len(line["text"]) for line in lines),
    }
    if capture_prediction:
        record["lines"] = lines
    return record


def _extract_lines(outputs: list) -> list[dict]:
    lines = []
    for output in outputs:
        payload = output.json
        if callable(payload):
            payload = payload()
        result = payload.get("res", payload)
        texts = _empty_if_none(result.get("rec_texts"))
        scores = _empty_if_none(result.get("rec_scores"))
        polygons = _empty_if_none(result.get("rec_polys"))
        for index, text in enumerate(texts):
            polygon = polygons[index] if index < len(polygons) else None
            lines.append(
                {
                    "text": str(text),
                    "confidence": float(scores[index]) if index < len(scores) else None,
                    "polygon": (
                        polygon.tolist()
                        if hasattr(polygon, "tolist")
                        else polygon
                    ),
                }
            )
    return lines


def _empty_if_none(value):
    return [] if value is None else value


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
