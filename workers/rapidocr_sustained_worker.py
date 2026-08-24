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
    _validated_opencv_threads(request["config"])
    backend = _backend_name(request["config"])
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
        ready_by_index = {}
        readiness_deadline = time.monotonic() + 600.0
        while len(ready_by_index) < process_count:
            remaining = readiness_deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("RapidOCR worker readiness timed out")
            try:
                message = ready_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                if any(
                    index not in ready_by_index and process.exitcode is not None
                    for index, process in enumerate(processes)
                ):
                    raise RuntimeError(
                        "RapidOCR worker exited before readiness"
                    ) from None
                continue
            process_index = _claim_process_index(
                message,
                process_count=process_count,
                claimed=ready_by_index,
                stage="readiness",
            )
            ready_by_index[process_index] = message
        ready = [ready_by_index[index] for index in range(process_count)]
        if any(not item["success"] for item in ready):
            raise RuntimeError("RapidOCR worker initialization failed")
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
                raise RuntimeError("RapidOCR result collection timed out")
            try:
                message = result_queue.get(timeout=min(5.0, remaining))
            except queue.Empty:
                if any(
                    index not in done_process_indices
                    and process.exitcode is not None
                    for index, process in enumerate(processes)
                ):
                    raise RuntimeError(
                        "RapidOCR worker exited before completion"
                    ) from None
                continue
            if not isinstance(message, dict):
                raise RuntimeError("RapidOCR worker result message is invalid")
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
                raise RuntimeError("RapidOCR worker failed during inference")
        steady_wall_seconds = time.perf_counter() - start_time.value
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                raise RuntimeError("RapidOCR worker did not exit after completion")
            if process.exitcode != 0:
                raise RuntimeError("RapidOCR worker exit code was nonzero")
        write_private_records(Path(request["private_records_path"]), records)
        public_summary = build_public_summary(
            candidate_id=request["candidate_id"],
            task="ocr",
            runtime_name="rapidocr",
            runtime_version=(
                f"rapidocr-{importlib.metadata.version('rapidocr')}+"
                f"{backend}-{importlib.metadata.version(backend)}"
            ),
            workload_class=request["workload"]["workload_class"],
            records=records,
            load_seconds=[item["load_seconds"] for item in ready],
            warmup_seconds=[item["warmup_seconds"] for item in ready],
            steady_wall_seconds=steady_wall_seconds,
            target_wall_seconds=float(request["target_wall_seconds"]),
            load_semantics="resident_model",
        )
        config = request["config"]
        public_summary["preprocessing"] = {
            "classifier_enabled": bool(config.get("use_cls", True)),
            "max_side_len": int(config.get("max_side_len", 2000)),
            "use_preprocess_img": True,
            "use_vertical_padding": True,
            "det_limit_side_len": 736,
            "det_use_dilation": True,
            "rec_batch_num": 6,
            "cls_batch_num": 6,
        }
        opencv_threads = _validated_opencv_threads(config)
        if opencv_threads is not None:
            public_summary["preprocessing"]["opencv_threads"] = opencv_threads
        public_summary["postprocessing"] = {
            "line_score_threshold": 0.5,
            "det_threshold": 0.3,
            "det_box_threshold": 0.5,
            "det_unclip_ratio": 1.6,
            "cls_threshold": 0.9,
        }
        public_summary["model"] = {
            "backend": backend,
            "device": "CPU",
            "model_revision": "rapidocr-v3.9.2-ppocrv6-small",
        }
        Path(request["response_path"]).write_text(
            json.dumps({"public_summary": public_summary}, indent=2),
            encoding="utf-8",
        )
    finally:
        _stop_processes(started_processes)


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
        opencv_threads = _validated_opencv_threads(config)
        if opencv_threads is not None:
            import cv2

            cv2.setNumThreads(opencv_threads)
        from rapidocr import EngineType, RapidOCR

        loaded_at = time.perf_counter()
        engine = RapidOCR(
            params=_engine_params(
                config,
                threads,
                engine_type=EngineType(_backend_name(config)),
            )
        )
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
        raise RuntimeError(f"RapidOCR worker {stage} message is invalid")
    process_index = message.get("process_index")
    if (
        type(process_index) is not int
        or not 0 <= process_index < process_count
        or process_index in claimed
    ):
        raise RuntimeError(f"RapidOCR worker {stage} index is invalid")
    return process_index


def _backend_name(config: dict) -> str:
    backend = str(config.get("backend", "onnxruntime")).casefold()
    if backend not in {"onnxruntime", "openvino"}:
        raise ValueError(f"unsupported RapidOCR backend: {backend}")
    return backend


def _validated_opencv_threads(config: dict) -> int | None:
    if "opencv_threads" not in config:
        return None
    threads = config["opencv_threads"]
    if type(threads) is not int or not 1 <= threads <= 24:
        raise ValueError("opencv_threads must be an integer in [1, 24]")
    return threads


def _engine_params(config: dict, threads: int, *, engine_type: object) -> dict:
    backend = _backend_name(config)
    params = {
        "Global.log_level": "critical",
        "Det.engine_type": engine_type,
        "Cls.engine_type": engine_type,
        "Rec.engine_type": engine_type,
    }
    if backend == "onnxruntime":
        params.update(
            {
                "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
    else:
        params.update(
            {
                "EngineConfig.openvino.inference_num_threads": threads,
                "EngineConfig.openvino.performance_hint": "LATENCY",
                "EngineConfig.openvino.num_streams": 1,
            }
        )
    if "use_cls" in config:
        params["Global.use_cls"] = bool(config["use_cls"])
    if "max_side_len" in config:
        params["Global.max_side_len"] = int(config["max_side_len"])
    return params


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
    output_is_valid = any(str(text).strip() for text in texts) or not item.get(
        "expected_text",
        True,
    )
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
