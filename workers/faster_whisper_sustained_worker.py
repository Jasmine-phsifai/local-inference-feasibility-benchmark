"""Measure CTranslate2 shared workers and independent process shapes."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import multiprocessing
import os
import queue
import threading
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


_MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"


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
    _validated_language(request["config"])
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
    readiness_deadline = time.monotonic() + 1200.0
    while len(ready_by_index) < len(processes):
        remaining = readiness_deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("faster-whisper worker readiness timed out")
        try:
            message = ready_queue.get(timeout=min(5.0, remaining))
        except queue.Empty:
            if any(
                index not in ready_by_index and process.exitcode is not None
                for index, process in enumerate(processes)
            ):
                raise RuntimeError(
                    "faster-whisper worker exited before readiness"
                ) from None
            continue
        process_index = _claim_process_index(
            message,
            process_count=process_count,
            claimed=ready_by_index,
            stage="readiness",
        )
        if type(message.get("success")) is not bool:
            raise RuntimeError("faster-whisper worker readiness message is invalid")
        ready_by_index[process_index] = message
    ready = [ready_by_index[index] for index in range(process_count)]
    if any(not item["success"] for item in ready):
        raise RuntimeError("faster-whisper worker initialization failed")

    start_time.value = time.perf_counter()
    start_event.set()
    records = []
    concurrency_diagnostics = []
    done_process_indices: set[int] = set()
    result_deadline = (
        time.monotonic() + float(request["target_wall_seconds"]) + 1800.0
    )
    while len(done_process_indices) < process_count:
        remaining = result_deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("faster-whisper result collection timed out")
        try:
            message = result_queue.get(timeout=min(5.0, remaining))
        except queue.Empty:
            if any(
                process_index not in done_process_indices
                and process.exitcode is not None
                for process_index, process in enumerate(processes)
            ):
                raise RuntimeError(
                    "faster-whisper worker exited before completion"
                ) from None
            continue
        if not isinstance(message, dict):
            raise RuntimeError("faster-whisper worker result message is invalid")
        if message.get("kind") == "done":
            process_index = _claim_process_index(
                message,
                process_count=process_count,
                claimed=done_process_indices,
                stage="completion",
            )
            done_process_indices.add(process_index)
            if isinstance(message.get("concurrency"), dict):
                concurrency_diagnostics.append(message["concurrency"])
        elif message.get("kind") == "record":
            records.append(message["record"])
        else:
            raise RuntimeError("faster-whisper worker failed during inference")
    steady_wall_seconds = time.perf_counter() - start_time.value
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            raise RuntimeError(
                "faster-whisper worker did not exit after completion"
            )
        if process.exitcode != 0:
            raise RuntimeError("faster-whisper worker exit code was nonzero")
    if len(concurrency_diagnostics) != process_count:
        raise RuntimeError("faster-whisper concurrency diagnostics are incomplete")
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
    public_summary["concurrency"] = _aggregate_concurrency_diagnostics(
        diagnostics=concurrency_diagnostics,
        config=request["config"],
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
    language = _validated_language(config)
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
        warmup = request["workload"]["warmup_item"]
        with ThreadPoolExecutor(max_workers=model_workers) as executor:
            warmup_records = list(
                executor.map(
                    lambda _: _transcribe(
                        model,
                        warmup,
                        capture_prediction=False,
                        language=language,
                    ),
                    range(model_workers),
                )
            )
        runtime_num_workers = int(model.model.num_workers)
        ready_queue.put(
            {
                "process_index": process_index,
                "success": (
                    runtime_num_workers == model_workers
                    and all(record["success"] for record in warmup_records)
                ),
                "load_seconds": load_seconds,
                "warmup_seconds": [
                    record["latency_seconds"] for record in warmup_records
                ],
                "runtime_num_workers": runtime_num_workers,
            }
        )
    except Exception as error:
        ready_queue.put(
            {
                "process_index": process_index,
                "success": False,
                "failure_kind": type(error).__name__,
                "load_seconds": 0.0,
                "warmup_seconds": [],
            }
        )
        return

    start_event.wait()
    capture = request["capture_predictions"]
    call_probe = _PythonCallConcurrencyProbe()
    saturation = _new_saturation_state()
    stop_sampler = threading.Event()
    sampler = threading.Thread(
        target=_sample_ctranslate2_saturation,
        args=(model.model, model_workers, stop_sampler, saturation),
        name=f"ctranslate2-saturation-{process_index}",
        daemon=True,
    )
    sampler.start()
    try:
        if request["phase"] in {"quality", "compatibility"}:
            assigned = items[process_index::int(config["processes"])]
            with ThreadPoolExecutor(max_workers=model_workers) as executor:
                process_records = list(
                    executor.map(
                        lambda item: _transcribe_measured(
                            model,
                            item,
                            capture_prediction=capture,
                            start_time=start_time.value,
                            call_probe=call_probe,
                            language=language,
                        ),
                        assigned,
                    )
                )
            for record in process_records:
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
                            lambda item: _transcribe_measured(
                                model,
                                item,
                                capture_prediction=capture,
                                start_time=start_time.value,
                                call_probe=call_probe,
                                language=language,
                            ),
                            batch,
                        )
                    )
                    for record in process_records:
                        result_queue.put({"kind": "record", "record": record})
                    if sum(
                        not record["success"] for record in process_records
                    ) == len(process_records):
                        break
                    item_index += model_workers * int(config["processes"])
    except BaseException as error:
        result_queue.put(
            {
                "kind": "worker_failed",
                "process_index": process_index,
                "failure_kind": type(error).__name__,
            }
        )
        return
    finally:
        stop_sampler.set()
        sampler.join(timeout=5)
        if sampler.is_alive():
            raise RuntimeError("CTranslate2 saturation sampler did not stop")
    result_queue.put(
        {
            "kind": "done",
            "process_index": process_index,
            "concurrency": _process_concurrency_diagnostics(
                runtime_num_workers=int(model.model.num_workers),
                python_peak=call_probe.peak,
                saturation=saturation,
            ),
        }
    )


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
        raise RuntimeError(f"faster-whisper worker {stage} message is invalid")
    process_index = message.get("process_index")
    if (
        type(process_index) is not int
        or not 0 <= process_index < process_count
        or process_index in claimed
    ):
        raise RuntimeError(f"faster-whisper worker {stage} index is invalid")
    return process_index


class _PythonCallConcurrencyProbe:
    """Track concurrent Python calls submitted to one shared Whisper model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)

    def exit(self) -> None:
        with self._lock:
            self._in_flight -= 1


def _transcribe_measured(
    model,
    item: dict,
    *,
    capture_prediction: bool,
    start_time: float,
    call_probe: _PythonCallConcurrencyProbe,
    language: str | None,
) -> dict:
    call_probe.enter()
    try:
        record = _transcribe(
            model,
            item,
            capture_prediction=capture_prediction,
            language=language,
        )
        record["completed_offset_seconds"] = time.perf_counter() - start_time
        return record
    finally:
        call_probe.exit()


def _new_saturation_state() -> dict[str, int]:
    return {
        "sample_count": 0,
        "busy_sample_count": 0,
        "fully_busy_sample_count": 0,
        "active_batches_peak": 0,
        "queued_batches_peak": 0,
        "processing_batches_peak": 0,
        "discarded_sample_count": 0,
        "sampler_failure_count": 0,
    }


def _sample_ctranslate2_saturation(
    runtime,
    configured_workers: int,
    stop_event: threading.Event,
    state: dict[str, int],
) -> None:
    while not stop_event.is_set():
        try:
            active_before = int(runtime.num_active_batches)
            queued = int(runtime.num_queued_batches)
            active = int(runtime.num_active_batches)
            if active_before != active or queued > active:
                state["discarded_sample_count"] += 1
                stop_event.wait(0.02)
                continue
            processing = max(0, active - queued)
            state["sample_count"] += 1
            state["busy_sample_count"] += int(active > 0)
            state["fully_busy_sample_count"] += int(
                processing >= configured_workers
            )
            state["active_batches_peak"] = max(
                state["active_batches_peak"], active
            )
            state["queued_batches_peak"] = max(
                state["queued_batches_peak"], queued
            )
            state["processing_batches_peak"] = max(
                state["processing_batches_peak"], processing
            )
        except Exception:
            state["sampler_failure_count"] += 1
        stop_event.wait(0.02)


def _process_concurrency_diagnostics(
    *,
    runtime_num_workers: int,
    python_peak: int,
    saturation: dict[str, int],
) -> dict:
    return {
        "runtime_num_workers": runtime_num_workers,
        "python_calls_in_flight_peak": python_peak,
        **saturation,
    }


def _aggregate_concurrency_diagnostics(
    *,
    diagnostics: list[dict],
    config: dict,
) -> dict:
    configured_processes = int(config["processes"])
    configured_workers = int(config["model_workers"])
    busy_samples = sum(int(item["busy_sample_count"]) for item in diagnostics)
    fully_busy_samples = sum(
        int(item["fully_busy_sample_count"]) for item in diagnostics
    )
    runtime_workers = [int(item["runtime_num_workers"]) for item in diagnostics]
    python_peaks = [
        int(item["python_calls_in_flight_peak"]) for item in diagnostics
    ]
    processing_peaks = [
        int(item["processing_batches_peak"]) for item in diagnostics
    ]
    sample_counts = [int(item["sample_count"]) for item in diagnostics]
    busy_counts = [int(item["busy_sample_count"]) for item in diagnostics]
    return {
        "configured_processes": configured_processes,
        "configured_model_workers_per_process": configured_workers,
        "configured_total_model_workers": configured_processes * configured_workers,
        "instrumented_process_count": len(diagnostics),
        "runtime_model_workers_min": min(runtime_workers, default=0),
        "runtime_model_workers_max": max(runtime_workers, default=0),
        "python_calls_in_flight_peak_per_process": max(
            python_peaks,
            default=0,
        ),
        "python_calls_in_flight_peak_min_per_process": min(
            python_peaks,
            default=0,
        ),
        "ctranslate2_active_batches_peak_per_process": max(
            (int(item["active_batches_peak"]) for item in diagnostics),
            default=0,
        ),
        "ctranslate2_queued_batches_peak_per_process": max(
            (int(item["queued_batches_peak"]) for item in diagnostics),
            default=0,
        ),
        "ctranslate2_processing_batches_peak_per_process": max(
            processing_peaks,
            default=0,
        ),
        "ctranslate2_processing_batches_peak_min_per_process": min(
            processing_peaks,
            default=0,
        ),
        "ctranslate2_sampler_sample_count": sum(
            sample_counts
        ),
        "ctranslate2_sampler_sample_count_min_per_process": min(
            sample_counts,
            default=0,
        ),
        "ctranslate2_busy_sample_count": busy_samples,
        "ctranslate2_busy_sample_count_min_per_process": min(
            busy_counts,
            default=0,
        ),
        "ctranslate2_fully_busy_sample_count": fully_busy_samples,
        "ctranslate2_fully_busy_fraction_when_busy": (
            fully_busy_samples / busy_samples if busy_samples else 0.0
        ),
        "ctranslate2_sampler_failure_count": sum(
            int(item["sampler_failure_count"]) for item in diagnostics
        ),
        "ctranslate2_discarded_sample_count": sum(
            int(item["discarded_sample_count"]) for item in diagnostics
        ),
    }


def _validated_language(config: dict) -> str | None:
    if "language" not in config:
        return None
    language = config["language"]
    if language != "zh":
        raise ValueError("language must be 'zh' when configured")
    return language


def _transcribe(
    model,
    item: dict,
    *,
    capture_prediction: bool,
    language: str | None,
) -> dict:
    started = time.perf_counter()
    try:
        transcribe_options = {
            "beam_size": 1,
            "temperature": 0.0,
            "vad_filter": False,
            "word_timestamps": True,
            "condition_on_previous_text": True,
        }
        if language is not None:
            transcribe_options["language"] = language
        segments_iterator, info = model.transcribe(
            item["path"],
            **transcribe_options,
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
