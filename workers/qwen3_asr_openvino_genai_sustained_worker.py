"""Measure the official OpenVINO GenAI Qwen3-ASR pipeline."""

from __future__ import annotations

import argparse
import json
import math
import time
from importlib.metadata import version
from pathlib import Path

try:
    from qwen3_asr_openvino_genai_export_manifest import verify_export
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.qwen3_asr_openvino_genai_export_manifest import verify_export
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


SOURCE_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
EXPORTER_REVISION = "f48d93fddff8c91e198389c47a6d5974789b67f4"
NATIVE_INTERNAL_CHUNK_SECONDS = 1200.0
BENCHMARK_ITEM_LIMIT_SECONDS = 7200.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    if int(config["processes"]) != 1:
        raise ValueError("OpenVINO GenAI Qwen3-ASR uses one resident process")
    device = str(config["device"]).upper()
    if device not in {"CPU", "GPU.0"}:
        raise ValueError("OpenVINO GenAI Qwen3-ASR requires CPU or GPU.0")
    threads = int(config["threads_per_process"])
    max_new_tokens = int(config["max_new_tokens"])
    capture_predictions = request["capture_predictions"]
    if type(capture_predictions) is not bool:
        raise ValueError("capture_predictions must be boolean")
    if capture_predictions and request["phase"] not in {"quality", "compatibility"}:
        raise ValueError("predictions may be captured only in a private quality phase")

    import openvino
    import openvino_genai

    project_root = Path(__file__).resolve().parents[1]
    source_model = project_root / "data/models/qwen3-asr-0.6b-original"
    export_model = (
        project_root
        / "data/models/qwen3-asr-0.6b-openvino-genai-official-f48d93f"
    )
    export_summary = verify_export(source_model, export_model)
    core = openvino.Core()
    available_devices = list(core.available_devices)
    if device == "CPU" and "CPU" not in available_devices:
        raise RuntimeError("requested OpenVINO CPU device is unavailable")
    if device.startswith("GPU") and not any(
        candidate.startswith("GPU") for candidate in available_devices
    ):
        raise RuntimeError("requested OpenVINO GPU device is unavailable")
    cache_dir = Path(request["response_path"]).parent / "openvino-compile-cache"
    cache_started_empty = not cache_dir.exists() or not any(cache_dir.rglob("*"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    properties: dict[str, object] = {
        "CACHE_DIR": str(cache_dir),
        "PERFORMANCE_HINT": "LATENCY",
    }
    if device == "CPU":
        properties["INFERENCE_NUM_THREADS"] = threads
    loaded_at = time.perf_counter()
    pipeline = openvino_genai.ASRPipeline(
        str(export_model),
        device,
        **properties,
    )
    load_seconds = time.perf_counter() - loaded_at
    generation_config = pipeline.get_generation_config()
    generation_config.max_new_tokens = max_new_tokens
    generation_config.return_timestamps = False

    warmup = _transcribe(
        pipeline,
        generation_config,
        core,
        device,
        request["workload"]["warmup_item"],
        capture_prediction=False,
        raise_on_error=True,
    )
    if not warmup["success"]:
        raise RuntimeError("OpenVINO GenAI Qwen3-ASR warmup produced empty output")

    started = time.perf_counter()
    records = []
    items = request["workload"]["items"]
    if request["phase"] in {"quality", "compatibility"}:
        for item in items:
            record = _transcribe(
                pipeline,
                generation_config,
                core,
                device,
                item,
                capture_prediction=capture_predictions,
            )
            record["completed_offset_seconds"] = time.perf_counter() - started
            records.append(record)
    else:
        deadline = started + float(request["target_wall_seconds"])
        item_index = 0
        consecutive_failures = 0
        while time.perf_counter() < deadline:
            record = _transcribe(
                pipeline,
                generation_config,
                core,
                device,
                items[item_index % len(items)],
                capture_prediction=False,
            )
            record["completed_offset_seconds"] = time.perf_counter() - started
            records.append(record)
            consecutive_failures = (
                0 if record["success"] else consecutive_failures + 1
            )
            if consecutive_failures >= 3:
                break
            item_index += 1
    steady_wall_seconds = time.perf_counter() - started

    write_private_records(Path(request["private_records_path"]), records)
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="asr",
        runtime_name="openvino_genai_asr_pipeline",
        runtime_version=(
            f"openvino-genai-{version('openvino-genai')}+"
            f"openvino-{version('openvino')}"
        ),
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[load_seconds],
        warmup_seconds=[warmup["latency_seconds"]],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="export_hash_verified_then_resident_compile",
    )
    public_summary["generation"] = {
        "max_new_tokens": max_new_tokens,
        "token_cap_hit_count": sum(
            bool(record.get("token_cap_hit")) for record in records
        ),
        "chunks_returned_count": sum(
            int(record.get("chunks_returned", 0)) for record in records
        ),
        "timestamps_available": False,
        "native_internal_chunk_seconds": NATIVE_INTERNAL_CHUNK_SECONDS,
        "benchmark_item_limit_seconds": BENCHMARK_ITEM_LIMIT_SECONDS,
        "native_long_audio_chunking": True,
        "post_release_tail_chunk_fix_present": False,
        "official_with_past_export": True,
    }
    public_summary["model"] = {
        "model_revision": (
            f"{SOURCE_REVISION}:optimum-intel:{EXPORTER_REVISION}:with-past"
        ),
        "compute_type": "fp16_ir",
        "backend": "openvino_genai",
        "device": device,
        "device_name": core.get_property(device, "FULL_DEVICE_NAME"),
        "requested_host_threads": threads,
        "cpu_inference_threads_applied": device == "CPU",
        "model_size_bytes": int(export_summary["total_bytes"]),
        "beam_idx_present": bool(export_summary["beam_idx_present"]),
        "stateful_decoder": bool(export_summary["stateful_decoder"]),
    }
    public_summary["compile_cache"] = {
        "started_empty": cache_started_empty,
        "entry_count_after_load": sum(
            1 for path in cache_dir.rglob("*") if path.is_file()
        ),
    }
    memory_snapshots = [
        value
        for value in [warmup.get("accelerator_memory_bytes")]
        + [record.get("accelerator_memory_bytes") for record in records]
        if isinstance(value, int)
    ]
    public_summary["accelerator_memory"] = {
        "maximum_observed_current_allocation_bytes": max(
            memory_snapshots,
            default=0,
        ),
        "snapshot_available": bool(memory_snapshots) and device.startswith("GPU"),
        "attributed_to_compiled_model": False,
    }
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _transcribe(
    pipeline,
    generation_config,
    core,
    device: str,
    item: dict,
    *,
    capture_prediction: bool,
    raise_on_error: bool = False,
) -> dict:
    started = time.perf_counter()
    try:
        audio, actual_duration_seconds = _read_audio(
            Path(item["path"]),
            declared_duration_seconds=float(item["duration_seconds"]),
        )
        result = pipeline.generate(audio, generation_config)
        text = result.texts[0].strip() if result.texts else ""
        output_tokens = int(result.perf_metrics.get_num_generated_tokens())
        chunks = result.chunks
        chunks_returned = len(chunks) if chunks is not None else 0
        accelerator_memory = _gpu_memory_snapshot_bytes(core, device)
    except Exception as error:
        if raise_on_error:
            raise
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
    latency = time.perf_counter() - started
    expects_speech = bool(item.get("expected_speech", True))
    output_is_valid = bool(text) or not expects_speech
    token_cap_hit = output_tokens >= int(generation_config.max_new_tokens)
    record = {
        "sample_id": item["id"],
        "success": output_is_valid and not token_cap_hit,
        "failure_kind": (
            None
            if output_is_valid and not token_cap_hit
            else "token_cap" if token_cap_hit else "empty_output"
        ),
        "latency_seconds": latency,
        "units": (
            actual_duration_seconds
            if output_is_valid and not token_cap_hit
            else 0.0
        ),
        "output_character_count": len(text),
        "output_tokens": output_tokens,
        "token_cap_hit": token_cap_hit,
        "chunks_returned": chunks_returned,
        "accelerator_memory_bytes": accelerator_memory,
        "perf_generate_milliseconds": _metric_mean(
            result.perf_metrics.get_generate_duration()
        ),
        "perf_inference_milliseconds": _metric_mean(
            result.perf_metrics.get_inference_duration()
        ),
    }
    if capture_prediction:
        record["prediction"] = text
    return record


def _read_audio(
    path: Path,
    *,
    declared_duration_seconds: float,
):
    import soundfile

    info = soundfile.info(str(path))
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.samplerate != 16000
        or info.channels != 1
    ):
        raise ValueError("Qwen3-ASR input must be PCM16 mono 16 kHz WAV")
    actual_duration_seconds = info.frames / info.samplerate
    if actual_duration_seconds > BENCHMARK_ITEM_LIMIT_SECONDS:
        raise ValueError("Qwen3-ASR benchmark item exceeds the 7200-second limit")
    if not math.isclose(
        actual_duration_seconds,
        declared_duration_seconds,
        rel_tol=0.0,
        abs_tol=1.0 / info.samplerate,
    ):
        raise ValueError("Qwen3-ASR declared duration does not match the WAV")
    audio, sample_rate = soundfile.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if sample_rate != info.samplerate or len(audio) != info.frames:
        raise ValueError("Qwen3-ASR decoded audio does not match its header")
    return audio, actual_duration_seconds


def _metric_mean(value) -> float:
    mean = getattr(value, "mean", value)
    return float(mean)


def _gpu_memory_snapshot_bytes(core, device: str) -> int | None:
    if not device.startswith("GPU"):
        return None
    try:
        statistics = core.get_property(device, "GPU_MEMORY_STATISTICS")
        return sum(int(value) for value in statistics.values())
    except Exception:
        return None


if __name__ == "__main__":
    main()
