"""Measure the official OpenVINO GenAI Qwen3-ASR pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import wave
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
NATIVE_MINIMUM_CHUNK_SECONDS = 0.5
NATIVE_INTERNAL_CHUNK_SECONDS = 1200.0
BENCHMARK_ITEM_LIMIT_SECONDS = 7200.0
MEL_HOP_SAMPLES = 160
ENCODER_CHUNK_FRAMES = 100
TOKENS_PER_FULL_CHUNK = 13
STABLE_ENVIRONMENT_NAME = "local-bench-qwen3-asr-openvino-genai-official"
STABLE_PRODUCT_VERSION = "2026.3.0.0-3277-bd8d6542e3c"
STABLE_RUNTIME_SOURCE_REVISION = (
    "bd8d6542e3ca1ac30042d5d8d4202ce00b5f4af0"
)
STABLE_PACKAGE_VERSIONS = {
    "openvino": "2026.3.0",
    "openvino-genai": "2026.3.0.0",
    "openvino-tokenizers": "2026.3.0.0",
}


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
    runtime_identity = _expected_runtime_identity(config, project_root)
    _verify_installed_runtime(runtime_identity, openvino_genai)
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
            f"openvino-genai-{runtime_identity['product_version']}+"
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
        "post_release_tail_chunk_fix_present": runtime_identity[
            "post_release_tail_chunk_fix_present"
        ],
        "runtime_identity_verified": True,
        "runtime_profile_applied": config.get("runtime_profile") is not None,
        "geometry_observation_count": sum(
            type(record.get("encoder_remainder_frames")) is int
            for record in records
        ),
        "encoder_remainder_frames_observed": sorted(
            {
                int(record["encoder_remainder_frames"])
                for record in records
                if type(record.get("encoder_remainder_frames")) is int
            }
        ),
        "tail_fix_sensitive_item_count": sum(
            record.get("tail_fix_sensitive_geometry") is True
            for record in records
        ),
        "official_with_past_export": True,
    }
    public_summary["model"] = {
        "model_revision": (
            f"{SOURCE_REVISION}:optimum-intel:{EXPORTER_REVISION}:with-past:"
            f"ovgenai:{runtime_identity['associated_source_revision']}"
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
        tail_geometry = _single_native_chunk_tail_geometry(
            len(audio),
            duration_seconds=actual_duration_seconds,
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
        **tail_geometry,
    }
    if capture_prediction:
        record["prediction"] = text
    return record


def _read_audio(
    path: Path,
    *,
    declared_duration_seconds: float,
):
    try:
        with wave.open(str(path), "rb") as stream:
            if (
                stream.getnchannels() != 1
                or stream.getsampwidth() != 2
                or stream.getframerate() != 16_000
                or stream.getcomptype() != "NONE"
            ):
                raise ValueError("Qwen3-ASR input must be PCM16 mono 16 kHz WAV")
            frame_count = stream.getnframes()
            frames = stream.readframes(frame_count)
            if len(frames) != frame_count * 2 or stream.readframes(1):
                raise ValueError("Qwen3-ASR WAV frame data is inconsistent")
    except (OSError, EOFError, wave.Error) as error:
        raise ValueError("Qwen3-ASR input must be a valid WAV") from error
    if frame_count <= 0:
        raise ValueError("Qwen3-ASR input WAV must contain audio frames")
    actual_duration_seconds = frame_count / 16_000
    if actual_duration_seconds > BENCHMARK_ITEM_LIMIT_SECONDS:
        raise ValueError("Qwen3-ASR benchmark item exceeds the 7200-second limit")
    if not math.isclose(
        actual_duration_seconds,
        declared_duration_seconds,
        rel_tol=0.0,
        abs_tol=1.0 / 16_000,
    ):
        raise ValueError("Qwen3-ASR declared duration does not match the WAV")
    import numpy

    audio = numpy.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if len(audio) != frame_count:
        raise ValueError("Qwen3-ASR decoded audio does not match its header")
    return audio, actual_duration_seconds


def _mel_frame_count(sample_count: int) -> int:
    """Match the official extractor's floor(sample_count / 160) geometry."""

    if type(sample_count) is not int or sample_count <= 0:
        raise ValueError("Qwen3-ASR sample count must be a positive integer")
    return sample_count // MEL_HOP_SAMPLES


def _tail_output_token_counts(remainder_frames: int) -> tuple[int, int]:
    if (
        type(remainder_frames) is not int
        or not 0 <= remainder_frames < ENCODER_CHUNK_FRAMES
    ):
        raise ValueError("Qwen3-ASR remainder frames must be in [0, 100)")
    if remainder_frames == 0:
        return 0, 0
    legacy = (
        remainder_frames * TOKENS_PER_FULL_CHUNK
        + ENCODER_CHUNK_FRAMES
        - 1
    ) // ENCODER_CHUNK_FRAMES
    fixed = remainder_frames
    for _ in range(3):
        fixed = (fixed + 1) // 2
    return legacy, fixed


def _single_native_chunk_tail_geometry(
    sample_count: int,
    *,
    duration_seconds: float,
) -> dict[str, int | bool]:
    """Report whole-input geometry only when the runtime keeps one native chunk."""

    if (
        not math.isfinite(duration_seconds)
        or duration_seconds < NATIVE_MINIMUM_CHUNK_SECONDS
        or duration_seconds > NATIVE_INTERNAL_CHUNK_SECONDS
    ):
        return {}
    mel_frame_count = _mel_frame_count(sample_count)
    encoder_remainder_frames = mel_frame_count % ENCODER_CHUNK_FRAMES
    legacy_tail_tokens, fixed_tail_tokens = _tail_output_token_counts(
        encoder_remainder_frames
    )
    return {
        "mel_frame_count": mel_frame_count,
        "encoder_remainder_frames": encoder_remainder_frames,
        "tail_fix_sensitive_geometry": legacy_tail_tokens != fixed_tail_tokens,
    }


def _expected_runtime_identity(config: dict, project_root: Path) -> dict:
    base_config_keys = {
        "processes",
        "device",
        "threads_per_process",
        "max_new_tokens",
    }
    runtime_profile = config.get("runtime_profile")
    if runtime_profile is None:
        if set(config) != base_config_keys:
            raise ValueError("Qwen3-ASR stable runtime config changed")
        return {
            "profile_id": "openvino_genai_stable_2026_3_0",
            "environment_name": STABLE_ENVIRONMENT_NAME,
            "package_versions": STABLE_PACKAGE_VERSIONS,
            "product_version": STABLE_PRODUCT_VERSION,
            "associated_source_revision": STABLE_RUNTIME_SOURCE_REVISION,
            "post_release_tail_chunk_fix_present": False,
        }

    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from local_inference_bench.qwen3_asr_tailfix_profile import (
        TAILFIX_PROFILE_RELATIVE_PATH,
        load_qwen3_asr_tailfix_profile,
    )

    if (
        set(config) != base_config_keys | {"runtime_profile"}
        or runtime_profile != TAILFIX_PROFILE_RELATIVE_PATH.as_posix()
    ):
        raise ValueError("Qwen3-ASR runtime profile is not pinned")
    profile_path = (project_root / runtime_profile).resolve(strict=True)
    if not profile_path.is_relative_to(project_root.resolve(strict=True)):
        raise ValueError("Qwen3-ASR runtime profile escaped the project root")
    profile = load_qwen3_asr_tailfix_profile(profile_path)
    return {
        "profile_id": profile["profile_id"],
        "environment_name": profile["environment_name"],
        "package_versions": profile["package_versions"],
        "product_version": profile["openvino_genai_product_version"],
        "associated_source_revision": profile["associated_source_revision"],
        "post_release_tail_chunk_fix_present": _tail_fix_presence_from_identity(
            profile["openvino_genai_product_version"],
            profile["associated_source_revision"],
        ),
    }


def _tail_fix_presence_from_identity(
    product_version: str,
    associated_source_revision: str,
) -> bool:
    known = {
        (STABLE_PRODUCT_VERSION, STABLE_RUNTIME_SOURCE_REVISION): False,
        (
            "2026.4.0.0-3387-98ae8c32197",
            "98ae8c32197d1afe88ebaff89968283493c25786",
        ): True,
    }
    try:
        return known[(product_version, associated_source_revision)]
    except KeyError as error:
        raise RuntimeError("Qwen3-ASR runtime source identity is not pinned") from error


def _verify_installed_runtime(identity: dict, openvino_genai) -> None:
    if Path(sys.prefix).name != identity["environment_name"]:
        raise RuntimeError("Qwen3-ASR runtime environment changed")
    for package, expected_version in identity["package_versions"].items():
        if version(package) != expected_version:
            raise RuntimeError("Qwen3-ASR runtime package version changed")
    product_version = identity["product_version"]
    if (
        openvino_genai.__version__ != product_version
        or openvino_genai.get_version() != product_version
    ):
        raise RuntimeError("Qwen3-ASR OpenVINO GenAI ProductVersion changed")
    derived_fix_presence = _tail_fix_presence_from_identity(
        product_version,
        identity["associated_source_revision"],
    )
    if derived_fix_presence != identity["post_release_tail_chunk_fix_present"]:
        raise RuntimeError("Qwen3-ASR tail-fix runtime identity is inconsistent")


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
