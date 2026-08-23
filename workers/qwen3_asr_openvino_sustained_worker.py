"""Measure one resident Qwen3-ASR model on an explicit OpenVINO device."""

from __future__ import annotations

import argparse
import json
import re
import time
from importlib.metadata import version
from pathlib import Path

try:
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


MODEL_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    if int(config["processes"]) != 1:
        raise ValueError("OpenVINO Qwen3-ASR uses one resident process")
    device = str(config["device"]).upper()
    if device not in {"CPU", "GPU"}:
        raise ValueError("OpenVINO Qwen3-ASR requires explicit CPU or GPU")
    threads = int(config["threads_per_process"])
    max_new_tokens = int(config["max_new_tokens"])

    import openvino
    import qwen_asr  # noqa: F401 - registers the original Qwen model type
    import torch
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoConfig, AutoProcessor, GenerationConfig

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    project_root = Path(__file__).resolve().parents[1]
    source_model_dir = project_root / "data" / "models" / "qwen3-asr-0.6b-original"
    openvino_model_dir = project_root / "data" / "models" / "qwen3-asr-0.6b-openvino-fp16"
    if not (source_model_dir / "model.safetensors").is_file():
        raise FileNotFoundError("pinned original Qwen3-ASR checkpoint is missing")
    if not (openvino_model_dir / "export-complete.json").is_file():
        raise FileNotFoundError("verified Qwen3-ASR OpenVINO export is missing")

    core = openvino.Core()
    if device not in core.available_devices:
        raise RuntimeError(f"requested OpenVINO device is unavailable: {device}")
    cache_dir = Path(request["response_path"]).parent / "openvino-compile-cache"
    cache_started_empty = not cache_dir.exists() or not any(cache_dir.rglob("*"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded_at = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(source_model_dir),
        trust_remote_code=True,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    generation_config = GenerationConfig.from_pretrained(
        str(source_model_dir),
        local_files_only=True,
    )
    if not generation_config.do_sample:
        generation_config.temperature = None
    model_config = AutoConfig.from_pretrained(
        str(openvino_model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    generation_config.decoder_start_token_id = model_config.decoder_start_token_id
    if generation_config.decoder_start_token_id is None:
        raise RuntimeError("exported Qwen3-ASR decoder start token is missing")
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        str(openvino_model_dir),
        trust_remote_code=True,
        local_files_only=True,
        device=device,
        ov_config=_openvino_config(device, threads, cache_dir),
        config=model_config,
        generation_config=generation_config,
        compile=True,
    )
    load_seconds = time.perf_counter() - loaded_at
    execution_devices = _execution_devices(model)
    _validate_execution_devices(execution_devices, device)

    warmup = _transcribe(
        processor,
        model,
        core,
        device,
        request["workload"]["warmup_item"],
        max_new_tokens=max_new_tokens,
        capture_prediction=False,
        raise_on_error=True,
    )
    if not warmup["success"]:
        raise RuntimeError("OpenVINO Qwen3-ASR warmup failed")
    started = time.perf_counter()
    records = []
    items = request["workload"]["items"]
    if request["phase"] in {"quality", "compatibility"}:
        for item in items:
            record = _transcribe(
                processor,
                model,
                core,
                device,
                item,
                max_new_tokens=max_new_tokens,
                capture_prediction=bool(request["capture_predictions"]),
            )
            record["completed_offset_seconds"] = time.perf_counter() - started
            records.append(record)
    else:
        deadline = started + float(request["target_wall_seconds"])
        item_index = 0
        consecutive_failures = 0
        while time.perf_counter() < deadline:
            record = _transcribe(
                processor,
                model,
                core,
                device,
                items[item_index % len(items)],
                max_new_tokens=max_new_tokens,
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
    runtime_version = (
        f"optimum-intel-{version('optimum-intel')}+"
        f"openvino-{version('openvino')}"
    )
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="asr",
        runtime_name="qwen3_asr_openvino",
        runtime_version=runtime_version,
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[load_seconds],
        warmup_seconds=[warmup["latency_seconds"]],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="resident_model_cold_compile_cache",
    )
    public_summary["generation"] = {
        "max_new_tokens": max_new_tokens,
        "token_cap_hit_count": sum(
            record.get("token_cap_hit", False) for record in records
        ),
        "timestamps_available": False,
        "eos_token_id": generation_config.eos_token_id,
        "pad_token_id": generation_config.pad_token_id,
        "decoder_start_token_id": generation_config.decoder_start_token_id,
        "source_generation_config": True,
    }
    public_summary["model"] = {
        "model_revision": MODEL_REVISION,
        "compute_type": "fp16_ir",
        "backend": "openvino",
        "device": device,
        "device_name": core.get_property(device, "FULL_DEVICE_NAME"),
        "execution_devices": sorted(
            {
                execution_device
                for component_devices in execution_devices.values()
                for execution_device in component_devices
            }
        ),
        "compiled_component_count": len(execution_devices),
        "host_threads": threads,
        "interop_threads": 1,
        "model_size_bytes": sum(
            path.stat().st_size
            for path in openvino_model_dir.rglob("*")
            if path.is_file()
        ),
    }
    public_summary["compile_cache"] = {
        "started_empty": cache_started_empty,
        "entry_count_after_load": sum(1 for path in cache_dir.rglob("*") if path.is_file()),
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
        "snapshot_available": bool(memory_snapshots) and device == "GPU",
        "attributed_to_compiled_model": False,
    }
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _openvino_config(device: str, threads: int, cache_dir: Path) -> dict[str, object]:
    config: dict[str, object] = {
        "CACHE_DIR": str(cache_dir),
        "PERFORMANCE_HINT": "LATENCY",
        "NUM_STREAMS": "1",
    }
    if device == "CPU":
        config["INFERENCE_NUM_THREADS"] = threads
    return config


def _transcribe(
    processor,
    model,
    core,
    device: str,
    item: dict,
    *,
    max_new_tokens: int,
    capture_prediction: bool,
    raise_on_error: bool = False,
) -> dict:
    import librosa
    import numpy as np
    import soundfile

    started = time.perf_counter()
    try:
        audio, sample_rate = soundfile.read(
            item["path"],
            dtype="float32",
            always_2d=False,
        )
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        if sample_rate != 16000:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=16000,
            )
            sample_rate = 16000
        prompt = processor.apply_chat_template(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=prompt,
            audio=audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        output_ids = model.generate(
            **_generation_kwargs(inputs, max_new_tokens=max_new_tokens)
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        raw_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        text = _extract_transcription(raw_text)
        output_tokens = int(generated_ids.numel())
    except Exception as error:
        if raise_on_error:
            raise
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
            "accelerator_memory_bytes": _gpu_memory_bytes(core, device),
        }
    latency = time.perf_counter() - started
    token_cap_hit = output_tokens >= max_new_tokens
    content_is_valid = bool(text) or not item.get("expected_speech", True)
    output_is_valid = content_is_valid and not token_cap_hit
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": (
            None
            if output_is_valid
            else "token_cap_hit"
            if token_cap_hit
            else "empty_output"
        ),
        "latency_seconds": latency,
        "units": float(item["duration_seconds"]) if output_is_valid else 0.0,
        "output_character_count": len(text),
        "output_tokens": output_tokens,
        "token_cap_hit": token_cap_hit,
        "accelerator_memory_bytes": _gpu_memory_bytes(core, device),
    }
    if capture_prediction:
        record["prediction"] = text
    return record


def _extract_transcription(raw_text: str) -> str:
    normalized = raw_text.replace("<|asr_text|>", "<asr_text>")
    match = re.search(r"<asr_text>(.*?)(?:<\||$)", normalized, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"^language\s+\S+", "", normalized).strip()


def _generation_kwargs(inputs, *, max_new_tokens: int) -> dict[str, object]:
    """Build the generation call used by the merged Optimum Intel adapter test."""
    return {
        "input_features": inputs["input_features"],
        "attention_mask": inputs.get("feature_attention_mask"),
        "decoder_input_ids": inputs["input_ids"],
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }


def _execution_devices(model) -> dict[str, list[str]]:
    result = {}
    component_names = tuple(getattr(model, "_component_names", ()))
    if not component_names:
        raise RuntimeError("OpenVINO model did not declare compiled components")
    for component_name in component_names:
        component = getattr(model, component_name, None)
        if component is None or getattr(component, "request", None) is None:
            raise RuntimeError(f"OpenVINO component is not compiled: {component_name}")
        compiled_model = component.request
        if not hasattr(compiled_model, "get_property"):
            compiled_model = compiled_model.get_compiled_model()
        devices = compiled_model.get_property("EXECUTION_DEVICES")
        result[component_name] = [str(device) for device in devices]
    return result


def _validate_execution_devices(
    execution_devices: dict[str, list[str]],
    requested_device: str,
) -> None:
    if not execution_devices:
        raise RuntimeError("OpenVINO execution-device evidence is empty")
    for component_name, component_devices in execution_devices.items():
        if not component_devices or any(
            not name.upper().startswith(requested_device)
            for name in component_devices
        ):
            raise RuntimeError(
                f"OpenVINO component did not execute only on requested "
                f"{requested_device}: {component_name}"
            )


def _gpu_memory_bytes(core, device: str) -> int | None:
    if device != "GPU":
        return None
    try:
        statistics = core.get_property("GPU", "GPU_MEMORY_STATISTICS")
        return sum(int(value) for value in statistics.values())
    except Exception:
        return None


if __name__ == "__main__":
    main()
