"""Measure HF-native Qwen3-ASR on an explicit OpenVINO device."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import distribution, version
from pathlib import Path

try:
    from sustained_worker_metrics import build_public_summary, write_private_records
    from qwen3_asr_hf_openvino_export_manifest import verify_export
except ModuleNotFoundError:
    from workers.qwen3_asr_hf_openvino_export_manifest import verify_export
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


MODEL_REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"
OPTIMUM_INTEL_REVISION = "4ca1144eafc3ef7d3d805a99c7b92953441437e5"
EXPECTED_EOS_TOKEN_IDS = (151643, 151645)
EXPECTED_PAD_TOKEN_ID = 151645
MAX_REPEATED_TRIGRAM_RATIO = 0.95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    if int(config["processes"]) != 1:
        raise ValueError("HF-native OpenVINO Qwen3-ASR uses one resident process")
    device = str(config["device"]).upper()
    if device not in {"CPU", "GPU"}:
        raise ValueError("HF-native OpenVINO Qwen3-ASR requires CPU or GPU")
    threads = int(config["threads_per_process"])
    max_new_tokens = int(config["max_new_tokens"])

    import openvino
    import torch
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    project_root = Path(__file__).resolve().parents[1]
    source_model_dir = project_root / "data" / "models" / "qwen3-asr-0.6b-hf"
    openvino_model_dir = (
        project_root / "data" / "models" / "qwen3-asr-0.6b-hf-openvino-with-past"
    )
    if not (source_model_dir / "model.safetensors").is_file():
        raise FileNotFoundError("pinned HF-native Qwen3-ASR checkpoint is missing")
    verify_export(source_model_dir, openvino_model_dir)

    optimum_intel = distribution("optimum-intel")
    direct_url_text = optimum_intel.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    installed_revision = direct_url.get("vcs_info", {}).get("commit_id")
    if installed_revision != OPTIMUM_INTEL_REVISION:
        raise RuntimeError("HF-native Optimum Intel revision mismatch")

    core = openvino.Core()
    if device not in core.available_devices:
        raise RuntimeError(f"requested OpenVINO device is unavailable: {device}")
    cache_dir = Path(request["response_path"]).parent / "openvino-compile-cache"
    cache_started_empty = not cache_dir.exists() or not any(cache_dir.rglob("*"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded_at = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(source_model_dir),
        local_files_only=True,
    )
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        str(openvino_model_dir),
        local_files_only=True,
        device=device,
        ov_config=_openvino_config(device, threads, cache_dir),
        compile=True,
    )
    load_seconds = time.perf_counter() - loaded_at
    eos_token_ids, pad_token_id = _validated_generation_tokens(
        model.generation_config
    )

    warmup = _transcribe(
        processor,
        model,
        core,
        device,
        request["workload"]["warmup_item"],
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        capture_prediction=False,
        raise_on_error=True,
    )
    if not warmup["success"]:
        raise RuntimeError("HF-native OpenVINO Qwen3-ASR warmup produced empty output")
    execution_devices = _execution_devices(model)
    _validate_execution_devices(execution_devices, device)
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
                eos_token_ids=eos_token_ids,
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
                eos_token_ids=eos_token_ids,
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
        runtime_name="qwen3_asr_hf_openvino",
        runtime_version=(
            f"optimum-intel-{optimum_intel.version}+"
            f"openvino-{version('openvino')}+transformers-{version('transformers')}"
        ),
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
        "eos_token_ids": eos_token_ids,
        "pad_token_id": pad_token_id,
        "token_cap_hit_count": sum(
            record.get("token_cap_hit", False) for record in records
        ),
        "unhealthy_generation_count": sum(
            not record.get("generation_is_healthy", False) for record in records
        ),
        "terminal_eos_count": sum(
            record.get("terminal_is_eos", False) for record in records
        ),
        "max_token_run": max(
            (record.get("max_token_run", 0) for record in records),
            default=0,
        ),
        "minimum_unique_token_ratio": min(
            (record.get("unique_token_ratio", 1.0) for record in records),
            default=1.0,
        ),
        "maximum_repeated_trigram_ratio": max(
            (record.get("repeated_trigram_ratio", 0.0) for record in records),
            default=0.0,
        ),
        "timestamps_available": False,
        "hf_native_asr_request": True,
    }
    public_summary["model"] = {
        "model_revision": (
            f"{MODEL_REVISION}:optimum-intel:{OPTIMUM_INTEL_REVISION}:with-past"
        ),
        "compute_type": "official_notebook_with_past_default_ir",
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
    eos_token_ids: list[int],
    capture_prediction: bool,
    raise_on_error: bool = False,
) -> dict:
    started = time.perf_counter()
    try:
        inputs = processor.apply_transcription_request(audio=item["path"])
        inputs = inputs.to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        output_ids = _generated_sequences(output_ids)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        token_diagnostics = _token_diagnostics(generated_ids, eos_token_ids)
        text, language_available = _decode_parsed_asr_output(
            processor,
            generated_ids,
        )
        output_tokens = token_diagnostics["output_tokens"]
    except Exception as error:
        if raise_on_error:
            raise
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
            "accelerator_memory_bytes": _gpu_memory_snapshot_bytes(core, device),
        }
    latency = time.perf_counter() - started
    token_cap_hit = (
        output_tokens >= max_new_tokens and not token_diagnostics["terminal_is_eos"]
    )
    generation_is_healthy = _generation_is_healthy(
        token_diagnostics,
        token_cap_hit=token_cap_hit,
    )
    expects_speech = bool(item.get("expected_speech", True))
    content_is_valid = (
        language_available and bool(text)
        if expects_speech
        else True
    )
    output_is_valid = content_is_valid and generation_is_healthy
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": (
            None
            if output_is_valid
            else "degenerate_generation"
            if content_is_valid
            else "empty_output"
        ),
        "latency_seconds": latency,
        "units": float(item["duration_seconds"]) if output_is_valid else 0.0,
        "output_character_count": len(text),
        "language_available": language_available,
        "output_tokens": output_tokens,
        "token_cap_hit": token_cap_hit,
        "generation_is_healthy": generation_is_healthy,
        "accelerator_memory_bytes": _gpu_memory_snapshot_bytes(core, device),
        **token_diagnostics,
    }
    if capture_prediction:
        record["prediction"] = text
    return record


def _generated_sequences(output_ids):
    return output_ids.sequences if hasattr(output_ids, "sequences") else output_ids


def _decode_parsed_asr_output(processor, generated_ids) -> tuple[str, bool]:
    parsed = processor.decode(generated_ids, return_format="parsed")
    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise RuntimeError("HF-native Qwen3-ASR returned an invalid batch")
        parsed = parsed[0]
    if not isinstance(parsed, dict) or "transcription" not in parsed:
        raise RuntimeError("HF-native Qwen3-ASR returned an invalid parsed output")
    transcription = parsed["transcription"]
    if not isinstance(transcription, str):
        raise RuntimeError("HF-native Qwen3-ASR transcription is not text")
    language = parsed.get("language")
    language_available = isinstance(language, str) and bool(language.strip())
    return transcription.strip(), language_available


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


def _validated_generation_tokens(generation_config) -> tuple[list[int], int]:
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, (list, tuple)):
        eos_token_ids = [eos_token_ids]
    normalized_eos = tuple(int(token_id) for token_id in eos_token_ids)
    if (
        normalized_eos != EXPECTED_EOS_TOKEN_IDS
        or int(generation_config.pad_token_id) != EXPECTED_PAD_TOKEN_ID
    ):
        raise RuntimeError("HF-native Qwen3-ASR generation-token mismatch")
    return list(normalized_eos), EXPECTED_PAD_TOKEN_ID


def _token_diagnostics(generated_ids, eos_token_ids: list[int]) -> dict:
    tokens = [int(token) for token in generated_ids.detach().cpu().reshape(-1).tolist()]
    max_token_run = 0
    current_run = 0
    previous = None
    for token in tokens:
        current_run = current_run + 1 if token == previous else 1
        max_token_run = max(max_token_run, current_run)
        previous = token
    trigrams = [tuple(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    repeated_trigram_ratio = (
        (len(trigrams) - len(set(trigrams))) / len(trigrams)
        if trigrams
        else 0.0
    )
    return {
        "output_tokens": len(tokens),
        "terminal_is_eos": bool(tokens and tokens[-1] in set(eos_token_ids)),
        "max_token_run": max_token_run,
        "unique_token_ratio": len(set(tokens)) / len(tokens) if tokens else 1.0,
        "repeated_trigram_ratio": repeated_trigram_ratio,
    }


def _generation_is_healthy(
    diagnostics: dict,
    *,
    token_cap_hit: bool,
) -> bool:
    return bool(
        diagnostics.get("terminal_is_eos")
        and not token_cap_hit
        and int(diagnostics.get("max_token_run", 0)) <= 32
        and float(diagnostics.get("repeated_trigram_ratio", 1.0))
        <= MAX_REPEATED_TRIGRAM_RATIO
    )


def _gpu_memory_snapshot_bytes(core, device: str) -> int | None:
    if device != "GPU":
        return None
    try:
        statistics = core.get_property("GPU", "GPU_MEMORY_STATISTICS")
        return sum(int(value) for value in statistics.values())
    except Exception:
        return None


if __name__ == "__main__":
    main()
