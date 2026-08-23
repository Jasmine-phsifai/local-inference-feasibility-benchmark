"""Measure one resident native-Transformers Qwen3-ASR model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


MODEL_REVISION = "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    if int(config["processes"]) != 1:
        raise ValueError("native Qwen3-ASR quality uses one resident process")
    threads = int(config["threads_per_process"])
    max_new_tokens = int(config["max_new_tokens"])

    import torch
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "models"
        / "qwen3-asr-0.6b-hf"
    )
    if not model_dir.is_dir():
        raise FileNotFoundError("pinned Qwen3-ASR model is missing")
    loaded_at = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - loaded_at

    warmup = _transcribe(
        processor,
        model,
        request["workload"]["warmup_item"],
        max_new_tokens=max_new_tokens,
        capture_prediction=False,
    )
    if not warmup["success"]:
        raise RuntimeError("Qwen3-ASR warmup failed")
    started = time.perf_counter()
    records = []
    items = request["workload"]["items"]
    if request["phase"] in {"quality", "compatibility"}:
        for item in items:
            record = _transcribe(
                processor,
                model,
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
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="asr",
        runtime_name="qwen3_asr_native_transformers",
        runtime_version=transformers.__version__,
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[load_seconds],
        warmup_seconds=[warmup["latency_seconds"]],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="resident_model",
    )
    public_summary["generation"] = {
        "max_new_tokens": max_new_tokens,
        "token_cap_hit_count": sum(
            record.get("token_cap_hit", False) for record in records
        ),
        "timestamps_available": False,
    }
    public_summary["model"] = {
        "model_revision": MODEL_REVISION,
        "compute_type": "float32",
        "backend": "cpu",
        "threads": threads,
        "interop_threads": 1,
    }
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _transcribe(
    processor,
    model,
    item: dict,
    *,
    max_new_tokens: int,
    capture_prediction: bool,
) -> dict:
    import torch

    started = time.perf_counter()
    try:
        inputs = processor.apply_transcription_request(
            audio=item["path"]
        ).to(model.device, model.dtype)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = processor.decode(
            generated_ids,
            return_format="transcription_only",
        )
        text = decoded[0].strip()
        output_tokens = int(generated_ids.numel())
    except Exception as error:
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
    latency = time.perf_counter() - started
    output_is_valid = bool(text) or not item.get("expected_speech", True)
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": None if output_is_valid else "empty_output",
        "latency_seconds": latency,
        "units": float(item["duration_seconds"]) if output_is_valid else 0.0,
        "output_character_count": len(text),
        "output_tokens": output_tokens,
        "token_cap_hit": output_tokens >= max_new_tokens,
    }
    if capture_prediction:
        record["prediction"] = text
    return record


if __name__ == "__main__":
    main()
