import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    audio = request["inputs"]["audio"][0]
    threads = int(request["config"]["threads"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model_dir = Path(__file__).resolve().parents[1] / "data" / "models" / "qwen3-asr-0.6b-hf"
    if not model_dir.exists():
        raise FileNotFoundError(f"Pinned Qwen3-ASR model is missing: {model_dir}")
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float32,
        local_files_only=True,
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    inputs = processor.apply_transcription_request(audio=audio["path"]).to(model.device, model.dtype)
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=256, do_sample=False)
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    inference_seconds = time.perf_counter() - started
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated_ids, return_format="transcription_only")[0].strip()
    if not text:
        raise RuntimeError("Qwen3-ASR produced no nonempty transcript")
    output_tokens = int(generated_ids.numel())
    token_count_method = "generated token IDs"
    tokens_per_second = output_tokens / inference_seconds
    cutoff = float(request.get("generation_stop_tokens_per_second", 1.0))
    response = {
        "runtime": {"name": "transformers", "version": transformers.__version__, "torch": torch.__version__, "threads": threads, "interop_threads": 1, "mkldnn_available": torch.backends.mkldnn.is_available(), "cpu_capability": torch.backends.cpu.get_cpu_capability()},
        "model": {"name": "Qwen/Qwen3-ASR-0.6B-hf", "revision": "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c", "dtype": "float32", "device": "cpu"},
        "model_size_bytes": sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()),
        "accelerator_memory_bytes": 0,
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "audio_seconds": audio["duration_seconds"],
        "output_tokens": output_tokens,
        "token_count_method": token_count_method,
        "tokens_per_second": tokens_per_second,
        "below_generation_cutoff": tokens_per_second <= cutoff,
        "generation_cutoff_tokens_per_second": cutoff,
        "throughput": {"real_time_factor": inference_seconds / audio["duration_seconds"], "audio_hours_per_wall_hour": audio["duration_seconds"] / inference_seconds, "tokens_per_second": tokens_per_second},
        "plausible_nonempty_output": True,
        "output_preview": text[:500],
    }
    Path(request["response_path"]).write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
