import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]
    backend = request["config"].get("backend", "cpu")
    runtime_name = "sensevoice-vulkan" if backend == "vulkan" else "sensevoice"
    runtime_root = project_root / "data" / "models" / runtime_name
    binary = runtime_root / "llama-funasr-sensevoice.exe"
    model = project_root / "data" / "models" / "sensevoice" / "sensevoice-small-q8.gguf"
    if not binary.exists() or not model.exists():
        raise FileNotFoundError(f"SenseVoice runtime/model missing: {binary}, {model}")
    environment = os.environ.copy()
    environment["PATH"] = f"D:/Anaconda/Library/bin;{environment['PATH']}"
    environment["OMP_NUM_THREADS"] = str(request["config"]["threads"])
    audio = request["inputs"]["audio"][0]
    command = [str(binary), "-m", str(model), "-a", audio["path"], "--backend", backend]

    def invoke() -> tuple[float, str, str]:
        started = time.perf_counter()
        result = subprocess.run(command, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        elapsed = time.perf_counter() - started
        if result.returncode != 0:
            raise RuntimeError(f"SenseVoice exited {result.returncode}: {result.stderr[-2000:]}")
        return elapsed, result.stdout, result.stderr

    warmup_seconds, _, _ = invoke()
    inference_seconds, stdout, stderr = invoke()
    combined = (stdout + "\n" + stderr).strip()
    if not combined:
        raise RuntimeError("SenseVoice produced no output")
    response = {
        "runtime": {"name": "FunASR llama.cpp SenseVoice", "release": "runtime-llamacpp-v0.1.9", "backend": backend, "threads": request["config"]["threads"], "intel_openmp_path_dependency": True},
        "model": {"name": "FunAudioLLM/SenseVoiceSmall-GGUF", "revision": "90c1c61912018b70ada0fcc024ea24aca62f2e63", "quantization": "q8"},
        "model_size_bytes": model.stat().st_size,
        "accelerator_memory_bytes": None if backend == "vulkan" else 0,
        "warmup_end_to_end_seconds": warmup_seconds,
        "load_seconds": None,
        "inference_seconds": inference_seconds,
        "audio_seconds": audio["duration_seconds"],
        "throughput": {"real_time_factor": inference_seconds / audio["duration_seconds"], "audio_hours_per_wall_hour": audio["duration_seconds"] / inference_seconds},
        "plausible_nonempty_output": True,
        "output_preview": combined[:1000],
        "limitation": "CLI process startup and model load are included in each measured file; runtime did not expose load-only timing separately.",
    }
    Path(request["response_path"]).write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
