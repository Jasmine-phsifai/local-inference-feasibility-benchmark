import argparse
import json
import time
import wave
from pathlib import Path

import ctranslate2
import faster_whisper
from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    audio = request["inputs"]["audio"][0]
    config = request["config"]
    started = time.perf_counter()
    model = WhisperModel("small", device="cpu", compute_type=config["compute_type"], cpu_threads=config["threads"], num_workers=config.get("workers", 1))
    load_seconds = time.perf_counter() - started
    list(model.transcribe(audio["path"], beam_size=1)[0])
    started = time.perf_counter()
    segments = list(model.transcribe(audio["path"], beam_size=1)[0])
    inference_seconds = time.perf_counter() - started
    text = " ".join(segment.text for segment in segments).strip()
    if not text:
        raise RuntimeError("faster-whisper produced no nonempty transcript")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-small"
    model_size_bytes = sum(path.stat().st_size for path in cache_root.rglob("*") if path.is_file() and not path.name.endswith(".incomplete"))
    response = {
        "runtime": {"name": "faster-whisper", "version": faster_whisper.__version__, "ctranslate2_version": ctranslate2.__version__, "compute_type": config["compute_type"], "threads": config["threads"]},
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "audio_seconds": audio["duration_seconds"],
        "model_size_bytes": model_size_bytes,
        "accelerator_memory_bytes": 0,
        "throughput": {"real_time_factor": inference_seconds / audio["duration_seconds"], "audio_hours_per_wall_hour": audio["duration_seconds"] / inference_seconds},
        "plausible_nonempty_output": bool(text),
        "output_preview": text[:500],
    }
    Path(request["response_path"]).write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
