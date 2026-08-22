import argparse
import json
import os
import time
from pathlib import Path


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def result_text(result: object) -> str:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    generated_blocks = []

    def collect_generated_text(value: object) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "block_content" and isinstance(child, str):
                    generated_blocks.append(child)
                else:
                    collect_generated_text(child)
        elif isinstance(value, list):
            for child in value:
                collect_generated_text(child)

    collect_generated_text(payload)
    if generated_blocks:
        return "\n".join(generated_blocks)
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False, default=str)
    return str(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    threads = int(request["config"]["threads"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    from paddleocr import PaddleOCRVL

    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "data" / "models" / "paddleocr-vl-1.6"
    started = time.perf_counter()
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_model_dir=str(model_dir),
        vl_rec_backend="native",
        device="cpu",
        cpu_threads=threads,
        enable_mkldnn=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
    )
    load_seconds = time.perf_counter() - started
    image = request["inputs"]["images"][0]["path"]

    def invoke() -> tuple[float, str]:
        begin = time.perf_counter()
        outputs = list(
            pipeline.predict(
                image,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=False,
                max_new_tokens=128,
                temperature=0.0,
            )
        )
        elapsed = time.perf_counter() - begin
        return elapsed, "\n".join(result_text(item) for item in outputs)

    warmup_seconds, _ = invoke()
    inference_seconds, text = invoke()
    if not text.strip():
        raise RuntimeError("PaddleOCR-VL produced no output")

    home = Path.home()
    cache_candidates = [
        model_dir,
        home / ".paddlex" / "official_models" / "PaddleOCR-VL-1.6",
        home / ".cache" / "huggingface" / "hub" / "models--PaddlePaddle--PaddleOCR-VL-1.6",
    ]
    model_size = sum(directory_size(path) for path in cache_candidates)
    estimated_tokens = max(1, len(text) / 4.0)
    tokens_per_second = estimated_tokens / inference_seconds
    cutoff = float(request.get("generation_stop_tokens_per_second", 1.0))
    response = {
        "runtime": {
            "name": "PaddleOCRVL native",
            "backend": "cpu",
            "threads": threads,
            "generation_limit": 128,
        },
        "model": {"name": "PaddlePaddle/PaddleOCR-VL-1.6", "parameters": "0.9B"},
        "model_size_bytes": model_size or None,
        "accelerator_memory_bytes": 0,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "inference_seconds": inference_seconds,
        "input_count": 1,
        "throughput": {
            "images_per_hour": 3600.0 / inference_seconds,
            "estimated_output_tokens_per_second": tokens_per_second,
        },
        "generation_cutoff": {
            "minimum_tokens_per_second": cutoff,
            "below_cutoff": tokens_per_second <= cutoff,
            "measurement": "Approximation from recognized block-content length divided by four; runtime exposes no token counter.",
        },
        "below_generation_cutoff": tokens_per_second <= cutoff,
        "plausible_nonempty_output": True,
        "output_preview": text[:1500],
    }
    Path(request["response_path"]).write_text(
        json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
