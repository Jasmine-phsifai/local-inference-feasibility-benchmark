import argparse
import importlib.metadata
import importlib.util
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rapidocr import RapidOCR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    images = [item["path"] for item in request["inputs"]["images"]]
    workers = int(request["config"].get("workers", 1))
    threads = int(request["config"]["threads"])
    started = time.perf_counter()
    engines = [RapidOCR(params={"EngineConfig.onnxruntime.intra_op_num_threads": threads}) for _ in range(workers)]
    load_seconds = time.perf_counter() - started
    for engine in engines:
        engine(images[0])
    benchmark_images = images if workers == 1 else images * 8

    def run_partition(worker_index: int) -> list:
        engine = engines[worker_index]
        return [engine(image) for image in benchmark_images[worker_index::workers]]

    started = time.perf_counter()
    if workers == 1:
        outputs = run_partition(0)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partitions = list(pool.map(run_partition, range(workers)))
        outputs = [output for partition in partitions for output in partition]
    inference_seconds = time.perf_counter() - started
    texts = [str(output.txts) for output in outputs]
    if not any(text.strip() for text in texts):
        raise RuntimeError("RapidOCR produced no nonempty output")
    package_root = Path(importlib.util.find_spec("rapidocr").origin).parent
    model_size_bytes = sum(path.stat().st_size for path in (package_root / "models").rglob("*") if path.is_file())
    response = {
        "runtime": {"name": "rapidocr", "version": importlib.metadata.version("rapidocr"), "threads_per_worker": threads, "workers": workers},
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "input_count": len(benchmark_images),
        "model_size_bytes": model_size_bytes,
        "accelerator_memory_bytes": 0,
        "throughput": {"seconds_per_image": inference_seconds / len(benchmark_images), "images_per_hour": len(benchmark_images) / inference_seconds * 3600},
        "plausible_nonempty_output": True,
        "output_preview": texts[:2],
        "pid": os.getpid(),
    }
    Path(request["response_path"]).write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
