import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from paddleocr import PaddleOCR


MODEL_NAMES = {
    "ppocrv6_tiny_cpu": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
    "ppocrv6_medium_cpu": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    candidate_id = request["candidate"]["id"]
    detector, recognizer = MODEL_NAMES[candidate_id]
    config = request["config"]
    images = [item["path"] for item in request["inputs"]["images"]]
    started = time.perf_counter()
    engine = PaddleOCR(
        device="cpu",
        text_detection_model_name=detector,
        text_recognition_model_name=recognizer,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        cpu_threads=config["threads"],
    )
    load_seconds = time.perf_counter() - started
    list(engine.predict(images[0]))
    started = time.perf_counter()
    batch_size = int(config.get("batch_size", 1))
    if batch_size > 1:
        outputs = list(engine.predict(images))
    else:
        outputs = [list(engine.predict(image)) for image in images]
    inference_seconds = time.perf_counter() - started
    previews = [str(output)[:1000] for output in outputs]
    if not any(preview.strip() for preview in previews):
        raise RuntimeError("PaddleOCR produced no nonempty output")
    model_root = Path.home() / ".paddlex" / "official_models"
    model_size_bytes = sum(
        path.stat().st_size
        for model_name in (detector, recognizer)
        for path in (model_root / model_name).rglob("*")
        if path.is_file()
    )
    response = {
        "runtime": {"name": "paddleocr", "version": importlib.metadata.version("paddleocr"), "threads": config["threads"], "input_batch_size": batch_size},
        "model": {"detector": detector, "recognizer": recognizer},
        "load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "input_count": len(images),
        "model_size_bytes": model_size_bytes,
        "accelerator_memory_bytes": 0,
        "throughput": {"seconds_per_image": inference_seconds / len(images), "images_per_hour": len(images) / inference_seconds * 3600},
        "plausible_nonempty_output": True,
        "output_preview": previews[:2],
    }
    Path(request["response_path"]).write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
