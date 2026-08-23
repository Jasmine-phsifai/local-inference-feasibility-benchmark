"""Measure PaddleOCR-VL with native document structure enabled."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import time
from pathlib import Path

try:
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    if int(config["processes"]) != 1:
        raise ValueError("structured PaddleOCR-VL quality uses one resident process")
    if request["phase"] not in {"quality", "compatibility"}:
        raise ValueError("structured PaddleOCR-VL worker is quality-only")
    threads = int(config["threads_per_process"])
    max_new_tokens = int(config["max_new_tokens"])
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    from paddleocr import PaddleOCRVL

    model_dir = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "models"
        / "paddleocr-vl-1.6"
    )
    loaded_at = time.perf_counter()
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_model_dir=str(model_dir),
        vl_rec_backend="native",
        device="cpu",
        cpu_threads=threads,
        enable_mkldnn=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=True,
    )
    load_seconds = time.perf_counter() - loaded_at
    warmup = _recognize(
        pipeline,
        request["workload"]["warmup_item"],
        max_new_tokens=max_new_tokens,
        capture_prediction=False,
    )
    if not warmup["success"]:
        raise RuntimeError("PaddleOCR-VL structured warmup failed")

    started = time.perf_counter()
    records = []
    for item in request["workload"]["items"]:
        record = _recognize(
            pipeline,
            item,
            max_new_tokens=max_new_tokens,
            capture_prediction=bool(request["capture_predictions"]),
        )
        record["completed_offset_seconds"] = time.perf_counter() - started
        records.append(record)
    steady_wall_seconds = time.perf_counter() - started
    write_private_records(Path(request["private_records_path"]), records)
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="ocr",
        runtime_name="paddleocr_vl_1_6_native_structured",
        runtime_version=importlib.metadata.version("paddleocr"),
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[load_seconds],
        warmup_seconds=[warmup["latency_seconds"]],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="resident_model",
    )
    public_summary["structured_output"] = {
        "layout_detection": True,
        "max_new_tokens": max_new_tokens,
        "block_count": sum(record.get("block_count", 0) for record in records),
        "formula_block_count": sum(
            record.get("formula_block_count", 0) for record in records
        ),
        "table_block_count": sum(
            record.get("table_block_count", 0) for record in records
        ),
        "confidence_available": False,
    }
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _recognize(
    pipeline,
    item: dict,
    *,
    max_new_tokens: int,
    capture_prediction: bool,
) -> dict:
    started = time.perf_counter()
    try:
        outputs = list(
            pipeline.predict(
                item["path"],
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=True,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
        )
        payloads = [_payload(output) for output in outputs]
        blocks = _extract_blocks(payloads)
        lines = _blocks_to_lines(blocks)
    except Exception as error:
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
    latency = time.perf_counter() - started
    output_is_valid = bool(lines) or not item.get("expected_text", True)
    labels = [block["label"].casefold() for block in blocks]
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": None if output_is_valid else "empty_output",
        "latency_seconds": latency,
        "units": 1.0 if output_is_valid else 0.0,
        "block_count": len(blocks),
        "formula_block_count": sum("formula" in label for label in labels),
        "table_block_count": sum("table" in label for label in labels),
        "output_character_count": sum(len(line["text"]) for line in lines),
    }
    if capture_prediction:
        record["lines"] = lines
        record["blocks"] = blocks
        record["raw_outputs"] = payloads
    return record


def _payload(output: object) -> object:
    payload = getattr(output, "json", output)
    if callable(payload):
        payload = payload()
    return _json_safe(payload)


def _extract_blocks(payloads: list[object]) -> list[dict]:
    blocks = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            parsing = value.get("parsing_res_list")
            if isinstance(parsing, list):
                for raw_block in parsing:
                    if not isinstance(raw_block, dict):
                        continue
                    blocks.append(
                        {
                            "label": str(raw_block.get("block_label", "")),
                            "content": str(raw_block.get("block_content", "")),
                            "bbox": _json_safe(raw_block.get("block_bbox")),
                            "block_id": raw_block.get("block_id"),
                            "block_order": raw_block.get("block_order"),
                        }
                    )
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payloads)
    return sorted(
        blocks,
        key=lambda block: (
            block["block_order"]
            if isinstance(block["block_order"], (int, float))
            else float("inf"),
            block["block_id"]
            if isinstance(block["block_id"], (int, float))
            else float("inf"),
        ),
    )


def _blocks_to_lines(blocks: list[dict]) -> list[dict]:
    lines = []
    for block in blocks:
        for line in block["content"].splitlines():
            if not line.strip():
                continue
            lines.append(
                {
                    "text": line,
                    "label": block["label"],
                    "bbox": block["bbox"],
                    "block_order": block["block_order"],
                }
            )
    return lines


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


if __name__ == "__main__":
    main()
