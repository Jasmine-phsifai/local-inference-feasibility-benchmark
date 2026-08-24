"""Build ignored hard-image subset manifests from deterministic OCR controls."""

from __future__ import annotations

import copy
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALITY_ROOT = PROJECT_ROOT / "data" / "inputs" / "generated" / "ocr_quality"
SOURCE_MANIFEST = QUALITY_ROOT / "manifest.json"
SUBSETS = {
    "paddle_vl_hard_quality.json": (
        "code_formula",
        "handwriting_board",
        "dense_table",
        "negative_diagram",
    ),
    "hunyuan_thread_screen.json": ("code_formula",),
    "hunyuan_doc_quality.json": (
        "code_formula",
        "dense_table",
        "negative_diagram",
    ),
    "hunyuan_structured_negative.json": ("negative_diagram",),
}


def main() -> None:
    document = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    for filename, sample_ids in SUBSETS.items():
        subset = build_subset(document, sample_ids)
        destination = QUALITY_ROOT / filename
        destination.write_text(
            json.dumps(subset, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(destination)


def build_subset(document: dict, sample_ids: tuple[str, ...]) -> dict:
    items = {item["id"]: item for item in document["items"]}
    references = document["references"]
    missing = set(sample_ids) - set(items)
    if missing or not set(sample_ids) <= set(references):
        raise ValueError(f"unknown OCR quality sample IDs: {sorted(missing)}")
    subset = {
        "schema_version": document["schema_version"],
        "task": "ocr",
        "workload_class": document["workload_class"],
        "disclosure": document.get("disclosure", ""),
        "warmup": copy.deepcopy(document["warmup"]),
        "items": [copy.deepcopy(items[sample_id]) for sample_id in sample_ids],
        "references": {
            sample_id: copy.deepcopy(references[sample_id])
            for sample_id in sample_ids
        },
        "subset": {
            "source": "deterministic_ocr_quality_controls",
            "sample_count": len(sample_ids),
        },
    }
    if "generator" in document:
        subset["generator"] = copy.deepcopy(document["generator"])
    return subset


if __name__ == "__main__":
    main()
