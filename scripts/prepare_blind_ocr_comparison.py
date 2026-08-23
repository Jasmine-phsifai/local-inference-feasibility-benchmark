"""Create an ignored, candidate-blinded OCR judging packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        help="Exactly two NAME=private-records.jsonl values.",
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    record_paths = _parse_candidate_specs(args.candidate)
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    records = {
        candidate_id: _read_records(path)
        for candidate_id, path in record_paths.items()
    }
    packet, mapping = build_blind_packet(
        workload,
        records,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "packet.json"
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mapping["packet_fingerprint"] = _sha256(packet_path)
    (args.output_dir / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(packet_path)


def build_blind_packet(
    workload: dict,
    records: dict[str, dict[str, dict]],
    *,
    sample_count: int,
    seed: int,
) -> tuple[dict, dict]:
    if workload.get("task") != "ocr":
        raise ValueError("blind comparison requires an OCR workload")
    if workload.get("workload_class") != "private_course":
        raise ValueError("blind comparison is restricted to ignored private-course data")
    if len(records) != 2:
        raise ValueError("blind comparison requires exactly two candidates")
    items = workload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("workload requires nonempty items")
    if not 1 <= sample_count <= len(items):
        raise ValueError("sample_count is outside the workload")

    candidate_ids = sorted(records)
    rng = random.Random(seed)
    selected = rng.sample(items, sample_count)
    packet_samples = []
    mapping_samples = []
    for index, item in enumerate(selected, start=1):
        sample_id = item.get("id")
        image_path = item.get("path")
        if not isinstance(sample_id, str) or not isinstance(image_path, str):
            raise ValueError("workload item is missing an opaque ID or path")
        labels = ["A", "B"]
        randomized_candidates = list(candidate_ids)
        rng.shuffle(randomized_candidates)
        blind_id = f"blind_{index:03d}"
        options = {}
        identities = {}
        for label, candidate_id in zip(labels, randomized_candidates, strict=True):
            record = records[candidate_id].get(sample_id)
            if record is None:
                raise ValueError(f"candidate is missing workload sample: {sample_id}")
            lines = record.get("lines", [])
            options[label] = [str(line.get("text", "")) for line in lines]
            identities[label] = candidate_id
        packet_samples.append(
            {
                "sample_id": blind_id,
                "image_path": image_path,
                "options": options,
            }
        )
        mapping_samples.append(
            {
                "sample_id": blind_id,
                "source_id": sample_id,
                "identities": identities,
            }
        )
    return (
        {
            "schema_version": 1,
            "protocol": "private-ocr-blind-v1",
            "instructions": {
                "winner_values": ["A", "B", "tie"],
                "severity_scale": [0, 1, 2, 3],
                "allowed_error_codes": sorted(_ALLOWED_ERROR_CODES),
            },
            "samples": packet_samples,
        },
        {
            "schema_version": 1,
            "protocol": "private-ocr-blind-v1",
            "candidate_ids": candidate_ids,
            "samples": mapping_samples,
        },
    )


def _parse_candidate_specs(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        candidate_id, separator, raw_path = value.partition("=")
        if not separator or not candidate_id or not raw_path or candidate_id in result:
            raise ValueError("candidate values must be unique NAME=PATH pairs")
        result[candidate_id] = Path(raw_path)
    if len(result) != 2:
        raise ValueError("exactly two candidate values are required")
    return result


def _read_records(path: Path) -> dict[str, dict]:
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or sample_id in records:
                raise ValueError(f"invalid record at line {line_number}")
            records[sample_id] = record
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ALLOWED_ERROR_CODES = {
    "false_positive",
    "formula_or_code",
    "garble",
    "missing_text",
    "reading_order",
    "small_text",
}


if __name__ == "__main__":
    main()
