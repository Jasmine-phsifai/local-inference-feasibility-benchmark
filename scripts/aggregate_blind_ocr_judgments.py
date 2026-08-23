"""Aggregate ignored blinded OCR judgments into privacy-safe evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from local_inference_bench.event_journal import append_event
from local_inference_bench.validate_public_summary import validate_public_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--judgment", required=True, action="append", type=Path)
    parser.add_argument("--append-journal", type=Path)
    args = parser.parse_args()
    event = aggregate_judgments(args.mapping, args.judgment)
    if args.append_journal is not None:
        append_event(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def aggregate_judgments(mapping_path: Path, judgment_paths: list[Path]) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if mapping.get("protocol") != "private-ocr-blind-v1":
        raise ValueError("unsupported blind mapping protocol")
    sample_mapping = {
        sample["sample_id"]: sample["identities"]
        for sample in mapping["samples"]
    }
    candidate_ids = list(mapping["candidate_ids"])
    judgments = [
        _load_judgment(path, mapping["packet_fingerprint"], set(sample_mapping))
        for path in judgment_paths
    ]
    if len(judgments) < 2:
        raise ValueError("at least two independent judgments are required")

    candidate_stats = {
        candidate_id: {"wins": 0, "severity_sum": 0, "usable": 0, "votes": 0}
        for candidate_id in candidate_ids
    }
    tie_votes = 0
    agreement_samples = 0
    consensus_wins = Counter()
    for sample_id, identities in sample_mapping.items():
        sample_votes = []
        for judgment in judgments:
            item = judgment[sample_id]
            winner = item["winner"]
            if winner == "tie":
                tie_votes += 1
                sample_votes.append("tie")
            else:
                candidate_id = identities[winner]
                candidate_stats[candidate_id]["wins"] += 1
                sample_votes.append(candidate_id)
            for label in ("A", "B"):
                candidate_id = identities[label]
                candidate_stats[candidate_id]["severity_sum"] += item[
                    f"{label.casefold()}_severity"
                ]
                candidate_stats[candidate_id]["usable"] += int(
                    item[f"{label.casefold()}_usable"]
                )
                candidate_stats[candidate_id]["votes"] += 1
        vote_counts = Counter(sample_votes)
        top_value, top_count = vote_counts.most_common(1)[0]
        if top_count >= 2:
            agreement_samples += 1
            if top_value != "tie":
                consensus_wins[top_value] += 1

    metrics = {
        "sample_count": len(sample_mapping),
        "judge_count": len(judgments),
        "vote_count": len(sample_mapping) * len(judgments),
        "tie_vote_count": tie_votes,
        "agreement_sample_fraction": agreement_samples / len(sample_mapping),
        "candidates": {
            candidate_id: {
                "win_votes": stats["wins"],
                "consensus_wins": consensus_wins[candidate_id],
                "mean_error_severity": stats["severity_sum"] / stats["votes"],
                "usable_vote_fraction": stats["usable"] / stats["votes"],
            }
            for candidate_id, stats in candidate_stats.items()
        },
    }
    return {
        "event": "blind_ocr_quality_scored",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "private_course_blind_ocr_comparison",
        "protocol": "private-ocr-blind-v1",
        "workload_class": "private_course",
        "judge_count": len(judgments),
        "judgment_set_fingerprint": _combined_fingerprint(judgment_paths),
        "metrics": validate_public_summary(metrics),
    }


def _load_judgment(
    path: Path,
    expected_fingerprint: str,
    expected_samples: set[str],
) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("protocol") != "private-ocr-blind-v1":
        raise ValueError("unsupported judgment protocol")
    fingerprint = document.get("packet_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or fingerprint.casefold() != expected_fingerprint.casefold()
    ):
        raise ValueError("judgment packet fingerprint does not match")
    result = {}
    for item in document.get("samples", []):
        sample_id = item.get("sample_id")
        if sample_id in result or sample_id not in expected_samples:
            raise ValueError("judgment contains an invalid sample ID")
        if item.get("winner") not in {"A", "B", "tie"}:
            raise ValueError("judgment winner must be A, B, or tie")
        for label in ("a", "b"):
            if item.get(f"{label}_severity") not in {0, 1, 2, 3}:
                raise ValueError("judgment severity must be in [0, 3]")
            if type(item.get(f"{label}_usable")) is not bool:
                raise ValueError("judgment usability must be boolean")
            codes = item.get(f"{label}_error_codes", [])
            if not isinstance(codes, list) or not set(codes) <= _ALLOWED_ERROR_CODES:
                raise ValueError("judgment contains an unsupported error code")
        result[sample_id] = item
    if set(result) != expected_samples:
        raise ValueError("judgment does not cover every blind sample")
    return result


def _combined_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(path.read_bytes())
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
