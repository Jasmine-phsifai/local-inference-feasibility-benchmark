import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def find_candidate(registry: dict, candidate_id: str) -> dict:
    for candidate in registry["candidates"]:
        if candidate["id"] == candidate_id:
            return candidate
    raise KeyError(f"Unknown candidate: {candidate_id}")
