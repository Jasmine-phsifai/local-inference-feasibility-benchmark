import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runnable_candidates_have_workers_and_configs():
    registry = json.loads((PROJECT_ROOT / "registries" / "candidates.json").read_text(encoding="utf-8"))
    for candidate in registry["candidates"]:
        assert candidate["configs"], candidate["id"]
        if candidate["status"] in {"enabled", "planned"}:
            assert (PROJECT_ROOT / candidate["worker"]).is_file(), candidate["id"]


def test_candidate_ids_are_unique():
    registry = json.loads((PROJECT_ROOT / "registries" / "candidates.json").read_text(encoding="utf-8"))
    candidate_ids = [candidate["id"] for candidate in registry["candidates"]]
    assert len(candidate_ids) == len(set(candidate_ids))
