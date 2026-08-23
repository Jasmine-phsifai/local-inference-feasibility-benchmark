import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _registry():
    return json.loads(
        (PROJECT_ROOT / "registries" / "sustained_candidates.json").read_text(
            encoding="utf-8"
        )
    )


def test_sustained_candidates_have_unique_configs_and_workers():
    for candidate in _registry()["candidates"]:
        assert (PROJECT_ROOT / candidate["worker"]).is_file(), candidate["id"]
        serialized = [json.dumps(config, sort_keys=True) for config in candidate["configs"]]
        assert len(serialized) == len(set(serialized)), candidate["id"]


def test_non_stress_candidates_do_not_exceed_visible_cpu_budget():
    for candidate in _registry()["candidates"]:
        if candidate["id"] == "sensevoice_small_gguf_cpu":
            continue
        for config in candidate["configs"]:
            processes = config["processes"]
            if candidate["task"] == "asr":
                budget = (
                    processes
                    * config["model_workers"]
                    * config["threads_per_worker"]
                )
            else:
                budget = processes * config["threads_per_process"]
            assert 1 <= budget <= 24, (candidate["id"], config)
