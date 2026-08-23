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
        assert (
            PROJECT_ROOT / candidate["environment_manifest"]
        ).is_file(), candidate["id"]
        if candidate.get("verify_script"):
            assert (PROJECT_ROOT / candidate["verify_script"]).is_file(), candidate["id"]
        for fingerprint_file in candidate.get("fingerprint_files", []):
            assert (PROJECT_ROOT / fingerprint_file).is_file(), candidate["id"]
        serialized = [json.dumps(config, sort_keys=True) for config in candidate["configs"]]
        assert len(serialized) == len(set(serialized)), candidate["id"]


def test_non_stress_candidates_do_not_exceed_visible_cpu_budget():
    for candidate in _registry()["candidates"]:
        if candidate["id"] == "sensevoice_small_gguf_cpu":
            continue
        for config in candidate["configs"]:
            processes = config["processes"]
            if candidate["task"] == "asr":
                if "model_workers" in config:
                    budget = (
                        processes
                        * config["model_workers"]
                        * config["threads_per_worker"]
                    )
                else:
                    budget = processes * config["threads_per_process"]
            else:
                budget = processes * config["threads_per_process"]
            assert 1 <= budget <= 24, (candidate["id"], config)


def test_native_hunyuan_attempts_bind_the_source_faithful_prompt() -> None:
    candidate = next(
        item
        for item in _registry()["candidates"]
        if item["id"] == "hunyuanocr_1_5_native_cpu"
    )

    assert "workers/build_source_faithful_ocr_prompt.py" in candidate[
        "fingerprint_files"
    ]
    source_faithful = next(
        config for config in candidate["configs"] if config["mode"] == "source_faithful"
    )
    assert source_faithful["max_items"] == 3
    assert "require_all_success" not in source_faithful
