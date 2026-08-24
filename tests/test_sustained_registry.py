import json
import re
from pathlib import Path
from pathlib import PurePosixPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
ALLOWED_PHASES = {"screen", "sustained", "quality", "compatibility"}


def _registry():
    return json.loads(
        (PROJECT_ROOT / "registries" / "sustained_candidates.json").read_text(
            encoding="utf-8"
        )
    )


def test_sustained_candidates_have_unique_configs_and_workers():
    candidates = _registry()["candidates"]
    candidate_ids = [candidate["id"] for candidate in candidates]
    assert len(candidate_ids) == len(set(candidate_ids))
    for candidate in candidates:
        assert PUBLIC_ID.fullmatch(candidate["id"])
        assert (PROJECT_ROOT / candidate["worker"]).is_file(), candidate["id"]
        assert (
            PROJECT_ROOT / candidate["environment_manifest"]
        ).is_file(), candidate["id"]
        if candidate.get("verify_script"):
            assert (PROJECT_ROOT / candidate["verify_script"]).is_file(), candidate["id"]
        if candidate.get("setup_script"):
            assert (PROJECT_ROOT / candidate["setup_script"]).is_file(), candidate["id"]
        for fingerprint_file in candidate.get("fingerprint_files", []):
            assert (PROJECT_ROOT / fingerprint_file).is_file(), candidate["id"]
        serialized = [json.dumps(config, sort_keys=True) for config in candidate["configs"]]
        assert len(serialized) == len(set(serialized)), candidate["id"]


def test_every_active_candidate_has_exact_setup_and_verification_policy() -> None:
    for candidate in _registry()["candidates"]:
        if str(candidate.get("status", "")).startswith("retired"):
            continue
        assert candidate.get("setup_script"), candidate["id"]
        assert candidate.get("verify_script"), candidate["id"]
        manifest = candidate["environment_manifest"]
        assert (
            manifest.endswith("requirements.lock.txt")
            or manifest == "environments/control/environment.yml"
        ), candidate["id"]


def test_artifact_groups_and_phase_scopes_are_bounded_and_referenced() -> None:
    for candidate in _registry()["candidates"]:
        allowed_phases = candidate.get("allowed_phases")
        if allowed_phases is not None:
            assert isinstance(allowed_phases, list) and allowed_phases
            assert len(allowed_phases) == len(set(allowed_phases))
            assert set(allowed_phases) <= ALLOWED_PHASES
        retired_indices = candidate.get("retired_config_indices", [])
        assert isinstance(retired_indices, list)
        assert len(retired_indices) == len(set(retired_indices))
        assert all(
            type(index) is int and 0 <= index < len(candidate["configs"])
            for index in retired_indices
        )
        groups = candidate.get("artifact_groups", {})
        assert isinstance(groups, dict)
        referenced_groups = set()
        default_group = candidate.get("default_artifact_group")
        if default_group is not None:
            assert default_group in groups
            referenced_groups.add(default_group)
        runtime_groups = candidate.get("artifact_group_by_runtime_variant", {})
        assert isinstance(runtime_groups, dict)
        for runtime_variant, group in runtime_groups.items():
            assert PUBLIC_ID.fullmatch(runtime_variant)
            assert group in groups
            referenced_groups.add(group)
        for config in candidate["configs"]:
            phases = config.get("phases")
            if phases is not None:
                assert isinstance(phases, list) and phases
                assert len(phases) == len(set(phases))
                assert set(phases) <= ALLOWED_PHASES
            group = config.get("artifact_group")
            if group is not None:
                assert group in groups
                referenced_groups.add(group)
        assert referenced_groups == set(groups)
        for group_name, paths in groups.items():
            assert PUBLIC_ID.fullmatch(group_name)
            _assert_bounded_relative_paths(paths)
        _assert_bounded_relative_paths(candidate.get("artifact_files", []))
        _assert_bounded_relative_paths(candidate.get("fingerprint_files", []))


def test_configs_remain_unique_after_artifact_group_resolution() -> None:
    for candidate in _registry()["candidates"]:
        effective_configs = []
        retired_indices = set(candidate.get("retired_config_indices", []))
        runtime_groups = candidate.get("artifact_group_by_runtime_variant", {})
        for index, config in enumerate(candidate["configs"]):
            if index in retired_indices:
                continue
            effective = dict(config)
            effective["artifact_group"] = config.get(
                "artifact_group",
                runtime_groups.get(
                    config.get("runtime_variant"),
                    candidate.get("default_artifact_group"),
                ),
            )
            effective_configs.append(json.dumps(effective, sort_keys=True))
        assert len(effective_configs) == len(set(effective_configs)), candidate["id"]


def _assert_bounded_relative_paths(paths: object) -> None:
    assert isinstance(paths, list)
    assert len(paths) == len(set(paths))
    for value in paths:
        assert isinstance(value, str) and 1 <= len(value) <= 240
        assert "\\" not in value and ":" not in value
        path = PurePosixPath(value)
        assert not path.is_absolute() and ".." not in path.parts


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
