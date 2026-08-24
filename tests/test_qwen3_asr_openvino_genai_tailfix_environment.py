import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

import scripts.verify_qwen3_asr_openvino_genai_tailfix_20260821_environment as verifier
from local_inference_bench.qwen3_asr_tailfix_profile import (
    EXPECTED_TAILFIX_PROFILE,
    TAILFIX_PACKAGE_VERSIONS,
    TAILFIX_PROFILE_RELATIVE_PATH,
    load_qwen3_asr_tailfix_profile,
)
from local_inference_bench.run_sustained import _select_config_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / TAILFIX_PROFILE_RELATIVE_PATH
LOCK_PATH = PROFILE_PATH.with_name("requirements.lock.txt")
SETUP_PATH = (
    PROJECT_ROOT
    / "scripts/prepare_qwen3_asr_openvino_genai_tailfix_20260821.ps1"
)


def _lock_versions() -> dict[str, str]:
    versions = {}
    for line in LOCK_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            package, package_version = line.split("==", 1)
            versions[package] = package_version
    return versions


def test_tracked_profile_and_lock_match_the_immutable_runtime_identity() -> None:
    assert load_qwen3_asr_tailfix_profile(PROFILE_PATH) == EXPECTED_TAILFIX_PROFILE
    assert _lock_versions() == TAILFIX_PACKAGE_VERSIONS
    assert len(TAILFIX_PACKAGE_VERSIONS) == 9
    assert "soundfile" not in TAILFIX_PACKAGE_VERSIONS


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("environment_name", "local-bench-injected"),
        ("openvino_genai_product_version", "2026.4.0.0-injected"),
        ("stable_source_revision", "d" * 40),
        ("associated_source_revision", "f" * 40),
        ("required_fix_revision", "e" * 40),
        ("source_repository", "https://example.invalid/repository.git"),
        ("schema_version", True),
        ("schema_version", 1.0),
    ],
)
def test_profile_loader_rejects_coordinated_identity_drift(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    mutated = copy.deepcopy(EXPECTED_TAILFIX_PROFILE)
    mutated[field] = replacement
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(mutated), encoding="utf-8")

    with pytest.raises(RuntimeError, match="profile changed"):
        load_qwen3_asr_tailfix_profile(path)


def test_source_ancestry_rejects_a_runtime_revision_without_the_fix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "data" / "vendor" / "source"
    checkout.mkdir(parents=True)
    profile = copy.deepcopy(EXPECTED_TAILFIX_PROFILE)
    profile["source_checkout"] = "data/vendor/source"
    monkeypatch.setattr(verifier, "PROJECT_ROOT", tmp_path)

    def fake_git(_checkout: Path, *arguments: str) -> str:
        if arguments == ("remote", "get-url", "origin"):
            return profile["source_repository"]
        if arguments == ("rev-parse", "--is-shallow-repository"):
            return "false"
        if arguments[0] == "rev-parse":
            return arguments[1].removesuffix("^{commit}")
        raise AssertionError(arguments)

    monkeypatch.setattr(verifier, "_run_git", fake_git)
    monkeypatch.setattr(verifier, "_git_is_ancestor", lambda *_args: False)

    with pytest.raises(RuntimeError, match="does not contain the fix"):
        verifier._verify_source_ancestry(profile)


def test_source_ancestry_rejects_checkout_outside_vendor_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "data" / "vendor").mkdir(parents=True)
    (tmp_path / "source").mkdir()
    profile = copy.deepcopy(EXPECTED_TAILFIX_PROFILE)
    profile["source_checkout"] = "source"
    monkeypatch.setattr(verifier, "PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="escaped vendor directory"):
        verifier._verify_source_ancestry(profile)


def test_pip_check_isolated_from_repository_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", str(PROJECT_ROOT / "src"))
    captured_environment = None

    def fake_run(command, **kwargs):
        nonlocal captured_environment
        captured_environment = kwargs["env"]
        assert command == [verifier.sys.executable, "-m", "pip", "check"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    verifier._verify_pip_dependencies()

    assert captured_environment is not None
    assert "PYTHONPATH" not in captured_environment
    assert captured_environment["PYTHONNOUSERSITE"] == "1"
    assert os.environ["PYTHONPATH"] == str(PROJECT_ROOT / "src")


def test_preparation_script_parses_and_validates_before_profile_driven_writes() -> None:
    escaped_setup_path = str(SETUP_PATH).replace("'", "''")
    command = (
        "$errors=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped_setup_path}',[ref]$null,[ref]$errors)|Out-Null;"
        "if($errors.Count){$errors|ForEach-Object{$_.Message};exit 1}"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    script = SETUP_PATH.read_text(encoding="utf-8")
    assert "$expectedProfileSha256" in script
    assert "$expectedLockSha256" in script
    assert "escaped the ignored vendor directory" in script
    assert script.index("$expectedProfileSha256") < script.index("New-Item")
    assert script.index("$expectedLockSha256") < script.index("New-Item")
    assert script.index("if (Test-Path -LiteralPath $targetPython") < script.index(
        "$officialPython ="
    )
    assert "Complete partial OpenVINO nightly wheel has the wrong hash" in script
    assert ".local-bench-tailfix-preparation-in-progress-v1" in script
    assert ".local-bench-tailfix-preparation-complete-v1" in script
    assert "$profile.stable_source_revision" in script
    assert "+refs/heads/*:refs/remotes/origin/*" in script
    assert script.index("$condaHistory =") < script.index("& $conda create")
    for destructive_command in (
        "Remove-Item",
        "conda remove",
        "git clean",
        "git reset",
    ):
        assert destructive_command not in script


def test_tailfix_registry_lane_is_bounded_and_stable_configs_are_unchanged() -> None:
    registry = json.loads(
        (PROJECT_ROOT / "registries/sustained_candidates.json").read_text(
            encoding="utf-8"
        )
    )
    by_id = {candidate["id"]: candidate for candidate in registry["candidates"]}
    stable = by_id["qwen3_asr_0_6b_openvino_genai_official"]
    assert stable["configs"] == [
        {"processes": 1, "device": "CPU", "threads_per_process": 24, "max_new_tokens": 512},
        {"processes": 1, "device": "GPU.0", "threads_per_process": 4, "max_new_tokens": 512},
        {"processes": 1, "device": "CPU", "threads_per_process": 24, "max_new_tokens": 4096},
        {"processes": 1, "device": "GPU.0", "threads_per_process": 4, "max_new_tokens": 4096},
    ]

    tailfix = by_id["qwen3_asr_0_6b_openvino_genai_tailfix_20260821"]
    assert tailfix["allowed_phases"] == ["quality", "compatibility"]
    assert tailfix["configs"] == [
        {
            "processes": 1,
            "device": "CPU",
            "threads_per_process": 24,
            "max_new_tokens": 512,
            "runtime_profile": TAILFIX_PROFILE_RELATIVE_PATH.as_posix(),
        }
    ]
    assert _select_config_indices(tailfix, "quality", None) == (0,)
    assert _select_config_indices(tailfix, "compatibility", None) == (0,)
    with pytest.raises(ValueError, match="no config"):
        _select_config_indices(tailfix, "sustained", None)
