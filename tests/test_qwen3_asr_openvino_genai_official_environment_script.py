import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "create_qwen3_asr_openvino_genai_official_environment.ps1"
)


def test_optimum_intel_vcs_install_disables_build_isolation() -> None:
    script = ENVIRONMENT_SCRIPT.read_text(encoding="utf-8")
    install = re.search(
        r"(?ms)^\$optimumIntelSource = \(.*?^\)\s*"
        r"(?P<command>^& \$targetPython -m pip install .*?"
        r"^\s+.*?\$optimumIntelSource\s*$)",
        script,
    )

    assert install is not None
    command = install.group("command")
    assert "--no-deps" in command
    assert "--force-reinstall" in command
    assert "--no-build-isolation" in command
