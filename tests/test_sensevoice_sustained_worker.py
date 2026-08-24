from pathlib import Path

import pytest

from workers.sensevoice_sustained_worker import (
    _build_command,
    _invocation_outcome,
    _select_runtime,
)


def test_selects_each_pinned_sensevoice_runtime() -> None:
    root = Path("project")

    official = _select_runtime(root, "official_fixed8", 8)
    source_v019 = _select_runtime(root, "source_thread_control", 3)
    source_v020 = _select_runtime(root, "source_thread_control_v020", 3)

    assert official[0].name == "llama-funasr-sensevoice.exe"
    assert official[2:] == ([], True)
    assert "v0.1.9" in source_v019[1]
    assert "legacy-qwenaudio" in source_v019[1]
    assert source_v019[2:] == (["--threads", "3"], False)
    assert "v0.2.0" in source_v020[1]
    assert "modelscope" in source_v020[1]
    assert source_v020[2:] == (["--threads", "3"], True)


@pytest.mark.parametrize(
    ("runtime_variant", "threads"),
    [
        ("official_fixed8", 7),
        ("source_thread_control", 0),
        ("source_thread_control", 25),
        ("source_thread_control_v020", 0),
        ("source_thread_control_v020", 25),
        ("unknown", 8),
    ],
)
def test_rejects_unsupported_sensevoice_runtime_settings(
    runtime_variant: str,
    threads: int,
) -> None:
    with pytest.raises(ValueError):
        _select_runtime(Path("project"), runtime_variant, threads)


def test_builds_explicit_cpu_threaded_vad_command() -> None:
    command = _build_command(
        binary=Path("sensevoice.exe"),
        model=Path("model.gguf"),
        audio=Path("sample.wav"),
        vad_model=Path("vad.gguf"),
        thread_arguments=["--threads", "3"],
        explicit_cpu_backend=True,
    )

    assert command == [
        "sensevoice.exe",
        "-m",
        "model.gguf",
        "-a",
        "sample.wav",
        "--backend",
        "cpu",
        "--keep-tags",
        "--threads",
        "3",
        "--vad",
        "vad.gguf",
        "--vad-maxseg",
        "30000",
    ]


@pytest.mark.parametrize(
    ("returncode", "stderr", "transcript", "expected_speech", "expected"),
    [
        (0, "[sensevoice] done 1.0s", "speech", True, (True, None)),
        (0, "", "", False, (True, None)),
        (1, "", "speech", True, (False, "runtime_exit")),
        (0, "compute failed", "speech", True, (False, "compute_failed")),
        (0, "", "", True, (False, "empty_output")),
        (
            0,
            "",
            "<|zh|><|NEUTRAL|><|Speech|><|woitn|>",
            True,
            (False, "empty_output"),
        ),
    ],
)
def test_classifies_sensevoice_runtime_outcomes(
    returncode: int,
    stderr: str,
    transcript: str,
    expected_speech: bool,
    expected: tuple[bool, str | None],
) -> None:
    assert _invocation_outcome(
        returncode=returncode,
        stderr=stderr,
        transcript=transcript,
        expected_speech=expected_speech,
    ) == expected
