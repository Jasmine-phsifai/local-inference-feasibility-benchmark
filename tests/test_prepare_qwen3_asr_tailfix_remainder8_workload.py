import json
import os
import subprocess
import wave
from pathlib import Path

import pytest

from local_inference_bench.load_sustained_workload import load_sustained_workload
from scripts.prepare_qwen3_asr_tailfix_remainder8_workload import (
    OUTPUT_MEL_FRAME_COUNT,
    OUTPUT_REMAINDER_FRAMES,
    OUTPUT_SAMPLE_COUNT,
    SAMPLE_RATE_HZ,
    SOURCE_SAMPLE_COUNT,
    _read_pcm16_wav,
    _require_ignored_or_external_output,
    main,
    prepare_remainder8_workload,
)


def _write_wav(
    path: Path,
    sample_count: int,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> bytes:
    frames = bytes(sample_count * channels * sample_width)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)
    return frames


def _write_source_workload(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source.wav"
    warmup = tmp_path / "source-warmup.wav"
    _write_wav(source, SOURCE_SAMPLE_COUNT)
    _write_wav(warmup, SAMPLE_RATE_HZ // 10)
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "asr",
                "workload_class": "private_course",
                "items": [
                    {
                        "id": "chunk_001",
                        "path": source.name,
                        "duration_seconds": 120.0,
                        "expected_speech": True,
                    },
                    {
                        "id": "chunk_002",
                        "path": source.name,
                        "duration_seconds": 120.0,
                        "expected_speech": True,
                    },
                ],
                "warmup": {
                    "id": "source_warmup",
                    "path": warmup.name,
                    "duration_seconds": 0.1,
                    "expected_speech": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, source, warmup


def _read_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as reader:
        return reader.readframes(reader.getnframes())


def test_prepares_exact_remainder8_workload_and_loader_accepts_it(
    tmp_path: Path,
) -> None:
    manifest, source, warmup = _write_source_workload(tmp_path)
    output = tmp_path / "external-output"

    geometry = prepare_remainder8_workload(manifest, output)

    assert geometry == {
        "duration_seconds": 120.08,
        "encoder_remainder_frames": 8,
        "item_count": 1,
        "mel_frame_count": 12_008,
        "sample_count": 1_921_280,
        "sample_rate_hz": 16_000,
    }
    frames = _read_frames(output / "chunk_001.wav")
    assert len(frames) == OUTPUT_SAMPLE_COUNT * 2
    assert frames[: SOURCE_SAMPLE_COUNT * 2] == _read_frames(source)
    assert frames[-2_560:] == bytes(2_560)
    assert (output / "warmup.wav").read_bytes() == warmup.read_bytes()
    loaded = load_sustained_workload(output / "manifest.json", expected_task="asr")
    assert loaded["public_summary"] == {
        "workload_class": "private_course",
        "item_count": 1,
        "total_duration_seconds": 120.08,
    }
    assert OUTPUT_MEL_FRAME_COUNT == 12_008
    assert OUTPUT_REMAINDER_FRAMES == 8


def test_valid_existing_outputs_are_idempotent(tmp_path: Path) -> None:
    manifest, _, _ = _write_source_workload(tmp_path)
    output = tmp_path / "external-output"
    prepare_remainder8_workload(manifest, output)
    mtimes = {path.name: path.stat().st_mtime_ns for path in output.iterdir()}

    prepare_remainder8_workload(manifest, output)

    assert {path.name: path.stat().st_mtime_ns for path in output.iterdir()} == mtimes


def test_legacy_partial_files_are_preserved_and_rejected(
    tmp_path: Path,
) -> None:
    manifest, _, _ = _write_source_workload(tmp_path)
    output = tmp_path / "external-output"
    output.mkdir()
    (output / "chunk_001.wav.part").write_bytes(b"stale")
    (output / "warmup.wav.part").write_bytes(b"stale")
    (output / "manifest.json.part").write_bytes(b"stale")

    originals = {
        path.name: path.read_bytes()
        for path in output.glob("*.part")
    }

    with pytest.raises(ValueError, match="requires manual review"):
        prepare_remainder8_workload(manifest, output)

    assert not (output / "manifest.json").exists()
    assert {
        path.name: path.read_bytes()
        for path in output.glob("*.part")
    } == originals


def test_atomic_promotion_never_overwrites_destination_that_appears(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, _ = _write_source_workload(tmp_path)
    output = tmp_path / "external-output"
    original_link = os.link
    appeared = b"valuable concurrent output"

    def inject_destination(source, destination, **kwargs):
        Path(destination).write_bytes(appeared)
        return original_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", inject_destination)

    with pytest.raises(FileExistsError, match="appeared"):
        prepare_remainder8_workload(manifest, output)

    assert (output / "chunk_001.wav").read_bytes() == appeared


def test_legacy_partial_hardlink_cannot_modify_source(tmp_path: Path) -> None:
    manifest, source, _ = _write_source_workload(tmp_path)
    original_source = source.read_bytes()
    output = tmp_path / "external-output"
    output.mkdir()
    os.link(source, output / "chunk_001.wav.part")

    with pytest.raises(ValueError, match="must not alias"):
        prepare_remainder8_workload(manifest, output)

    assert source.read_bytes() == original_source


def test_legacy_partial_symlink_cannot_modify_source(tmp_path: Path) -> None:
    manifest, source, _ = _write_source_workload(tmp_path)
    original_source = source.read_bytes()
    output = tmp_path / "external-output"
    output.mkdir()
    partial = output / "chunk_001.wav.part"
    try:
        partial.symlink_to(source)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="reparse point|must not alias"):
        prepare_remainder8_workload(manifest, output)

    assert source.read_bytes() == original_source


def test_output_manifest_cannot_alias_source_manifest(tmp_path: Path) -> None:
    manifest, _, _ = _write_source_workload(tmp_path)
    manifest = manifest.rename(tmp_path / "manifest.json")
    original_manifest = manifest.read_bytes()

    with pytest.raises(ValueError, match="must not alias"):
        prepare_remainder8_workload(manifest, tmp_path)

    assert manifest.read_bytes() == original_manifest


def test_warmup_media_must_be_distinct_from_source(tmp_path: Path) -> None:
    manifest, source, _ = _write_source_workload(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["warmup"] = {
        "id": "source_warmup",
        "path": source.name,
        "duration_seconds": 120.0,
        "expected_speech": True,
    }
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="separate warmup media"):
        prepare_remainder8_workload(manifest, tmp_path / "external-output")


def test_incompatible_committed_output_is_not_overwritten(tmp_path: Path) -> None:
    manifest, _, _ = _write_source_workload(tmp_path)
    output = tmp_path / "external-output"
    prepare_remainder8_workload(manifest, output)
    audio = output / "chunk_001.wav"
    corrupted = bytearray(audio.read_bytes())
    corrupted[-1] = 1
    audio.write_bytes(corrupted)

    with pytest.raises(ValueError, match="incompatible"):
        prepare_remainder8_workload(manifest, output)

    assert audio.read_bytes() == corrupted


def test_in_repository_output_requires_git_ignore(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("private/\n", encoding="utf-8")

    _require_ignored_or_external_output(
        tmp_path / "private" / "tailfix",
        project_root=tmp_path,
    )
    with pytest.raises(ValueError, match="must be ignored"):
        _require_ignored_or_external_output(
            tmp_path / "tracked" / "tailfix",
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    "wav_options",
    [
        {"channels": 2},
        {"sample_width": 1},
        {"sample_rate": 8_000},
    ],
)
def test_rejects_wrong_source_wav_format(
    tmp_path: Path,
    wav_options: dict,
) -> None:
    path = tmp_path / "invalid.wav"
    _write_wav(path, 100, **wav_options)

    with pytest.raises(ValueError, match="PCM16 mono 16 kHz"):
        _read_pcm16_wav(path, declared_duration_seconds=100 / SAMPLE_RATE_HZ)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "private ASR workload"),
        ("task", "ocr", "private ASR workload"),
        ("workload_class", "public_course", "private ASR workload"),
        ("warmup", None, "separate explicit warmup"),
    ],
)
def test_rejects_invalid_source_manifest(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest, _, _ = _write_source_workload(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document[field] = value
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        prepare_remainder8_workload(manifest, tmp_path / "external-output")


def test_successful_cli_stdout_contains_numeric_geometry_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest, _, _ = _write_source_workload(tmp_path)
    output = tmp_path / "external-output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_qwen3_asr_tailfix_remainder8_workload.py",
            "--source-workload",
            str(manifest),
            "--output-dir",
            str(output),
        ],
    )

    main()

    stdout = capsys.readouterr().out
    published = json.loads(stdout)
    assert all(type(value) in {int, float} for value in published.values())
    assert str(tmp_path) not in stdout
    assert "sha256" not in stdout
