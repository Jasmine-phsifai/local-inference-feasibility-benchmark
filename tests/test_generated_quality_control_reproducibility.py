import hashlib
import json
import wave
from importlib.metadata import version
from pathlib import Path

import pytest
from PIL import Image

from scripts.prepare_generated_quality_controls import (
    _isolated_subprocess_environment,
    _promote_staged_output,
    _unique_suites,
    _verify_staged_pinned_outputs,
)
from scripts.verify_generated_quality_controls import (
    PROJECT_ROOT,
    verify_generated_quality_controls,
)


def test_ocr_manifest_verifies_media_hash_format_and_regeneration_dependencies(
    tmp_path: Path,
):
    font_root = tmp_path / "fonts"
    font_root.mkdir()
    font_path = font_root / "fixture.ttf"
    font_path.write_bytes(b"public synthetic font identity")
    manifest_path = _write_ocr_fixture(
        tmp_path,
        generator={
            "canvas_width": 32,
            "canvas_height": 24,
            "pillow_version": version("pillow"),
            "font_files": {"fixture.ttf": _sha256(font_path)},
        },
    )

    summary = verify_generated_quality_controls(
        manifest_path,
        verify_regeneration_host=True,
        font_root=font_root,
    )

    assert summary["protocol"] == "generated-quality-control-verification-v1"
    assert summary["task"] == "ocr"
    assert summary["item_count"] == 1
    assert summary["regeneration_dependency_checks"] == [
        "font_file_sha256",
        "pillow_version",
    ]
    assert "fixture text" not in json.dumps(summary)
    assert str(tmp_path) not in json.dumps(summary)


def test_ocr_verifier_rejects_media_mutation(tmp_path: Path):
    manifest_path = _write_ocr_fixture(tmp_path)
    with (tmp_path / "fixture.png").open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(ValueError, match="hash"):
        verify_generated_quality_controls(manifest_path)


def test_verifier_rejects_media_path_escape(tmp_path: Path):
    manifest_path = _write_ocr_fixture(tmp_path)
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["items"][0]["path"] = "../fixture.png"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="relative filename"):
        verify_generated_quality_controls(manifest_path)


def test_asr_manifest_verifies_pcm_format_hash_and_exact_duration(tmp_path: Path):
    manifest_path = _write_asr_fixture(tmp_path, declared_seconds=1)

    summary = verify_generated_quality_controls(manifest_path)

    assert summary["task"] == "asr"
    assert summary["item_count"] == 1
    assert summary["warmup"]["sample_rate_hz"] == 16_000
    assert summary["warmup"]["duration_seconds"] == 1.0


def test_asr_verifier_rejects_declared_duration_drift(tmp_path: Path):
    manifest_path = _write_asr_fixture(tmp_path, declared_seconds=2)

    with pytest.raises(ValueError, match="duration"):
        verify_generated_quality_controls(manifest_path)


def test_staged_preparation_refuses_to_replace_pinned_fixture(tmp_path: Path):
    staging_output = tmp_path / "staging"
    staging_output.mkdir()
    staged_fixture = staging_output / "control.png"
    staged_fixture.write_bytes(b"expected public fixture")
    registry_path = tmp_path / "registries" / "bounded_vlm_b10598_assets.json"
    registry_path.parent.mkdir()
    registry_path.write_text(
        json.dumps(
            {
                "candidates": {
                    "candidate": {
                        "fixtures": {
                            "manifest": None,
                            "images": {
                                "control": {
                                    "path": (
                                        "data/inputs/generated/ocr_quality/"
                                        "control.png"
                                    ),
                                    "bytes": staged_fixture.stat().st_size,
                                    "sha256": _sha256(staged_fixture),
                                }
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    _verify_staged_pinned_outputs(
        suite="ocr",
        staging_output=staging_output,
        project_root=tmp_path,
    )
    staged_fixture.write_bytes(b"different public fixture")

    with pytest.raises(ValueError, match="pinned tracked public fixture"):
        _verify_staged_pinned_outputs(
            suite="ocr",
            staging_output=staging_output,
            project_root=tmp_path,
        )


def test_staged_promotion_preserves_unrelated_generated_files(tmp_path: Path):
    staging_output = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging_output.mkdir()
    destination.mkdir()
    (staging_output / "new.json").write_bytes(b"new")
    (destination / "unrelated.json").write_bytes(b"preserve")

    _promote_staged_output(
        staging_output=staging_output,
        destination=destination,
    )

    assert (destination / "new.json").read_bytes() == b"new"
    assert (destination / "unrelated.json").read_bytes() == b"preserve"


def test_preparation_requires_unique_explicit_suites():
    assert _unique_suites(["ocr", "asr"]) == ["ocr", "asr"]
    with pytest.raises(ValueError, match="selected once"):
        _unique_suites(["ocr", "ocr"])


def test_preparation_subprocesses_disable_user_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "0")

    environment = _isolated_subprocess_environment()

    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["CI"] == "true"


def test_bounded_vlm_fixture_records_match_public_files_and_producers():
    registry = json.loads(
        (PROJECT_ROOT / "registries" / "bounded_vlm_b10598_assets.json").read_text(
            encoding="utf-8"
        )
    )
    checked = 0
    for candidate in registry["candidates"].values():
        fixtures = candidate["fixtures"]
        records = [fixtures["manifest"]]
        records.extend(fixtures["images"].values())
        records.extend(fixtures["generator_files"].values())
        for record in records:
            path = PROJECT_ROOT / record["path"]
            assert path.stat().st_size == record["bytes"]
            assert _sha256(path) == record["sha256"]
            checked += 1
    assert checked == 10


def test_tracked_ocr_subset_manifest_has_platform_independent_newlines():
    content = (
        PROJECT_ROOT
        / "data"
        / "inputs"
        / "generated"
        / "ocr_quality"
        / "hunyuan_doc_quality.json"
    ).read_bytes()
    assert b"\r\n" not in content
    assert content.endswith(b"\n")


def _write_ocr_fixture(tmp_path: Path, *, generator: dict | None = None) -> Path:
    fixture_path = tmp_path / "fixture.png"
    warmup_path = tmp_path / "warmup.png"
    Image.new("RGB", (32, 24), "white").save(fixture_path, format="PNG")
    Image.new("RGB", (16, 12), "white").save(warmup_path, format="PNG")
    document = {
        "schema_version": 1,
        "task": "ocr",
        "workload_class": "generated_quality_control",
        "warmup": {
            "id": "warmup",
            "path": warmup_path.name,
            "expected_text": True,
        },
        "items": [
            {"id": "fixture", "path": fixture_path.name, "expected_text": True}
        ],
        "references": {
            "fixture": {
                "category": "public_synthetic",
                "lines": ["fixture text"],
                "required_tokens": ["fixture text"],
                "image_sha256": _sha256(fixture_path),
            }
        },
        "generator": generator or {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    return manifest_path


def _write_asr_fixture(tmp_path: Path, *, declared_seconds: int) -> Path:
    fixture_path = tmp_path / "fixture.wav"
    warmup_path = tmp_path / "warmup.wav"
    _write_pcm_wav(fixture_path, seconds=1)
    _write_pcm_wav(warmup_path, seconds=1)
    document = {
        "schema_version": 1,
        "task": "asr",
        "workload_class": "generated_quality_control",
        "warmup": {
            "id": "warmup",
            "path": warmup_path.name,
            "duration_seconds": 1,
            "expected_speech": True,
        },
        "items": [
            {
                "id": "fixture",
                "path": fixture_path.name,
                "duration_seconds": declared_seconds,
                "expected_speech": True,
            }
        ],
        "references": {
            "fixture": {
                "category": "public_synthetic",
                "transcript": "fixture speech",
                "required_terms": [],
                "expected_speech": True,
                "speech_intervals": [[0, 1]],
                "audio_sha256": _sha256(fixture_path),
            }
        },
        "generator": {
            "channels": 1,
            "sample_width_bytes": 2,
            "sample_rate_hz": 16_000,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    return manifest_path


def _write_pcm_wav(path: Path, *, seconds: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * (16_000 * seconds))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
