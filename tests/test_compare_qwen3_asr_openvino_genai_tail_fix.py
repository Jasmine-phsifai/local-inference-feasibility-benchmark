import json
import wave
from pathlib import Path

import pytest

from scripts.compare_qwen3_asr_openvino_genai_tail_fix import (
    EVENT_PROTOCOL,
    NIGHTLY_SPEC,
    PROJECT_ROOT,
    STABLE_SPEC,
    TAIL_CONTROL_SHA256,
    _append_pcm16_silence,
    _build_event,
    _control_descriptor,
    _read_pcm16_float32,
    _producer_sha256,
    _sha256,
    _spec_from_request,
    _spec_to_request,
    _summarize_transcript,
    _tail_output_token_counts,
    _validate_child_response,
    _validate_child_failure,
    _validate_timeout,
    _verify_source_ancestry,
    _write_tail_control,
)


def _write_public_shape(path: Path) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(bytes(176_000 * 2))


def _fake_child(label: str, digest_suffix: str) -> dict:
    return {
        "schema": "qwen3-asr-openvino-genai-tail-fix-child-v1",
        "runtime": {
            "label": label,
            "python_version": "3.11.15",
            "openvino_genai_source_repository": "https://example.invalid/repo",
            "associated_openvino_genai_source_revision": "0" * 40,
            "packages": {
                name: {"version": "1", "wheel_sha256": "f" * 64}
                for name in (
                    "openvino",
                    "openvino-genai",
                    "openvino-tokenizers",
                )
            },
        },
        "outputs": [
            {
                "control_id": "control",
                "transcript_sha256": "a" * 63 + digest_suffix,
                "unicode_character_count": 5,
                "utf8_byte_count": 7,
                "generated_token_count": 3,
            }
        ],
    }


def test_remainder_eight_exposes_the_fixed_geometry() -> None:
    assert _tail_output_token_counts(0) == (0, 0)
    assert _tail_output_token_counts(8) == (2, 1)
    with pytest.raises(ValueError, match="remainder"):
        _tail_output_token_counts(100)


def test_tail_control_adds_exactly_eighty_milliseconds(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    target = tmp_path / "tail.wav"
    _write_public_shape(source)

    _append_pcm16_silence(source, target, silence_samples=1_280)
    with wave.open(str(target), "rb") as reader:
        assert reader.getnframes() == 177_280
        assert reader.getframerate() == 16_000


def test_standard_library_audio_reader_returns_mono_float32(tmp_path: Path) -> None:
    numpy = pytest.importorskip("numpy")
    source = tmp_path / "source.wav"
    _write_public_shape(source)

    audio = _read_pcm16_float32(source, numpy)

    assert audio.dtype.name == "float32"
    assert audio.shape == (176_000,)
    assert float(audio.max()) == 0.0


def test_checked_in_public_control_derives_expected_tail_hash(tmp_path: Path) -> None:
    source = Path("data/inputs/public/jfk.wav")
    if not source.is_file():
        pytest.skip("ignored public fixture is unavailable")
    target = tmp_path / "tail.wav"

    _write_tail_control(source, target)

    assert _sha256(target) == TAIL_CONTROL_SHA256
    descriptor = _control_descriptor("tail", target, mel_frames=1_108)
    assert descriptor["sample_count"] == 177_280
    assert descriptor["legacy_tail_output_token_count"] == 2
    assert descriptor["fixed_tail_output_token_count"] == 1


def test_transcript_summary_never_contains_raw_text() -> None:
    class Metrics:
        def get_num_generated_tokens(self):
            return 4

    class Result:
        perf_metrics = Metrics()

    summary = _summarize_transcript("control", "private words", Result())

    assert "private words" not in json.dumps(summary)
    assert set(summary) == {
        "control_id",
        "transcript_sha256",
        "unicode_character_count",
        "utf8_byte_count",
        "generated_token_count",
    }


def test_event_distinguishes_containment_from_observed_effect() -> None:
    stable = _fake_child("stable", "0")
    nightly = _fake_child("nightly", "1")
    controls = [
        {
            "id": "control",
            "path": "ignored.wav",
            "sha256": "b" * 64,
            "sample_count": 177_280,
            "sample_rate_hz": 16_000,
            "mel_frame_count": 1_108,
            "encoder_remainder_frames": 8,
            "legacy_tail_output_token_count": 2,
            "fixed_tail_output_token_count": 1,
        }
    ]

    ancestry = {
        "repository": "https://example.invalid/repo",
        "fix_revision": "f" * 40,
        "stable_source_revision": "0" * 40,
        "nightly_source_revision": "1" * 40,
        "stable_contains_fix": False,
        "nightly_contains_fix": True,
        "evidence_method": "executed_git_merge_base_is_ancestor_full_clone",
        "verified_conclusion": "bounded test",
    }
    event = _build_event(stable, nightly, controls, ancestry)

    assert event["protocol"] == EVENT_PROTOCOL
    assert event["static_implementation_containment"]["stable_contains_fix"] is False
    assert event["static_implementation_containment"]["nightly_contains_fix"] is True
    effect = event["transcript_level_effect"]
    assert effect["effect_observed_on_bounded_controls"] is True
    assert effect["all_transcripts_exactly_equal"] is False
    assert "path" not in effect["controls"][0]["input"]


def test_event_binds_all_repository_owned_direct_producers() -> None:
    producer_sha256 = _producer_sha256()

    assert set(producer_sha256) == {
        "scripts/compare_qwen3_asr_openvino_genai_tail_fix.py",
        "workers/qwen3_asr_openvino_genai_export_manifest.py",
    }
    for relative_path, digest in producer_sha256.items():
        assert len(digest) == 64
        assert digest == _sha256(PROJECT_ROOT / relative_path)


def test_event_producer_binding_contains_no_local_path() -> None:
    stable = _fake_child("stable", "0")
    nightly = _fake_child("nightly", "0")
    controls = [
        {
            "id": "control",
            "path": "must-not-be-published.wav",
            "sha256": "b" * 64,
            "sample_count": 177_280,
            "sample_rate_hz": 16_000,
            "mel_frame_count": 1_108,
            "encoder_remainder_frames": 8,
            "legacy_tail_output_token_count": 2,
            "fixed_tail_output_token_count": 1,
        }
    ]
    event = _build_event(stable, nightly, controls, {})
    serialized = json.dumps(event, sort_keys=True)

    assert "must-not-be-published.wav" not in serialized
    assert str(PROJECT_ROOT) not in serialized
    assert str(PROJECT_ROOT).replace("\\", "/") not in serialized


def test_child_response_rejects_transcript_field() -> None:
    response = _fake_child("stable", "0")
    response["outputs"][0]["transcript"] = "must not escape"

    with pytest.raises(RuntimeError, match="unapproved"):
        _validate_child_response(response, expected_control_count=1)


def test_child_failure_allows_only_stage_and_exception_kind() -> None:
    response = {
        "schema": "qwen3-asr-openvino-genai-tail-fix-child-failure-v1",
        "failure": {"stage": "inference", "kind": "RuntimeError"},
    }
    _validate_child_failure(response)

    response["failure"]["message"] = "decoded content"
    with pytest.raises(RuntimeError, match="unapproved|malformed"):
        _validate_child_failure(response)


def test_runtime_request_must_match_a_pinned_specification() -> None:
    assert _spec_from_request(_spec_to_request(STABLE_SPEC)) == STABLE_SPEC
    assert _spec_from_request(_spec_to_request(NIGHTLY_SPEC)) == NIGHTLY_SPEC
    changed = _spec_to_request(STABLE_SPEC)
    changed["source_revision"] = "0" * 40
    with pytest.raises(RuntimeError, match="not pinned"):
        _spec_from_request(changed)


def test_source_ancestry_is_executed_and_not_inferred_from_wheel_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "source"
    checkout.mkdir()

    def fake_git(_checkout: Path, *arguments: str) -> str:
        if arguments == ("remote", "get-url", "origin"):
            return "https://github.com/openvinotoolkit/openvino.genai.git"
        if arguments == ("rev-parse", "--is-shallow-repository"):
            return "false"
        if arguments[0] == "rev-parse":
            return arguments[1].removesuffix("^{commit}")
        raise AssertionError(arguments)

    def fake_is_ancestor(_checkout: Path, _ancestor: str, descendant: str) -> bool:
        return descendant == NIGHTLY_SPEC.source_revision

    monkeypatch.setattr(
        "scripts.compare_qwen3_asr_openvino_genai_tail_fix._run_git",
        fake_git,
    )
    monkeypatch.setattr(
        "scripts.compare_qwen3_asr_openvino_genai_tail_fix._git_is_ancestor",
        fake_is_ancestor,
    )

    evidence = _verify_source_ancestry(checkout)

    assert evidence["stable_contains_fix"] is False
    assert evidence["nightly_contains_fix"] is True
    assert evidence["evidence_method"].startswith("executed_git_")


def test_timeout_cannot_exceed_three_minutes() -> None:
    assert _validate_timeout(180) == 180
    for invalid in (True, 0, 181):
        with pytest.raises(ValueError, match="timeout"):
            _validate_timeout(invalid)
