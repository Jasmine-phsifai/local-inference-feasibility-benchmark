import array
import wave
from pathlib import Path

import pytest

from workers.qwen3_asr_openvino_genai_sustained_worker import (
    BENCHMARK_ITEM_LIMIT_SECONDS,
    NATIVE_MINIMUM_CHUNK_SECONDS,
    STABLE_ENVIRONMENT_NAME,
    STABLE_PACKAGE_VERSIONS,
    STABLE_PRODUCT_VERSION,
    STABLE_RUNTIME_SOURCE_REVISION,
    NATIVE_INTERNAL_CHUNK_SECONDS,
    _expected_runtime_identity,
    _gpu_memory_snapshot_bytes,
    _mel_frame_count,
    _metric_mean,
    _read_audio,
    _single_native_chunk_tail_geometry,
    _tail_fix_presence_from_identity,
    _tail_output_token_counts,
    _verify_installed_runtime,
)
from local_inference_bench.qwen3_asr_tailfix_profile import (
    TAILFIX_ENVIRONMENT_NAME,
    TAILFIX_PACKAGE_VERSIONS,
    TAILFIX_PRODUCT_VERSION,
    TAILFIX_PROFILE_RELATIVE_PATH,
    TAILFIX_SOURCE_REVISION,
)


def _write_pcm16_wav(
    path: Path,
    *,
    seconds: float = 0.1,
    channels: int = 1,
    sample_rate: int = 16_000,
    sample_width: int = 2,
) -> None:
    sample_count = round(sample_rate * seconds) * channels
    frames = (
        array.array("h", [0] * sample_count).tobytes()
        if sample_width == 2
        else bytes(sample_count * sample_width)
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(sample_width)
        stream.setframerate(sample_rate)
        stream.writeframes(frames)


def test_reads_compact_audio_and_verifies_declared_duration(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    path = tmp_path / "sample.wav"
    _write_pcm16_wav(path)

    audio, duration = _read_audio(path, declared_duration_seconds=0.1)

    assert audio.dtype.name == "float32"
    assert len(audio) == 1600
    assert duration == pytest.approx(0.1)
    with pytest.raises(ValueError, match="declared duration"):
        _read_audio(path, declared_duration_seconds=0.2)


def test_pcm16_scaling_uses_the_full_signed_range(tmp_path: Path) -> None:
    pytest.importorskip("numpy")
    path = tmp_path / "scale.wav"
    samples = array.array("h", [-32768, -1, 0, 1, 32767])
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16_000)
        stream.writeframes(samples.tobytes())

    audio, duration = _read_audio(
        path,
        declared_duration_seconds=5 / 16_000,
    )

    assert audio.dtype.name == "float32"
    assert audio.tolist() == pytest.approx(
        [-1.0, -1 / 32768, 0.0, 1 / 32768, 32767 / 32768]
    )
    assert duration == 5 / 16_000


@pytest.mark.parametrize(
    "options",
    [
        {"channels": 2},
        {"sample_rate": 8_000},
        {"sample_width": 1},
    ],
)
def test_rejects_noncanonical_wav_formats(
    tmp_path: Path,
    options: dict,
) -> None:
    path = tmp_path / "invalid.wav"
    _write_pcm16_wav(path, **options)

    with pytest.raises(ValueError, match="PCM16 mono 16 kHz"):
        _read_audio(path, declared_duration_seconds=0.1)


def test_rejects_truncated_wav_payload(tmp_path: Path) -> None:
    path = tmp_path / "truncated.wav"
    _write_pcm16_wav(path)
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="frame data"):
        _read_audio(path, declared_duration_seconds=0.1)


def test_mel_geometry_matches_the_official_floor_formula() -> None:
    assert _mel_frame_count(1_920_000) == 12_000
    assert _mel_frame_count(1_921_280) == 12_008
    assert _mel_frame_count(1_921_281) == 12_008
    assert _tail_output_token_counts(0) == (0, 0)
    assert _tail_output_token_counts(8) == (2, 1)
    with pytest.raises(ValueError, match="sample count"):
        _mel_frame_count(0)


def test_tail_geometry_is_reported_only_for_one_native_runtime_chunk() -> None:
    assert _single_native_chunk_tail_geometry(
        1_921_280,
        duration_seconds=120.08,
    ) == {
        "mel_frame_count": 12_008,
        "encoder_remainder_frames": 8,
        "tail_fix_sensitive_geometry": True,
    }
    assert _single_native_chunk_tail_geometry(
        1_280,
        duration_seconds=0.08,
    ) == {}
    assert _single_native_chunk_tail_geometry(
        19_216_000,
        duration_seconds=1201.0,
    ) == {}
    assert _single_native_chunk_tail_geometry(
        8_000,
        duration_seconds=NATIVE_MINIMUM_CHUNK_SECONDS,
    )["mel_frame_count"] == 50
    assert _single_native_chunk_tail_geometry(
        19_200_000,
        duration_seconds=NATIVE_INTERNAL_CHUNK_SECONDS,
    )["mel_frame_count"] == 120_000


def test_runtime_profile_selects_only_the_two_pinned_identities() -> None:
    project_root = Path(__file__).resolve().parents[1]
    stable = _expected_runtime_identity(
        {
            "processes": 1,
            "device": "CPU",
            "threads_per_process": 24,
            "max_new_tokens": 512,
        },
        project_root,
    )
    nightly = _expected_runtime_identity(
        {
            "processes": 1,
            "device": "CPU",
            "threads_per_process": 24,
            "max_new_tokens": 512,
            "runtime_profile": TAILFIX_PROFILE_RELATIVE_PATH.as_posix(),
        },
        project_root,
    )

    assert stable == {
        "profile_id": "openvino_genai_stable_2026_3_0",
        "environment_name": STABLE_ENVIRONMENT_NAME,
        "package_versions": STABLE_PACKAGE_VERSIONS,
        "product_version": STABLE_PRODUCT_VERSION,
        "associated_source_revision": STABLE_RUNTIME_SOURCE_REVISION,
        "post_release_tail_chunk_fix_present": False,
    }
    assert nightly["environment_name"] == TAILFIX_ENVIRONMENT_NAME
    assert nightly["package_versions"] == TAILFIX_PACKAGE_VERSIONS
    assert nightly["product_version"] == TAILFIX_PRODUCT_VERSION
    assert nightly["associated_source_revision"] == TAILFIX_SOURCE_REVISION
    assert nightly["post_release_tail_chunk_fix_present"] is True
    with pytest.raises(ValueError, match="not pinned"):
        _expected_runtime_identity(
            {
                "processes": 1,
                "device": "CPU",
                "threads_per_process": 24,
                "max_new_tokens": 512,
                "runtime_profile": "environments/injected.json",
            },
            project_root,
        )


def test_tail_fix_flag_is_derived_from_exact_runtime_tuple() -> None:
    assert (
        _tail_fix_presence_from_identity(
            STABLE_PRODUCT_VERSION,
            STABLE_RUNTIME_SOURCE_REVISION,
        )
        is False
    )
    assert (
        _tail_fix_presence_from_identity(
            TAILFIX_PRODUCT_VERSION,
            TAILFIX_SOURCE_REVISION,
        )
        is True
    )
    with pytest.raises(RuntimeError, match="not pinned"):
        _tail_fix_presence_from_identity(
            TAILFIX_PRODUCT_VERSION,
            "f" * 40,
        )


def test_installed_runtime_verification_rejects_version_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GenAI:
        __version__ = STABLE_PRODUCT_VERSION

        @staticmethod
        def get_version():
            return STABLE_PRODUCT_VERSION

    identity = {
        "environment_name": STABLE_ENVIRONMENT_NAME,
        "package_versions": STABLE_PACKAGE_VERSIONS,
        "product_version": STABLE_PRODUCT_VERSION,
        "associated_source_revision": STABLE_RUNTIME_SOURCE_REVISION,
        "post_release_tail_chunk_fix_present": False,
    }
    monkeypatch.setattr(
        "workers.qwen3_asr_openvino_genai_sustained_worker.sys.prefix",
        str(Path("D:/Anaconda/envs") / STABLE_ENVIRONMENT_NAME),
    )
    monkeypatch.setattr(
        "workers.qwen3_asr_openvino_genai_sustained_worker.version",
        lambda package: STABLE_PACKAGE_VERSIONS[package],
    )
    _verify_installed_runtime(identity, GenAI)

    monkeypatch.setattr(
        "workers.qwen3_asr_openvino_genai_sustained_worker.version",
        lambda package: "0" if package == "openvino" else STABLE_PACKAGE_VERSIONS[package],
    )
    with pytest.raises(RuntimeError, match="package version"):
        _verify_installed_runtime(identity, GenAI)


def test_distinguishes_native_chunking_from_benchmark_limit() -> None:
    assert NATIVE_MINIMUM_CHUNK_SECONDS == 0.5
    assert NATIVE_INTERNAL_CHUNK_SECONDS == 1200.0
    assert BENCHMARK_ITEM_LIMIT_SECONDS == 7200.0


def test_reads_gpu_memory_property_from_exact_device() -> None:
    class Core:
        def get_property(self, device, name):
            assert device == "GPU.0"
            assert name == "GPU_MEMORY_STATISTICS"
            return {"current": 10, "other": 20}

    assert _gpu_memory_snapshot_bytes(Core(), "GPU.0") == 30
    assert _gpu_memory_snapshot_bytes(Core(), "CPU") is None


def test_metric_mean_accepts_runtime_property_or_number() -> None:
    class Metric:
        mean = 12.5

    assert _metric_mean(Metric()) == 12.5
    assert _metric_mean(7) == 7.0
