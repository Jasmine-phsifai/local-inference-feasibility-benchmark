import array
import wave
from pathlib import Path

import pytest

from workers.qwen3_asr_openvino_genai_sustained_worker import (
    BENCHMARK_ITEM_LIMIT_SECONDS,
    NATIVE_INTERNAL_CHUNK_SECONDS,
    _gpu_memory_snapshot_bytes,
    _metric_mean,
    _read_audio,
)


def _write_pcm16_wav(path: Path, *, seconds: float = 0.1) -> None:
    samples = array.array("h", [0] * round(16000 * seconds))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(samples.tobytes())


def test_reads_compact_audio_and_verifies_declared_duration(tmp_path: Path) -> None:
    pytest.importorskip("soundfile")
    path = tmp_path / "sample.wav"
    _write_pcm16_wav(path)

    audio, duration = _read_audio(path, declared_duration_seconds=0.1)

    assert audio.dtype.name == "float32"
    assert len(audio) == 1600
    assert duration == pytest.approx(0.1)
    with pytest.raises(ValueError, match="declared duration"):
        _read_audio(path, declared_duration_seconds=0.2)


def test_distinguishes_native_chunking_from_benchmark_limit() -> None:
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
