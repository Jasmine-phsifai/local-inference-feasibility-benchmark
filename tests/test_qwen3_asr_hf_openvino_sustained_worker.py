from pathlib import Path
from types import SimpleNamespace

import pytest

from workers.qwen3_asr_hf_openvino_sustained_worker import (
    _execution_devices,
    _decode_parsed_asr_output,
    _generated_sequences,
    _generation_is_healthy,
    _openvino_config,
    _token_diagnostics,
    _validate_execution_devices,
    _validated_generation_tokens,
)


def test_hf_openvino_cpu_config_pins_threads_and_single_stream() -> None:
    config = _openvino_config("CPU", 24, Path("cache"))
    assert config["INFERENCE_NUM_THREADS"] == 24
    assert config["NUM_STREAMS"] == "1"
    assert config["PERFORMANCE_HINT"] == "LATENCY"


def test_hf_openvino_gpu_config_does_not_set_cpu_thread_property() -> None:
    config = _openvino_config("GPU", 4, Path("cache"))
    assert "INFERENCE_NUM_THREADS" not in config
    assert config["NUM_STREAMS"] == "1"


def test_generated_sequences_accepts_tensor_like_and_generate_output() -> None:
    class GenerateOutput:
        sequences = "sequences"

    assert _generated_sequences("tensor") == "tensor"
    assert _generated_sequences(GenerateOutput()) == "sequences"


def test_hf_execution_devices_handles_infer_requests() -> None:
    class CompiledModel:
        def get_property(self, name: str) -> list[str]:
            assert name == "EXECUTION_DEVICES"
            return ["GPU.0"]

    class InferRequest:
        def get_compiled_model(self) -> CompiledModel:
            return CompiledModel()

    class Component:
        request = InferRequest()

    class Model:
        _component_names = ("encoder", "decoder", "decoder_with_past")
        encoder = Component()
        decoder = Component()
        decoder_with_past = Component()

    assert _execution_devices(Model()) == {
        "encoder": ["GPU.0"],
        "decoder": ["GPU.0"],
        "decoder_with_past": ["GPU.0"],
    }


def test_hf_execution_devices_rejects_missing_declared_component() -> None:
    class Model:
        _component_names = ("encoder", "decoder")
        encoder = object()

    with pytest.raises(RuntimeError, match="not compiled"):
        _execution_devices(Model())


def test_hf_execution_devices_rejects_mixed_fallback() -> None:
    with pytest.raises(RuntimeError, match="did not execute only"):
        _validate_execution_devices(
            {"encoder": ["GPU.0"], "decoder": ["GPU.0", "CPU"]},
            "GPU",
        )


def test_hf_generation_tokens_match_pinned_native_checkpoint() -> None:
    assert _validated_generation_tokens(
        SimpleNamespace(eos_token_id=[151643, 151645], pad_token_id=151645)
    ) == ([151643, 151645], 151645)


def test_hf_raw_token_diagnostics_expose_repetition_and_eos() -> None:
    class FakeTokens:
        def detach(self):
            return self

        def cpu(self):
            return self

        def reshape(self, *shape):
            return self

        def tolist(self):
            return [7, 7, 7, 8, 7, 7, 7, 151645]

    diagnostics = _token_diagnostics(FakeTokens(), [151643, 151645])

    assert diagnostics["terminal_is_eos"] is True
    assert diagnostics["max_token_run"] == 3
    assert diagnostics["unique_token_ratio"] == 3 / 8
    assert diagnostics["repeated_trigram_ratio"] > 0


def test_hf_generation_health_rejects_cap_and_repetition() -> None:
    healthy = {
        "terminal_is_eos": True,
        "max_token_run": 2,
        "repeated_trigram_ratio": 0.65,
    }
    degenerate = {
        "terminal_is_eos": False,
        "max_token_run": 508,
        "repeated_trigram_ratio": 0.99,
    }

    assert _generation_is_healthy(healthy, token_cap_hit=False) is True
    assert _generation_is_healthy(healthy, token_cap_hit=True) is False
    assert _generation_is_healthy(degenerate, token_cap_hit=True) is False


def test_hf_parsed_output_requires_structured_language_and_transcription() -> None:
    class Processor:
        def decode(self, generated_ids, *, return_format):
            assert generated_ids == "ids"
            assert return_format == "parsed"
            return [{"language": "English", "transcription": "  speech  "}]

    assert _decode_parsed_asr_output(Processor(), "ids") == ("speech", True)


def test_hf_parsed_output_exposes_missing_language() -> None:
    class Processor:
        def decode(self, generated_ids, *, return_format):
            return {"language": None, "transcription": "malformed raw output"}

    assert _decode_parsed_asr_output(Processor(), "ids") == (
        "malformed raw output",
        False,
    )
