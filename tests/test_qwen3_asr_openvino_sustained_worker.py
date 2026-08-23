from pathlib import Path

from workers.qwen3_asr_openvino_sustained_worker import (
    _execution_devices,
    _extract_transcription,
    _generation_kwargs,
    _openvino_config,
    _validate_execution_devices,
)


def test_extract_transcription_accepts_both_qwen_marker_spellings() -> None:
    expected = "mixed 中文 and English 42"
    assert _extract_transcription(
        f"language English<asr_text>{expected}<|im_end|>"
    ) == expected
    assert _extract_transcription(
        f"language English<|asr_text|>{expected}<|im_end|>"
    ) == expected


def test_extract_transcription_has_bounded_prefix_fallback() -> None:
    assert _extract_transcription("language Chinese 简短内容") == "简短内容"


def test_openvino_cpu_config_pins_threads_and_single_stream() -> None:
    config = _openvino_config("CPU", 24, Path("cache"))
    assert config["INFERENCE_NUM_THREADS"] == 24
    assert config["NUM_STREAMS"] == "1"
    assert config["PERFORMANCE_HINT"] == "LATENCY"


def test_openvino_gpu_config_does_not_set_cpu_thread_property() -> None:
    config = _openvino_config("GPU", 4, Path("cache"))
    assert "INFERENCE_NUM_THREADS" not in config
    assert config["NUM_STREAMS"] == "1"


def test_generation_kwargs_forward_the_processor_audio_attention_mask() -> None:
    inputs = {
        "input_features": "features",
        "feature_attention_mask": "audio-mask",
        "input_ids": "decoder-prompt",
    }

    assert _generation_kwargs(inputs, max_new_tokens=64) == {
        "input_features": "features",
        "attention_mask": "audio-mask",
        "decoder_input_ids": "decoder-prompt",
        "max_new_tokens": 64,
        "do_sample": False,
    }


def test_execution_devices_handles_stateful_decoder_infer_request() -> None:
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
        _component_names = ("encoder", "decoder")
        encoder = Component()
        decoder = Component()

    assert _execution_devices(Model()) == {
        "encoder": ["GPU.0"],
        "decoder": ["GPU.0"],
    }


def test_execution_device_gate_rejects_mixed_fallback() -> None:
    try:
        _validate_execution_devices({"encoder": ["GPU.0", "CPU"]}, "GPU")
    except RuntimeError as error:
        assert "only on requested GPU" in str(error)
    else:
        raise AssertionError("mixed OpenVINO execution devices were accepted")
