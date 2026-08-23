import pytest

from workers.rapidocr_sustained_worker import _backend_name, _engine_params


def test_engine_params_apply_bounded_preprocessing_variants():
    engine_type = object()
    params = _engine_params(
        {"use_cls": False, "max_side_len": 1536},
        threads=2,
        engine_type=engine_type,
    )

    assert params["Det.engine_type"] is engine_type
    assert params["Cls.engine_type"] is engine_type
    assert params["Rec.engine_type"] is engine_type
    assert params["Global.use_cls"] is False
    assert params["Global.max_side_len"] == 1536
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 2
    assert params["EngineConfig.onnxruntime.inter_op_num_threads"] == 1


def test_engine_params_select_openvino_with_explicit_threading() -> None:
    engine_type = object()
    params = _engine_params(
        {"backend": "openvino", "use_cls": True, "max_side_len": 2000},
        threads=2,
        engine_type=engine_type,
    )

    assert params["Det.engine_type"] is engine_type
    assert params["Cls.engine_type"] is engine_type
    assert params["Rec.engine_type"] is engine_type
    assert params["EngineConfig.openvino.inference_num_threads"] == 2
    assert params["EngineConfig.openvino.performance_hint"] == "LATENCY"
    assert params["EngineConfig.openvino.num_streams"] == 1
    assert "EngineConfig.onnxruntime.intra_op_num_threads" not in params


def test_backend_name_is_backward_compatible_and_bounded() -> None:
    assert _backend_name({}) == "onnxruntime"
    assert _backend_name({"backend": "OpenVINO"}) == "openvino"
    with pytest.raises(ValueError, match="unsupported RapidOCR backend"):
        _backend_name({"backend": "auto"})
