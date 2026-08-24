import pytest

from workers.rapidocr_sustained_worker import (
    _backend_name,
    _claim_process_index,
    _engine_params,
    _recognize,
    _validated_opencv_threads,
)


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


def test_validates_optional_opencv_thread_count() -> None:
    assert _validated_opencv_threads({}) is None
    assert _validated_opencv_threads({"opencv_threads": 1}) == 1
    for invalid in (True, "1", 0, 25):
        with pytest.raises(ValueError, match="opencv_threads"):
            _validated_opencv_threads({"opencv_threads": invalid})


def test_process_index_is_bounded_and_unique() -> None:
    assert (
        _claim_process_index(
            {"process_index": 1},
            process_count=2,
            claimed={0},
            stage="completion",
        )
        == 1
    )
    for invalid in (True, -1, 2, "1"):
        with pytest.raises(RuntimeError, match="completion index"):
            _claim_process_index(
                {"process_index": invalid},
                process_count=2,
                claimed=set(),
                stage="completion",
            )
    with pytest.raises(RuntimeError, match="completion index"):
        _claim_process_index(
            {"process_index": 1},
            process_count=2,
            claimed={1},
            stage="completion",
        )
    with pytest.raises(RuntimeError, match="completion message"):
        _claim_process_index(
            [],
            process_count=2,
            claimed=set(),
            stage="completion",
        )


def test_expected_text_rejects_whitespace_only_rapidocr_output() -> None:
    class WhitespaceOnlyOutput:
        txts = ["  ", "\t"]
        scores = [0.9, 0.8]
        boxes = None

    class FakeEngine:
        def __call__(self, _path):
            return WhitespaceOnlyOutput()

    record = _recognize(
        FakeEngine(),
        {"id": "expected-text", "path": "unused.png", "expected_text": True},
        capture_prediction=True,
    )

    assert record["success"] is False
    assert record["failure_kind"] == "empty_output"
    assert record["units"] == 0.0
    assert [line["text"] for line in record["lines"]] == ["  ", "\t"]
