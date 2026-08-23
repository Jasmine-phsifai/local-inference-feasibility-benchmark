from workers.rapidocr_sustained_worker import _engine_params


def test_engine_params_apply_bounded_preprocessing_variants():
    params = _engine_params(
        {"use_cls": False, "max_side_len": 1536},
        threads=2,
    )

    assert params["Global.use_cls"] is False
    assert params["Global.max_side_len"] == 1536
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 2
    assert params["EngineConfig.onnxruntime.inter_op_num_threads"] == 1
