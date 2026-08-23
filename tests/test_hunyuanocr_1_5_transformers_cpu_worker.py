from types import SimpleNamespace

import pytest

from workers.hunyuanocr_1_5_transformers_cpu_worker import (
    PROMPTS,
    _all_records_pass_safety,
    _bounded_aligned_dimensions,
    _chat_messages,
    _decoded_output,
    _decode_semantics,
    _generation_summary,
    _prompt_for_item,
    _token_id_set,
    _validate_native_cpu_bounds,
    _validated_generation_tokens,
    _verify_model_bundle,
)


def test_native_hunyuan_resize_is_bounded_and_does_not_upscale() -> None:
    assert _bounded_aligned_dimensions(1920, 1080, 1024, 1024 * 576) == (
        1024,
        576,
    )
    assert _bounded_aligned_dimensions(640, 480, 1024, 1024 * 576) == (
        640,
        480,
    )
    square = _bounded_aligned_dimensions(1920, 1920, 1920, 1920 * 1088)
    assert square[0] * square[1] <= 1920 * 1088
    smoke_square = _bounded_aligned_dimensions(4096, 4096, 960, 522240)
    assert smoke_square[0] * smoke_square[1] <= 522240
    assert _bounded_aligned_dimensions(1000, 600, 1920, 2088960) == (992, 608)


def test_native_hunyuan_messages_match_official_image_then_text_order() -> None:
    messages = _chat_messages("control.png", PROMPTS["doc_parse"])

    assert messages[0] == {"role": "system", "content": ""}
    assert messages[1]["content"][0] == {
        "type": "image",
        "image": "control.png",
    }
    assert messages[1]["content"][1]["type"] == "text"


def test_native_hunyuan_source_faithful_mode_binds_item_marker() -> None:
    marker = "<!-- meta:page number=7 -->"
    prompt = _prompt_for_item(
        "source_faithful",
        {"output_marker": marker},
    )

    assert marker in prompt
    assert "leading indentation exactly" in prompt


def test_native_hunyuan_source_faithful_mode_requires_marker() -> None:
    with pytest.raises(ValueError, match="output_marker"):
        _prompt_for_item("source_faithful", {})


def test_native_hunyuan_source_faithful_prediction_preserves_decode_boundaries() -> None:
    decoded = ["\n  <!-- meta:page number=7 -->\nbody  \n"]

    assert _decoded_output(decoded, preserve_raw=True) == decoded[0]
    assert _decoded_output(decoded, preserve_raw=False) == (
        "<!-- meta:page number=7 -->\nbody"
    )
    assert _decode_semantics("source_faithful") == {
        "raw_decode_scored": True,
        "outer_whitespace_trimmed": False,
    }
    assert _decode_semantics("doc_parse") == {
        "raw_decode_scored": False,
        "outer_whitespace_trimmed": True,
    }


def test_native_hunyuan_generation_summary_counts_caps_and_eos() -> None:
    summary = _generation_summary(
        [
            {
                "completion_tokens": 64,
                "token_cap_hit": True,
                "eos_finish": False,
                "latex_marker": True,
                "complete_html_table": False,
                "end_to_end_completion_tokens_per_second": 2.0,
            },
            {
                "completion_tokens": 20,
                "token_cap_hit": False,
                "eos_finish": True,
                "latex_marker": False,
                "complete_html_table": True,
                "end_to_end_completion_tokens_per_second": 4.0,
            },
        ],
        64,
    )

    assert summary["completion_tokens_total"] == 84
    assert summary["token_cap_hit_count"] == 1
    assert summary["eos_finish_count"] == 1
    assert summary["mean_end_to_end_completion_tokens_per_second"] == 3.0


def test_native_hunyuan_safety_probe_rejects_token_cap() -> None:
    assert _all_records_pass_safety(
        [{"success": True, "token_cap_hit": False}]
    )
    assert not _all_records_pass_safety(
        [{"success": True, "token_cap_hit": True}]
    )


def test_native_hunyuan_rejects_panorama_that_conflicts_with_bounds() -> None:
    try:
        _bounded_aligned_dimensions(960, 100, 960, 522240)
    except ValueError as error:
        assert "aspect ratio" in str(error)
    else:
        raise AssertionError("unsafe panorama was accepted")


def test_native_hunyuan_token_id_set_accepts_scalar_and_sequence() -> None:
    assert _token_id_set(None) == set()
    assert _token_id_set(7) == {7}
    assert _token_id_set([7, 8]) == {7, 8}


def test_native_hunyuan_preserves_the_pinned_two_eos_ids() -> None:
    eos_ids, pad_id = _validated_generation_tokens(
        SimpleNamespace(
            bos_token_id=120000,
            eos_token_id=[120007, 120020],
            pad_token_id=120002,
        )
    )

    assert eos_ids == [120007, 120020]
    assert pad_id == 120002


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("threads", 25),
        ("max_new_tokens", 513),
        ("max_side_len", 1921),
        ("max_pixels", 1920 * 1088 + 1),
        ("max_vision_patches", 8161),
        ("max_items", 4),
    ],
)
def test_native_hunyuan_rejects_registry_values_above_cpu_ceiling(
    field: str,
    unsafe_value: int,
) -> None:
    values = {
        "threads": 24,
        "max_new_tokens": 512,
        "max_side_len": 1920,
        "max_pixels": 1920 * 1088,
        "max_vision_patches": 8160,
        "max_items": 3,
    }
    values[field] = unsafe_value

    with pytest.raises(ValueError, match=field):
        _validate_native_cpu_bounds(**values)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("max_side_len", 480),
        ("max_pixels", 512 * 512 - 1),
        ("max_vision_patches", 1023),
    ],
)
def test_native_hunyuan_rejects_processor_impossible_bounds(
    field: str,
    unsafe_value: int,
) -> None:
    values = {
        "threads": 24,
        "max_new_tokens": 64,
        "max_side_len": 960,
        "max_pixels": 960 * 544,
        "max_vision_patches": 2040,
        "max_items": 1,
    }
    values[field] = unsafe_value

    with pytest.raises(ValueError, match="processor minimum"):
        _validate_native_cpu_bounds(**values)


def test_native_hunyuan_bundle_rejects_mutated_auxiliary_asset(tmp_path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"weight")
    weight_hash = __import__("hashlib").sha256(b"weight").hexdigest()
    (tmp_path / "aux.json").write_bytes(b"expected")
    expected_hash = __import__("hashlib").sha256(b"expected").hexdigest()
    _verify_model_bundle(
        tmp_path,
        expected_auxiliary_sha256={"aux.json": expected_hash},
        expected_weight_sha256=weight_hash,
    )
    (tmp_path / "aux.json").write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="bundle mismatch"):
        _verify_model_bundle(
            tmp_path,
            expected_auxiliary_sha256={"aux.json": expected_hash},
            expected_weight_sha256=weight_hash,
        )
