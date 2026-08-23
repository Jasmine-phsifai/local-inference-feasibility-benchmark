from workers.hunyuanocr_1_5_server_worker import (
    PROMPTS,
    _format_metrics,
    _generation_summary,
    _raw_completion_payload,
    _raw_hunyuan_prompt,
)


def test_format_metrics_detect_complete_table_and_latex():
    metrics = _format_metrics(
        "<table><tr><td>x</td></tr></table>\n\\[x^2\\]"
    )

    assert metrics == {
        "latex_marker": True,
        "complete_html_table": True,
    }


def test_generation_summary_counts_confirmed_caps_and_finish_modes():
    summary = _generation_summary(
        [
            {
                "completion_tokens": 4096,
                "token_cap_hit": True,
                "length_finish": True,
                "stop_finish": False,
                "latex_marker": True,
                "complete_html_table": False,
                "generated_tokens_per_second": 8.0,
            },
            {
                "completion_tokens": 128,
                "token_cap_hit": False,
                "length_finish": False,
                "stop_finish": True,
                "latex_marker": False,
                "complete_html_table": True,
                "generated_tokens_per_second": 10.0,
            },
        ],
        4096,
    )

    assert summary["token_cap_hit_count"] == 1
    assert summary["completion_tokens_total"] == 4224
    assert summary["mean_generated_tokens_per_second"] == 9.0


def test_hunyuan_raw_completion_pairs_one_marker_with_one_image(tmp_path):
    image_path = tmp_path / "control.png"
    image_path.write_bytes(b"image")
    marker = "<__media_test__>"

    prompt = _raw_hunyuan_prompt(marker, PROMPTS["doc_parse"])
    payload = _raw_completion_payload(
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=256,
    )

    assert payload["prompt"]["prompt_string"].count(marker) == 1
    assert prompt.startswith("<\uff5chy_begin\u2581of\u2581sentence\uff5c>")
    assert prompt.endswith("<\uff5chy_User\uff5c>")
    assert len(payload["prompt"]["multimodal_data"]) == 1
    assert payload["n_predict"] == 256
