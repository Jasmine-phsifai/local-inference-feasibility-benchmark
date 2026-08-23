from workers.hunyuanocr_1_5_server_worker import (
    _format_metrics,
    _generation_summary,
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
