import json

from local_inference_bench.score_asr_quality import (
    _contains_mixed_token_sequence,
    _mixed_tokens,
    _repeated_ngram_ratio,
    score_asr_quality,
)


def test_mixed_tokenizer_keeps_cjk_words_and_decimals():
    assert _mixed_tokens("CPU 二十四, 0.975") == ["cpu", "二", "十", "四", "0.975"]


def test_repetition_ratio_counts_extra_trigrams():
    assert _repeated_ngram_ratio("a b c a b c".split(), 3) == 0.25


def test_term_matching_does_not_credit_abbreviation_substrings():
    tokens = _mixed_tokens("the system must remain stable for milliseconds")

    assert not _contains_mixed_token_sequence(tokens, "AI")
    assert not _contains_mixed_token_sequence(tokens, "MS")
    assert _contains_mixed_token_sequence(
        _mixed_tokens("人工智能 AI"),
        "人工智能",
    )


def test_asr_quality_event_contains_aggregate_metrics_only(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "task": "asr",
                "workload_class": "generated_quality_control",
                "items": [
                    {"id": "speech", "duration_seconds": 10},
                    {"id": "silence", "duration_seconds": 60},
                ],
                "references": {
                    "speech": {
                        "category": "mixed",
                        "transcript": "CPU 二十四",
                        "speech_intervals": [[1.0, 5.0]],
                        "required_terms": [
                            {"aliases": ["CPU"]},
                            {"aliases": ["二十四", "24"]},
                        ],
                    },
                    "silence": {
                        "category": "silence",
                        "transcript": "",
                        "expected_speech": False,
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                {
                    "sample_id": "speech",
                    "success": True,
                    "prediction": "<|zh|> CPU 二十四",
                    "segments": [
                        {"start": 1.0, "end": 3.0},
                        {"start": 3.0, "end": 5.0},
                    ],
                },
                {
                    "sample_id": "silence",
                    "success": True,
                    "prediction": "invented",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    event = score_asr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )

    overall = event["metrics"]["overall"]
    assert event["scorer_protocol"] == "asr-quality-v3"
    assert event["metrics"]["categories"]["mixed"]["mixed_token_error_rate"] == 0.0
    assert overall["mixed_token_error_rate"] == 0.25
    assert overall["required_term_recall"] == 1.0
    assert overall["silence_false_positive_characters_per_minute"] == 8.0
    assert overall["timestamp_metrics_available_fraction"] == 0.5
    assert overall["timestamp_speech_recall_when_available"] == 1.0
    assert overall["timestamp_speech_precision_when_available"] == 1.0
    assert overall["timestamp_invalid_segment_count"] == 0
    assert overall["timestamp_nonmonotonic_segment_count"] == 0
    assert overall["mean_excess_repeated_trigram_ratio"] == 0.0
    serialized = json.dumps(event)
    assert "invented" not in serialized
    assert "二十四" not in serialized
