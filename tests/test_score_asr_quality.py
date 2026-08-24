import hashlib
import json
import uuid
from pathlib import Path

import pytest

from local_inference_bench.load_sustained_workload import load_sustained_workload
from local_inference_bench.score_asr_quality import (
    _contains_mixed_token_sequence,
    _mixed_tokens,
    _repeated_ngram_ratio,
    _score_timestamps,
    score_asr_quality,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_records(
    manifest_path: Path,
    records_path: Path,
    *,
    candidate_id: str = "candidate",
) -> None:
    workload = load_sustained_workload(manifest_path, expected_task="asr")
    expected_ids = {item["id"] for item in workload["items"]}
    successful_ids = {
        record["sample_id"]
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for record in [json.loads(line)]
        if isinstance(record, dict)
        and type(record.get("sample_id")) is str
        and record["sample_id"] in expected_ids
        and record.get("success") is True
    }
    status = (
        "succeeded"
        if successful_ids == expected_ids
        else "all_failed" if not successful_ids else "partial_failure"
    )
    records_path.with_name("records-provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "sustained-process-v1",
                "status": status,
                "attempt_id": str(uuid.UUID("00000000-0000-4000-8000-000000000001")),
                "attempt_key": "1" * 16,
                "candidate_id": candidate_id,
                "task": "asr",
                "config": {"processes": 1},
                "config_index": 2,
                "phase": "quality",
                "trial_index": 0,
                "workload_class": "generated_quality_control",
                "workload_fingerprint": workload["fingerprint"],
                "code_fingerprint": "a" * 16,
                "environment_fingerprint": "b" * 16,
                "controller_environment_fingerprint": "c" * 16,
                "execution_policy_fingerprint": "d" * 16,
                "records_sha256": _sha256(records_path),
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_quality_fixture(
    tmp_path: Path,
    record: dict,
) -> tuple[Path, Path]:
    input_path = tmp_path / "sample.wav"
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "private-records.jsonl"
    input_path.write_bytes(b"public-control")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "asr",
                "workload_class": "generated_quality_control",
                "items": [
                    {
                        "id": "sample",
                        "path": input_path.name,
                        "duration_seconds": 1,
                    }
                ],
                "references": {
                    "sample": {
                        "audio_sha256": _sha256(input_path),
                        "category": "control",
                        "expected_speech": True,
                        "required_terms": [],
                        "speech_intervals": [],
                        "transcript": "reference",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    _bind_records(manifest_path, records_path)
    return manifest_path, records_path


def test_mixed_tokenizer_keeps_cjk_words_and_decimals():
    value = "CPU \u4e8c\u5341\u56db, 0.975"
    assert _mixed_tokens(value) == ["cpu", "\u4e8c", "\u5341", "\u56db", "0.975"]


def test_repetition_ratio_counts_extra_trigrams():
    assert _repeated_ngram_ratio("a b c a b c".split(), 3) == 0.25


def test_timestamp_scoring_rejects_nonfinite_bounds() -> None:
    result = _score_timestamps(
        reference_intervals=[[float("nan"), 1.0], [0.0, float("inf")]],
        predicted_segments=[
            {"start": float("nan"), "end": float("nan")},
            {"start": 0.0, "end": float("inf")},
        ],
        duration_seconds=10.0,
    )

    assert result["timestamp_reference_seconds"] == 0.0
    assert result["timestamp_predicted_seconds"] == 0.0
    assert result["timestamp_overlap_seconds"] == 0.0
    assert result["timestamp_invalid_segments"] == 2


def test_term_matching_does_not_credit_abbreviation_substrings():
    tokens = _mixed_tokens("the system must remain stable for milliseconds")

    assert not _contains_mixed_token_sequence(tokens, "AI")
    assert not _contains_mixed_token_sequence(tokens, "MS")
    assert _contains_mixed_token_sequence(
        _mixed_tokens("\u4eba\u5de5\u667a\u80fd AI"),
        "\u4eba\u5de5\u667a\u80fd",
    )


def test_asr_quality_event_contains_aggregate_metrics_only(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    (tmp_path / "speech.wav").write_bytes(b"speech")
    (tmp_path / "silence.wav").write_bytes(b"silence")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "asr",
                "workload_class": "generated_quality_control",
                "items": [
                    {
                        "id": "speech",
                        "path": "speech.wav",
                        "duration_seconds": 10,
                    },
                    {
                        "id": "silence",
                        "path": "silence.wav",
                        "duration_seconds": 60,
                        "expected_speech": False,
                    },
                ],
                "references": {
                    "speech": {
                        "audio_sha256": _sha256(tmp_path / "speech.wav"),
                        "category": "mixed",
                        "expected_speech": True,
                        "transcript": "CPU \u4e8c\u5341\u56db",
                        "speech_intervals": [[1.0, 5.0]],
                        "required_terms": [
                            {"aliases": ["CPU"]},
                            {"aliases": ["\u4e8c\u5341\u56db", "24"]},
                        ],
                    },
                    "silence": {
                        "audio_sha256": _sha256(tmp_path / "silence.wav"),
                        "category": "silence",
                        "expected_speech": False,
                        "required_terms": [],
                        "speech_intervals": [],
                        "transcript": "",
                    },
                },
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=True) + "\n"
            for record in (
                {
                    "sample_id": "speech",
                    "success": True,
                    "prediction": "<|zh|> CPU \u4e8c\u5341\u56db",
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
        ),
        encoding="utf-8",
    )
    _bind_records(manifest_path, records_path)

    event = score_asr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )

    overall = event["metrics"]["overall"]
    assert event["scorer_protocol"] == "asr-quality-v6"
    assert event["source_attempt"] == {
        "status": "succeeded",
        "candidate_id": "candidate",
        "attempt_id": "00000000-0000-4000-8000-000000000001",
        "attempt_key": "1111111111111111",
        "config_fingerprint": "2266b7bfde15dc37",
        "config_index": 2,
        "trial_index": 0,
        "code_fingerprint": "a" * 16,
        "environment_fingerprint": "b" * 16,
        "controller_environment_fingerprint": "c" * 16,
        "execution_policy_fingerprint": "d" * 16,
    }
    assert event["workload_fingerprint"] == load_sustained_workload(
        manifest_path,
        expected_task="asr",
    )["fingerprint"]
    assert len(event["scorer_fingerprint"]) == 16
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
    assert overall["availability"] == {
        "sample_denominator": 2,
        "successful_sample_count": 2,
        "unavailable_sample_count": 0,
        "explicit_failed_record_count": 0,
        "missing_record_count": 0,
    }
    assert overall["quality_denominators"]["mixed_token_error_denominator"] == 4
    assert (
        overall["quality_denominators"]
        ["timestamp_availability_sample_denominator"]
        == 2
    )
    serialized = json.dumps(event)
    assert "invented" not in serialized
    assert "\u4e8c\u5341\u56db" not in serialized


def test_rejects_records_mutated_after_provenance_binding(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "prediction": "reference"},
    )
    records_path.write_text(
        json.dumps(
            {"sample_id": "sample", "success": True, "prediction": "changed"}
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provenance is invalid"):
        score_asr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


def test_rejects_replaced_audio_with_stale_reference_hash(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "prediction": "reference"},
    )
    (tmp_path / "sample.wav").write_bytes(b"different-public-control")
    _bind_records(manifest_path, records_path)

    with pytest.raises(ValueError, match="reference media hash"):
        score_asr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


def test_public_registry_config_is_fingerprinted_without_copying_raw_config(
    tmp_path: Path,
) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "prediction": "reference"},
    )
    provenance_path = records_path.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["config"] = {"processes": 4, "effective_threads_per_process": 6}
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    event = score_asr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )
    serialized = json.dumps(event)
    assert "effective_threads_per_process" not in serialized
    assert set(event["source_attempt"]) == {
        "candidate_id",
        "status",
        "attempt_id",
        "attempt_key",
        "config_fingerprint",
        "config_index",
        "trial_index",
        "code_fingerprint",
        "environment_fingerprint",
        "controller_environment_fingerprint",
        "execution_policy_fingerprint",
    }


def test_partial_failure_scores_failed_and_missing_samples_as_empty(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    items = []
    references = {}
    for sample_id in ("success", "failed", "missing"):
        media_path = tmp_path / f"{sample_id}.wav"
        media_path.write_bytes(sample_id.encode("ascii"))
        items.append(
            {"id": sample_id, "path": media_path.name, "duration_seconds": 1}
        )
        references[sample_id] = {
            "audio_sha256": _sha256(media_path),
            "category": "control",
            "expected_speech": True,
            "required_terms": [{"aliases": [sample_id]}],
            "speech_intervals": [],
            "transcript": sample_id,
        }
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "asr",
                "workload_class": "generated_quality_control",
                "items": items,
                "references": references,
            }
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                {
                    "sample_id": "success",
                    "success": True,
                    "prediction": "success",
                },
                {
                    "sample_id": "failed",
                    "success": False,
                    "prediction": "must be ignored",
                },
            )
        ),
        encoding="utf-8",
    )
    _bind_records(manifest_path, records_path)

    event = score_asr_quality(
        manifest_path=manifest_path,
        records_path=records_path,
        candidate_id="candidate",
    )

    overall = event["metrics"]["overall"]
    assert event["source_attempt"]["status"] == "partial_failure"
    assert overall["availability"] == {
        "sample_denominator": 3,
        "successful_sample_count": 1,
        "unavailable_sample_count": 2,
        "explicit_failed_record_count": 1,
        "missing_record_count": 1,
    }
    assert overall["quality_denominators"] == {
        "mixed_token_error_denominator": 3,
        "normalized_character_error_denominator": 20,
        "required_term_recall_denominator": 3,
        "silence_minutes_denominator": 0.0,
        "repetition_sample_denominator": 3,
        "timestamp_availability_sample_denominator": 3,
        "timestamp_recall_seconds_denominator": 0,
        "timestamp_precision_seconds_denominator": 0,
    }
    assert overall["mixed_token_error_rate"] == pytest.approx(2 / 3)
    assert overall["required_term_recall"] == pytest.approx(1 / 3)
    assert "must be ignored" not in json.dumps(event)


def test_rejects_source_status_that_disagrees_with_records(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "prediction": "reference"},
    )
    provenance_path = records_path.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["status"] = "partial_failure"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="source status does not match records"):
        score_asr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


@pytest.mark.parametrize(
    "record",
    [
        {"sample_id": "sample", "success": "true", "prediction": "text"},
        {"sample_id": "sample", "success": True},
        {"sample_id": "sample", "success": True, "prediction": None},
        {"sample_id": "sample", "success": False, "prediction": []},
    ],
)
def test_rejects_malformed_quality_records(tmp_path: Path, record: dict) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(tmp_path, record)

    with pytest.raises(ValueError, match="invalid ASR quality record"):
        score_asr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )


def test_rejects_private_workload_and_unbounded_candidate_id(tmp_path: Path) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "prediction": "reference"},
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["workload_class"] = "private_course"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="generated controls only"):
        score_asr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )

    document["workload_class"] = "generated_quality_control"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="public identifier"):
        score_asr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="D:\\private\\course",
        )


@pytest.mark.parametrize(
    ("reference_update", "message"),
    [
        ({"transcript": 123}, "transcript"),
        ({"required_terms": "CPU"}, "required terms"),
        ({"required_terms": ["CPU"]}, "required term"),
        ({"required_terms": [{"aliases": []}]}, "aliases"),
        ({"required_terms": [{"aliases": [123]}]}, "aliases"),
        ({"required_terms": [{"aliases": ["x" * 257]}]}, "aliases"),
        ({"speech_intervals": "0-1"}, "speech intervals"),
        ({"speech_intervals": [[0.5, 0.25]]}, "speech interval"),
        (
            {"speech_intervals": [[0.0, float("nan")]]},
            "sustained workload manifest is invalid",
        ),
        ({"expected_speech": False}, "speech expectation"),
    ],
)
def test_rejects_malformed_reference_schema(
    tmp_path: Path,
    reference_update: dict,
    message: str,
) -> None:
    manifest_path, records_path = _write_minimal_quality_fixture(
        tmp_path,
        {"sample_id": "sample", "success": True, "prediction": "reference"},
    )
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["references"]["sample"].update(reference_update)
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _bind_records(manifest_path, records_path)
        score_asr_quality(
            manifest_path=manifest_path,
            records_path=records_path,
            candidate_id="candidate",
        )
