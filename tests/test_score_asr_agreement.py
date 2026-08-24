import hashlib
import json
import os
import threading
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

import local_inference_bench.score_asr_agreement as agreement_scorer
from local_inference_bench.event_journal import _locked_journal
from local_inference_bench.load_sustained_workload import load_sustained_workload
from local_inference_bench.private_records_commitment import (
    PRIVATE_RECORDS_COMMITMENT_SCHEME,
    create_private_records_commitment,
)
from local_inference_bench.score_asr_agreement import _bounded_levenshtein
from local_inference_bench.score_asr_agreement import _append_public_event_once
from local_inference_bench.score_asr_agreement import _parse_candidate_specs
from local_inference_bench.score_asr_agreement import _public_event_sha256
from local_inference_bench.score_asr_agreement import score_asr_agreement


_REGISTERED_TEST_SOURCES = (
    ("faster_whisper_cpu", 19),
    ("sensevoice_small_gguf_cpu", 14),
    ("qwen3_asr_0_6b_openvino_genai_official", 0),
    ("faster_whisper_cpu", 20),
    ("sensevoice_small_gguf_cpu", 13),
    ("qwen3_asr_0_6b_openvino_genai_official", 1),
    ("faster_whisper_cpu", 18),
    ("sensevoice_small_gguf_cpu", 12),
)


@pytest.fixture(autouse=True)
def _use_test_sustained_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agreement_scorer,
        "SUSTAINED_EVENTS_PATH",
        tmp_path / "sustained-events.jsonl",
    )


def _registered_candidate(candidate_id: str) -> dict:
    registry = json.loads(
        agreement_scorer.SUSTAINED_REGISTRY_PATH.read_text(encoding="utf-8")
    )
    return next(
        candidate
        for candidate in registry["candidates"]
        if candidate["id"] == candidate_id
    )


def _write_workload(path: Path, items: list[dict], **overrides: object) -> None:
    document = {
        "schema_version": 1,
        "task": "asr",
        "workload_class": "private_course",
        "items": items,
        **overrides,
    }
    for item in document.get("items", []):
        if "path" not in item:
            input_path = path.parent / f"{item['id']}.wav"
            _write_test_wav(
                input_path,
                duration_seconds=float(item["duration_seconds"]),
                seed=item["id"],
            )
            item["path"] = input_path.name
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_test_wav(
    path: Path,
    *,
    duration_seconds: float,
    seed: str,
) -> None:
    frame_count = round(duration_seconds * 16_000)
    marker = hashlib.sha256(seed.encode("utf-8")).digest()
    frames = marker + bytes(frame_count * 2 - len(marker))
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(frames)


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recommit_records(records_path: Path) -> None:
    provenance_path = records_path.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop("records_sha256", None)
    provenance.pop("private_records_commitment", None)
    commitment = create_private_records_commitment(records_path, provenance)
    provenance["records_sha256"] = commitment["records_sha256"]
    provenance["private_records_commitment"] = commitment["private"]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")


def _bind_records(
    workload_path: Path,
    candidate_record_paths: dict[str, Path],
) -> None:
    workload = load_sustained_workload(workload_path, expected_task="asr")
    expected_ids = {item["id"] for item in workload["items"]}
    for index, (_, records_path) in enumerate(
        candidate_record_paths.items()
    ):
        candidate_id, config_index = _REGISTERED_TEST_SOURCES[index]
        config = _registered_candidate(candidate_id)["configs"][config_index]
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
        provenance = {
            "schema_version": 1,
            "protocol": "sustained-process-v1",
            "status": status,
            "attempt_id": str(
                uuid.UUID(f"00000000-0000-4000-8000-{index + 1:012d}")
            ),
            "attempt_key": f"{index + 1:016x}",
            "candidate_id": candidate_id,
            "task": "asr",
            "config": config,
            "config_index": config_index,
            "phase": "quality",
            "target_wall_seconds": 1.0,
            "trial_index": 0,
            "workload_class": "private_course",
            "workload_fingerprint": workload["fingerprint"],
            "code_fingerprint": "a" * 16,
            "environment_fingerprint": "b" * 16,
            "controller_environment_fingerprint": "c" * 16,
            "execution_policy_fingerprint": "d" * 16,
        }
        commitment = create_private_records_commitment(records_path, provenance)
        provenance["records_sha256"] = commitment["records_sha256"]
        provenance["private_records_commitment"] = commitment["private"]
        records_path.with_name("records-provenance.json").write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )
    _rewrite_sustained_journal(workload_path, candidate_record_paths.values())


def _rewrite_sustained_journal(
    workload_path: Path,
    record_paths,
) -> None:
    workload = load_sustained_workload(workload_path, expected_task="asr")
    events = []
    for records_path in record_paths:
        provenance = json.loads(
            records_path.with_name("records-provenance.json").read_text(
                encoding="utf-8"
            )
        )
        common = {
            key: provenance[key]
            for key in (
                "protocol",
                "attempt_id",
                "candidate_id",
                "task",
                "config",
                "config_index",
                "phase",
                "target_wall_seconds",
                "trial_index",
                "code_fingerprint",
                "environment_fingerprint",
                "controller_environment_fingerprint",
                "execution_policy_fingerprint",
            )
        }
        common["workload"] = workload["public_summary"]
        common["private_records_commitment_scheme"] = (
            PRIVATE_RECORDS_COMMITMENT_SCHEME
        )
        status = provenance["status"]
        event_name, result_status = {
            "succeeded": ("sustained_attempt_succeeded", "complete"),
            "partial_failure": ("sustained_attempt_partial", "partial_failure"),
            "all_failed": ("sustained_attempt_failed", "all_failed"),
        }[status]
        source_records = [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        attempted = len(source_records)
        completed = sum(record.get("success") is True for record in source_records)
        events.extend(
            [
                {**common, "event": "sustained_attempt_started"},
                {
                    **common,
                    "event": event_name,
                    "private_artifact_commitment": {
                        "scheme": PRIVATE_RECORDS_COMMITMENT_SCHEME,
                        "hmac_sha256": provenance[
                            "private_records_commitment"
                        ]["hmac_sha256"],
                    },
                    "result": {
                        "candidate_id": provenance["candidate_id"],
                        "task": "asr",
                        "workload_class": "private_course",
                        "status": result_status,
                        "counts": {
                            "attempted": attempted,
                            "completed": completed,
                            "failed": attempted - completed,
                        },
                    },
                },
            ]
        )
    agreement_scorer.SUSTAINED_EVENTS_PATH.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _score(
    workload_path: Path,
    candidate_record_paths: dict[str, Path],
) -> dict:
    _bind_records(workload_path, candidate_record_paths)
    return score_asr_agreement(
        workload_path=workload_path,
        candidate_record_paths=candidate_record_paths,
    )


def _two_candidate_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    return (
        tmp_path / "workload.json",
        tmp_path / "left" / "private-records.jsonl",
        tmp_path / "right" / "private-records.jsonl",
    )


def test_small_private_cohort_uses_fixed_arrays_and_suppresses_exact_lengths(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(
        workload,
        [
            {"id": "speech", "duration_seconds": 10, "expected_speech": True},
            {"id": "silence", "duration_seconds": 30, "expected_speech": False},
        ],
    )
    _write_records(
        left,
        [
            {"sample_id": "speech", "success": True, "prediction": "Hello 世界"},
            {"sample_id": "silence", "success": True, "prediction": ""},
        ],
    )
    _write_records(
        right,
        [
            {"sample_id": "speech", "success": True, "prediction": "hello 世介"},
            {"sample_id": "silence", "success": True, "prediction": "noise"},
        ],
    )

    event = _score(workload, {"left": left, "right": right})

    assert event["protocol"] == "asr-text-agreement-v10"
    assert event["public_event_sha256"] == _public_event_sha256(event)
    assert event["scorer_fingerprint"] == agreement_scorer._scorer_fingerprint()
    assert len(event["scorer_fingerprint"]) == 16
    assert set(event["scorer_fingerprint"]) <= set("0123456789abcdef")
    assert event["interpretation"] == {
        "agreement_is_not_ground_truth": True,
        "trusted_gold_used": False,
            "failed_outputs_excluded_from_success_metrics": True,
            "text_only": True,
            "timestamps_compared": False,
            "exact_decoded_pcm_content_uniqueness_verified": True,
            "semantic_audio_independence_verified": False,
            "source_authority_lock_honoring_writers_required": True,
        }
    assert (
        event["privacy"]["workload_meets_minimum_exact_aggregate_denominator"]
        is False
    )
    assert event["privacy"]["private_fingerprints_published"] is False
    assert event["privacy"]["private_run_identifiers_published"] is False
    assert [source["candidate_evidence_id"] for source in event["source_candidates"]] == [
        1,
        2,
    ]
    assert all(
        set(source)
        == {
            "candidate_evidence_id",
            "candidate_id",
            "status",
            "config_index",
            "config_fingerprint",
        }
        for source in event["source_candidates"]
    )
    assert all(
        len(source["config_fingerprint"]) == 16
        and set(source["config_fingerprint"]) <= set("0123456789abcdef")
        for source in event["source_candidates"]
    )

    metrics = event["metrics"]
    assert isinstance(metrics["candidates"], list)
    assert isinstance(metrics["pairs"], list)
    assert [candidate["candidate_evidence_id"] for candidate in metrics["candidates"]] == [
        1,
        2,
    ]
    pair = metrics["pairs"][0]
    assert pair["pair_evidence_id"] == 3
    assert pair["left_candidate_evidence_id"] == 1
    assert pair["right_candidate_evidence_id"] == 2
    assert pair["availability"] == {
        "comparable_sample_count": 2,
        "unavailable_sample_count": 0,
    }
    agreement = pair["successful_output_agreement"]
    assert agreement["exact_character_aggregates_published"] is False
    assert agreement["normalized_character_similarity"] is None
    assert agreement["mean_length_agreement"] is None
    assert agreement["exact_match_count"] is None
    assert agreement["one_empty_disagreement_count"] is None
    assert agreement["normalized_character_similarity_bucket"] is not None
    assert agreement["any_one_empty_disagreement"] is True

    left_metrics, right_metrics = metrics["candidates"]
    for candidate in (left_metrics, right_metrics):
        successful = candidate["successful_output_metrics"]
        assert successful["sample_denominator"] == 2
        assert successful["speech_sample_denominator"] == 1
        assert successful["near_silence_sample_denominator"] == 1
        assert successful["exact_character_aggregates_published"] is False
        assert (
            successful["near_silence_exact_character_aggregates_published"]
            is False
        )
        assert successful["near_silence_successful_seconds"] is None
        assert successful["near_silence_characters_per_minute"] is None
        assert successful["mean_normalized_character_count"] is None
        assert successful["mean_observed_repeated_trigram_ratio"] is None
    assert (
        left_metrics["successful_output_metrics"]
        ["near_silence_normalized_character_count_bucket"]
        == 0
    )
    assert (
        right_metrics["successful_output_metrics"]
        ["near_silence_normalized_character_count_bucket"]
        == 1
    )

    serialized = json.dumps(event, sort_keys=True)
    bound_workload = load_sustained_workload(workload, expected_task="asr")
    assert bound_workload["fingerprint"] not in serialized
    assert _sha256(left) not in serialized
    assert _sha256(right) not in serialized
    assert "records_sha256" not in serialized
    assert "key_hex" not in serialized
    assert "private_records_commitment" not in serialized
    assert "workload_fingerprint" not in serialized
    assert "attempt_id" not in serialized
    assert "attempt_key" not in serialized
    assert "code_fingerprint" not in serialized
    assert "environment_fingerprint" not in serialized
    assert "controller_environment_fingerprint" not in serialized
    assert "execution_policy_fingerprint" not in serialized
    assert "left__right" not in serialized


def test_three_sample_cohort_suppresses_aggregate_character_metrics(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(
        workload,
        [
            {"id": "one", "duration_seconds": 10, "expected_speech": True},
            {"id": "two", "duration_seconds": 60, "expected_speech": False},
            {"id": "three", "duration_seconds": 10, "expected_speech": True},
        ],
    )
    _write_records(
        left,
        [
            {"sample_id": "one", "success": True, "prediction": "abc"},
            {"sample_id": "two", "success": True, "prediction": ""},
            {"sample_id": "three", "success": True, "prediction": "foo"},
        ],
    )
    _write_records(
        right,
        [
            {"sample_id": "one", "success": True, "prediction": "abd"},
            {"sample_id": "two", "success": True, "prediction": "noise"},
            {"sample_id": "three", "success": True, "prediction": "foo"},
        ],
    )

    event = _score(workload, {"left": left, "right": right})

    assert (
        event["privacy"]["workload_meets_minimum_exact_aggregate_denominator"]
        is False
    )
    left_metrics, right_metrics = event["metrics"]["candidates"]
    left_success = left_metrics["successful_output_metrics"]
    right_success = right_metrics["successful_output_metrics"]
    assert left_success["mean_normalized_character_count"] is None
    assert left_success["near_silence_characters_per_minute"] is None
    assert right_success["near_silence_characters_per_minute"] is None
    assert left_success["near_silence_exact_character_aggregates_published"] is False
    pair = event["metrics"]["pairs"][0]["successful_output_agreement"]
    assert pair["exact_character_aggregates_published"] is False
    assert pair["exact_match_count"] is None
    assert pair["one_empty_disagreement_count"] is None
    assert pair["normalized_character_similarity"] is None


def test_ten_sample_cohort_publishes_aggregate_character_metrics(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    items = [
        {"id": f"sample_{index}", "duration_seconds": 10, "expected_speech": True}
        for index in range(10)
    ]
    _write_workload(workload, items)
    _write_records(
        left,
        [
            {"sample_id": item["id"], "success": True, "prediction": "abc"}
            for item in items
        ],
    )
    _write_records(
        right,
        [
            {
                "sample_id": item["id"],
                "success": True,
                "prediction": "abd" if index == 0 else "abc",
            }
            for index, item in enumerate(items)
        ],
    )

    event = _score(workload, {"left": left, "right": right})

    assert (
        event["privacy"]["workload_meets_minimum_exact_aggregate_denominator"]
        is True
    )
    left_success = event["metrics"]["candidates"][0]["successful_output_metrics"]
    assert left_success["mean_normalized_character_count"] == 3
    pair = event["metrics"]["pairs"][0]["successful_output_agreement"]
    assert pair["exact_character_aggregates_published"] is True
    assert pair["exact_match_count"] == 9
    assert pair["one_empty_disagreement_count"] == 0
    assert 0 < pair["normalized_character_similarity"] < 1


def test_near_silence_rate_requires_ten_successful_near_silence_samples(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(
        workload,
        [
            {
                "id": f"silence_{index}",
                "duration_seconds": 60,
                "expected_speech": False,
            }
            for index in range(10)
        ],
    )
    records = [
        {
            "sample_id": f"silence_{index}",
            "success": True,
            "prediction": "x",
        }
        for index in range(10)
    ]
    _write_records(left, records)
    _write_records(right, records)

    event = _score(workload, {"left": left, "right": right})
    successful = event["metrics"]["candidates"][0]["successful_output_metrics"]

    assert successful["near_silence_exact_character_aggregates_published"] is True
    assert successful["near_silence_successful_seconds"] == 600
    assert successful["near_silence_characters_per_minute"] == 1


def test_failed_and_missing_outputs_are_separate_from_success_metrics(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(
        workload,
        [
            {"id": "failed", "duration_seconds": 30, "expected_speech": False},
            {"id": "speech", "duration_seconds": 10, "expected_speech": True},
            {"id": "missing", "duration_seconds": 10, "expected_speech": True},
        ],
    )
    records = [
        {"sample_id": "failed", "success": False, "prediction": "private noise"},
        {"sample_id": "speech", "success": True, "prediction": "ok"},
    ]
    _write_records(left, records)
    _write_records(right, records)

    event = _score(workload, {"left": left, "right": right})
    candidate = event["metrics"]["candidates"][0]

    assert candidate["availability"] == {
        "attempted_sample_count": 3,
        "successful_sample_count": 1,
        "unavailable_sample_count": 2,
    }
    assert candidate["failed_output_diagnostics"] == {
        "explicit_failed_record_count": 1,
        "missing_record_count": 1,
        "any_explicit_failed_output_nonempty": True,
    }
    successful = candidate["successful_output_metrics"]
    assert successful["exact_character_aggregates_published"] is False
    assert successful["mean_normalized_character_count"] is None
    assert successful["near_silence_successful_seconds"] is None
    assert successful["near_silence_characters_per_minute"] is None
    pair = event["metrics"]["pairs"][0]
    assert pair["availability"] == {
        "comparable_sample_count": 1,
        "unavailable_sample_count": 2,
    }
    assert pair["successful_output_agreement"]["sample_denominator"] == 1
    assert candidate["successful_output_metrics"]["sample_denominator"] == 1
    assert (
        candidate["successful_output_metrics"]["speech_sample_denominator"]
        == 1
    )
    assert (
        candidate["successful_output_metrics"]
        ["near_silence_sample_denominator"]
        == 0
    )
    assert "private noise" not in json.dumps(event)


@pytest.mark.parametrize(
    "invalid_record",
    [
        {"sample_id": "sample", "success": "false", "prediction": ""},
        {"sample_id": "sample", "success": 1, "prediction": ""},
        {"sample_id": "sample", "success": True},
        {"sample_id": "sample", "success": True, "prediction": None},
        {"sample_id": "sample", "success": False, "prediction": None},
        {"sample_id": "sample", "success": True, "prediction": ["text"]},
        {"sample_id": ["sample"], "success": True, "prediction": "text"},
    ],
)
def test_requires_strict_boolean_success_and_string_prediction(
    tmp_path: Path,
    invalid_record: dict,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    _write_records(left, [invalid_record])
    _write_records(
        right,
        [{"sample_id": "sample", "success": True, "prediction": "ok"}],
    )
    _bind_records(workload, {"left": left, "right": right})

    with pytest.raises(ValueError, match="invalid ASR agreement record"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_failed_record_may_omit_prediction(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    _write_records(left, [{"sample_id": "sample", "success": False}])
    _write_records(
        right,
        [{"sample_id": "sample", "success": True, "prediction": "ok"}],
    )

    event = _score(workload, {"left": left, "right": right})

    left_metrics = event["metrics"]["candidates"][0]
    assert left_metrics["availability"]["successful_sample_count"] == 0
    assert (
        left_metrics["failed_output_diagnostics"]
        ["any_explicit_failed_output_nonempty"]
        is False
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("status", "complete"),
        ("candidate_id", "D:\\private\\candidate"),
        ("task", "ocr"),
        ("phase", "compatibility"),
        ("workload_class", "public_course"),
        ("workload_fingerprint", "f" * 64),
        ("records_sha256", "f" * 64),
        ("attempt_id", "not-a-uuid"),
        ("controller_environment_fingerprint", None),
        ("execution_policy_fingerprint", "f" * 64),
    ],
)
def test_rejects_unbound_or_unsuccessful_runner_provenance(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[field] = invalid_value
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance is invalid"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_accepts_and_publishes_all_exact_runner_statuses(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    _write_workload(
        workload,
        [
            {"id": "one", "duration_seconds": 1},
            {"id": "two", "duration_seconds": 1},
        ],
    )
    record_paths = {
        name: tmp_path / name / "private-records.jsonl"
        for name in ("all_failed", "partial", "succeeded")
    }
    _write_records(
        record_paths["succeeded"],
        [
            {"sample_id": "one", "success": True, "prediction": "one"},
            {"sample_id": "two", "success": True, "prediction": "two"},
        ],
    )
    _write_records(
        record_paths["partial"],
        [
            {"sample_id": "one", "success": True, "prediction": "one"},
            {"sample_id": "two", "success": False},
        ],
    )
    _write_records(
        record_paths["all_failed"],
        [
            {"sample_id": "one", "success": False},
            {"sample_id": "two", "success": False},
        ],
    )

    event = _score(workload, record_paths)

    assert {
        source["candidate_id"]: source["status"]
        for source in event["source_candidates"]
    } == {
        "faster_whisper_cpu": "all_failed",
        "sensevoice_small_gguf_cpu": "partial_failure",
        "qwen3_asr_0_6b_openvino_genai_official": "succeeded",
    }


def test_rejects_source_status_that_disagrees_with_records(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["status"] = "partial_failure"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="source status does not match records"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_records_mutated_after_provenance_binding(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    _write_records(left, [{**record, "prediction": "mutated"}])

    with pytest.raises(ValueError, match="provenance is invalid"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_records_and_sidecar_recommitted_after_terminal_journal(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    _write_records(left, [{**record, "prediction": "attacker rewrite"}])
    _recommit_records(left)

    with pytest.raises(ValueError, match="source commitment mismatch"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_duplicate_attempt_ids_and_attempt_keys(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    left_provenance = json.loads(
        left.with_name("records-provenance.json").read_text(encoding="utf-8")
    )
    right_path = right.with_name("records-provenance.json")
    right_provenance = json.loads(right_path.read_text(encoding="utf-8"))
    right_provenance["attempt_id"] = left_provenance["attempt_id"]
    right_path.write_text(json.dumps(right_provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="distinct attempts"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )

    right_provenance["attempt_id"] = str(
        uuid.UUID("00000000-0000-4000-8000-000000000002")
    )
    right_provenance["attempt_key"] = left_provenance["attempt_key"]
    right_path.write_text(json.dumps(right_provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct attempt keys"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_duplicate_record_files_and_provenance_sidecars(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _bind_records(workload, {"left": left})

    with pytest.raises(ValueError, match="distinct record files"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": left},
        )

    shared = tmp_path / "shared"
    first = shared / "one.jsonl"
    second = shared / "two.jsonl"
    _write_records(first, [record])
    _write_records(second, [record])
    _bind_records(workload, {"left": first})
    with pytest.raises(ValueError, match="distinct provenance sidecars"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": first, "right": second},
        )


def test_repeated_runner_candidate_ids_remain_distinct_comparison_candidates(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"cpu": left, "gpu": right})
    faster = _registered_candidate("faster_whisper_cpu")
    for path, config_index in (
        (left.with_name("records-provenance.json"), 19),
        (right.with_name("records-provenance.json"), 20),
    ):
        provenance = json.loads(path.read_text(encoding="utf-8"))
        provenance["candidate_id"] = "faster_whisper_cpu"
        provenance["config_index"] = config_index
        provenance["config"] = faster["configs"][config_index]
        path.write_text(json.dumps(provenance), encoding="utf-8")
        _recommit_records(path.with_name("private-records.jsonl"))
    _rewrite_sustained_journal(workload, (left, right))

    event = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"cpu": left, "gpu": right},
    )

    assert [source["candidate_id"] for source in event["source_candidates"]] == [
        "faster_whisper_cpu",
        "faster_whisper_cpu",
    ]
    assert [source["candidate_evidence_id"] for source in event["source_candidates"]] == [
        1,
        2,
    ]


def test_source_config_fingerprint_changes_without_publishing_raw_config(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    baseline = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"left": left, "right": right},
    )
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["config_index"] = 20
    provenance["config"] = _registered_candidate("faster_whisper_cpu")["configs"][20]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    _recommit_records(left)
    _rewrite_sustained_journal(workload, (left, right))

    changed = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"left": left, "right": right},
    )

    assert (
        baseline["source_candidates"][0]["config_fingerprint"]
        != changed["source_candidates"][0]["config_fingerprint"]
    )
    assert "threads_per_worker" not in json.dumps(changed)


def test_rejects_bounded_but_unregistered_candidate_identity(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["candidate_id"] = "teacher_alice"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate is not registered"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


@pytest.mark.parametrize("mutation", ["config_index", "config"])
def test_rejects_candidate_config_that_does_not_match_registry(
    tmp_path: Path,
    mutation: str,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if mutation == "config_index":
        provenance["config_index"] = 20
    else:
        provenance["config"] = {**provenance["config"], "threads_per_worker": 99}
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the registry"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


@pytest.mark.parametrize("substitute", [True, 1.0])
def test_registry_config_binding_is_type_strict(
    tmp_path: Path,
    substitute: object,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["config"]["processes"] = substitute
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the registry"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_candidate_level_retired_source(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    retired = _registered_candidate("qwen3_asr_0_6b_cpu")
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["candidate_id"] = retired["id"]
    provenance["config_index"] = 0
    provenance["config"] = retired["configs"][0]
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the registry"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_config_level_quality_phase_exclusion() -> None:
    provenance = {
        "candidate_id": "active_candidate",
        "config_index": 0,
        "config": {"processes": 1, "phases": ["sustained"]},
    }
    registered = {
        "active_candidate": {
            "id": "active_candidate",
            "task": "asr",
            "configs": [provenance["config"]],
        }
    }

    with pytest.raises(ValueError, match="does not match the registry"):
        agreement_scorer._validate_registered_source(
            provenance,
            registered_sources=registered,
        )


def test_rejects_duplicate_private_workload_content(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    shared = tmp_path / "shared.wav"
    _write_test_wav(shared, duration_seconds=1.0, seed="shared")
    items = [
        {
            "id": f"sample_{index}",
            "path": shared.name,
            "duration_seconds": 1,
        }
        for index in range(10)
    ]
    _write_workload(workload, items)
    records = [
        {"sample_id": item["id"], "success": True, "prediction": "ok"}
        for item in items
    ]
    _write_records(left, records)
    _write_records(right, records)

    with pytest.raises(ValueError, match="distinct workload item content"):
        _score(workload, {"left": left, "right": right})


def test_rejects_source_attempt_absent_from_sustained_journal(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    agreement_scorer.SUSTAINED_EVENTS_PATH.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="absent from the journal"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_raw_distinct_wavs_with_identical_decoded_pcm(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    items = []
    for index in range(10):
        audio = tmp_path / f"wrapped_{index}.wav"
        _write_test_wav(audio, duration_seconds=1.0, seed="identical-pcm")
        with audio.open("ab") as handle:
            handle.write(f"wrapper-{index}".encode("ascii"))
        items.append(
            {
                "id": f"sample_{index}",
                "path": audio.name,
                "duration_seconds": 1,
            }
        )
    _write_workload(workload, items)
    records = [
        {"sample_id": item["id"], "success": True, "prediction": "ok"}
        for item in items
    ]
    _write_records(left, records)
    _write_records(right, records)
    _bind_records(workload, {"left": left, "right": right})

    with pytest.raises(ValueError, match="distinct workload item content"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_source_journal_identity_mismatch(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    events = [
        json.loads(line)
        for line in agreement_scorer.SUSTAINED_EVENTS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    events[1]["code_fingerprint"] = "f" * 16
    agreement_scorer.SUSTAINED_EVENTS_PATH.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="journal identity mismatch"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("protocol", "sustained-process-forged"),
        ("target_wall_seconds", 999.0),
    ],
)
def test_rejects_lifecycle_fields_changed_on_start_and_terminal(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    events = [
        json.loads(line)
        for line in agreement_scorer.SUSTAINED_EVENTS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    events[0][field] = replacement
    events[1][field] = replacement
    agreement_scorer.SUSTAINED_EVENTS_PATH.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="journal identity mismatch"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_terminal_counts_that_disagree_with_bound_records(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    events = [
        json.loads(line)
        for line in agreement_scorer.SUSTAINED_EVENTS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    events[1]["result"]["counts"] = {
        "attempted": 1,
        "completed": 0,
        "failed": 1,
    }
    agreement_scorer.SUSTAINED_EVENTS_PATH.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="journal outcome mismatch"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_records_are_parsed_and_committed_from_one_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "bound"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    original_reader = agreement_scorer._read_records
    reader_calls = 0

    def mutate_path_after_snapshot(records_bytes: bytes, expected_ids: set[str]):
        nonlocal reader_calls
        reader_calls += 1
        if reader_calls == 1:
            _write_records(
                left,
                [
                    {
                        "sample_id": "sample",
                        "success": True,
                        "prediction": "forged",
                    }
                ],
            )
        return original_reader(records_bytes, expected_ids)

    monkeypatch.setattr(
        agreement_scorer,
        "_read_records",
        mutate_path_after_snapshot,
    )

    event = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"left": left, "right": right},
    )

    assert event["metrics"]["candidates"][0]["availability"][
        "successful_sample_count"
    ] == 1


def test_pcm_uniqueness_uses_the_workload_bound_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    items = [
        {"id": f"sample_{index}", "duration_seconds": 1}
        for index in range(10)
    ]
    _write_workload(workload, items)
    records = [
        {"sample_id": item["id"], "success": True, "prediction": "ok"}
        for item in items
    ]
    _write_records(left, records)
    _write_records(right, records)
    _bind_records(workload, {"left": left, "right": right})
    original_reader = agreement_scorer._read_bound_workload_bytes
    reader_calls = 0

    def mutate_path_after_binding(path: Path, content_binding: object) -> bytes:
        nonlocal reader_calls
        reader_calls += 1
        if reader_calls == 1:
            _write_test_wav(
                path,
                duration_seconds=1.0,
                seed="changed-after-binding",
            )
        return original_reader(path, content_binding)

    monkeypatch.setattr(
        agreement_scorer,
        "_read_bound_workload_bytes",
        mutate_path_after_binding,
    )

    with pytest.raises(ValueError, match="content changed after binding"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_effectively_invalidated_source_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    left_provenance = json.loads(
        left.with_name("records-provenance.json").read_text(encoding="utf-8")
    )
    snapshot = agreement_scorer.capture_sustained_journal_snapshot(
        agreement_scorer.SUSTAINED_EVENTS_PATH
    )
    monkeypatch.setattr(
        agreement_scorer,
        "capture_sustained_journal_snapshot",
        lambda _path: replace(
            snapshot,
            invalidated_attempt_ids=frozenset({left_provenance["attempt_id"]}),
        ),
    )

    with pytest.raises(ValueError, match="source attempt is invalidated"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_malformed_active_correction_targets(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    provenance = json.loads(
        left.with_name("records-provenance.json").read_text(encoding="utf-8")
    )
    existing = agreement_scorer.SUSTAINED_EVENTS_PATH.read_text(encoding="utf-8")
    malformed = {
        "event": "sustained_attempts_reclassified",
        "reclassified_attempt_ids": provenance["attempt_id"],
        "reclassified_status": "partial_failure",
        "reason_kind": "terminal_event_name_did_not_reflect_item_failures",
    }
    agreement_scorer.SUSTAINED_EVENTS_PATH.write_text(
        existing + json.dumps(malformed) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid_active_correction_targets"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_registry_change_during_scoring_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "sustained-registry.json"
    registry_path.write_bytes(agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes())
    monkeypatch.setattr(agreement_scorer, "SUSTAINED_REGISTRY_PATH", registry_path)
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    original_loader = agreement_scorer._load_registered_asr_sources

    def mutate_after_snapshot(path: Path, *, registry_bytes: bytes | None = None):
        sources = original_loader(path, registry_bytes=registry_bytes)
        registry = json.loads(path.read_text(encoding="utf-8"))
        registry["candidates"][0]["status"] = "retired_by_test"
        path.write_text(json.dumps(registry), encoding="utf-8")
        return sources

    monkeypatch.setattr(
        agreement_scorer,
        "_load_registered_asr_sources",
        mutate_after_snapshot,
    )

    with pytest.raises(ValueError, match="authority changed during scoring"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_journal_change_during_final_event_construction_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    original_fingerprint = agreement_scorer._scorer_fingerprint

    def mutate_inside_fingerprint(**kwargs):
        existing = agreement_scorer.SUSTAINED_EVENTS_PATH.read_text(encoding="utf-8")
        agreement_scorer.SUSTAINED_EVENTS_PATH.write_text(
            existing + json.dumps({"event": "authority_changed_by_test"}) + "\n",
            encoding="utf-8",
        )
        return original_fingerprint(**kwargs)

    monkeypatch.setattr(
        agreement_scorer,
        "_scorer_fingerprint",
        mutate_inside_fingerprint,
    )

    with pytest.raises(ValueError, match="authority changed during scoring"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_workload_gate_and_loader_share_one_manifest_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    original_loader = agreement_scorer.load_sustained_workload_from_bytes

    def mutate_path_before_load(snapshot: bytes, *, manifest_path: Path, expected_task: str):
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        document["workload_class"] = "public_course"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")
        return original_loader(
            snapshot,
            manifest_path=manifest_path,
            expected_task=expected_task,
        )

    monkeypatch.setattr(
        agreement_scorer,
        "load_sustained_workload_from_bytes",
        mutate_path_before_load,
    )

    event = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"left": left, "right": right},
    )

    assert event["workload_class"] == "private_course"


def test_rejects_extra_provenance_fields_and_duplicate_record_keys(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})
    provenance_path = left.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["private_path"] = "secret"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="records provenance is invalid"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )

    provenance.pop("private_path")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    left.write_text(
        '{"sample_id":"sample","sample_id":"sample","success":true,"prediction":"ok"}\n',
        encoding="utf-8",
    )
    _recommit_records(left)
    with pytest.raises(ValueError, match="invalid ASR agreement record"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_alias_order_does_not_change_public_event_identity(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"first_alias": left, "second_alias": right})

    first = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"first_alias": left, "second_alias": right},
    )
    reordered = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"z_alias": right, "a_alias": left},
    )

    assert first["source_candidates"] == reordered["source_candidates"]
    assert first["metrics"] == reordered["metrics"]
    assert first["public_event_sha256"] == reordered["public_event_sha256"]


def test_public_event_append_is_semantically_deduplicated(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    event = _score(workload, {"left": left, "right": right})
    journal = tmp_path / "quality-events.jsonl"

    assert _append_public_event_once(journal, event) is True
    assert _append_public_event_once(journal, event) is False
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 1


def test_public_event_append_rejects_rebound_or_copied_event(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    event = _score(workload, {"left": left, "right": right})
    copied = dict(event)
    copied["source_authority_fingerprint"] = "0" * 16
    copied["public_event_sha256"] = _public_event_sha256(copied)

    with pytest.raises(ValueError, match="not authorized by this scoring process"):
        _append_public_event_once(tmp_path / "quality-events.jsonl", copied)


def test_public_event_append_rejects_stale_source_authority(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    event = _score(workload, {"left": left, "right": right})
    quality_journal = tmp_path / "quality-events.jsonl"
    existing = agreement_scorer.SUSTAINED_EVENTS_PATH.read_text(encoding="utf-8")
    agreement_scorer.SUSTAINED_EVENTS_PATH.write_text(
        existing + json.dumps({"event": "authority_changed_by_test"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="authority changed before append"):
        _append_public_event_once(quality_journal, event)

    assert not quality_journal.exists()


def test_public_event_append_holds_registry_lock_through_quality_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_path = tmp_path / "sustained-candidates.json"
    registry_path.write_bytes(agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes())
    monkeypatch.setattr(agreement_scorer, "SUSTAINED_REGISTRY_PATH", registry_path)
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    event = _score(workload, {"left": left, "right": right})
    quality_journal = tmp_path / "quality-events.jsonl"
    real_quality_append = agreement_scorer.append_event_once
    executor = ThreadPoolExecutor(max_workers=1)
    writer_started = threading.Event()
    writer_future = None
    writer_was_blocked: list[bool] = []

    def write_registry() -> None:
        writer_started.set()
        with _locked_journal(registry_path, exclusive=True, create=False) as handle:
            handle.seek(0)
            registry = json.loads(handle.read().decode("utf-8"))
            registry["candidates"][0]["configs"][0]["workers"] = 99
            encoded = (json.dumps(registry, sort_keys=True) + "\n").encode("utf-8")
            handle.seek(0)
            handle.write(encoded)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())

    def observed_quality_append(path, public_event, **kwargs):
        nonlocal writer_future
        writer_future = executor.submit(write_registry)
        assert writer_started.wait(timeout=2)
        time.sleep(0.1)
        writer_was_blocked.append(not writer_future.done())
        return real_quality_append(path, public_event, **kwargs)

    monkeypatch.setattr(
        agreement_scorer,
        "append_event_once",
        observed_quality_append,
    )
    try:
        assert _append_public_event_once(quality_journal, event) is True
    finally:
        if writer_future is not None:
            writer_future.result(timeout=5)
        executor.shutdown(wait=True)

    assert writer_was_blocked == [True]


def test_public_event_append_rejects_sustained_journal_as_output(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    event = _score(workload, {"left": left, "right": right})
    before = agreement_scorer.SUSTAINED_EVENTS_PATH.read_bytes()

    with pytest.raises(ValueError, match="cannot be the sustained journal"):
        _append_public_event_once(agreement_scorer.SUSTAINED_EVENTS_PATH, event)

    assert agreement_scorer.SUSTAINED_EVENTS_PATH.read_bytes() == before


def test_public_event_append_rejects_sustained_registry_as_output(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    event = _score(workload, {"left": left, "right": right})
    before = agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes()

    with pytest.raises(ValueError, match="cannot be the sustained registry"):
        _append_public_event_once(agreement_scorer.SUSTAINED_REGISTRY_PATH, event)

    assert agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes() == before


def test_public_event_append_rejects_registry_hardlink_as_output(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    event = _score(workload, {"left": left, "right": right})
    registry_path = agreement_scorer.SUSTAINED_REGISTRY_PATH
    registry_hardlink = tmp_path / "registry-hardlink.json"
    try:
        os.link(registry_path, registry_hardlink)
    except OSError as error:
        pytest.skip(f"hardlinks unavailable: {error}")
    before = registry_path.read_bytes()

    with pytest.raises(ValueError, match="cannot be the sustained registry"):
        _append_public_event_once(registry_hardlink, event)

    assert registry_path.read_bytes() == before


def _exact_asr_event(tmp_path: Path) -> dict:
    workload, left, right = _two_candidate_paths(tmp_path)
    items = [
        {
            "id": f"sample_{index}",
            "duration_seconds": 1,
            "expected_speech": index < 10,
        }
        for index in range(20)
    ]
    _write_workload(workload, items)
    records = [
        {"sample_id": item["id"], "success": True, "prediction": "abc"}
        for item in items
    ]
    _write_records(left, records)
    _write_records(right, records)
    return _score(workload, {"left": left, "right": right})


@pytest.mark.parametrize(
    "contradiction",
    [
        "character_mean_bucket",
        "repetition_mean_flag",
        "zero_near_silence_duration",
        "negative_zero",
        "zero_near_silence_cpm",
        "huge_near_silence_cpm",
        "nonintegral_near_silence_cpm",
        "too_few_characters_for_nonempty_outputs",
    ],
)
def test_public_validator_rejects_exact_metric_contradictions(
    tmp_path: Path,
    contradiction: str,
) -> None:
    event = _exact_asr_event(tmp_path)
    successful = event["metrics"]["candidates"][0]["successful_output_metrics"]
    if contradiction == "character_mean_bucket":
        successful["mean_normalized_character_count"] = 0.0
    elif contradiction == "repetition_mean_flag":
        successful["mean_observed_repeated_trigram_ratio"] = 0.1
    elif contradiction == "zero_near_silence_duration":
        successful["near_silence_successful_seconds"] = 0.0
    elif contradiction == "negative_zero":
        successful["mean_observed_repeated_trigram_ratio"] = -0.0
    elif contradiction == "zero_near_silence_cpm":
        successful["near_silence_characters_per_minute"] = 0.0
    elif contradiction == "huge_near_silence_cpm":
        successful["near_silence_characters_per_minute"] = 1_000_000_000.0
    elif contradiction == "nonintegral_near_silence_cpm":
        successful["near_silence_characters_per_minute"] = 1.0
    else:
        successful["mean_normalized_character_count"] = 0.5
        successful["normalized_character_count_bucket"] = 1
    event["public_event_sha256"] = _public_event_sha256(event)

    with pytest.raises(ValueError, match="public event is invalid"):
        agreement_scorer._validate_asr_public_event(
            event,
            registry_bytes=agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes(),
        )


def test_public_validator_rejects_jointly_impossible_three_source_overlaps(
    tmp_path: Path,
) -> None:
    workload = tmp_path / "workload.json"
    items = [
        {"id": "sample_1", "duration_seconds": 1},
        {"id": "sample_2", "duration_seconds": 1},
    ]
    _write_workload(workload, items)
    paths = {
        alias: tmp_path / alias / "private-records.jsonl"
        for alias in ("first", "second", "third")
    }
    success_patterns = ((True, False), (True, False), (False, True))
    for path, successes in zip(paths.values(), success_patterns):
        _write_records(
            path,
            [
                {
                    "sample_id": item["id"],
                    "success": success,
                    **({"prediction": "ok"} if success else {}),
                }
                for item, success in zip(items, successes)
            ],
        )
    event = _score(workload, paths)
    assert [
        pair["availability"]["comparable_sample_count"]
        for pair in event["metrics"]["pairs"]
    ] == [0, 1, 0]
    forged_pair = event["metrics"]["pairs"][0]
    forged_pair["availability"] = {
        "comparable_sample_count": 1,
        "unavailable_sample_count": 1,
    }
    forged_pair["successful_output_agreement"].update(
        {
            "sample_denominator": 1,
            "normalized_character_similarity_bucket": 4,
            "mean_length_agreement_bucket": 4,
        }
    )
    event["public_event_sha256"] = _public_event_sha256(event)

    with pytest.raises(ValueError, match="public event is invalid"):
        agreement_scorer._validate_asr_public_event(
            event,
            registry_bytes=agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes(),
        )


@pytest.mark.parametrize(
    "contradiction",
    [
        "zero_denominator",
        "oversized_denominator",
        "one_empty_with_perfect_length",
        "nonintegral_edit_distance",
        "too_many_exact_matches_for_length_mean",
        "too_little_edit_distance_for_nonmatches",
    ],
)
def test_public_validator_rejects_exact_pair_arithmetic_contradictions(
    tmp_path: Path,
    contradiction: str,
) -> None:
    event = _exact_asr_event(tmp_path)
    agreement = event["metrics"]["pairs"][0]["successful_output_agreement"]
    if contradiction == "zero_denominator":
        agreement["normalized_character_denominator"] = 0
    elif contradiction == "oversized_denominator":
        agreement["normalized_character_denominator"] = 1_000
    elif contradiction == "one_empty_with_perfect_length":
        agreement.update(
            {
                "normalized_character_similarity": 59 / 60,
                "normalized_character_similarity_bucket": 4,
                "exact_match_count": 19,
                "one_empty_disagreement_count": 1,
                "any_one_empty_disagreement": True,
                "all_comparable_exact_matches": False,
            }
        )
    elif contradiction == "nonintegral_edit_distance":
        agreement.update(
            {
                "normalized_character_similarity": 0.991,
                "normalized_character_similarity_bucket": 4,
                "exact_match_count": 19,
                "all_comparable_exact_matches": False,
            }
        )
    elif contradiction == "too_many_exact_matches_for_length_mean":
        agreement.update(
            {
                "normalized_character_similarity": 59 / 60,
                "normalized_character_similarity_bucket": 4,
                "mean_length_agreement": 0.1,
                "mean_length_agreement_bucket": 0,
                "exact_match_count": 19,
                "all_comparable_exact_matches": False,
            }
        )
    else:
        agreement.update(
            {
                "normalized_character_similarity": 59 / 60,
                "normalized_character_similarity_bucket": 4,
                "mean_length_agreement": 0.5,
                "mean_length_agreement_bucket": 2,
                "exact_match_count": 0,
                "all_comparable_exact_matches": False,
            }
        )
    event["public_event_sha256"] = _public_event_sha256(event)

    with pytest.raises(ValueError, match="public event is invalid"):
        agreement_scorer._validate_asr_public_event(
            event,
            registry_bytes=agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes(),
        )


@pytest.mark.parametrize(
    "contradiction",
    [
        "overall_zero_bucket",
        "near_zero_bucket",
        "near_bucket_above_overall",
        "one_empty_with_perfect_length_bucket",
    ],
)
def test_public_validator_rejects_suppressed_exact_bucket_contradictions(
    tmp_path: Path,
    contradiction: str,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    items = [
        {
            "id": "speech",
            "duration_seconds": 1,
            "expected_speech": True,
        },
        {
            "id": "near_silence",
            "duration_seconds": 1,
            "expected_speech": False,
        },
    ]
    _write_workload(workload, items)
    left_records = [
        {"sample_id": "speech", "success": True, "prediction": "a"},
        {"sample_id": "near_silence", "success": True, "prediction": "b"},
    ]
    right_records = [dict(record) for record in left_records]
    _write_records(left, left_records)
    _write_records(right, right_records)
    event = _score(workload, {"left": left, "right": right})
    successful = event["metrics"]["candidates"][0]["successful_output_metrics"]
    agreement = event["metrics"]["pairs"][0]["successful_output_agreement"]
    if contradiction == "overall_zero_bucket":
        successful["normalized_character_count_bucket"] = 0
    elif contradiction == "near_zero_bucket":
        successful["near_silence_normalized_character_count_bucket"] = 0
    elif contradiction == "near_bucket_above_overall":
        successful["near_silence_normalized_character_count_bucket"] = 5
    else:
        agreement["any_one_empty_disagreement"] = True
    event["public_event_sha256"] = _public_event_sha256(event)

    with pytest.raises(ValueError, match="public event is invalid"):
        agreement_scorer._validate_asr_public_event(
            event,
            registry_bytes=agreement_scorer.SUSTAINED_REGISTRY_PATH.read_bytes(),
        )


def test_private_asr_record_decoder_normalizes_excessive_nesting() -> None:
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    records = b'{"sample_id":"sample","success":true,"prediction":' + nested + b"}\n"

    with pytest.raises(ValueError, match="invalid ASR agreement record"):
        agreement_scorer._read_records(records, {"sample"})


def test_ten_items_with_nine_comparable_outputs_keep_exact_metrics_suppressed(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    items = [
        {"id": f"sample_{index}", "duration_seconds": 1}
        for index in range(10)
    ]
    _write_workload(workload, items)
    _write_records(
        left,
        [
            {
                "sample_id": item["id"],
                "success": index != 9,
                **({"prediction": "ok"} if index != 9 else {}),
            }
            for index, item in enumerate(items)
        ],
    )
    _write_records(
        right,
        [
            {"sample_id": item["id"], "success": True, "prediction": "ok"}
            for item in items
        ],
    )

    event = _score(workload, {"left": left, "right": right})

    assert (
        event["privacy"]["workload_meets_minimum_exact_aggregate_denominator"]
        is True
    )
    assert (
        event["metrics"]["candidates"][0]["successful_output_metrics"]
        ["exact_character_aggregates_published"]
        is False
    )
    pair = event["metrics"]["pairs"][0]["successful_output_agreement"]
    assert pair["sample_denominator"] == 9
    assert pair["exact_character_aggregates_published"] is False
    assert pair["normalized_character_denominator"] is None


def test_similarity_publishes_character_micro_denominator(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    items = [
        {"id": f"sample_{index}", "duration_seconds": 1}
        for index in range(10)
    ]
    _write_workload(workload, items)
    _write_records(
        left,
        [
            {"sample_id": item["id"], "success": True, "prediction": "a"}
            for item in items
        ],
    )
    _write_records(
        right,
        [
            {
                "sample_id": item["id"],
                "success": True,
                "prediction": "aaaaaaaaa" if index == 0 else "a",
            }
            for index, item in enumerate(items)
        ],
    )

    event = _score(workload, {"left": left, "right": right})
    pair = event["metrics"]["pairs"][0]["successful_output_agreement"]

    assert pair["normalized_character_similarity_is_character_micro_weighted"] is True
    assert pair["normalized_character_denominator"] == 18
    assert pair["normalized_character_similarity"] == pytest.approx(10 / 18)


def test_scorer_fingerprint_frames_dependency_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = tmp_path / "a.py"
    right = tmp_path / "b.py"
    left.write_bytes(b"X")
    right.write_bytes(b"b.py\0Y")
    monkeypatch.setattr(
        agreement_scorer,
        "_scorer_dependency_paths",
        lambda: [left, right],
    )
    initial = agreement_scorer._scorer_fingerprint()

    # This collides under the former path-name/NUL/raw-content concatenation.
    left.write_bytes(b"Xb.py\0")
    right.write_bytes(b"Y")

    assert agreement_scorer._scorer_fingerprint() != initial


def test_pair_ids_do_not_depend_on_alias_delimiters(tmp_path: Path) -> None:
    workload = tmp_path / "workload.json"
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    aliases = ["a", "a__b", "b__c"]
    paths = {
        alias: tmp_path / alias / "private-records.jsonl" for alias in aliases
    }
    for path in paths.values():
        _write_records(
            path,
            [{"sample_id": "sample", "success": True, "prediction": "ok"}],
        )

    event = _score(workload, paths)

    assert len(event["source_candidates"]) == len(aliases)
    assert [pair["pair_evidence_id"] for pair in event["metrics"]["pairs"]] == [
        4,
        5,
        6,
    ]
    assert [
        (pair["left_candidate_evidence_id"], pair["right_candidate_evidence_id"])
        for pair in event["metrics"]["pairs"]
    ] == [(1, 2), (1, 3), (2, 3)]


def test_rejects_unknown_and_duplicate_record_ids(tmp_path: Path) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    _write_records(
        left,
        [{"sample_id": "unknown", "success": True, "prediction": "ok"}],
    )
    _write_records(
        right,
        [{"sample_id": "sample", "success": True, "prediction": "ok"}],
    )
    _bind_records(workload, {"left": left, "right": right})
    with pytest.raises(ValueError, match="invalid ASR agreement record"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )

    duplicate = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [duplicate, duplicate])
    _bind_records(workload, {"left": left, "right": right})
    with pytest.raises(ValueError, match="invalid ASR agreement record"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_bounds_candidate_and_workload_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = tmp_path / "workload.json"
    _write_workload(
        workload,
        [
            {"id": "one", "duration_seconds": 1},
            {"id": "two", "duration_seconds": 1},
        ],
    )
    monkeypatch.setattr(agreement_scorer, "_MAX_WORKLOAD_ITEMS", 1)
    with pytest.raises(ValueError, match="workload item budget"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={
                "left": tmp_path / "left.jsonl",
                "right": tmp_path / "right.jsonl",
            },
        )

    with pytest.raises(ValueError, match="two to eight"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"only": tmp_path / "only.jsonl"},
        )
    with pytest.raises(ValueError, match="two to eight"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={
                f"candidate_{index}": tmp_path / f"{index}.jsonl"
                for index in range(9)
            },
        )


@pytest.mark.parametrize(
    ("constant", "limit", "expected_message"),
    [
        ("_MAX_PREDICTION_CHARACTERS", 2, "invalid ASR agreement record"),
        ("_MAX_TOTAL_PREDICTION_CHARACTERS", 3, "total prediction budget"),
        ("_MAX_RECORDS_PER_CANDIDATE", 1, "record count budget"),
    ],
)
def test_bounds_record_and_prediction_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    constant: str,
    limit: int,
    expected_message: str,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(
        workload,
        [
            {"id": "one", "duration_seconds": 1},
            {"id": "two", "duration_seconds": 1},
        ],
    )
    records = [
        {"sample_id": "one", "success": True, "prediction": "abc"},
        {"sample_id": "two", "success": True, "prediction": "abc"},
    ]
    _write_records(left, records)
    _write_records(right, records)
    _bind_records(workload, {"left": left, "right": right})
    monkeypatch.setattr(agreement_scorer, constant, limit)

    with pytest.raises(ValueError, match=expected_message):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_bounds_record_and_provenance_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(workload, [{"id": "sample", "duration_seconds": 1}])
    record = {"sample_id": "sample", "success": True, "prediction": "ok"}
    _write_records(left, [record])
    _write_records(right, [record])
    _bind_records(workload, {"left": left, "right": right})

    monkeypatch.setattr(agreement_scorer, "_MAX_RECORD_FILE_BYTES", left.stat().st_size - 1)
    with pytest.raises(ValueError, match="record file byte budget"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )

    monkeypatch.setattr(agreement_scorer, "_MAX_RECORD_FILE_BYTES", 1_000_000)
    provenance_path = left.with_name("records-provenance.json")
    monkeypatch.setattr(
        agreement_scorer,
        "_MAX_PROVENANCE_BYTES",
        provenance_path.stat().st_size - 1,
    )
    with pytest.raises(ValueError, match="provenance byte budget"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )


def test_rejects_invalid_candidate_specs_and_private_workload_class(
    tmp_path: Path,
) -> None:
    workload, left, right = _two_candidate_paths(tmp_path)
    _write_workload(
        workload,
        [{"id": "sample", "duration_seconds": 1}],
        workload_class="public_course",
    )
    with pytest.raises(ValueError, match="private_course"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": left, "right": right},
        )
    with pytest.raises(ValueError, match="bounded public values"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"D:\\private\\course": left, "right": right},
        )
    with pytest.raises(ValueError, match="bounded public values"):
        score_asr_agreement(
            workload_path=workload,
            candidate_record_paths={"left": "left.jsonl", "right": right},
        )


def test_parse_candidate_specs_is_bounded_and_delimiter_independent() -> None:
    parsed = _parse_candidate_specs(["left__cpu=left.jsonl", "right=right.jsonl"])
    assert parsed == {
        "left__cpu": Path("left.jsonl"),
        "right": Path("right.jsonl"),
    }
    with pytest.raises(ValueError, match="two to eight"):
        _parse_candidate_specs(["only=one.jsonl"])
    with pytest.raises(ValueError, match="unique"):
        _parse_candidate_specs(["same=one.jsonl", "same=two.jsonl"])


def test_bounds_edit_distance_work() -> None:
    budget = [3]

    with pytest.raises(ValueError, match="edit-distance budget"):
        _bounded_levenshtein("ab", "cd", budget)

    assert _bounded_levenshtein("same", "same", budget) == 0
    assert budget == [3]
