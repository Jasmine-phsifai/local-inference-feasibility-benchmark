import hashlib
import json
import uuid
from pathlib import Path

import pytest

import local_inference_bench.score_asr_agreement as agreement_scorer
from local_inference_bench.load_sustained_workload import load_sustained_workload
from local_inference_bench.score_asr_agreement import _bounded_levenshtein
from local_inference_bench.score_asr_agreement import _parse_candidate_specs
from local_inference_bench.score_asr_agreement import score_asr_agreement


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
            input_path = path.parent / f"{item['id']}.bin"
            input_path.write_bytes(item["id"].encode("ascii"))
            item["path"] = input_path.name
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_records(
    workload_path: Path,
    candidate_record_paths: dict[str, Path],
) -> None:
    workload = load_sustained_workload(workload_path, expected_task="asr")
    expected_ids = {item["id"] for item in workload["items"]}
    for index, (candidate_id, records_path) in enumerate(
        candidate_record_paths.items()
    ):
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
            "config": {"processes": 1},
            "config_index": index,
            "phase": "quality",
            "trial_index": 0,
            "workload_class": "private_course",
            "workload_fingerprint": workload["fingerprint"],
            "code_fingerprint": "a" * 16,
            "environment_fingerprint": "b" * 16,
            "controller_environment_fingerprint": "c" * 16,
            "execution_policy_fingerprint": "d" * 16,
            "records_sha256": _sha256(records_path),
        }
        records_path.with_name("records-provenance.json").write_text(
            json.dumps(provenance),
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

    assert event["protocol"] == "asr-text-agreement-v8"
    assert event["scorer_fingerprint"] == agreement_scorer._scorer_fingerprint()
    assert len(event["scorer_fingerprint"]) == 16
    assert set(event["scorer_fingerprint"]) <= set("0123456789abcdef")
    assert event["interpretation"] == {
        "agreement_is_not_ground_truth": True,
        "trusted_gold_used": False,
        "failed_outputs_excluded_from_success_metrics": True,
        "text_only": True,
        "timestamps_compared": False,
    }
    assert event["privacy"]["exact_character_aggregates_suppressed"] is True
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

    assert event["privacy"]["exact_character_aggregates_suppressed"] is True
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

    assert event["privacy"]["exact_character_aggregates_suppressed"] is False
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
        "all_failed": "all_failed",
        "partial": "partial_failure",
        "succeeded": "succeeded",
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
    for path in (
        left.with_name("records-provenance.json"),
        right.with_name("records-provenance.json"),
    ):
        provenance = json.loads(path.read_text(encoding="utf-8"))
        provenance["candidate_id"] = "same_runner_candidate"
        path.write_text(json.dumps(provenance), encoding="utf-8")

    event = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"cpu": left, "gpu": right},
    )

    assert [source["candidate_id"] for source in event["source_candidates"]] == [
        "same_runner_candidate",
        "same_runner_candidate",
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
    provenance["config"] = {"processes": 4, "effective_threads_per_process": 6}
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    changed = score_asr_agreement(
        workload_path=workload,
        candidate_record_paths={"left": left, "right": right},
    )

    assert (
        baseline["source_candidates"][0]["config_fingerprint"]
        != changed["source_candidates"][0]["config_fingerprint"]
    )
    assert "effective_threads_per_process" not in json.dumps(changed)


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
