import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.validate_event_journals as journal_validator_script
import local_inference_bench.journal_integrity as journal_integrity_module

from local_inference_bench.asr_agreement_public_protocol import (
    INTERPRETATION as ASR_INTERPRETATION,
    PROTOCOL as ASR_PROTOCOL,
    PUBLIC_PRIVACY as ASR_PUBLIC_PRIVACY,
    public_event_sha256 as asr_public_event_sha256,
)
from local_inference_bench.journal_integrity import (
    effective_sustained_invalidated_attempt_ids,
    read_sustained_journal_snapshot,
    validate_append_only_record_prefix,
    validate_repository_journals,
)
from local_inference_bench.fingerprint import fingerprint_json
from local_inference_bench.verified_blind_ocr_protocol import (
    COMMITMENT_SCHEME,
    PRECOMMIT_PROTOCOL,
    PROTOCOL as BLIND_PROTOCOL,
    SCORE_INTERPRETATION,
    public_event_sha256,
    utc_timestamp_now,
)


QUALITY_INVALIDATION_REASON = "stale_authority_and_insufficient_provenance"
BOUNDED_INVALIDATION_REASON = "source_ancestry_claim_not_executed_by_harness"


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def _registry(
    path: Path,
    candidate_id: str = "candidate",
    configs: list[dict] | None = None,
    *,
    candidates: list[dict] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "candidates": candidates
                or [{"id": candidate_id, "configs": configs or [{"workers": 1}]}]
            }
        ),
        encoding="utf-8",
    )


def _attempt(attempt_id: str = "attempt", *, terminal: str = "sustained_attempt_succeeded") -> list[dict]:
    common = {
        "attempt_id": attempt_id,
        "attempt_key": f"key-{attempt_id}",
        "candidate_id": "candidate",
        "config": {"workers": 1},
        "config_index": 0,
        "trial_index": 0,
    }
    return [
        {"event": "sustained_attempt_started", **common},
        {
            "event": terminal,
            **common,
            "result": {
                "status": {
                    "sustained_attempt_succeeded": "complete",
                    "sustained_attempt_partial": "partial_failure",
                    "sustained_attempt_failed": "all_failed",
                }[terminal],
                "counts": {"attempted": 1, "completed": 1, "failed": 0},
            },
        },
    ]


def _validate(
    tmp_path: Path,
    sustained: list[dict],
    quality: list[dict] | None = None,
    bounded: list[dict] | None = None,
    *,
    registry_candidates: list[dict] | None = None,
):
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(sustained_path, sustained)
    _write_jsonl(quality_path, quality or [])
    _write_jsonl(bounded_path, bounded or [])
    _registry(registry_path, candidates=registry_candidates)
    return validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    )


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def _blind_preparation_event(preparation_id: str) -> dict:
    event = {
        "event": "blind_ocr_packet_prepared",
        "protocol": PRECOMMIT_PROTOCOL,
        "candidate_id": "private_course_blind_ocr_comparison",
        "workload_class": "private_course",
        "preparation_id": preparation_id,
        "mapping_protocol": BLIND_PROTOCOL,
        "sample_count": 1,
        "source_count": 2,
        "selected_source_status_counts": {
            "available": 2,
            "failed": 0,
            "unavailable": 0,
        },
        "producer_fingerprint": "a" * 16,
        "private_packet_commitment": {
            "scheme": COMMITMENT_SCHEME,
            "hmac_sha256": "b" * 64,
        },
        "privacy": {
            "private_commitment_key_published": False,
            "private_paths_or_text_published": False,
            "private_hashes_or_attempt_ids_published": False,
        },
    }
    event["public_event_sha256"] = public_event_sha256(event)
    event["timestamp_utc"] = utc_timestamp_now()
    return event


def _asr_score_event() -> tuple[dict, list[dict]]:
    candidates = [
        {
            "id": "asr_alpha",
            "task": "asr",
            "configs": [{"workers": 1, "phases": ["quality"]}],
        },
        {
            "id": "asr_beta",
            "task": "asr",
            "configs": [{"workers": 2, "phases": ["quality"]}],
        },
    ]
    candidate_metrics = []
    sources = []
    for evidence_id, candidate in enumerate(candidates, start=1):
        candidate_metrics.append(
            {
                "candidate_evidence_id": evidence_id,
                "availability": {
                    "attempted_sample_count": 1,
                    "successful_sample_count": 1,
                    "unavailable_sample_count": 0,
                },
                "successful_output_metrics": {
                    "sample_denominator": 1,
                    "speech_sample_denominator": 1,
                    "near_silence_sample_denominator": 0,
                    "exact_character_aggregates_published": False,
                    "near_silence_exact_character_aggregates_published": False,
                    "speech_sample_count": 1,
                    "successful_speech_sample_count": 1,
                    "successful_speech_nonempty_count": 1,
                    "near_silence_sample_count": 0,
                    "successful_near_silence_sample_count": 0,
                    "successful_near_silence_nonempty_count": 0,
                    "normalized_character_count_bucket": 1,
                    "near_silence_normalized_character_count_bucket": 0,
                    "repeated_trigram_observed": False,
                    "near_silence_successful_seconds": None,
                    "near_silence_characters_per_minute": None,
                    "mean_normalized_character_count": None,
                    "mean_observed_repeated_trigram_ratio": None,
                },
                "failed_output_diagnostics": {
                    "explicit_failed_record_count": 0,
                    "missing_record_count": 0,
                    "any_explicit_failed_output_nonempty": False,
                },
            }
        )
        sources.append(
            {
                "candidate_evidence_id": evidence_id,
                "candidate_id": candidate["id"],
                "status": "succeeded",
                "config_index": 0,
                "config_fingerprint": fingerprint_json(candidate["configs"][0]),
            }
        )
    event = {
        "event": "asr_agreement_scored",
        "candidate_id": "private_course_asr_agreement",
        "protocol": ASR_PROTOCOL,
        "scorer_fingerprint": "a" * 16,
        "source_authority_fingerprint": "b" * 16,
        "workload_class": "private_course",
        "source_candidates": sources,
        "privacy": {
            **ASR_PUBLIC_PRIVACY,
            "workload_meets_minimum_exact_aggregate_denominator": False,
        },
        "interpretation": dict(ASR_INTERPRETATION),
        "metrics": {
            "sample_count": 1,
            "candidate_count": 2,
            "small_private_cohort": True,
            "candidates": candidate_metrics,
            "pairs": [
                {
                    "pair_evidence_id": 3,
                    "left_candidate_evidence_id": 1,
                    "right_candidate_evidence_id": 2,
                    "availability": {
                        "comparable_sample_count": 1,
                        "unavailable_sample_count": 0,
                    },
                    "successful_output_agreement": {
                        "sample_denominator": 1,
                        "exact_character_aggregates_published": False,
                        "normalized_character_similarity_is_character_micro_weighted": True,
                        "normalized_character_denominator": None,
                        "normalized_character_similarity": None,
                        "normalized_character_similarity_bucket": 5,
                        "mean_length_agreement": None,
                        "mean_length_agreement_bucket": 5,
                        "exact_match_count": None,
                        "one_empty_disagreement_count": None,
                        "any_exact_match": True,
                        "all_comparable_exact_matches": True,
                        "any_one_empty_disagreement": False,
                    },
                }
            ],
        },
        "timestamp_utc": "2026-08-24T00:00:00.000000+00:00",
    }
    event["public_event_sha256"] = asr_public_event_sha256(event)
    return event, candidates


def _change_second_asr_workload_split(event: dict) -> None:
    successful = event["metrics"]["candidates"][1]["successful_output_metrics"]
    successful.update(
        {
            "speech_sample_count": 0,
            "successful_speech_sample_count": 0,
            "successful_speech_nonempty_count": 0,
            "speech_sample_denominator": 0,
            "near_silence_sample_count": 1,
            "successful_near_silence_sample_count": 1,
            "successful_near_silence_nonempty_count": 1,
            "near_silence_sample_denominator": 1,
            "near_silence_normalized_character_count_bucket": 1,
        }
    )


def _make_first_asr_source_all_failed(event: dict) -> None:
    candidate = event["metrics"]["candidates"][0]
    candidate["availability"].update(
        {"successful_sample_count": 0, "unavailable_sample_count": 1}
    )
    candidate["successful_output_metrics"].update(
        {
            "sample_denominator": 0,
            "speech_sample_denominator": 0,
            "successful_speech_sample_count": 0,
            "successful_speech_nonempty_count": 0,
            "normalized_character_count_bucket": 0,
        }
    )
    candidate["failed_output_diagnostics"].update(
        {"explicit_failed_record_count": 1}
    )
    event["source_candidates"][0]["status"] = "all_failed"


def _make_first_asr_source_all_failed_consistently(event: dict) -> None:
    _make_first_asr_source_all_failed(event)
    pair = event["metrics"]["pairs"][0]
    pair["availability"] = {
        "comparable_sample_count": 0,
        "unavailable_sample_count": 1,
    }
    pair["successful_output_agreement"].update(
        {
            "sample_denominator": 0,
            "normalized_character_similarity_bucket": None,
            "mean_length_agreement_bucket": None,
            "any_exact_match": False,
            "all_comparable_exact_matches": False,
        }
    )


def _make_zero_success_asr_bucket_nonzero(event: dict) -> None:
    _make_first_asr_source_all_failed_consistently(event)
    event["metrics"]["candidates"][0]["successful_output_metrics"][
        "normalized_character_count_bucket"
    ] = 1


def _make_zero_success_asr_repetition_true(event: dict) -> None:
    _make_first_asr_source_all_failed_consistently(event)
    event["metrics"]["candidates"][0]["successful_output_metrics"][
        "repeated_trigram_observed"
    ] = True


def _make_exact_near_silence_asr_cpm_inconsistent(event: dict) -> None:
    metrics = event["metrics"]
    metrics.update(
        {
            "sample_count": 10,
            "small_private_cohort": False,
        }
    )
    event["privacy"][
        "workload_meets_minimum_exact_aggregate_denominator"
    ] = True
    for candidate in metrics["candidates"]:
        candidate["availability"].update(
            {
                "attempted_sample_count": 10,
                "successful_sample_count": 10,
                "unavailable_sample_count": 0,
            }
        )
        candidate["successful_output_metrics"].update(
            {
                "sample_denominator": 10,
                "speech_sample_denominator": 0,
                "near_silence_sample_denominator": 10,
                "exact_character_aggregates_published": True,
                "near_silence_exact_character_aggregates_published": True,
                "speech_sample_count": 0,
                "successful_speech_sample_count": 0,
                "successful_speech_nonempty_count": 0,
                "near_silence_sample_count": 10,
                "successful_near_silence_sample_count": 10,
                "successful_near_silence_nonempty_count": 10,
                "normalized_character_count_bucket": 1,
                "near_silence_normalized_character_count_bucket": 1,
                "near_silence_successful_seconds": 10.0,
                "near_silence_characters_per_minute": 0.0,
                "mean_normalized_character_count": 1.0,
                "mean_observed_repeated_trigram_ratio": 0.0,
            }
        )
    pair = metrics["pairs"][0]
    pair["availability"] = {
        "comparable_sample_count": 10,
        "unavailable_sample_count": 0,
    }
    pair["successful_output_agreement"].update(
        {
            "sample_denominator": 10,
            "exact_character_aggregates_published": True,
            "normalized_character_denominator": 10,
            "normalized_character_similarity": 1.0,
            "normalized_character_similarity_bucket": 5,
            "mean_length_agreement": 1.0,
            "mean_length_agreement_bucket": 5,
            "exact_match_count": 10,
            "one_empty_disagreement_count": 0,
        }
    )


def _blind_three_judge_all_tie_with_false_unanimity(event: dict) -> None:
    event["judgment_file_count"] = 3
    metrics = event["metrics"]
    metrics.update(
        {
            "judgment_file_count": 3,
            "vote_count": 3,
            "tie_vote_count": 3,
            "consensus_tie_sample_count": 1,
            "unanimous_sample_fraction": 0.0,
            "pairwise_winner_agreement_fraction": 1 / 3,
        }
    )
    metrics["comparison_sample_denominators"].update(
        {
            "individual_winner_vote_denominator": 3,
            "pairwise_winner_agreement_denominator": 3,
        }
    )
    for candidate in metrics["candidates"]:
        candidate["mean_error_severity_vote_denominator"] = 3
        candidate["usable_vote_denominator"] = 3


def _blind_three_judge_split_with_false_pair_agreement(event: dict) -> None:
    _blind_three_judge_all_tie_with_false_unanimity(event)
    metrics = event["metrics"]
    metrics.update(
        {
            "tie_vote_count": 1,
            "consensus_tie_sample_count": 0,
            "no_strict_majority_sample_count": 1,
            "strict_majority_sample_fraction": 0.0,
        }
    )
    for candidate in metrics["candidates"]:
        candidate["win_votes"] = 1


def _blind_four_judge_split_with_too_few_pair_agreements(event: dict) -> None:
    event["judgment_file_count"] = 4
    metrics = event["metrics"]
    metrics.update(
        {
            "judgment_file_count": 4,
            "vote_count": 4,
            "tie_vote_count": 0,
            "consensus_tie_sample_count": 0,
            "no_strict_majority_sample_count": 1,
            "strict_majority_sample_fraction": 0.0,
            "unanimous_sample_fraction": 0.0,
            "pairwise_winner_agreement_fraction": 1 / 6,
        }
    )
    metrics["comparison_sample_denominators"].update(
        {
            "individual_winner_vote_denominator": 4,
            "pairwise_winner_agreement_denominator": 6,
        }
    )
    for candidate in metrics["candidates"]:
        candidate.update(
            {
                "win_votes": 2,
                "consensus_wins": 0,
                "mean_error_severity_vote_denominator": 4,
                "usable_vote_denominator": 4,
            }
        )


def _blind_four_judge_category_totals_with_too_many_pair_agreements(
    event: dict,
) -> None:
    _blind_four_judge_split_with_too_few_pair_agreements(event)
    metrics = event["metrics"]
    metrics["tie_vote_count"] = 2
    metrics["pairwise_winner_agreement_fraction"] = 2 / 6
    metrics["candidates"][0]["win_votes"] = 1
    metrics["candidates"][1]["win_votes"] = 1


def _blind_score_event(preparation_event: dict) -> tuple[dict, list[dict]]:
    candidates = [
        {
            "id": "candidate",
            "task": "ocr",
            "configs": [{"workers": 1, "phases": ["quality"]}],
        },
        {
            "id": "candidate_two",
            "task": "ocr",
            "configs": [{"workers": 2, "phases": ["quality"]}],
        },
    ]
    sources = [
        {
            "candidate_evidence_id": index,
            "candidate_id": candidate["id"],
            "config_index": 0,
            "config_fingerprint": fingerprint_json(candidate["configs"][0]),
            "attempt_status": "succeeded",
            "selected_available_record_count": 1,
            "selected_failed_record_count": 0,
            "selected_unavailable_record_count": 0,
        }
        for index, candidate in enumerate(candidates, start=1)
    ]
    event = {
        "event": "blind_ocr_quality_scored",
        "protocol": BLIND_PROTOCOL,
        "candidate_id": "private_course_blind_ocr_comparison",
        "workload_class": "private_course",
        "judgment_file_count": 2,
        "preparation_public_event_sha256": preparation_event["public_event_sha256"],
        "source_candidates": sources,
        "producer_fingerprint": "c" * 16,
        "metrics": {
            "sample_count": 1,
            "judgment_file_count": 2,
            "vote_count": 2,
            "tie_vote_count": 2,
            "consensus_tie_sample_count": 1,
            "no_strict_majority_sample_count": 0,
            "strict_majority_sample_fraction": 1.0,
            "unanimous_sample_fraction": 1.0,
            "pairwise_winner_agreement_fraction": 1.0,
            "comparison_sample_denominators": {
                "total_selected_sample_count": 1,
                "individual_winner_vote_denominator": 2,
                "consensus_winner_sample_denominator": 1,
                "strict_majority_sample_denominator": 1,
                "unanimous_sample_denominator": 1,
                "pairwise_winner_agreement_denominator": 1,
                "fully_available_comparison_count": 1,
                "not_fully_available_comparison_count": 0,
            },
            "source_record_availability": {
                "candidate_sample_count": 2,
                "available_record_count": 2,
                "failed_record_count": 0,
                "unavailable_record_count": 0,
            },
            "candidates": [
                {
                    "candidate_evidence_id": index,
                    "win_votes": 0,
                    "consensus_wins": 0,
                    "mean_error_severity": 0.0,
                    "usable_vote_fraction": 1.0,
                    "mean_error_severity_vote_denominator": 2,
                    "usable_vote_denominator": 2,
                }
                for index in (1, 2)
            ],
        },
        "interpretation": dict(SCORE_INTERPRETATION),
    }
    event["public_event_sha256"] = public_event_sha256(event)
    event["timestamp_utc"] = utc_timestamp_now()
    return event, candidates


def test_valid_lifecycle_and_registry_identity_pass(tmp_path: Path) -> None:
    assert _validate(tmp_path, _attempt()) == []


def test_excessively_nested_journal_record_is_reported_as_invalid_json(
    tmp_path: Path,
) -> None:
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    sustained_path.write_bytes(b'{"event":"deep","value":' + nested + b"}\n")
    quality_path.write_bytes(b"")
    bounded_path.write_bytes(b"")
    _registry(registry_path)

    issues = validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    )

    assert "invalid_json" in _codes(issues)


def test_excessively_nested_registry_is_reported_as_invalid_registry(
    tmp_path: Path,
) -> None:
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(sustained_path, _attempt())
    quality_path.write_bytes(b"")
    bounded_path.write_bytes(b"")
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    registry_path.write_bytes(b'{"candidates":' + nested + b"}")

    issues = validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    )

    assert "invalid_registry" in _codes(issues)


def test_lifecycle_detects_orphan_duplicate_and_identity_change(tmp_path: Path) -> None:
    events = _attempt()
    events[1]["candidate_id"] = "changed"
    events.append(dict(events[1]))
    events.append({"event": "sustained_attempt_failed", "attempt_id": "orphan"})

    codes = _codes(_validate(tmp_path, events))

    assert "attempt_identity_mismatch" in codes
    assert "duplicate_attempt_terminal" in codes
    assert "orphan_attempt_terminal" in codes


def test_unterminated_attempt_is_reported(tmp_path: Path) -> None:
    issues = _validate(tmp_path, _attempt()[:1])
    assert _codes(issues) == {"unterminated_attempt"}


def test_registry_rejects_unbound_historical_candidate_and_config(tmp_path: Path) -> None:
    events = _attempt()
    for event in events:
        event["candidate_id"] = "retired_candidate"
        event["config_index"] = 7
        event["config"] = {"old": True}

    issues = _validate(tmp_path, events)
    assert "unregistered_candidate_identity" in _codes(issues)


def test_legacy_registry_accepts_only_exact_explicit_historical_config(tmp_path: Path) -> None:
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    legacy_path = tmp_path / "legacy.jsonl"
    sustained_registry = tmp_path / "sustained-registry.json"
    candidate_registry = tmp_path / "candidate-registry.json"
    _write_jsonl(sustained_path, _attempt())
    _write_jsonl(quality_path, [])
    _write_jsonl(bounded_path, [])
    legacy_common = {
        "attempt_id": "legacy",
        "candidate_id": "candidate",
        "config": {"threads": 8},
    }
    _write_jsonl(
        legacy_path,
        [
            {"event": "attempt_started", **legacy_common},
            {"event": "attempt_succeeded", **legacy_common},
        ],
    )
    _registry(sustained_registry)
    _registry(candidate_registry, configs=[{"threads": 8, "backend": "cpu"}])

    common = dict(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=sustained_registry,
        legacy_journal=legacy_path,
        candidate_registry=candidate_registry,
    )
    assert "registry_config_mismatch" in _codes(validate_repository_journals(**common))
    historical_hash = hashlib.sha256(b'{"threads":8}').hexdigest()
    assert validate_repository_journals(
        **common,
        historical_legacy_configs={("candidate", historical_hash)},
    ) == []


def test_legacy_registry_accepts_tracked_nonrunnable_historical_config(
    tmp_path: Path,
) -> None:
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    legacy_path = tmp_path / "legacy.jsonl"
    sustained_registry = tmp_path / "sustained-registry.json"
    candidate_registry = tmp_path / "candidate-registry.json"
    _write_jsonl(sustained_path, _attempt())
    _write_jsonl(quality_path, [])
    _write_jsonl(bounded_path, [])
    legacy_common = {
        "attempt_id": "legacy",
        "candidate_id": "candidate",
        "config": {"threads": 8},
    }
    _write_jsonl(
        legacy_path,
        [
            {"event": "attempt_started", **legacy_common},
            {"event": "attempt_succeeded", **legacy_common},
        ],
    )
    _registry(sustained_registry)
    candidate_registry.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "id": "candidate",
                        "configs": [{"threads": 8, "backend": "cpu"}],
                        "historical_configs": [{"threads": 8}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=sustained_registry,
        legacy_journal=legacy_path,
        candidate_registry=candidate_registry,
    ) == []


def test_registry_detects_out_of_range_and_config_drift(tmp_path: Path) -> None:
    out_of_range = _attempt("out-of-range")
    for event in out_of_range:
        event["config_index"] = 3
    drift = _attempt("drift")
    for event in drift:
        event["config"] = {"workers": 2}

    codes = _codes(_validate(tmp_path, out_of_range + drift))

    assert "unresolved_config_identity" in codes
    assert "registry_config_mismatch" in codes


def test_invalidation_target_and_reason_must_match_terminal(tmp_path: Path) -> None:
    events = _attempt(terminal="sustained_attempt_failed")
    events[1]["result"] = {
        "status": "all_failed",
        "counts": {"attempted": 1, "completed": 0, "failed": 1},
    }
    events.append(
        {
            "event": "sustained_attempts_invalidated",
            "candidate_id": "candidate",
            "invalidated_attempt_ids": ["attempt", "missing"],
            "reason_kind": "all_failed_attempt_mislabeled_as_succeeded",
            "replacement_kind": "fail_closed_status_and_count_invariant",
        }
    )

    codes = _codes(_validate(tmp_path, events))

    assert "invalidation_reason_terminal_mismatch" in codes
    assert "unresolved_attempt_target" in codes


def test_reclassification_rejects_already_projected_terminal(tmp_path: Path) -> None:
    events = _attempt(terminal="sustained_attempt_partial")
    events[1]["result"] = {
        "status": "partial_failure",
        "counts": {"attempted": 2, "completed": 1, "failed": 1},
    }
    events.append(
        {
            "event": "sustained_attempts_reclassified",
            "reclassified_attempt_ids": ["attempt"],
            "reclassified_status": "partial_failure",
            "reason_kind": "terminal_event_name_did_not_reflect_item_failures",
        }
    )

    assert "reclassification_already_applied" in _codes(_validate(tmp_path, events))


def test_config_reclassification_requires_prior_state_and_resolved_replacement(tmp_path: Path) -> None:
    events = _attempt()
    events.append(
        {
            "event": "sustained_config_indices_reclassified",
            "candidate_id": "candidate",
            "reclassified_attempt_ids": ["attempt"],
            "prior_config_index": 0,
            "replacement_config_index": 4,
        }
    )
    for event in events[:2]:
        event["config_index"] = 4

    codes = _codes(_validate(tmp_path, events))

    assert "unresolved_config_identity" in codes
    assert "unresolved_reclassified_config_identity" in codes
    assert "config_reclassification_already_applied" in codes


def test_exact_payload_bound_config_reclassification_resolves_registry_drift(
    tmp_path: Path,
) -> None:
    events = _attempt()
    for event in events:
        event["config_index"] = 0
        event["config"] = {"workers": 2}
    events.append(
        {
            "event": "sustained_config_indices_reclassified",
            "candidate_id": "candidate",
            "reclassified_attempt_ids": ["attempt"],
            "prior_config_index": 0,
            "replacement_config_index": 1,
            "reason_kind": "historical_registry_index_drift",
        }
    )
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(sustained_path, events)
    _write_jsonl(quality_path, [])
    _write_jsonl(bounded_path, [])
    _registry(registry_path, configs=[{"workers": 1}, {"workers": 2}])

    assert validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    ) == []


def test_config_reclassification_rejects_replacement_payload_mismatch(
    tmp_path: Path,
) -> None:
    events = _attempt()
    events.append(
        {
            "event": "sustained_config_indices_reclassified",
            "candidate_id": "candidate",
            "reclassified_attempt_ids": ["attempt"],
            "prior_config_index": 0,
            "replacement_config_index": 1,
            "reason_kind": "historical_registry_index_drift",
        }
    )
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(sustained_path, events)
    _write_jsonl(quality_path, [])
    _write_jsonl(bounded_path, [])
    _registry(registry_path, configs=[{"workers": 1}, {"workers": 2}])

    issues = validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    )

    assert "reclassified_config_payload_mismatch" in _codes(issues)


def test_quality_invalidation_resolves_both_exact_events(tmp_path: Path) -> None:
    quality = [
        {"event": "ocr_quality_scored", "protocol": "score-v1", "timestamp_utc": "old"},
        {"event": "ocr_quality_scored", "protocol": "score-v2", "timestamp_utc": "new"},
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]
    assert _validate(tmp_path, _attempt(), quality=quality) == []


def test_quality_invalidation_rejects_missing_replacement(tmp_path: Path) -> None:
    quality = [
        {"event": "ocr_quality_scored", "protocol": "score-v1", "timestamp_utc": "old"},
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "missing",
        },
    ]
    assert "unresolved_replacement_event" in _codes(
        _validate(tmp_path, _attempt(), quality=quality)
    )


def test_quality_invalidation_binds_top_level_and_nested_identities(
    tmp_path: Path,
) -> None:
    quality = [
        {
            "event": "ocrllm_compatibility_checked",
            "protocol": "score-v1",
            "timestamp_utc": "old",
            "candidate_id": "candidate-old",
            "workload_class": "generated_quality_control",
        },
        {
            "event": "ocrllm_compatibility_checked",
            "protocol": "score-v2",
            "timestamp_utc": "new",
            "result": {
                "candidate_id": "candidate-new",
                "workload_class": "generated_quality_control",
            },
        },
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "candidate_id": "candidate-new",
            "workload_class": "generated_quality_control",
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]

    codes = _codes(_validate(tmp_path, _attempt(), quality=quality))

    assert "invalidated_candidate_id_mismatch" in codes
    assert "replacement_candidate_id_mismatch" not in codes


def test_quality_invalidation_allows_explicit_identity_transition(
    tmp_path: Path,
) -> None:
    quality = [
        {
            "event": "ocrllm_compatibility_checked",
            "protocol": "score-v1",
            "timestamp_utc": "old",
            "candidate_id": "candidate-old",
            "workload_class": "generated_quality_control",
        },
        {
            "event": "ocrllm_compatibility_checked",
            "protocol": "score-v2",
            "timestamp_utc": "new",
            "result": {
                "candidate_id": "candidate-new",
                "workload_class": "generated_quality_control",
            },
        },
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "candidate_id": "candidate-old",
            "replacement_candidate_id": "candidate-new",
            "workload_class": "generated_quality_control",
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]

    assert _validate(tmp_path, _attempt(), quality=quality) == []


def test_quality_invalidation_requires_explicit_replacement_identity_transition(
    tmp_path: Path,
) -> None:
    quality = [
        {
            "event": "ocr_quality_scored",
            "protocol": "score-v1",
            "timestamp_utc": "old",
            "candidate_id": "candidate-old",
        },
        {
            "event": "ocr_quality_scored",
            "protocol": "score-v2",
            "timestamp_utc": "new",
            "candidate_id": "candidate-new",
        },
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "candidate_id": "candidate-old",
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]

    assert "replacement_candidate_id_mismatch" in _codes(
        _validate(tmp_path, _attempt(), quality=quality)
    )


def test_quality_invalidation_rejects_workload_identity_mismatch(
    tmp_path: Path,
) -> None:
    quality = [
        {
            "event": "ocr_quality_scored",
            "protocol": "score-v1",
            "timestamp_utc": "old",
            "candidate_id": "candidate",
            "workload_class": "generated_quality_control",
        },
        {
            "event": "ocr_quality_scored",
            "protocol": "score-v2",
            "timestamp_utc": "new",
            "candidate_id": "candidate",
            "workload_class": "private_course",
        },
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "candidate_id": "candidate",
            "workload_class": "private_course",
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]

    codes = _codes(_validate(tmp_path, _attempt(), quality=quality))

    assert "invalidated_workload_class_mismatch" in codes
    assert "replacement_workload_class_mismatch" not in codes


def test_quality_invalidation_rejects_conflicting_nested_identity(
    tmp_path: Path,
) -> None:
    quality = [
        {
            "event": "ocr_quality_scored",
            "protocol": "score-v1",
            "timestamp_utc": "old",
            "candidate_id": "candidate",
            "result": {"candidate_id": "different-candidate"},
        },
        {
            "event": "ocr_quality_scored",
            "protocol": "score-v2",
            "timestamp_utc": "new",
        },
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "candidate_id": "candidate",
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]

    assert "ambiguous_invalidated_candidate_id" in _codes(
        _validate(tmp_path, _attempt(), quality=quality)
    )


def test_bounded_invalidation_can_resolve_unique_replacement_by_protocol(tmp_path: Path) -> None:
    bounded = [
        {
            "event": "openvino_genai_qwen3_asr_tail_fix_compared",
            "protocol": "compare-v1",
            "timestamp_utc": "old",
        },
        {
            "event": "bounded_event_invalidated",
            "reason_kind": BOUNDED_INVALIDATION_REASON,
            "invalidated_protocol": "compare-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "compare-v2",
        },
        {
            "event": "openvino_genai_qwen3_asr_tail_fix_compared",
            "protocol": "compare-v2",
            "timestamp_utc": "new",
        },
    ]
    assert _validate(tmp_path, _attempt(), bounded=bounded) == []


def test_bounded_scorer_invalidation_can_bind_same_protocol_by_timestamp(
    tmp_path: Path,
) -> None:
    bounded = [
        {
            "event": "bounded_candidate_quality_verified",
            "protocol": "bounded-community-screen-v4",
            "timestamp_utc": "old",
            "candidate_id": "candidate",
            "workload_class": "generated_quality_control",
        },
        {
            "event": "bounded_candidate_quality_verified",
            "protocol": "bounded-community-screen-v4",
            "timestamp_utc": "new",
            "candidate_id": "candidate",
            "workload_class": "generated_quality_control",
        },
        {
            "event": "bounded_event_invalidated",
            "reason_kind": "structure_aware_scorer_failed_adversarial_validation",
            "candidate_id": "candidate",
            "workload_class": "generated_quality_control",
            "invalidated_protocol": "bounded-community-screen-v4",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "bounded-community-screen-v4",
            "replacement_event_timestamp_utc": "new",
            "timestamp_utc": "correction",
        },
    ]

    assert _validate(tmp_path, _attempt(), bounded=bounded) == []


def test_hash_bound_supersession_can_retire_only_a_correction(tmp_path: Path) -> None:
    events = _attempt(terminal="sustained_attempt_failed")
    events[1]["result"] = {
        "status": "all_failed",
        "counts": {"attempted": 1, "completed": 0, "failed": 1},
    }
    bad_correction = {
        "event": "sustained_attempts_invalidated",
        "candidate_id": "candidate",
        "invalidated_attempt_ids": ["attempt"],
        "reason_kind": "all_failed_attempt_mislabeled_as_succeeded",
        "replacement_kind": "fixed",
    }
    encoded = json.dumps(bad_correction, sort_keys=True).encode("utf-8")
    events.extend(
        [
            bad_correction,
            {
                "event": "journal_event_superseded",
                "protocol": "append-only-journal-integrity-v1",
                "superseded_event_line": 3,
                "superseded_event_sha256": hashlib.sha256(encoded).hexdigest(),
                "reason_kind": "correction_target_already_projected",
            },
        ]
    )

    assert _validate(tmp_path, events) == []


def test_supersession_cannot_hide_measurement_event(tmp_path: Path) -> None:
    events = _attempt()
    encoded = json.dumps(events[1], sort_keys=True).encode("utf-8")
    events.append(
        {
            "event": "journal_event_superseded",
            "protocol": "append-only-journal-integrity-v1",
            "superseded_event_line": 2,
            "superseded_event_sha256": hashlib.sha256(encoded).hexdigest(),
            "reason_kind": "not_allowed",
        }
    )

    assert "forbidden_supersession_target" in _codes(_validate(tmp_path, events))


def test_effective_invalidations_exclude_only_hash_bound_superseded_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sustained.jsonl"
    current_attempt = _attempt("current-attempt")
    current_attempt[1]["result"]["generation"] = {"token_cap_hit_count": 1}
    invalidation = {
        "event": "sustained_attempts_invalidated",
        "invalidated_attempt_ids": ["old-attempt"],
    }
    encoded = json.dumps(invalidation, sort_keys=True).encode("utf-8")
    _write_jsonl(
        path,
        [
            *current_attempt,
            invalidation,
            {
                "event": "journal_event_superseded",
                "protocol": "append-only-journal-integrity-v1",
                "superseded_event_line": 3,
                "superseded_event_sha256": hashlib.sha256(encoded).hexdigest(),
                "reason_kind": "correction_target_already_projected",
            },
            {
                "event": "sustained_attempts_invalidated",
                "candidate_id": "candidate",
                "invalidated_attempt_ids": ["current-attempt"],
                "reason_kind": "token_cap_accepted_by_compatibility_gate",
                "replacement_kind": "token_cap_fail_closed_256_token_gate",
            },
        ],
    )

    assert effective_sustained_invalidated_attempt_ids(path) == {"current-attempt"}

    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    events[3]["superseded_event_sha256"] = "0" * 64
    _write_jsonl(path, events)
    with pytest.raises(ValueError, match="supersession_hash_mismatch"):
        effective_sustained_invalidated_attempt_ids(path)


def test_sustained_snapshot_returns_events_and_invalidations_together(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sustained.jsonl"
    attempt = _attempt("target")
    attempt[1]["result"]["generation"] = {"token_cap_hit_count": 1}
    invalidation = {
        "event": "sustained_attempts_invalidated",
        "candidate_id": "candidate",
        "invalidated_attempt_ids": ["target"],
        "reason_kind": "token_cap_accepted_by_compatibility_gate",
        "replacement_kind": "token_cap_fail_closed_256_token_gate",
    }
    _write_jsonl(path, [*attempt, invalidation])

    events, invalidated, corrected = read_sustained_journal_snapshot(path)

    assert events == [*attempt, invalidation]
    assert invalidated == {"target"}
    assert corrected == set()


def test_sustained_snapshot_reads_journal_bytes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sustained.jsonl"
    events = _attempt("target")
    raw = "".join(json.dumps(event, sort_keys=True) + "\n" for event in events).encode()
    calls = 0

    def read_once(_path: Path) -> bytes:
        nonlocal calls
        calls += 1
        if calls != 1:
            raise AssertionError("journal snapshot reopened")
        return raw

    monkeypatch.setattr(journal_integrity_module, "read_journal_bytes", read_once)

    snapshot, invalidated, corrected = read_sustained_journal_snapshot(path)

    assert snapshot == events
    assert invalidated == set()
    assert corrected == set()
    assert calls == 1


def test_valid_asr_v10_score_matches_closed_schema_and_registry(tmp_path: Path) -> None:
    score, candidates = _asr_score_event()

    issues = _validate(
        tmp_path,
        _attempt(),
        quality=[score],
        registry_candidates=candidates,
    )

    assert "invalid_asr_agreement_score_event" not in _codes(issues)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event.update({"private_path": "D:/private/course.wav"}),
        lambda event: event["metrics"].update({"forged_count": 7}),
        lambda event: event["metrics"].update({"sample_count": 999}),
        lambda event: event["source_candidates"][0].update(
            {"candidate_id": "unregistered_asr"}
        ),
    ],
)
def test_asr_v10_score_rejects_malformed_or_unregistered_claims(
    tmp_path: Path,
    mutate,
) -> None:
    score, candidates = _asr_score_event()
    mutate(score)
    score["public_event_sha256"] = asr_public_event_sha256(score)

    assert "invalid_asr_agreement_score_event" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[score],
            registry_candidates=candidates,
        )
    )


@pytest.mark.parametrize(
    "mutate",
    [
        _change_second_asr_workload_split,
        _make_first_asr_source_all_failed,
        lambda event: event["metrics"]["candidates"][0][
            "failed_output_diagnostics"
        ].update({"any_explicit_failed_output_nonempty": True}),
        lambda event: event["metrics"]["pairs"][0][
            "successful_output_agreement"
        ].update(
            {
                "normalized_character_similarity_bucket": 4,
                "mean_length_agreement_bucket": 4,
            }
        ),
        lambda event: event["metrics"]["candidates"][0][
            "successful_output_metrics"
        ].update({"sample_denominator": True}),
        lambda event: event["metrics"]["pairs"][0].update(
            {"left_candidate_evidence_id": True}
        ),
        lambda event: event["metrics"]["pairs"][0][
            "successful_output_agreement"
        ].update({"sample_denominator": True}),
        lambda event: event["metrics"]["pairs"][0][
            "successful_output_agreement"
        ].update({"all_comparable_exact_matches": False}),
        lambda event: event["metrics"]["pairs"][0][
            "successful_output_agreement"
        ].update(
            {
                "normalized_character_similarity_bucket": 4,
                "mean_length_agreement_bucket": 4,
                "all_comparable_exact_matches": False,
            }
        ),
        _make_zero_success_asr_bucket_nonzero,
        _make_zero_success_asr_repetition_true,
        _make_exact_near_silence_asr_cpm_inconsistent,
    ],
)
def test_asr_v10_score_rejects_cross_metric_contradictions(
    tmp_path: Path,
    mutate,
) -> None:
    score, candidates = _asr_score_event()
    mutate(score)
    score["public_event_sha256"] = asr_public_event_sha256(score)

    assert "invalid_asr_agreement_score_event" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[score],
            registry_candidates=candidates,
        )
    )


def test_asr_v10_duplicate_public_hash_is_invalid(tmp_path: Path) -> None:
    score, candidates = _asr_score_event()

    assert "duplicate_asr_agreement_score_hash" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[score, dict(score)],
            registry_candidates=candidates,
        )
    )


def test_historical_asr_v10_score_survives_later_config_retirement(
    tmp_path: Path,
) -> None:
    score, candidates = _asr_score_event()
    candidates[0]["retired_config_indices"] = [0]

    assert "invalid_asr_agreement_score_event" not in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[score],
            registry_candidates=candidates,
        )
    )


def test_duplicate_blind_ocr_preparation_id_is_invalid(tmp_path: Path) -> None:
    preparation_id = "b2d37ca7-eb41-4e6f-af0e-544167c7ea22"
    quality = [
        _blind_preparation_event(preparation_id),
        _blind_preparation_event(preparation_id),
    ]

    assert "duplicate_blind_ocr_preparation_id" in _codes(
        _validate(tmp_path, _attempt(), quality=quality)
    )


def test_blind_ocr_preparation_id_must_be_canonical_uuid(tmp_path: Path) -> None:
    quality = [{"event": "blind_ocr_packet_prepared", "preparation_id": "not-a-uuid"}]

    assert "invalid_blind_ocr_preparation_event" in _codes(
        _validate(tmp_path, _attempt(), quality=quality)
    )


def test_valid_blind_ocr_v10_score_resolves_earlier_preparation_and_registry(
    tmp_path: Path,
) -> None:
    preparation = _blind_preparation_event("d70a1baf-ec5f-4991-9b0a-8a35d357fe90")
    score, candidates = _blind_score_event(preparation)

    issues = _validate(
        tmp_path,
        _attempt(),
        quality=[preparation, score],
        registry_candidates=candidates,
    )

    assert not {
        "invalid_blind_ocr_score_event",
        "unresolved_blind_ocr_preparation_anchor",
        "duplicate_blind_ocr_score_anchor",
        "blind_ocr_score_registry_mismatch",
    } & _codes(issues)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event.update({"private_path": "secret"}),
        lambda event: event["metrics"].update({"raw_text": "secret"}),
        lambda event: event["interpretation"].update(
            {"raw_text_or_images_published": 0}
        ),
        lambda event: event.update({"timestamp_utc": "2026-01-01T00:00:00+00:00"}),
    ],
)
def test_blind_ocr_v10_score_schema_rejects_private_or_type_lax_fields(
    tmp_path: Path,
    mutate,
) -> None:
    preparation = _blind_preparation_event("15ddbdc5-99b0-48b1-82e2-0788cf530ece")
    score, candidates = _blind_score_event(preparation)
    mutate(score)
    score["public_event_sha256"] = public_event_sha256(score)

    assert "invalid_blind_ocr_score_event" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[preparation, score],
            registry_candidates=candidates,
        )
    )


def test_blind_ocr_v10_score_requires_one_earlier_anchor(tmp_path: Path) -> None:
    preparation = _blind_preparation_event("08663f9c-2a9d-4821-b615-a1ed20484c2c")
    score, candidates = _blind_score_event(preparation)

    assert "unresolved_blind_ocr_preparation_anchor" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[score, preparation],
            registry_candidates=candidates,
        )
    )


def test_blind_ocr_v10_score_anchor_is_unique(tmp_path: Path) -> None:
    preparation = _blind_preparation_event("1c08071f-3bc6-4748-aa78-c61fbb079351")
    score, candidates = _blind_score_event(preparation)
    duplicate = {**score, "timestamp_utc": utc_timestamp_now()}

    assert "duplicate_blind_ocr_score_anchor" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[preparation, score, duplicate],
            registry_candidates=candidates,
        )
    )


def test_blind_ocr_v10_score_source_must_match_tracked_registry(tmp_path: Path) -> None:
    preparation = _blind_preparation_event("96144e8e-38f2-4f0a-96cc-fc3df52d1654")
    score, candidates = _blind_score_event(preparation)
    score["source_candidates"][0]["candidate_id"] = "private_teacher_name"
    score["public_event_sha256"] = public_event_sha256(score)

    assert "blind_ocr_score_registry_mismatch" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[preparation, score],
            registry_candidates=candidates,
        )
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event["metrics"].update(
            {"strict_majority_sample_fraction": 0.125}
        ),
        lambda event: event["source_candidates"][0].update(
            {
                "selected_available_record_count": 0,
                "selected_failed_record_count": 1,
            }
        ),
        lambda event: event["metrics"]["comparison_sample_denominators"].update(
            {"fully_available_comparison_count": 0, "not_fully_available_comparison_count": 1}
        ),
        lambda event: event["metrics"].update(
            {"unanimous_sample_fraction": 1.0, "pairwise_winner_agreement_fraction": 0.0}
        ),
        lambda event: event["metrics"]["candidates"][0].update(
            {"consensus_wins": 1, "win_votes": 0}
        ),
        lambda event: event["metrics"].update(
            {"unanimous_sample_fraction": 0.125}
        ),
        lambda event: event["metrics"]["candidates"][0].update(
            {"mean_error_severity": 0.123}
        ),
        lambda event: event["metrics"]["candidates"][0].update(
            {"mean_error_severity": -0.0}
        ),
        lambda event: event["metrics"].update(
            {
                "no_strict_majority_sample_count": 1,
                "strict_majority_sample_fraction": 0.0,
                "consensus_tie_sample_count": 0,
            }
        ),
        lambda event: event["metrics"].update(
            {
                "consensus_tie_sample_count": 0,
                "no_strict_majority_sample_count": 1,
                "strict_majority_sample_fraction": 0.0,
                "unanimous_sample_fraction": 0.0,
                "pairwise_winner_agreement_fraction": 0.0,
            }
        ),
        _blind_three_judge_all_tie_with_false_unanimity,
        _blind_three_judge_split_with_false_pair_agreement,
        _blind_four_judge_split_with_too_few_pair_agreements,
        _blind_four_judge_category_totals_with_too_many_pair_agreements,
    ],
)
def test_blind_ocr_v10_score_rejects_arithmetic_or_status_contradictions(
    tmp_path: Path,
    mutate,
) -> None:
    preparation = _blind_preparation_event("c8c0b76e-ce6c-435c-a6a5-934b21fb57bd")
    score, candidates = _blind_score_event(preparation)
    mutate(score)
    score["public_event_sha256"] = public_event_sha256(score)

    assert "invalid_blind_ocr_score_event" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[preparation, score],
            registry_candidates=candidates,
        )
    )


@pytest.mark.parametrize("mismatch", ["sample_count", "status_counts"])
def test_blind_ocr_v10_score_counts_match_preparation_anchor(
    tmp_path: Path,
    mismatch: str,
) -> None:
    preparation = _blind_preparation_event("00bfa264-7531-4232-bcc3-2ce85e4448aa")
    if mismatch == "sample_count":
        preparation["sample_count"] = 2
        preparation["selected_source_status_counts"]["available"] = 4
    else:
        preparation["selected_source_status_counts"] = {
            "available": 1,
            "failed": 1,
            "unavailable": 0,
        }
    preparation["public_event_sha256"] = public_event_sha256(preparation)
    score, candidates = _blind_score_event(preparation)

    assert "blind_ocr_score_preparation_mismatch" in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[preparation, score],
            registry_candidates=candidates,
        )
    )


def test_blind_ocr_v10_historical_score_survives_later_registry_retirement(
    tmp_path: Path,
) -> None:
    preparation = _blind_preparation_event("53901d63-a382-46ab-a507-d14bfed95072")
    score, candidates = _blind_score_event(preparation)
    candidates[0]["status"] = "retired_after_measurement"
    candidates[0]["retired_config_indices"] = [0]

    assert "blind_ocr_score_registry_mismatch" not in _codes(
        _validate(
            tmp_path,
            _attempt(),
            quality=[preparation, score],
            registry_candidates=candidates,
        )
    )


def test_effective_invalidations_reject_malformed_active_row(tmp_path: Path) -> None:
    path = tmp_path / "sustained.jsonl"
    _write_jsonl(
        path,
        [
            *_attempt("target"),
            {
                "event": "sustained_attempts_invalidated",
                "invalidated_attempt_ids": ["target"],
            },
        ],
    )

    with pytest.raises(ValueError, match="missing_invalidation_reason"):
        effective_sustained_invalidated_attempt_ids(path)


def test_effective_invalidations_ignore_unrelated_in_flight_attempt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sustained.jsonl"
    target_attempt = _attempt("target")
    target_attempt[1]["result"]["generation"] = {"token_cap_hit_count": 1}
    _write_jsonl(
        path,
        [
            *target_attempt,
            {
                "event": "sustained_attempts_invalidated",
                "candidate_id": "candidate",
                "invalidated_attempt_ids": ["target"],
                "reason_kind": "token_cap_accepted_by_compatibility_gate",
                "replacement_kind": "token_cap_fail_closed_256_token_gate",
            },
            _attempt("still-running")[0],
        ],
    )

    assert effective_sustained_invalidated_attempt_ids(path) == {"target"}


def test_append_only_prefix_accepts_append_and_line_ending_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    head = b'{"event":"first"}\n{"event":"second"}\n'

    path.write_bytes(head + b'{"event":"third"}\n')
    assert validate_append_only_record_prefix(path, head) == []

    path.write_bytes(head.replace(b"\n", b"\r\n") + b'{"event":"third"}\r\n')
    assert validate_append_only_record_prefix(path, head) == []


@pytest.mark.parametrize(
    "current",
    [
        b'{"event":"mutated"}\n{"event":"second"}\n',
        b'{"event":"first"}\n',
    ],
    ids=["mutation", "deletion"],
)
def test_append_only_prefix_rejects_head_mutation_or_deletion(
    tmp_path: Path,
    current: bytes,
) -> None:
    path = tmp_path / "events.jsonl"
    head = b'{"event":"first"}\n{"event":"second"}\n'
    path.write_bytes(current)

    issues = validate_append_only_record_prefix(path, head)

    assert [issue.code for issue in issues] == ["append_only_prefix_mismatch"]


def test_append_only_prefix_rejects_non_git_line_separator(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    head = b'{"event":"first"}\n{"event":"second"}\n'
    path.write_bytes(b'{"event":"first"}\v{"event":"second"}\n')

    assert [
        issue.code for issue in validate_append_only_record_prefix(path, head)
    ] == ["append_only_prefix_mismatch"]


def test_terminal_identity_must_be_present_on_both_lifecycle_rows(
    tmp_path: Path,
) -> None:
    events = _attempt()
    events[1].pop("candidate_id")

    assert "incomplete_attempt_identity" in _codes(_validate(tmp_path, events))


def test_sustained_identity_requires_complete_fields_and_exact_integer_types(
    tmp_path: Path,
) -> None:
    missing = _attempt("missing")
    for event in missing:
        event.pop("trial_index")
    boolean = _attempt("boolean")
    for event in boolean:
        event["trial_index"] = True

    codes = _codes(_validate(tmp_path, missing + boolean))

    assert "incomplete_attempt_identity" in codes
    assert "invalid_attempt_identity" in codes


def test_registry_config_identity_is_type_exact(tmp_path: Path) -> None:
    events = _attempt()
    for event in events:
        event["config"] = {"workers": True}

    assert "registry_config_mismatch" in _codes(_validate(tmp_path, events))


def test_unknown_invalidation_reason_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "sustained.jsonl"
    _write_jsonl(
        path,
        [
            *_attempt("target"),
            {
                "event": "sustained_attempts_invalidated",
                "candidate_id": "candidate",
                "invalidated_attempt_ids": ["target"],
                "reason_kind": "invented_reason",
                "replacement_kind": "invented_replacement",
            },
        ],
    )

    with pytest.raises(ValueError, match="unsupported_invalidation_reason"):
        effective_sustained_invalidated_attempt_ids(path)


def test_non_string_invalidation_target_fails_closed_without_crashing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sustained.jsonl"
    _write_jsonl(
        path,
        [
            *_attempt("target"),
            {
                "event": "sustained_attempts_invalidated",
                "candidate_id": "candidate",
                "invalidated_attempt_ids": [{"not": "hashable"}],
                "reason_kind": "overstrict_repetition_gate",
                "replacement_kind": "calibrated_near_total_loop_gate",
            },
        ],
    )

    with pytest.raises(ValueError, match="invalid_attempt_target_list"):
        effective_sustained_invalidated_attempt_ids(path)


def test_terminal_status_and_counts_must_agree_with_event(tmp_path: Path) -> None:
    events = _attempt()
    events[1]["result"] = {
        "status": "all_failed",
        "counts": {"attempted": 1, "completed": 0, "failed": 1},
    }

    codes = _codes(_validate(tmp_path, events))

    assert "terminal_result_status_mismatch" in codes
    assert "terminal_status_count_mismatch" in codes


def test_conflicting_active_status_reclassifications_are_rejected(
    tmp_path: Path,
) -> None:
    events = _attempt()
    events[1]["result"] = {
        "counts": {"attempted": 2, "completed": 1, "failed": 1},
    }
    events.extend(
        [
            {
                "event": "sustained_attempts_reclassified",
                "reclassified_attempt_ids": ["attempt"],
                "reclassified_status": "partial_failure",
                "reason_kind": "terminal_event_name_did_not_reflect_item_failures",
            },
            {
                "event": "sustained_attempts_reclassified",
                "reclassified_attempt_ids": ["attempt"],
                "reclassified_status": "all_failed",
                "reason_kind": "second_projection",
            },
        ]
    )

    assert "conflicting_active_correction" in _codes(_validate(tmp_path, events))


def test_unknown_status_reclassification_reason_is_rejected(
    tmp_path: Path,
) -> None:
    events = _attempt()
    events[1]["result"] = {
        "counts": {"attempted": 2, "completed": 1, "failed": 1},
    }
    events.append(
        {
            "event": "sustained_attempts_reclassified",
            "reclassified_attempt_ids": ["attempt"],
            "reclassified_status": "partial_failure",
            "reason_kind": "invented_reason",
        }
    )

    assert "unsupported_reclassification_reason" in _codes(
        _validate(tmp_path, events)
    )


def test_quality_replacement_must_preserve_event_kind(tmp_path: Path) -> None:
    quality = [
        {"event": "ocr_quality_scored", "protocol": "score-v1", "timestamp_utc": "old"},
        {"event": "asr_agreement_scored", "protocol": "score-v2", "timestamp_utc": "new"},
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]

    assert "replacement_event_kind_mismatch" in _codes(
        _validate(tmp_path, _attempt(), quality=quality)
    )


def test_quality_event_cannot_have_two_active_replacements(tmp_path: Path) -> None:
    quality = [
        {"event": "ocr_quality_scored", "protocol": "score-v1", "timestamp_utc": "old"},
        {"event": "ocr_quality_scored", "protocol": "score-v2", "timestamp_utc": "new-a"},
        {"event": "ocr_quality_scored", "protocol": "score-v3", "timestamp_utc": "new-b"},
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new-a",
        },
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v3",
            "replacement_event_timestamp_utc": "new-b",
        },
    ]

    assert "conflicting_active_event_invalidation" in _codes(
        _validate(tmp_path, _attempt(), quality=quality)
    )


@pytest.mark.parametrize(
    ("reason", "expected_code"),
    [
        (None, "missing_event_invalidation_reason"),
        ("invented_reason", "unsupported_event_invalidation_reason"),
    ],
    ids=["missing", "unsupported"],
)
def test_quality_event_invalidation_reason_is_fail_closed(
    tmp_path: Path,
    reason: str | None,
    expected_code: str,
) -> None:
    correction = {
        "event": "quality_event_invalidated",
        "invalidated_protocol": "score-v1",
        "invalidated_event_timestamp_utc": "old",
        "replacement_protocol": "score-v2",
        "replacement_event_timestamp_utc": "new",
    }
    if reason is not None:
        correction["reason_kind"] = reason
    quality = [
        {"event": "ocr_quality_scored", "protocol": "score-v1", "timestamp_utc": "old"},
        {"event": "ocr_quality_scored", "protocol": "score-v2", "timestamp_utc": "new"},
        correction,
    ]

    assert expected_code in _codes(_validate(tmp_path, _attempt(), quality=quality))


@pytest.mark.parametrize(
    "event_kind",
    [None, "bounded_candidate_screened"],
    ids=["missing-kind", "foreign-domain"],
)
def test_quality_replacement_events_must_belong_to_quality_domain(
    tmp_path: Path,
    event_kind: str | None,
) -> None:
    invalidated = {"protocol": "score-v1", "timestamp_utc": "old"}
    replacement = {"protocol": "score-v2", "timestamp_utc": "new"}
    if event_kind is not None:
        invalidated["event"] = event_kind
        replacement["event"] = event_kind
    quality = [
        invalidated,
        replacement,
        {
            "event": "quality_event_invalidated",
            "reason_kind": QUALITY_INVALIDATION_REASON,
            "invalidated_protocol": "score-v1",
            "invalidated_event_timestamp_utc": "old",
            "replacement_protocol": "score-v2",
            "replacement_event_timestamp_utc": "new",
        },
    ]

    codes = _codes(_validate(tmp_path, _attempt(), quality=quality))

    assert "invalidated_event_outside_journal_domain" in codes
    assert "replacement_event_outside_journal_domain" in codes


def test_duplicate_json_keys_and_non_finite_numbers_are_rejected(
    tmp_path: Path,
) -> None:
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    sustained_path.write_bytes(
        b'{"event":"sustained_attempt_started","attempt_id":"a","candidate_id":"first","candidate_id":"candidate"}\n'
    )
    quality_path.write_bytes(b'{"event":"score","metric":NaN}\n')
    _write_jsonl(bounded_path, [])
    _registry(registry_path)

    codes = _codes(
        validate_repository_journals(
            sustained_journal=sustained_path,
            quality_journal=quality_path,
            bounded_journal=bounded_path,
            sustained_registry=registry_path,
        )
    )

    assert "duplicate_json_key" in codes
    assert "non_finite_json_number" in codes


def test_numeric_overflow_is_rejected_as_non_finite_json(tmp_path: Path) -> None:
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(sustained_path, _attempt())
    quality_path.write_bytes(b'{"event":"ocr_quality_scored","metric":1e9999}\n')
    _write_jsonl(bounded_path, [])
    _registry(registry_path)

    issues = validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    )

    assert "non_finite_json_number" in _codes(issues)


@pytest.mark.parametrize(
    ("registry_contents", "expected_code"),
    [
        (
            b'{"candidates":[{"id":"first","id":"candidate","configs":[{"workers":1}]}]}',
            "duplicate_registry_json_key",
        ),
        (
            b'{"candidates":[{"id":"candidate","configs":[{"workers":NaN}]}]}',
            "non_finite_registry_json_number",
        ),
        (
            b'{"candidates":[{"id":"candidate","configs":[{"workers":1e9999}]}]}',
            "non_finite_registry_json_number",
        ),
        (
            b'{"candidates":[{"id":"candidate","configs":[1]}]}',
            "invalid_registry_configs",
        ),
    ],
    ids=[
        "duplicate-key",
        "non-finite-number",
        "overflowing-number",
        "non-object-config",
    ],
)
def test_candidate_registry_json_and_config_shapes_are_strict(
    tmp_path: Path,
    registry_contents: bytes,
    expected_code: str,
) -> None:
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(sustained_path, _attempt())
    _write_jsonl(quality_path, [])
    _write_jsonl(bounded_path, [])
    registry_path.write_bytes(registry_contents)

    issues = validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    )

    assert expected_code in _codes(issues)


def test_boolean_supersession_line_is_not_an_integer_line_number(
    tmp_path: Path,
) -> None:
    correction = {
        "event": "sustained_attempts_invalidated",
        "invalidated_attempt_ids": ["attempt"],
    }
    encoded = json.dumps(correction, sort_keys=True).encode("utf-8")
    events = [
        correction,
        {
            "event": "journal_event_superseded",
            "protocol": "append-only-journal-integrity-v1",
            "superseded_event_line": True,
            "superseded_event_sha256": hashlib.sha256(encoded).hexdigest(),
            "reason_kind": "not_a_real_line_number",
        },
    ]

    assert "unresolved_supersession_target" in _codes(_validate(tmp_path, events))


def test_tracked_head_gate_fails_closed_when_git_head_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    journal_path = tmp_path / "results" / "events.jsonl"
    journal_path.parent.mkdir()
    journal_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(journal_validator_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        journal_validator_script.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=128,
            stdout=b"",
            stderr=b"fatal: invalid HEAD",
        ),
    )

    issues = journal_validator_script._tracked_head_prefix_issues([journal_path])

    assert [issue.code for issue in issues] == ["git_head_unavailable"]
