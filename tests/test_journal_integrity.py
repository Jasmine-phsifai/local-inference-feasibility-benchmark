import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.validate_event_journals as journal_validator_script

from local_inference_bench.journal_integrity import (
    effective_sustained_invalidated_attempt_ids,
    validate_append_only_record_prefix,
    validate_repository_journals,
)


QUALITY_INVALIDATION_REASON = "stale_authority_and_insufficient_provenance"
BOUNDED_INVALIDATION_REASON = "source_ancestry_claim_not_executed_by_harness"


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def _registry(path: Path, candidate_id: str = "candidate", configs: list[dict] | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "candidates": [
                    {"id": candidate_id, "configs": configs or [{"workers": 1}]}
                ]
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


def _validate(tmp_path: Path, sustained: list[dict], quality: list[dict] | None = None, bounded: list[dict] | None = None):
    sustained_path = tmp_path / "sustained.jsonl"
    quality_path = tmp_path / "quality.jsonl"
    bounded_path = tmp_path / "bounded.jsonl"
    registry_path = tmp_path / "registry.json"
    _write_jsonl(sustained_path, sustained)
    _write_jsonl(quality_path, quality or [])
    _write_jsonl(bounded_path, bounded or [])
    _registry(registry_path)
    return validate_repository_journals(
        sustained_journal=sustained_path,
        quality_journal=quality_path,
        bounded_journal=bounded_path,
        sustained_registry=registry_path,
    )


def _codes(issues) -> set[str]:
    return {issue.code for issue in issues}


def test_valid_lifecycle_and_registry_identity_pass(tmp_path: Path) -> None:
    assert _validate(tmp_path, _attempt()) == []


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
