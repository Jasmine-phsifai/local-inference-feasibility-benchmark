"""Semantic integrity checks for the repository's append-only event journals.

The validator deliberately does not repair evidence.  Exact historical smoke
configs may be admitted by payload hash, and an erroneous correction event may
be superseded by a later, hash-bound ``journal_event_superseded`` row.  Measurement
and score events themselves cannot be hidden by that mechanism.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Iterable

from .asr_agreement_public_protocol import (
    PROTOCOL as ASR_AGREEMENT_PROTOCOL,
    validate_public_event as validate_asr_agreement_event,
)
from .event_journal import read_journal_bytes
from .fingerprint import fingerprint_json
from .verified_blind_ocr_protocol import (
    PROTOCOL as VERIFIED_BLIND_OCR_PROTOCOL,
    validate_preparation_event,
    validate_score_event,
)


SUSTAINED_START_EVENT = "sustained_attempt_started"
SUSTAINED_TERMINAL_EVENTS = frozenset(
    {
        "sustained_attempt_succeeded",
        "sustained_attempt_partial",
        "sustained_attempt_failed",
    }
)
LEGACY_START_EVENT = "attempt_started"
LEGACY_TERMINAL_EVENTS = frozenset(
    {"attempt_succeeded", "attempt_failed", "attempt_interrupted"}
)
SUPERSESSION_EVENT = "journal_event_superseded"
SUPERSESSION_PROTOCOL = "append-only-journal-integrity-v1"
SUPERSEDEABLE_CORRECTION_EVENTS = frozenset(
    {
        "sustained_attempts_invalidated",
        "sustained_attempts_reclassified",
        "sustained_config_indices_reclassified",
        "sustained_telemetry_fields_invalidated",
        "quality_event_invalidated",
        "bounded_event_invalidated",
    }
)
SUSTAINED_REQUIRED_IDENTITY_FIELDS = frozenset(
    {"candidate_id", "config", "config_index", "trial_index"}
)
SUSTAINED_INVALIDATION_REPLACEMENTS = {
    "all_failed_attempt_mislabeled_as_succeeded": "fail_closed_status_and_count_invariant",
    "overstrict_repetition_gate": "calibrated_near_total_loop_gate",
    "stateless_decoder_cache_topology": "verified_stateful_with_past",
    "token_cap_accepted_by_compatibility_gate": "token_cap_fail_closed_256_token_gate",
}
EVENT_INVALIDATION_REASONS = {
    "quality_event_invalidated": frozenset(
        {
            "candidate_mapping_not_committed_to_judged_packet",
            "configured_limit_was_overstated_as_verified_maximum",
            "ignored_manifest_not_clean_clone_reproducible",
            "stale_authority_and_insufficient_provenance",
        }
    ),
    "bounded_event_invalidated": frozenset(
        {"source_ancestry_claim_not_executed_by_harness"}
    ),
}
REPLACEABLE_EVENT_KINDS = {
    "quality_event_invalidated": frozenset(
        {
            "asr_agreement_scored",
            "asr_quality_scored",
            "blind_ocr_quality_scored",
            "document_fidelity_scored",
            "ocr_quality_scored",
            "ocrllm_compatibility_checked",
        }
    ),
    "bounded_event_invalidated": frozenset(
        {
            "bounded_candidate_blocker_verified",
            "bounded_candidate_quality_verified",
            "bounded_candidate_screened",
            "candidate_acquisition_deferred",
            "openvino_genai_qwen3_asr_tail_fix_compared",
        }
    ),
}


@dataclass(frozen=True)
class JournalRecord:
    path: Path
    line_number: int
    event: dict
    line_sha256: str


@dataclass(frozen=True)
class JournalIssue:
    path: Path
    line_number: int | None
    code: str
    message: str

    def format(self) -> str:
        location = str(self.path)
        if self.line_number is not None:
            location = f"{location}:{self.line_number}"
        return f"{location} [{self.code}] {self.message}"


@dataclass(frozen=True)
class AttemptLifecycle:
    starts: dict[str, tuple[JournalRecord, ...]]
    terminals: dict[str, tuple[JournalRecord, ...]]


@dataclass(frozen=True)
class SustainedJournalSnapshot:
    events: tuple[dict, ...]
    invalidated_attempt_ids: frozenset[str]
    corrected_attempt_ids: frozenset[str]
    contents_sha256: str


HistoricalLegacyConfig = tuple[str, str]


class _DuplicateJsonKey(ValueError):
    pass


class _NonFiniteJsonNumber(ValueError):
    pass


def validate_append_only_record_prefix(
    path: Path,
    expected_head_contents: bytes,
) -> list[JournalIssue]:
    """Verify immutable HEAD records while tolerating Git line-ending conversion."""

    try:
        current_contents = path.read_bytes()
    except OSError as error:
        return [
            JournalIssue(
                path,
                None,
                "append_only_prefix_unreadable",
                f"cannot read journal for HEAD-prefix validation: {error}",
            )
        ]
    if current_contents.startswith(expected_head_contents):
        return []
    normalized_expected = expected_head_contents.replace(b"\r\n", b"\n")
    normalized_current = current_contents.replace(b"\r\n", b"\n")
    if normalized_current.startswith(normalized_expected):
        return []
    expected_records = _lf_records(normalized_expected)
    current_records = _lf_records(normalized_current)
    mismatch_index = next(
        (
            index
            for index, (expected, current) in enumerate(
                zip(expected_records, current_records),
                start=1,
            )
            if expected != current
        ),
        min(len(expected_records), len(current_records)) + 1,
    )
    return [
        JournalIssue(
            path,
            mismatch_index,
            "append_only_prefix_mismatch",
            "working journal does not preserve every HEAD record in order",
        )
    ]


def effective_sustained_invalidated_attempt_ids(path: Path) -> set[str]:
    """Return invalidated ids only after their correction rows validate."""

    records, issues = _read_journal(path)
    return _effective_sustained_invalidated_attempt_ids(records, issues)


def read_sustained_journal_snapshot(
    path: Path,
) -> tuple[list[dict], set[str], set[str]]:
    """Return events, invalidations, and corrected ids from one byte snapshot."""

    snapshot = capture_sustained_journal_snapshot(path)
    return (
        list(snapshot.events),
        set(snapshot.invalidated_attempt_ids),
        set(snapshot.corrected_attempt_ids),
    )


def capture_sustained_journal_snapshot(path: Path) -> SustainedJournalSnapshot:
    """Capture one writer-coherent sustained authority snapshot with its digest."""

    contents = read_journal_bytes(path)
    records, issues = _read_journal(path, contents=contents)
    invalidated = _effective_sustained_invalidated_attempt_ids(records, issues)
    superseded_lines = _validated_superseded_lines(records, [])
    correction_fields = {
        "sustained_attempts_reclassified": "reclassified_attempt_ids",
        "sustained_config_indices_reclassified": "reclassified_attempt_ids",
    }
    corrected = {
        attempt_id
        for record in records
        if record.line_number not in superseded_lines
        for target_field in [correction_fields.get(record.event.get("event"))]
        if target_field is not None
        and isinstance(record.event.get(target_field), list)
        for attempt_id in record.event[target_field]
        if type(attempt_id) is str and attempt_id
    }
    return SustainedJournalSnapshot(
        events=tuple(record.event for record in records),
        invalidated_attempt_ids=frozenset(invalidated),
        corrected_attempt_ids=frozenset(corrected),
        contents_sha256=sha256(contents).hexdigest(),
    )


def _effective_sustained_invalidated_attempt_ids(
    records: list[JournalRecord],
    issues: list[JournalIssue],
) -> set[str]:
    superseded_lines = _validated_superseded_lines(records, issues)
    active_invalidation_targets = {
        attempt_id
        for record in records
        if record.line_number not in superseded_lines
        and record.event.get("event") == "sustained_attempts_invalidated"
        and isinstance(record.event.get("invalidated_attempt_ids"), list)
        for attempt_id in record.event["invalidated_attempt_ids"]
        if isinstance(attempt_id, str) and attempt_id
    }
    lifecycle = _validate_attempt_lifecycle(
        records,
        start_event=SUSTAINED_START_EVENT,
        terminal_events=SUSTAINED_TERMINAL_EVENTS,
        issues=issues,
        issue_attempt_ids=active_invalidation_targets,
    )
    _validate_active_sustained_correction_conflicts(
        records,
        superseded_lines,
        issues,
    )
    invalidated: set[str] = set()
    for record in records:
        if (
            record.line_number in superseded_lines
            or record.event.get("event") != "sustained_attempts_invalidated"
        ):
            continue
        issue_count = len(issues)
        _validate_attempt_invalidation(record, lifecycle, issues)
        if len(issues) == issue_count:
            invalidated.update(record.event["invalidated_attempt_ids"])
    if issues:
        details = "; ".join(issue.format() for issue in issues)
        raise ValueError(f"sustained invalidation journal is invalid: {details}")
    return invalidated


def validate_repository_journals(
    *,
    sustained_journal: Path,
    quality_journal: Path,
    bounded_journal: Path,
    sustained_registry: Path,
    legacy_journal: Path | None = None,
    candidate_registry: Path | None = None,
    historical_legacy_configs: Iterable[HistoricalLegacyConfig] = (),
) -> list[JournalIssue]:
    """Return every detected contradiction without changing any input file."""

    issues: list[JournalIssue] = []
    historical_legacy = frozenset(historical_legacy_configs)

    sustained_records, read_issues = _read_journal(sustained_journal)
    issues.extend(read_issues)
    quality_records, read_issues = _read_journal(quality_journal)
    issues.extend(read_issues)
    bounded_records, read_issues = _read_journal(bounded_journal)
    issues.extend(read_issues)

    sustained_superseded = _validated_superseded_lines(sustained_records, issues)
    quality_superseded = _validated_superseded_lines(quality_records, issues)
    bounded_superseded = _validated_superseded_lines(bounded_records, issues)

    sustained_lifecycle = _validate_attempt_lifecycle(
        sustained_records,
        start_event=SUSTAINED_START_EVENT,
        terminal_events=SUSTAINED_TERMINAL_EVENTS,
        issues=issues,
    )
    sustained_candidates = _read_candidate_registry(sustained_registry, issues)
    projected_statuses = _validate_active_sustained_correction_conflicts(
        sustained_records,
        sustained_superseded,
        issues,
    )
    resolved_config_indices = _resolved_config_reclassifications(
        sustained_records,
        sustained_lifecycle,
        sustained_candidates,
        sustained_superseded,
    )
    _validate_sustained_terminal_outcomes(
        sustained_lifecycle,
        projected_statuses,
        issues,
    )
    _validate_sustained_registry_identities(
        sustained_lifecycle,
        sustained_candidates,
        resolved_config_indices,
        issues,
    )
    _validate_sustained_corrections(
        sustained_records,
        sustained_lifecycle,
        sustained_candidates,
        sustained_superseded,
        issues,
    )

    _validate_event_replacements(
        quality_records,
        correction_event="quality_event_invalidated",
        superseded_lines=quality_superseded,
        issues=issues,
        require_replacement_timestamp=True,
    )
    _validate_asr_agreement_score_events(
        quality_records,
        quality_superseded,
        sustained_candidates,
        issues,
    )
    _validate_blind_ocr_preparation_events(
        quality_records,
        quality_superseded,
        issues,
    )
    _validate_blind_ocr_score_events(
        quality_records,
        quality_superseded,
        sustained_candidates,
        issues,
    )
    _validate_event_replacements(
        bounded_records,
        correction_event="bounded_event_invalidated",
        superseded_lines=bounded_superseded,
        issues=issues,
        require_replacement_timestamp=False,
    )

    if legacy_journal is not None:
        legacy_records, read_issues = _read_journal(legacy_journal)
        issues.extend(read_issues)
        _validated_superseded_lines(legacy_records, issues)
        legacy_lifecycle = _validate_attempt_lifecycle(
            legacy_records,
            start_event=LEGACY_START_EVENT,
            terminal_events=LEGACY_TERMINAL_EVENTS,
            issues=issues,
        )
        if candidate_registry is not None:
            legacy_candidates = _read_candidate_registry(candidate_registry, issues)
            _validate_legacy_registry_identities(
                legacy_lifecycle,
                legacy_candidates,
                historical_legacy,
                issues,
            )

    return sorted(
        issues,
        key=lambda issue: (
            str(issue.path),
            issue.line_number if issue.line_number is not None else -1,
            issue.code,
            issue.message,
        ),
    )


def _read_journal(
    path: Path,
    *,
    contents: bytes | None = None,
) -> tuple[list[JournalRecord], list[JournalIssue]]:
    issues: list[JournalIssue] = []
    if contents is None:
        try:
            contents = read_journal_bytes(path)
        except FileNotFoundError:
            return [], [JournalIssue(path, None, "journal_missing", "journal does not exist")]
        except OSError as error:
            return [], [
                JournalIssue(path, None, "journal_unreadable", f"cannot read journal: {error}")
            ]
    if contents and not contents.endswith(b"\n"):
        issues.append(
            JournalIssue(
                path,
                len(contents.splitlines()),
                "missing_final_newline",
                "append-only journal ends without a newline",
            )
        )

    records: list[JournalRecord] = []
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        if not raw_line.strip():
            issues.append(
                JournalIssue(path, line_number, "blank_journal_line", "blank JSONL record")
            )
            continue
        try:
            decoded = raw_line.decode("utf-8")
            event = json.loads(
                decoded,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_non_finite_json_number,
                parse_float=_parse_finite_json_float,
            )
        except _DuplicateJsonKey as error:
            issues.append(
                JournalIssue(
                    path,
                    line_number,
                    "duplicate_json_key",
                    f"JSON object repeats key {str(error)!r}",
                )
            )
            continue
        except _NonFiniteJsonNumber as error:
            issues.append(
                JournalIssue(
                    path,
                    line_number,
                    "non_finite_json_number",
                    f"JSON number {str(error)!r} is not finite",
                )
            )
            continue
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
            issues.append(
                JournalIssue(path, line_number, "invalid_json", f"invalid UTF-8 JSON: {error}")
            )
            continue
        if not isinstance(event, dict):
            issues.append(
                JournalIssue(path, line_number, "non_object_event", "event must be a JSON object")
            )
            continue
        records.append(
            JournalRecord(
                path=path,
                line_number=line_number,
                event=event,
                line_sha256=sha256(raw_line).hexdigest(),
            )
        )
    return records, issues


def _read_candidate_registry(
    path: Path,
    issues: list[JournalIssue],
) -> dict[str, dict]:
    try:
        registry = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_non_finite_json_number,
            parse_float=_parse_finite_json_float,
        )
    except _DuplicateJsonKey as error:
        issues.append(
            JournalIssue(
                path,
                None,
                "duplicate_registry_json_key",
                f"candidate registry repeats key {str(error)!r}",
            )
        )
        return {}
    except _NonFiniteJsonNumber as error:
        issues.append(
            JournalIssue(
                path,
                None,
                "non_finite_registry_json_number",
                f"candidate registry number {str(error)!r} is not finite",
            )
        )
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        issues.append(
            JournalIssue(path, None, "invalid_registry", f"cannot read candidate registry: {error}")
        )
        return {}
    candidates = registry.get("candidates") if isinstance(registry, dict) else None
    if not isinstance(candidates, list):
        issues.append(
            JournalIssue(path, None, "invalid_registry", "registry candidates must be a list")
        )
        return {}
    result: dict[str, dict] = {}
    for candidate in candidates:
        candidate_id = candidate.get("id") if isinstance(candidate, dict) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            issues.append(
                JournalIssue(path, None, "invalid_registry_candidate", "candidate has no string id")
            )
            continue
        if candidate_id in result:
            issues.append(
                JournalIssue(
                    path,
                    None,
                    "duplicate_registry_candidate",
                    f"candidate id {candidate_id!r} appears more than once",
                )
            )
            continue
        if not isinstance(candidate.get("configs"), list) or not all(
            isinstance(config, dict) for config in candidate["configs"]
        ):
            issues.append(
                JournalIssue(
                    path,
                    None,
                    "invalid_registry_configs",
                    f"candidate {candidate_id!r} configs must be a list of objects",
                )
            )
            continue
        historical_configs = candidate.get("historical_configs", [])
        if not isinstance(historical_configs, list) or not all(
            isinstance(config, dict) for config in historical_configs
        ):
            issues.append(
                JournalIssue(
                    path,
                    None,
                    "invalid_historical_registry_configs",
                    f"candidate {candidate_id!r} historical_configs must be a list of objects",
                )
            )
            continue
        result[candidate_id] = candidate
    return result


def _validated_superseded_lines(
    records: list[JournalRecord],
    issues: list[JournalIssue],
) -> frozenset[int]:
    by_line = {record.line_number: record for record in records}
    superseded: set[int] = set()
    for record in records:
        event = record.event
        if event.get("event") != SUPERSESSION_EVENT:
            continue
        valid = True
        if event.get("protocol") != SUPERSESSION_PROTOCOL:
            _add_issue(
                issues,
                record,
                "invalid_supersession_protocol",
                f"protocol must be {SUPERSESSION_PROTOCOL!r}",
            )
            valid = False
        target_line = event.get("superseded_event_line")
        target = by_line.get(target_line) if _is_config_index(target_line) else None
        if target is None or target.line_number >= record.line_number:
            _add_issue(
                issues,
                record,
                "unresolved_supersession_target",
                "superseded_event_line must identify an earlier retained event",
            )
            valid = False
        elif target.event.get("event") not in SUPERSEDEABLE_CORRECTION_EVENTS:
            _add_issue(
                issues,
                record,
                "forbidden_supersession_target",
                "only a correction event may be superseded",
            )
            valid = False
        expected_hash = event.get("superseded_event_sha256")
        if target is not None and expected_hash != target.line_sha256:
            _add_issue(
                issues,
                record,
                "supersession_hash_mismatch",
                "superseded_event_sha256 does not bind the retained target line",
            )
            valid = False
        if not isinstance(event.get("reason_kind"), str) or not event["reason_kind"]:
            _add_issue(
                issues,
                record,
                "missing_supersession_reason",
                "reason_kind must be a non-empty string",
            )
            valid = False
        if _is_config_index(target_line) and target_line in superseded:
            _add_issue(
                issues,
                record,
                "duplicate_supersession",
                f"event line {target_line} was already superseded",
            )
            valid = False
        if valid:
            superseded.add(target_line)
    return frozenset(superseded)


def _validate_asr_agreement_score_events(
    records: list[JournalRecord],
    superseded_lines: frozenset[int],
    sustained_candidates: dict[str, dict],
    issues: list[JournalIssue],
) -> None:
    seen_hashes: dict[str, JournalRecord] = {}
    for record in records:
        event = record.event
        if (
            record.line_number in superseded_lines
            or event.get("event") != "asr_agreement_scored"
            or event.get("protocol") != ASR_AGREEMENT_PROTOCOL
        ):
            continue
        try:
            score = validate_asr_agreement_event(
                event,
                source_matches_registry=lambda source: (
                    _asr_score_source_matches_registry(
                        source,
                        sustained_candidates,
                    )
                ),
            )
        except ValueError as error:
            _add_issue(
                issues,
                record,
                "invalid_asr_agreement_score_event",
                str(error),
            )
            continue

        public_hash = score["public_event_sha256"]
        previous_hash = seen_hashes.get(public_hash)
        if previous_hash is not None:
            _add_issue(
                issues,
                record,
                "duplicate_asr_agreement_score_hash",
                f"ASR agreement public hash already appears on line {previous_hash.line_number}",
            )
        else:
            seen_hashes[public_hash] = record


def _asr_score_source_matches_registry(
    source: dict,
    sustained_candidates: dict[str, dict],
) -> bool:
    candidate = sustained_candidates.get(source["candidate_id"])
    config_index = source["config_index"]
    if not isinstance(candidate, dict) or candidate.get("task") != "asr":
        return False
    configs = candidate.get("configs")
    if not isinstance(configs, list) or config_index >= len(configs):
        return False
    config = configs[config_index]
    return (
        isinstance(config, dict)
        and source["config_fingerprint"] == fingerprint_json(config)
    )


def _validate_blind_ocr_preparation_events(
    records: list[JournalRecord],
    superseded_lines: frozenset[int],
    issues: list[JournalIssue],
) -> None:
    seen: dict[str, JournalRecord] = {}
    for record in records:
        if (
            record.line_number in superseded_lines
            or record.event.get("event") != "blind_ocr_packet_prepared"
        ):
            continue
        try:
            validate_preparation_event(record.event)
        except ValueError as error:
            _add_issue(
                issues,
                record,
                "invalid_blind_ocr_preparation_event",
                str(error),
            )
            continue
        preparation_id = record.event.get("preparation_id")
        try:
            canonical_id = str(uuid.UUID(preparation_id))
        except (ValueError, TypeError, AttributeError):
            canonical_id = None
        if canonical_id is None or canonical_id != preparation_id:
            _add_issue(
                issues,
                record,
                "invalid_blind_ocr_preparation_id",
                "blind OCR preparation_id must be a canonical UUID",
            )
            continue
        previous = seen.get(preparation_id)
        if previous is not None:
            _add_issue(
                issues,
                record,
                "duplicate_blind_ocr_preparation_id",
                f"blind OCR preparation_id already appears on line {previous.line_number}",
            )
            continue
        seen[preparation_id] = record


def _validate_blind_ocr_score_events(
    records: list[JournalRecord],
    superseded_lines: frozenset[int],
    sustained_candidates: dict[str, dict],
    issues: list[JournalIssue],
) -> None:
    preparations_by_hash: dict[str, list[JournalRecord]] = {}
    for record in records:
        if (
            record.line_number in superseded_lines
            or record.event.get("event") != "blind_ocr_packet_prepared"
        ):
            continue
        try:
            preparation = validate_preparation_event(record.event)
        except ValueError:
            continue
        preparations_by_hash.setdefault(
            preparation["public_event_sha256"],
            [],
        ).append(record)

    seen_score_anchors: dict[str, JournalRecord] = {}
    for record in records:
        event = record.event
        if (
            record.line_number in superseded_lines
            or event.get("event") != "blind_ocr_quality_scored"
            or event.get("protocol") != VERIFIED_BLIND_OCR_PROTOCOL
        ):
            continue
        try:
            score = validate_score_event(event)
        except ValueError as error:
            _add_issue(
                issues,
                record,
                "invalid_blind_ocr_score_event",
                str(error),
            )
            continue
        anchor = score["preparation_public_event_sha256"]
        preparation_matches = preparations_by_hash.get(anchor, [])
        if (
            len(preparation_matches) != 1
            or preparation_matches[0].line_number >= record.line_number
        ):
            _add_issue(
                issues,
                record,
                "unresolved_blind_ocr_preparation_anchor",
                "v10 score must reference exactly one earlier valid preparation event",
            )
        else:
            preparation = preparation_matches[0].event
            availability = score["metrics"]["source_record_availability"]
            preparation_counts = preparation["selected_source_status_counts"]
            if (
                score["metrics"]["sample_count"] != preparation["sample_count"]
                or availability["available_record_count"]
                != preparation_counts["available"]
                or availability["failed_record_count"]
                != preparation_counts["failed"]
                or availability["unavailable_record_count"]
                != preparation_counts["unavailable"]
            ):
                _add_issue(
                    issues,
                    record,
                    "blind_ocr_score_preparation_mismatch",
                    "v10 score counts do not match its public preparation anchor",
                )
        previous = seen_score_anchors.get(anchor)
        if previous is not None:
            _add_issue(
                issues,
                record,
                "duplicate_blind_ocr_score_anchor",
                f"preparation anchor already has a score on line {previous.line_number}",
            )
        else:
            seen_score_anchors[anchor] = record
        for source in score["source_candidates"]:
            if not _score_source_matches_registry(source, sustained_candidates):
                _add_issue(
                    issues,
                    record,
                    "blind_ocr_score_registry_mismatch",
                    "v10 score source candidate/config is absent from the sustained registry",
                )


def _score_source_matches_registry(
    source: dict,
    sustained_candidates: dict[str, dict],
) -> bool:
    candidate = sustained_candidates.get(source["candidate_id"])
    config_index = source["config_index"]
    if not isinstance(candidate, dict) or candidate.get("task") != "ocr":
        return False
    configs = candidate.get("configs")
    if (
        not isinstance(configs, list)
        or config_index >= len(configs)
    ):
        return False
    config = configs[config_index]
    if (
        not isinstance(config, dict)
        or source["config_fingerprint"] != fingerprint_json(config)
    ):
        return False
    return True


def _validate_attempt_lifecycle(
    records: list[JournalRecord],
    *,
    start_event: str,
    terminal_events: frozenset[str],
    issues: list[JournalIssue],
    issue_attempt_ids: set[str] | None = None,
) -> AttemptLifecycle:
    starts: dict[str, list[JournalRecord]] = {}
    terminals: dict[str, list[JournalRecord]] = {}
    for record in records:
        event_name = record.event.get("event")
        if event_name != start_event and event_name not in terminal_events:
            continue
        attempt_id = record.event.get("attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id:
            if issue_attempt_ids is None:
                _add_issue(
                    issues,
                    record,
                    "missing_attempt_id",
                    "attempt event has no string id",
                )
            continue
        destination = starts if event_name == start_event else terminals
        destination.setdefault(attempt_id, []).append(record)

    for attempt_id, start_records in starts.items():
        if issue_attempt_ids is not None and attempt_id not in issue_attempt_ids:
            continue
        if len(start_records) != 1:
            for record in start_records[1:]:
                _add_issue(
                    issues,
                    record,
                    "duplicate_attempt_start",
                    f"attempt {attempt_id!r} has more than one start",
                )
        terminal_records = terminals.get(attempt_id, [])
        if not terminal_records:
            _add_issue(
                issues,
                start_records[0],
                "unterminated_attempt",
                f"attempt {attempt_id!r} has no terminal event",
            )
            continue
        if len(terminal_records) != 1:
            for record in terminal_records[1:]:
                _add_issue(
                    issues,
                    record,
                    "duplicate_attempt_terminal",
                    f"attempt {attempt_id!r} has more than one terminal event",
                )
        start = start_records[0]
        terminal = terminal_records[0]
        if terminal.line_number <= start.line_number:
            _add_issue(
                issues,
                terminal,
                "terminal_precedes_start",
                f"attempt {attempt_id!r} terminal does not follow its start",
            )
        for field in ("candidate_id", "attempt_key", "config", "config_index", "trial_index"):
            start_has_field = field in start.event
            terminal_has_field = field in terminal.event
            required = (
                start_event == SUSTAINED_START_EVENT
                and field in SUSTAINED_REQUIRED_IDENTITY_FIELDS
            )
            if start_has_field != terminal_has_field or (
                required and not start_has_field
            ):
                _add_issue(
                    issues,
                    terminal,
                    "incomplete_attempt_identity",
                    f"attempt {attempt_id!r} does not record {field!r} on both lifecycle events",
                )
            elif start_has_field and not _json_values_equal(
                start.event[field], terminal.event[field]
            ):
                _add_issue(
                    issues,
                    terminal,
                    "attempt_identity_mismatch",
                    f"attempt {attempt_id!r} changed {field!r} between start and terminal",
                )
        if start_event == SUSTAINED_START_EVENT:
            _validate_sustained_identity_types(start, issues)
            _validate_sustained_identity_types(terminal, issues)

    for attempt_id, terminal_records in terminals.items():
        if issue_attempt_ids is not None and attempt_id not in issue_attempt_ids:
            continue
        if attempt_id in starts:
            continue
        for record in terminal_records:
            _add_issue(
                issues,
                record,
                "orphan_attempt_terminal",
                f"attempt {attempt_id!r} has no start event",
            )

    return AttemptLifecycle(
        starts={key: tuple(value) for key, value in starts.items()},
        terminals={key: tuple(value) for key, value in terminals.items()},
    )


def _validate_sustained_identity_types(
    record: JournalRecord,
    issues: list[JournalIssue],
) -> None:
    validators = {
        "candidate_id": lambda value: isinstance(value, str) and bool(value),
        "attempt_key": lambda value: isinstance(value, str) and bool(value),
        "config": lambda value: isinstance(value, dict),
        "config_index": _is_nonnegative_integer,
        "trial_index": _is_nonnegative_integer,
    }
    for field, validator in validators.items():
        if field in record.event and not validator(record.event[field]):
            _add_issue(
                issues,
                record,
                "invalid_attempt_identity",
                f"sustained attempt identity field {field!r} has an invalid type or value",
            )


def _validate_active_sustained_correction_conflicts(
    records: list[JournalRecord],
    superseded_lines: frozenset[int],
    issues: list[JournalIssue],
) -> dict[str, str]:
    """Reject overlapping active corrections and return unique status projections."""

    target_field_by_event = {
        "sustained_attempts_invalidated": "invalidated_attempt_ids",
        "sustained_attempts_reclassified": "reclassified_attempt_ids",
        "sustained_config_indices_reclassified": "reclassified_attempt_ids",
    }
    owners: dict[tuple[str, str], JournalRecord] = {}
    projected_statuses: dict[str, str] = {}
    conflicted_status_ids: set[str] = set()
    for record in records:
        if record.line_number in superseded_lines:
            continue
        event_name = record.event.get("event")
        target_field = target_field_by_event.get(event_name)
        if target_field is None:
            continue
        target_ids = record.event.get(target_field)
        if (
            not isinstance(target_ids, list)
            or not target_ids
            or not all(type(attempt_id) is str and attempt_id for attempt_id in target_ids)
            or len(target_ids) != len(set(target_ids))
        ):
            _add_issue(
                issues,
                record,
                "invalid_active_correction_targets",
                f"{target_field} must be a nonempty list of unique nonempty strings",
            )
            continue
        for attempt_id in target_ids:
            key = (event_name, attempt_id)
            previous = owners.get(key)
            if previous is not None:
                _add_issue(
                    issues,
                    record,
                    "conflicting_active_correction",
                    f"attempt {attempt_id!r} is already targeted by active {event_name!r} line {previous.line_number}",
                )
                if event_name == "sustained_attempts_reclassified":
                    conflicted_status_ids.add(attempt_id)
                    projected_statuses.pop(attempt_id, None)
                continue
            owners[key] = record
            if event_name == "sustained_attempts_reclassified":
                status = record.event.get("reclassified_status")
                if status in {"complete", "partial_failure", "all_failed"}:
                    projected_statuses[attempt_id] = status

    for attempt_id in conflicted_status_ids:
        projected_statuses.pop(attempt_id, None)
    return projected_statuses


def _validate_sustained_terminal_outcomes(
    lifecycle: AttemptLifecycle,
    projected_statuses: dict[str, str],
    issues: list[JournalIssue],
) -> None:
    """Bind terminal event names, public status, and item counts."""

    for attempt_id, terminal_records in lifecycle.terminals.items():
        if len(terminal_records) != 1:
            continue
        terminal = terminal_records[0]
        original_status = _terminal_status(terminal)
        if original_status is None:
            continue
        effective_status = projected_statuses.get(attempt_id, original_status)
        result = terminal.event.get("result")
        if not isinstance(result, dict):
            if original_status != "all_failed":
                _add_issue(
                    issues,
                    terminal,
                    "missing_terminal_result",
                    f"attempt {attempt_id!r} has no result object",
                )
            continue

        nested_candidate = result.get("candidate_id")
        if (
            nested_candidate is not None
            and nested_candidate != terminal.event.get("candidate_id")
        ):
            _add_issue(
                issues,
                terminal,
                "terminal_result_identity_mismatch",
                f"attempt {attempt_id!r} result candidate does not match its terminal identity",
            )

        reported_status = result.get("status")
        if reported_status is not None and reported_status != effective_status:
            _add_issue(
                issues,
                terminal,
                "terminal_result_status_mismatch",
                f"attempt {attempt_id!r} result status {reported_status!r} does not match effective status {effective_status!r}",
            )

        counts = result.get("counts")
        if not isinstance(counts, dict):
            if effective_status in {"complete", "partial_failure"}:
                _add_issue(
                    issues,
                    terminal,
                    "missing_terminal_counts",
                    f"attempt {attempt_id!r} has no result counts",
                )
            continue
        values = [counts.get(field) for field in ("attempted", "completed", "failed")]
        if not all(_is_nonnegative_integer(value) for value in values):
            _add_issue(
                issues,
                terminal,
                "invalid_terminal_counts",
                f"attempt {attempt_id!r} counts must be non-negative integers",
            )
            continue
        attempted, completed, failed = values
        if attempted != completed + failed:
            _add_issue(
                issues,
                terminal,
                "terminal_count_total_mismatch",
                f"attempt {attempt_id!r} attempted count does not equal completed plus failed",
            )
            continue
        status_counts_match = {
            "complete": attempted > 0 and completed == attempted and failed == 0,
            "partial_failure": attempted > 0 and completed > 0 and failed > 0,
            "all_failed": attempted > 0 and completed == 0 and failed == attempted,
        }[effective_status]
        if not status_counts_match:
            _add_issue(
                issues,
                terminal,
                "terminal_status_count_mismatch",
                f"attempt {attempt_id!r} counts do not match effective status {effective_status!r}",
            )


def _validate_sustained_registry_identities(
    lifecycle: AttemptLifecycle,
    candidates: dict[str, dict],
    resolved_config_indices: dict[str, int],
    issues: list[JournalIssue],
) -> None:
    for attempt_id, start_records in lifecycle.starts.items():
        if not start_records:
            continue
        record = start_records[0]
        event = record.event
        candidate_id = event.get("candidate_id")
        config_index = event.get("config_index")
        candidate = candidates.get(candidate_id) if isinstance(candidate_id, str) else None
        if candidate is None:
            _add_issue(
                issues,
                record,
                "unregistered_candidate_identity",
                f"candidate {candidate_id!r} is not in the sustained registry",
            )
            continue
        configs = candidate["configs"]
        corrected_index = resolved_config_indices.get(attempt_id)
        if corrected_index is not None:
            if _json_values_equal(event.get("config"), configs[corrected_index]):
                continue
        if not _is_config_index(config_index) or not 0 <= config_index < len(configs):
            _add_issue(
                issues,
                record,
                "unresolved_config_identity",
                f"candidate {candidate_id!r} config index {config_index!r} is not registered",
            )
            continue
        if not _json_values_equal(event.get("config"), configs[config_index]):
            _add_issue(
                issues,
                record,
                "registry_config_mismatch",
                f"candidate {candidate_id!r} config index {config_index} no longer maps to the recorded config",
            )


def _resolved_config_reclassifications(
    records: list[JournalRecord],
    lifecycle: AttemptLifecycle,
    candidates: dict[str, dict],
    superseded_lines: frozenset[int],
) -> dict[str, int]:
    """Return only exact payload-bound config-index corrections."""

    resolved: dict[str, int] = {}
    for record in records:
        event = record.event
        if (
            record.line_number in superseded_lines
            or event.get("event") != "sustained_config_indices_reclassified"
        ):
            continue
        candidate_id = event.get("candidate_id")
        prior_index = event.get("prior_config_index")
        replacement_index = event.get("replacement_config_index")
        target_ids = event.get("reclassified_attempt_ids")
        candidate = (
            candidates.get(candidate_id) if isinstance(candidate_id, str) else None
        )
        if (
            candidate is None
            or not _is_config_index(prior_index)
            or not _is_config_index(replacement_index)
            or not 0 <= replacement_index < len(candidate["configs"])
            or not isinstance(target_ids, list)
            or not target_ids
            or not all(isinstance(attempt_id, str) for attempt_id in target_ids)
            or len(target_ids) != len(set(target_ids))
        ):
            continue
        replacement_config = candidate["configs"][replacement_index]
        for attempt_id in target_ids:
            starts = lifecycle.starts.get(attempt_id, ())
            terminals = lifecycle.terminals.get(attempt_id, ())
            if len(starts) != 1 or len(terminals) != 1:
                continue
            start = starts[0]
            terminal = terminals[0]
            if (
                terminal.line_number >= record.line_number
                or start.event.get("candidate_id") != candidate_id
                or terminal.event.get("candidate_id") != candidate_id
                or start.event.get("config_index") != prior_index
                or terminal.event.get("config_index") != prior_index
                or not _json_values_equal(start.event.get("config"), replacement_config)
                or not _json_values_equal(terminal.event.get("config"), replacement_config)
                or attempt_id in resolved
            ):
                continue
            resolved[attempt_id] = replacement_index
    return resolved


def _validate_legacy_registry_identities(
    lifecycle: AttemptLifecycle,
    candidates: dict[str, dict],
    historical: frozenset[HistoricalLegacyConfig],
    issues: list[JournalIssue],
) -> None:
    for start_records in lifecycle.starts.values():
        record = start_records[0]
        candidate_id = record.event.get("candidate_id")
        recorded_config = record.event.get("config")
        historical_identity = (
            candidate_id,
            _config_sha256(recorded_config),
        )
        if isinstance(candidate_id, str) and historical_identity in historical:
            continue
        candidate = candidates.get(candidate_id) if isinstance(candidate_id, str) else None
        if candidate is None:
            _add_issue(
                issues,
                record,
                "unregistered_candidate_identity",
                f"candidate {candidate_id!r} is not in the smoke registry",
            )
            continue
        registered_configs = [
            *candidate["configs"],
            *candidate.get("historical_configs", []),
        ]
        if not any(
            _json_values_equal(recorded_config, registered_config)
            for registered_config in registered_configs
        ):
            _add_issue(
                issues,
                record,
                "registry_config_mismatch",
                f"candidate {candidate_id!r} recorded config is not in the smoke registry; config_sha256={historical_identity[1]}",
            )


def _validate_sustained_corrections(
    records: list[JournalRecord],
    lifecycle: AttemptLifecycle,
    candidates: dict[str, dict],
    superseded_lines: frozenset[int],
    issues: list[JournalIssue],
) -> None:
    for record in records:
        if record.line_number in superseded_lines:
            continue
        event_name = record.event.get("event")
        if event_name == "sustained_attempts_invalidated":
            _validate_attempt_invalidation(record, lifecycle, issues)
        elif event_name == "sustained_attempts_reclassified":
            _validate_attempt_reclassification(record, lifecycle, issues)
        elif event_name == "sustained_config_indices_reclassified":
            _validate_config_reclassification(
                record,
                lifecycle,
                candidates,
                issues,
            )
        elif event_name == "sustained_telemetry_fields_invalidated":
            _validate_telemetry_invalidation(record, lifecycle, issues)


def _validate_attempt_invalidation(
    record: JournalRecord,
    lifecycle: AttemptLifecycle,
    issues: list[JournalIssue],
) -> None:
    event = record.event
    target_ids = _validated_id_list(record, "invalidated_attempt_ids", issues)
    reason = event.get("reason_kind")
    if not isinstance(reason, str) or not reason:
        _add_issue(issues, record, "missing_invalidation_reason", "reason_kind is required")
        reason_supported = False
    elif reason not in SUSTAINED_INVALIDATION_REPLACEMENTS:
        _add_issue(
            issues,
            record,
            "unsupported_invalidation_reason",
            f"reason_kind {reason!r} is not part of the sustained invalidation protocol",
        )
        reason_supported = False
    else:
        reason_supported = True
    replacement_kind = event.get("replacement_kind")
    if not isinstance(replacement_kind, str) or not replacement_kind:
        _add_issue(issues, record, "missing_replacement_kind", "replacement_kind is required")
    elif (
        reason_supported
        and replacement_kind != SUSTAINED_INVALIDATION_REPLACEMENTS[reason]
    ):
        _add_issue(
            issues,
            record,
            "invalidation_replacement_kind_mismatch",
            f"reason {reason!r} requires replacement_kind {SUSTAINED_INVALIDATION_REPLACEMENTS[reason]!r}",
        )

    expected_terminal_by_reason = {
        "all_failed_attempt_mislabeled_as_succeeded": "sustained_attempt_succeeded",
        "overstrict_repetition_gate": "sustained_attempt_succeeded",
        "token_cap_accepted_by_compatibility_gate": "sustained_attempt_succeeded",
    }
    for attempt_id in target_ids:
        terminal = _resolved_terminal(record, attempt_id, lifecycle, issues)
        if terminal is None:
            continue
        correction_candidate = event.get("candidate_id")
        if correction_candidate != terminal.event.get("candidate_id"):
            _add_issue(
                issues,
                record,
                "invalidation_candidate_mismatch",
                f"attempt {attempt_id!r} belongs to {terminal.event.get('candidate_id')!r}",
            )
        expected_terminal = expected_terminal_by_reason.get(reason)
        if expected_terminal is not None and terminal.event.get("event") != expected_terminal:
            _add_issue(
                issues,
                record,
                "invalidation_reason_terminal_mismatch",
                f"reason {reason!r} requires a {expected_terminal!r} target, but attempt {attempt_id!r} ended as {terminal.event.get('event')!r}",
            )
            continue
        if reason == "all_failed_attempt_mislabeled_as_succeeded":
            counts = _result_counts(terminal)
            if not (
                counts is not None
                and _is_positive_integer(counts.get("attempted"))
                and _is_nonnegative_integer(counts.get("completed"))
                and counts.get("completed") == 0
                and _is_positive_integer(counts.get("failed"))
                and counts.get("failed") == counts.get("attempted")
            ):
                _add_issue(
                    issues,
                    record,
                    "invalidation_reason_evidence_mismatch",
                    f"attempt {attempt_id!r} does not contain an all-failed result count",
                )
        elif reason == "overstrict_repetition_gate":
            counts = _result_counts(terminal)
            result = terminal.event.get("result")
            generation = result.get("generation") if isinstance(result, dict) else None
            if not (
                counts is not None
                and _is_positive_integer(counts.get("completed"))
                and _is_positive_integer(counts.get("failed"))
                and isinstance(generation, dict)
                and _is_positive_integer(
                    generation.get("unhealthy_generation_count")
                )
                and generation.get("unhealthy_generation_count")
                == counts.get("failed")
            ):
                _add_issue(
                    issues,
                    record,
                    "invalidation_reason_evidence_mismatch",
                    f"attempt {attempt_id!r} does not contain the mixed-result repetition-gate evidence",
                )
        elif reason == "token_cap_accepted_by_compatibility_gate":
            result = terminal.event.get("result")
            generation = result.get("generation") if isinstance(result, dict) else None
            if not (
                isinstance(generation, dict)
                and _is_positive_integer(generation.get("token_cap_hit_count"))
            ):
                _add_issue(
                    issues,
                    record,
                    "invalidation_reason_evidence_mismatch",
                    f"attempt {attempt_id!r} does not record a token-cap hit",
                )
        elif reason == "stateless_decoder_cache_topology":
            result = terminal.event.get("result")
            generation = result.get("generation") if isinstance(result, dict) else None
            failed_compatibility = (
                terminal.event.get("event") == "sustained_attempt_failed"
                and terminal.event.get("phase") == "compatibility"
                and terminal.event.get("failure_kind") == "invalid_response"
            )
            degenerate_compatibility = (
                terminal.event.get("event") == "sustained_attempt_succeeded"
                and terminal.event.get("phase") == "compatibility"
                and isinstance(generation, dict)
                and _is_positive_integer(generation.get("token_cap_hit_count"))
                and _is_nonnegative_integer(generation.get("terminal_eos_count"))
                and generation.get("terminal_eos_count") == 0
            )
            if not (failed_compatibility or degenerate_compatibility):
                _add_issue(
                    issues,
                    record,
                    "invalidation_reason_evidence_mismatch",
                    f"attempt {attempt_id!r} does not contain stateless-decoder compatibility evidence",
                )


def _validate_attempt_reclassification(
    record: JournalRecord,
    lifecycle: AttemptLifecycle,
    issues: list[JournalIssue],
) -> None:
    event = record.event
    target_ids = _validated_id_list(record, "reclassified_attempt_ids", issues)
    replacement_status = event.get("reclassified_status")
    if replacement_status not in {"complete", "partial_failure", "all_failed"}:
        _add_issue(
            issues,
            record,
            "invalid_reclassified_status",
            "reclassified_status must be complete, partial_failure, or all_failed",
        )
    reason = event.get("reason_kind")
    if not isinstance(reason, str) or not reason:
        _add_issue(issues, record, "missing_reclassification_reason", "reason_kind is required")
    elif reason != "terminal_event_name_did_not_reflect_item_failures":
        _add_issue(
            issues,
            record,
            "unsupported_reclassification_reason",
            f"reason_kind {reason!r} is not part of the status reclassification protocol",
        )

    for attempt_id in target_ids:
        terminal = _resolved_terminal(record, attempt_id, lifecycle, issues)
        if terminal is None:
            continue
        if _terminal_status(terminal) == replacement_status:
            _add_issue(
                issues,
                record,
                "reclassification_already_applied",
                f"attempt {attempt_id!r} already has terminal status {replacement_status!r}",
            )
            continue
        if reason == "terminal_event_name_did_not_reflect_item_failures":
            if terminal.event.get("event") != "sustained_attempt_succeeded":
                _add_issue(
                    issues,
                    record,
                    "reclassification_reason_terminal_mismatch",
                    f"attempt {attempt_id!r} was not recorded as succeeded",
                )
            counts = _result_counts(terminal)
            partial_evidence = (
                counts is not None
                and _is_positive_integer(counts.get("attempted"))
                and _is_positive_integer(counts.get("completed"))
                and _is_positive_integer(counts.get("failed"))
                and counts.get("attempted")
                == counts.get("completed") + counts.get("failed")
            )
            all_failed_evidence = (
                counts is not None
                and _is_positive_integer(counts.get("attempted"))
                and _is_nonnegative_integer(counts.get("completed"))
                and counts.get("completed") == 0
                and _is_positive_integer(counts.get("failed"))
                and counts.get("failed") == counts.get("attempted")
            )
            evidence_matches = {
                "partial_failure": partial_evidence,
                "all_failed": all_failed_evidence,
            }.get(replacement_status, False)
            if not evidence_matches:
                _add_issue(
                    issues,
                    record,
                    "reclassification_reason_evidence_mismatch",
                    f"attempt {attempt_id!r} counts do not support status {replacement_status!r}",
                )


def _validate_config_reclassification(
    record: JournalRecord,
    lifecycle: AttemptLifecycle,
    candidates: dict[str, dict],
    issues: list[JournalIssue],
) -> None:
    event = record.event
    target_ids = _validated_id_list(record, "reclassified_attempt_ids", issues)
    candidate_id = event.get("candidate_id")
    prior_index = event.get("prior_config_index")
    replacement_index = event.get("replacement_config_index")
    _validate_declared_identity(
        record,
        candidate_id,
        prior_index,
        "prior",
        candidates,
        issues,
    )
    _validate_declared_identity(
        record,
        candidate_id,
        replacement_index,
        "replacement",
        candidates,
        issues,
    )
    for attempt_id in target_ids:
        terminal = _resolved_terminal(record, attempt_id, lifecycle, issues)
        if terminal is None:
            continue
        if terminal.event.get("candidate_id") != candidate_id:
            _add_issue(
                issues,
                record,
                "reclassification_candidate_mismatch",
                f"attempt {attempt_id!r} belongs to {terminal.event.get('candidate_id')!r}",
            )
        actual_index = terminal.event.get("config_index")
        if _json_values_equal(actual_index, replacement_index):
            _add_issue(
                issues,
                record,
                "config_reclassification_already_applied",
                f"attempt {attempt_id!r} already records replacement config index {replacement_index!r}",
            )
        elif not _json_values_equal(actual_index, prior_index):
            _add_issue(
                issues,
                record,
                "config_reclassification_prior_mismatch",
                f"attempt {attempt_id!r} records config index {actual_index!r}, not prior index {prior_index!r}",
            )
        candidate = candidates.get(candidate_id) if isinstance(candidate_id, str) else None
        if (
            candidate is not None
            and _is_config_index(replacement_index)
            and 0 <= replacement_index < len(candidate["configs"])
            and not _json_values_equal(
                terminal.event.get("config"), candidate["configs"][replacement_index]
            )
        ):
            _add_issue(
                issues,
                record,
                "reclassified_config_payload_mismatch",
                f"attempt {attempt_id!r} config payload does not match replacement index {replacement_index!r}",
            )


def _validate_declared_identity(
    record: JournalRecord,
    candidate_id: object,
    config_index: object,
    label: str,
    candidates: dict[str, dict],
    issues: list[JournalIssue],
) -> None:
    candidate = candidates.get(candidate_id) if isinstance(candidate_id, str) else None
    if candidate is None or not _is_config_index(config_index):
        _add_issue(
            issues,
            record,
            "unresolved_reclassified_config_identity",
            f"{label} identity ({candidate_id!r}, {config_index!r}) is not registered or explicitly historical",
        )
        return
    if not 0 <= config_index < len(candidate["configs"]):
        _add_issue(
            issues,
            record,
            "unresolved_reclassified_config_identity",
            f"{label} identity ({candidate_id!r}, {config_index!r}) is not registered or explicitly historical",
        )


def _validate_telemetry_invalidation(
    record: JournalRecord,
    lifecycle: AttemptLifecycle,
    issues: list[JournalIssue],
) -> None:
    event = record.event
    target_ids = _validated_id_list(record, "invalidated_attempt_ids", issues)
    fields = event.get("invalidated_fields")
    if not isinstance(fields, list) or not fields or not all(
        isinstance(field, str) and field for field in fields
    ):
        _add_issue(
            issues,
            record,
            "invalid_telemetry_field_list",
            "invalidated_fields must be a non-empty string list",
        )
        return
    for attempt_id in target_ids:
        terminal = _resolved_terminal(record, attempt_id, lifecycle, issues)
        if terminal is None:
            continue
        telemetry = terminal.event.get("host_telemetry")
        if not isinstance(telemetry, dict) or not any(field in telemetry for field in fields):
            _add_issue(
                issues,
                record,
                "telemetry_invalidation_field_missing",
                f"attempt {attempt_id!r} has none of the invalidated telemetry fields",
            )


def _validate_event_replacements(
    records: list[JournalRecord],
    *,
    correction_event: str,
    superseded_lines: frozenset[int],
    issues: list[JournalIssue],
    require_replacement_timestamp: bool,
) -> None:
    ordinary_records = [
        record
        for record in records
        if record.event.get("event")
        not in {*SUPERSEDEABLE_CORRECTION_EVENTS, SUPERSESSION_EVENT}
    ]
    claimed_invalidated_lines: dict[int, JournalRecord] = {}
    resolved_replacements: list[
        tuple[JournalRecord, JournalRecord, JournalRecord]
    ] = []
    for correction in records:
        if (
            correction.event.get("event") != correction_event
            or correction.line_number in superseded_lines
        ):
            continue
        event = correction.event
        reason = event.get("reason_kind")
        allowed_reasons = EVENT_INVALIDATION_REASONS[correction_event]
        if not isinstance(reason, str) or not reason:
            _add_issue(
                issues,
                correction,
                "missing_event_invalidation_reason",
                "reason_kind must be a non-empty string",
            )
        elif reason not in allowed_reasons:
            _add_issue(
                issues,
                correction,
                "unsupported_event_invalidation_reason",
                f"reason_kind {reason!r} is not part of the {correction_event!r} protocol",
            )
        invalidated = _resolve_event_reference(
            correction,
            ordinary_records,
            timestamp=event.get("invalidated_event_timestamp_utc"),
            protocol=event.get("invalidated_protocol"),
            label="invalidated",
            issues=issues,
        )
        if invalidated is not None and invalidated.line_number >= correction.line_number:
            _add_issue(
                issues,
                correction,
                "invalidation_target_not_earlier",
                "invalidated event must precede its correction",
            )
        if invalidated is not None:
            previous = claimed_invalidated_lines.get(invalidated.line_number)
            if previous is not None:
                _add_issue(
                    issues,
                    correction,
                    "conflicting_active_event_invalidation",
                    f"event line {invalidated.line_number} is already invalidated by active correction line {previous.line_number}",
                )
            else:
                claimed_invalidated_lines[invalidated.line_number] = correction

        replacement_timestamp = event.get("replacement_event_timestamp_utc")
        replacement_protocol = event.get("replacement_protocol")
        if require_replacement_timestamp and not isinstance(replacement_timestamp, str):
            _add_issue(
                issues,
                correction,
                "missing_replacement_timestamp",
                "replacement_event_timestamp_utc is required",
            )
            continue
        replacement = _resolve_event_reference(
            correction,
            ordinary_records,
            timestamp=replacement_timestamp,
            protocol=replacement_protocol,
            label="replacement",
            issues=issues,
            timestamp_optional=not require_replacement_timestamp,
        )
        _validate_replacement_identities(
            correction,
            invalidated=invalidated,
            replacement=replacement,
            issues=issues,
        )
        if invalidated is not None and replacement is not None:
            allowed_event_kinds = REPLACEABLE_EVENT_KINDS[correction_event]
            for label, referenced in (
                ("invalidated", invalidated),
                ("replacement", replacement),
            ):
                referenced_kind = referenced.event.get("event")
                if referenced_kind not in allowed_event_kinds:
                    _add_issue(
                        issues,
                        correction,
                        f"{label}_event_outside_journal_domain",
                        f"{label} event kind {referenced_kind!r} is not replaceable in this journal",
                    )
            if invalidated.line_number == replacement.line_number:
                _add_issue(
                    issues,
                    correction,
                    "replacement_equals_invalidated_event",
                    "replacement must identify a different retained event",
                )
            if invalidated.event.get("event") != replacement.event.get("event"):
                _add_issue(
                    issues,
                    correction,
                    "replacement_event_kind_mismatch",
                    f"invalidated event kind {invalidated.event.get('event')!r} does not match replacement kind {replacement.event.get('event')!r}",
                )
            resolved_replacements.append((correction, invalidated, replacement))

    invalidated_lines = set(claimed_invalidated_lines)
    for correction, _invalidated, replacement in resolved_replacements:
        if replacement.line_number in invalidated_lines:
            _add_issue(
                issues,
                correction,
                "replacement_event_is_invalidated",
                f"replacement event line {replacement.line_number} is itself actively invalidated",
            )


def _validate_replacement_identities(
    correction: JournalRecord,
    *,
    invalidated: JournalRecord | None,
    replacement: JournalRecord | None,
    issues: list[JournalIssue],
) -> None:
    """Bind correction identities to referenced events when identity is available.

    Historical events are not required to expose these fields.  A correction
    that replaces one public candidate or workload with another must make that
    transition explicit instead of overloading the invalidated identity.
    """

    for field in ("candidate_id", "workload_class"):
        invalidated_identity = _referenced_event_identity(
            correction,
            invalidated,
            field=field,
            label="invalidated",
            issues=issues,
        )
        replacement_identity = _referenced_event_identity(
            correction,
            replacement,
            field=field,
            label="replacement",
            issues=issues,
        )
        declared_invalidated = correction.event.get(field)
        replacement_field = f"replacement_{field}"
        declared_replacement = correction.event.get(
            replacement_field,
            declared_invalidated,
        )

        _validate_declared_replacement_identity(
            correction,
            field=field,
            label="invalidated",
            actual=invalidated_identity,
            declared=declared_invalidated,
            issues=issues,
        )
        _validate_declared_replacement_identity(
            correction,
            field=field,
            label="replacement",
            actual=replacement_identity,
            declared=declared_replacement,
            issues=issues,
        )


def _referenced_event_identity(
    correction: JournalRecord,
    referenced: JournalRecord | None,
    *,
    field: str,
    label: str,
    issues: list[JournalIssue],
) -> str | None:
    if referenced is None:
        return None
    values = []
    top_level = referenced.event.get(field)
    if isinstance(top_level, str) and top_level:
        values.append(top_level)
    result = referenced.event.get("result")
    nested = result.get(field) if isinstance(result, dict) else None
    if isinstance(nested, str) and nested:
        values.append(nested)
    unique = set(values)
    if len(unique) > 1:
        _add_issue(
            issues,
            correction,
            f"ambiguous_{label}_{field}",
            f"{label} event exposes conflicting {field!r} values",
        )
        return None
    return values[0] if values else None


def _validate_declared_replacement_identity(
    correction: JournalRecord,
    *,
    field: str,
    label: str,
    actual: str | None,
    declared: object,
    issues: list[JournalIssue],
) -> None:
    if actual is None:
        return
    if not isinstance(declared, str) or not declared:
        _add_issue(
            issues,
            correction,
            f"missing_{label}_{field}",
            f"correction must declare the referenced {label} event's {field!r}",
        )
    elif declared != actual:
        _add_issue(
            issues,
            correction,
            f"{label}_{field}_mismatch",
            f"correction declares {field} {declared!r}, but {label} event records {actual!r}",
        )


def _resolve_event_reference(
    correction: JournalRecord,
    records: list[JournalRecord],
    *,
    timestamp: object,
    protocol: object,
    label: str,
    issues: list[JournalIssue],
    timestamp_optional: bool = False,
) -> JournalRecord | None:
    if not isinstance(protocol, str) or not protocol:
        _add_issue(
            issues,
            correction,
            f"missing_{label}_protocol",
            f"{label}_protocol must be a non-empty string",
        )
        return None
    if not timestamp_optional and (not isinstance(timestamp, str) or not timestamp):
        _add_issue(
            issues,
            correction,
            f"missing_{label}_timestamp",
            f"{label}_event_timestamp_utc must be a non-empty string",
        )
        return None
    if timestamp_optional and timestamp is not None and (
        not isinstance(timestamp, str) or not timestamp
    ):
        _add_issue(
            issues,
            correction,
            f"invalid_{label}_timestamp",
            f"{label}_event_timestamp_utc must be a non-empty string when provided",
        )
        return None
    matches = [record for record in records if record.event.get("protocol") == protocol]
    if isinstance(timestamp, str) and timestamp:
        matches = [record for record in matches if record.event.get("timestamp_utc") == timestamp]
    if len(matches) != 1:
        _add_issue(
            issues,
            correction,
            f"unresolved_{label}_event",
            f"{label} reference resolved to {len(matches)} retained events",
        )
        return None
    return matches[0]


def _validated_id_list(
    record: JournalRecord,
    field: str,
    issues: list[JournalIssue],
) -> list[str]:
    value = record.event.get(field)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ):
        _add_issue(
            issues,
            record,
            "invalid_attempt_target_list",
            f"{field} must be a non-empty string list",
        )
        return []
    if len(set(value)) != len(value):
        _add_issue(
            issues,
            record,
            "duplicate_attempt_target",
            f"{field} contains duplicate attempt ids",
        )
    return list(dict.fromkeys(value))


def _resolved_terminal(
    correction: JournalRecord,
    attempt_id: str,
    lifecycle: AttemptLifecycle,
    issues: list[JournalIssue],
) -> JournalRecord | None:
    starts = lifecycle.starts.get(attempt_id, ())
    terminals = lifecycle.terminals.get(attempt_id, ())
    if len(starts) != 1 or len(terminals) != 1:
        _add_issue(
            issues,
            correction,
            "unresolved_attempt_target",
            f"attempt {attempt_id!r} does not resolve to exactly one start and terminal",
        )
        return None
    terminal = terminals[0]
    if terminal.line_number >= correction.line_number:
        _add_issue(
            issues,
            correction,
            "attempt_target_not_earlier",
            f"attempt {attempt_id!r} terminal does not precede the correction",
        )
        return None
    return terminal


def _terminal_status(record: JournalRecord) -> str | None:
    return {
        "sustained_attempt_succeeded": "complete",
        "sustained_attempt_partial": "partial_failure",
        "sustained_attempt_failed": "all_failed",
    }.get(record.event.get("event"))


def _result_counts(record: JournalRecord) -> dict | None:
    result = record.event.get("result")
    counts = result.get("counts") if isinstance(result, dict) else None
    return counts if isinstance(counts, dict) else None


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_non_finite_json_number(value: str) -> object:
    raise _NonFiniteJsonNumber(value)


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not isfinite(parsed):
        raise _NonFiniteJsonNumber(value)
    return parsed


def _lf_records(contents: bytes) -> list[bytes]:
    if not contents:
        return []
    if contents.endswith(b"\n"):
        contents = contents[:-1]
    return contents.split(b"\n")


def _json_values_equal(left: object, right: object) -> bool:
    try:
        return json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _config_sha256(config: object) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _is_config_index(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_nonnegative_integer(value: object) -> bool:
    return _is_config_index(value) and value >= 0


def _is_positive_integer(value: object) -> bool:
    return _is_nonnegative_integer(value) and value > 0


def _add_issue(
    issues: list[JournalIssue],
    record: JournalRecord,
    code: str,
    message: str,
) -> None:
    issues.append(JournalIssue(record.path, record.line_number, code, message))
