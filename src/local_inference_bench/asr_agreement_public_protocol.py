"""Validate privacy-bounded ASR agreement v10 public events."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from datetime import datetime

from .validate_public_summary import validate_public_summary


PROTOCOL = "asr-text-agreement-v10"
MINIMUM_EXACT_AGGREGATE_DENOMINATOR = 10
CHARACTER_COUNT_BUCKET_UPPER_BOUNDS = (0, 32, 128, 512, 2_048)
FRACTION_BUCKET_LOWER_BOUNDS = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
MAX_CANDIDATES = 8
MAX_SAMPLE_COUNT = 4_096

PUBLIC_PRIVACY = {
    "small_private_cohort_threshold": MINIMUM_EXACT_AGGREGATE_DENOMINATOR,
    "minimum_exact_numeric_character_metric_denominator": (
        MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    ),
    "exact_numeric_character_metrics_require_per_metric_denominator": True,
    "character_count_bucket_upper_bounds": [
        *CHARACTER_COUNT_BUCKET_UPPER_BOUNDS,
        None,
    ],
    "fraction_bucket_lower_bounds": list(FRACTION_BUCKET_LOWER_BOUNDS),
    "private_fingerprints_published": False,
    "private_run_identifiers_published": False,
    "private_commitment_keys_published": False,
    "opaque_keyed_source_commitments_verified": True,
    "decoded_pcm_fingerprints_published": False,
}
INTERPRETATION = {
    "agreement_is_not_ground_truth": True,
    "trusted_gold_used": False,
    "failed_outputs_excluded_from_success_metrics": True,
    "text_only": True,
    "timestamps_compared": False,
    "exact_decoded_pcm_content_uniqueness_verified": True,
    "semantic_audio_independence_verified": False,
    "source_authority_lock_honoring_writers_required": True,
}

_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_HEX_16 = re.compile(r"[0-9a-f]{16}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})
_EVENT_FIELDS = frozenset(
    {
        "event",
        "candidate_id",
        "protocol",
        "scorer_fingerprint",
        "source_authority_fingerprint",
        "workload_class",
        "source_candidates",
        "privacy",
        "interpretation",
        "metrics",
        "public_event_sha256",
        "timestamp_utc",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "candidate_evidence_id",
        "candidate_id",
        "status",
        "config_index",
        "config_fingerprint",
    }
)
_METRIC_FIELDS = frozenset(
    {"sample_count", "candidate_count", "small_private_cohort", "candidates", "pairs"}
)
_CANDIDATE_METRIC_FIELDS = frozenset(
    {
        "candidate_evidence_id",
        "availability",
        "successful_output_metrics",
        "failed_output_diagnostics",
    }
)
_CANDIDATE_AVAILABILITY_FIELDS = frozenset(
    {"attempted_sample_count", "successful_sample_count", "unavailable_sample_count"}
)
_SUCCESSFUL_OUTPUT_FIELDS = frozenset(
    {
        "sample_denominator",
        "speech_sample_denominator",
        "near_silence_sample_denominator",
        "exact_character_aggregates_published",
        "near_silence_exact_character_aggregates_published",
        "speech_sample_count",
        "successful_speech_sample_count",
        "successful_speech_nonempty_count",
        "near_silence_sample_count",
        "successful_near_silence_sample_count",
        "successful_near_silence_nonempty_count",
        "normalized_character_count_bucket",
        "near_silence_normalized_character_count_bucket",
        "repeated_trigram_observed",
        "near_silence_successful_seconds",
        "near_silence_characters_per_minute",
        "mean_normalized_character_count",
        "mean_observed_repeated_trigram_ratio",
    }
)
_FAILED_OUTPUT_FIELDS = frozenset(
    {"explicit_failed_record_count", "missing_record_count", "any_explicit_failed_output_nonempty"}
)
_PAIR_FIELDS = frozenset(
    {
        "pair_evidence_id",
        "left_candidate_evidence_id",
        "right_candidate_evidence_id",
        "availability",
        "successful_output_agreement",
    }
)
_PAIR_AVAILABILITY_FIELDS = frozenset(
    {"comparable_sample_count", "unavailable_sample_count"}
)
_PAIR_AGREEMENT_FIELDS = frozenset(
    {
        "sample_denominator",
        "exact_character_aggregates_published",
        "normalized_character_similarity_is_character_micro_weighted",
        "normalized_character_denominator",
        "normalized_character_similarity",
        "normalized_character_similarity_bucket",
        "mean_length_agreement",
        "mean_length_agreement_bucket",
        "exact_match_count",
        "one_empty_disagreement_count",
        "any_exact_match",
        "all_comparable_exact_matches",
        "any_one_empty_disagreement",
    }
)


def public_event_sha256(event: Mapping[str, object]) -> str:
    """Return the semantic event digest; publication time is not identity."""

    body = {
        key: value
        for key, value in event.items()
        if key not in {"public_event_sha256", "timestamp_utc"}
    }
    serialized = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def validate_public_event(
    event: object,
    *,
    source_matches_registry: Callable[[dict], bool],
) -> dict:
    """Return a closed-schema v10 event whose sources match a registry snapshot."""

    if not isinstance(event, dict) or set(event) != _EVENT_FIELDS:
        raise ValueError("ASR agreement public event is invalid")
    metrics = event.get("metrics")
    sources = event.get("source_candidates")
    timestamp_value = event.get("timestamp_utc")
    try:
        timestamp = datetime.fromisoformat(timestamp_value)
        validated_metrics = validate_public_summary(metrics)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("ASR agreement public event is invalid") from error
    if (
        event.get("event") != "asr_agreement_scored"
        or event.get("candidate_id") != "private_course_asr_agreement"
        or event.get("protocol") != PROTOCOL
        or event.get("workload_class") != "private_course"
        or type(event.get("scorer_fingerprint")) is not str
        or _HEX_16.fullmatch(event["scorer_fingerprint"]) is None
        or type(event.get("source_authority_fingerprint")) is not str
        or _HEX_16.fullmatch(event["source_authority_fingerprint"]) is None
        or not _json_values_equal(metrics, validated_metrics)
        or not _validate_metrics(metrics)
        or not isinstance(sources, list)
        or len(sources) != metrics["candidate_count"]
        or not _validate_sources(
            sources,
            metrics=metrics,
            source_matches_registry=source_matches_registry,
        )
        or not _json_values_equal(
            event.get("privacy"),
            {
                **PUBLIC_PRIVACY,
                "workload_meets_minimum_exact_aggregate_denominator": (
                    not metrics["small_private_cohort"]
                ),
            },
        )
        or not _json_values_equal(event.get("interpretation"), INTERPRETATION)
        or type(timestamp_value) is not str
        or timestamp.isoformat(timespec="microseconds") != timestamp_value
        or timestamp.utcoffset() is None
        or timestamp.utcoffset().total_seconds() != 0
        or type(event.get("public_event_sha256")) is not str
        or _HEX_64.fullmatch(event["public_event_sha256"]) is None
        or event["public_event_sha256"] != public_event_sha256(event)
    ):
        raise ValueError("ASR agreement public event is invalid")
    return event


def _validate_metrics(metrics: object) -> bool:
    if not isinstance(metrics, dict) or set(metrics) != _METRIC_FIELDS:
        return False
    sample_count = metrics.get("sample_count")
    candidate_count = metrics.get("candidate_count")
    small = metrics.get("small_private_cohort")
    candidates = metrics.get("candidates")
    pairs = metrics.get("pairs")
    if (
        type(sample_count) is not int
        or not 1 <= sample_count <= MAX_SAMPLE_COUNT
        or type(candidate_count) is not int
        or not 2 <= candidate_count <= MAX_CANDIDATES
        or type(small) is not bool
        or small != (sample_count < MINIMUM_EXACT_AGGREGATE_DENOMINATOR)
        or not isinstance(candidates, list)
        or len(candidates) != candidate_count
        or not isinstance(pairs, list)
        or len(pairs) != candidate_count * (candidate_count - 1) // 2
    ):
        return False
    for expected_id, candidate in enumerate(candidates, start=1):
        if not _validate_candidate_metrics(
            candidate,
            expected_id=expected_id,
            sample_count=sample_count,
            small_private_cohort=small,
        ):
            return False
    candidate_total_characters = [
        _integer_total_from_mean(
            candidate["successful_output_metrics"][
                "mean_normalized_character_count"
            ],
            candidate["availability"]["successful_sample_count"],
        )
        if candidate["successful_output_metrics"][
            "exact_character_aggregates_published"
        ]
        else None
        for candidate in candidates
    ]
    workload_splits = {
        (
            candidate["successful_output_metrics"]["speech_sample_count"],
            candidate["successful_output_metrics"]["near_silence_sample_count"],
        )
        for candidate in candidates
    }
    if len(workload_splits) != 1:
        return False
    expected_pairs = [
        (left, right)
        for left in range(1, candidate_count + 1)
        for right in range(left + 1, candidate_count + 1)
    ]
    pairwise_success_overlaps: dict[tuple[int, int], int] = {}
    for offset, (pair, (left, right)) in enumerate(zip(pairs, expected_pairs)):
        left_successes = candidates[left - 1]["availability"]["successful_sample_count"]
        right_successes = candidates[right - 1]["availability"]["successful_sample_count"]
        left_successful_metrics = candidates[left - 1]["successful_output_metrics"]
        right_successful_metrics = candidates[right - 1]["successful_output_metrics"]
        left_empty_successes = left_successes - (
            left_successful_metrics["successful_speech_nonempty_count"]
            + left_successful_metrics["successful_near_silence_nonempty_count"]
        )
        right_empty_successes = right_successes - (
            right_successful_metrics["successful_speech_nonempty_count"]
            + right_successful_metrics["successful_near_silence_nonempty_count"]
        )
        if not _validate_pair_metrics(
            pair,
            expected_pair_id=candidate_count + 1 + offset,
            expected_left=left,
            expected_right=right,
            sample_count=sample_count,
            small_private_cohort=small,
            left_successful_sample_count=left_successes,
            right_successful_sample_count=right_successes,
            left_total_characters=candidate_total_characters[left - 1],
            right_total_characters=candidate_total_characters[right - 1],
            left_empty_successes=left_empty_successes,
            right_empty_successes=right_empty_successes,
        ):
            return False
        pairwise_success_overlaps[(left, right)] = pair["availability"][
            "comparable_sample_count"
        ]
    success_counts = [
        candidate["availability"]["successful_sample_count"]
        for candidate in candidates
    ]
    for left in range(1, candidate_count - 1):
        for middle in range(left + 1, candidate_count):
            for right in range(middle + 1, candidate_count + 1):
                left_middle = pairwise_success_overlaps[(left, middle)]
                left_right = pairwise_success_overlaps[(left, right)]
                middle_right = pairwise_success_overlaps[(middle, right)]
                left_successes = success_counts[left - 1]
                middle_successes = success_counts[middle - 1]
                right_successes = success_counts[right - 1]
                minimum_triple_overlap = max(
                    0,
                    left_middle + left_right - left_successes,
                    left_middle + middle_right - middle_successes,
                    left_right + middle_right - right_successes,
                )
                maximum_triple_overlap = min(
                    left_middle,
                    left_right,
                    middle_right,
                    sample_count
                    - left_successes
                    - middle_successes
                    - right_successes
                    + left_middle
                    + left_right
                    + middle_right,
                )
                if minimum_triple_overlap > maximum_triple_overlap:
                    return False
    return True


def _validate_candidate_metrics(
    candidate: object,
    *,
    expected_id: int,
    sample_count: int,
    small_private_cohort: bool,
) -> bool:
    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_METRIC_FIELDS:
        return False
    availability = candidate.get("availability")
    successful = candidate.get("successful_output_metrics")
    failed = candidate.get("failed_output_diagnostics")
    if (
        type(candidate.get("candidate_evidence_id")) is not int
        or candidate["candidate_evidence_id"] != expected_id
        or not isinstance(availability, dict)
        or set(availability) != _CANDIDATE_AVAILABILITY_FIELDS
        or not isinstance(successful, dict)
        or set(successful) != _SUCCESSFUL_OUTPUT_FIELDS
        or not isinstance(failed, dict)
        or set(failed) != _FAILED_OUTPUT_FIELDS
    ):
        return False
    attempted = availability.get("attempted_sample_count")
    succeeded = availability.get("successful_sample_count")
    unavailable = availability.get("unavailable_sample_count")
    explicit_failed = failed.get("explicit_failed_record_count")
    missing = failed.get("missing_record_count")
    speech = successful.get("speech_sample_count")
    near_silence = successful.get("near_silence_sample_count")
    successful_speech = successful.get("successful_speech_sample_count")
    successful_near_silence = successful.get("successful_near_silence_sample_count")
    sample_denominator = successful.get("sample_denominator")
    speech_sample_denominator = successful.get("speech_sample_denominator")
    near_silence_sample_denominator = successful.get(
        "near_silence_sample_denominator"
    )
    integer_values = (
        attempted,
        succeeded,
        unavailable,
        explicit_failed,
        missing,
        speech,
        near_silence,
        successful_speech,
        successful_near_silence,
        successful.get("successful_speech_nonempty_count"),
        successful.get("successful_near_silence_nonempty_count"),
    )
    if any(type(value) is not int or value < 0 for value in integer_values):
        return False
    if any(
        type(value) is not int or value < 0
        for value in (
            sample_denominator,
            speech_sample_denominator,
            near_silence_sample_denominator,
        )
    ):
        return False
    exact = not small_private_cohort and succeeded >= MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    near_exact = (
        not small_private_cohort
        and successful_near_silence >= MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    )
    if (
        attempted != sample_count
        or succeeded + unavailable != sample_count
        or explicit_failed + missing != unavailable
        or type(failed.get("any_explicit_failed_output_nonempty")) is not bool
        or (
            explicit_failed == 0
            and failed["any_explicit_failed_output_nonempty"]
        )
        or speech + near_silence != sample_count
        or successful_speech + successful_near_silence != succeeded
        or successful_speech > speech
        or successful_near_silence > near_silence
        or successful["successful_speech_nonempty_count"] > successful_speech
        or successful["successful_near_silence_nonempty_count"] > successful_near_silence
        or sample_denominator != succeeded
        or speech_sample_denominator != successful_speech
        or near_silence_sample_denominator != successful_near_silence
        or type(successful.get("exact_character_aggregates_published")) is not bool
        or successful["exact_character_aggregates_published"] != exact
        or type(successful.get("near_silence_exact_character_aggregates_published"))
        is not bool
        or successful["near_silence_exact_character_aggregates_published"] != near_exact
        or not _valid_bucket(successful.get("normalized_character_count_bucket"), 5)
        or not _valid_bucket(
            successful.get("near_silence_normalized_character_count_bucket"), 5
        )
        or type(successful.get("repeated_trigram_observed")) is not bool
        or not _optional_nonnegative_number(
            successful.get("near_silence_successful_seconds"), published=near_exact
        )
        or not _optional_nonnegative_number(
            successful.get("near_silence_characters_per_minute"), published=near_exact
        )
        or not _optional_nonnegative_number(
            successful.get("mean_normalized_character_count"), published=exact
        )
        or not _optional_fraction(
            successful.get("mean_observed_repeated_trigram_ratio"), published=exact
        )
    ):
        return False
    normalized_character_count_bucket = successful[
        "normalized_character_count_bucket"
    ]
    near_silence_character_count_bucket = successful[
        "near_silence_normalized_character_count_bucket"
    ]
    repeated_trigram_observed = successful["repeated_trigram_observed"]
    total_nonempty_successes = (
        successful["successful_speech_nonempty_count"]
        + successful["successful_near_silence_nonempty_count"]
    )
    if succeeded == 0 and (
        normalized_character_count_bucket != 0 or repeated_trigram_observed
    ):
        return False
    if (
        (total_nonempty_successes == 0)
        != (normalized_character_count_bucket == 0)
        or not _bucket_can_contain_at_least(
            normalized_character_count_bucket,
            total_nonempty_successes,
        )
        or (successful["successful_near_silence_nonempty_count"] == 0)
        != (near_silence_character_count_bucket == 0)
        or not _bucket_can_contain_at_least(
            near_silence_character_count_bucket,
            successful["successful_near_silence_nonempty_count"],
        )
        or near_silence_character_count_bucket > normalized_character_count_bucket
    ):
        return False
    total_characters: int | None = None
    if exact:
        total_characters = _integer_total_from_mean(
            successful["mean_normalized_character_count"],
            succeeded,
        )
        if (
            total_characters is None
            or _character_count_bucket(total_characters)
            != normalized_character_count_bucket
            or total_characters < total_nonempty_successes
            or (total_characters == 0) != (total_nonempty_successes == 0)
            or repeated_trigram_observed
            != (successful["mean_observed_repeated_trigram_ratio"] > 0.0)
        ):
            return False
    if near_exact:
        near_silence_seconds = successful["near_silence_successful_seconds"]
        near_silence_characters = _integer_total_from_rate(
            successful["near_silence_characters_per_minute"],
            near_silence_seconds,
        )
        near_silence_nonempty = successful[
            "successful_near_silence_nonempty_count"
        ]
        if (
            near_silence_seconds <= 0.0
            or near_silence_characters is None
            or total_characters is None
            or _character_count_bucket(near_silence_characters)
            != near_silence_character_count_bucket
            or near_silence_characters > total_characters
            or near_silence_characters < near_silence_nonempty
            or (near_silence_characters == 0) != (near_silence_nonempty == 0)
        ):
            return False
    return True


def _validate_pair_metrics(
    pair: object,
    *,
    expected_pair_id: int,
    expected_left: int,
    expected_right: int,
    sample_count: int,
    small_private_cohort: bool,
    left_successful_sample_count: int,
    right_successful_sample_count: int,
    left_total_characters: int | None,
    right_total_characters: int | None,
    left_empty_successes: int,
    right_empty_successes: int,
) -> bool:
    if not isinstance(pair, dict) or set(pair) != _PAIR_FIELDS:
        return False
    availability = pair.get("availability")
    agreement = pair.get("successful_output_agreement")
    if (
        type(pair.get("pair_evidence_id")) is not int
        or pair["pair_evidence_id"] != expected_pair_id
        or type(pair.get("left_candidate_evidence_id")) is not int
        or pair["left_candidate_evidence_id"] != expected_left
        or type(pair.get("right_candidate_evidence_id")) is not int
        or pair["right_candidate_evidence_id"] != expected_right
        or not isinstance(availability, dict)
        or set(availability) != _PAIR_AVAILABILITY_FIELDS
        or not isinstance(agreement, dict)
        or set(agreement) != _PAIR_AGREEMENT_FIELDS
    ):
        return False
    comparable = availability.get("comparable_sample_count")
    unavailable = availability.get("unavailable_sample_count")
    if (
        type(comparable) is not int
        or comparable < 0
        or type(unavailable) is not int
        or unavailable < 0
        or comparable + unavailable != sample_count
        or comparable
        < max(
            0,
            left_successful_sample_count
            + right_successful_sample_count
            - sample_count,
        )
        or comparable
        > min(left_successful_sample_count, right_successful_sample_count)
    ):
        return False
    exact = (
        not small_private_cohort
        and comparable >= MINIMUM_EXACT_AGGREGATE_DENOMINATOR
    )
    normalized_bucket = agreement.get("normalized_character_similarity_bucket")
    length_bucket = agreement.get("mean_length_agreement_bucket")
    sample_denominator = agreement.get("sample_denominator")
    if (
        type(sample_denominator) is not int
        or sample_denominator != comparable
        or type(agreement.get("exact_character_aggregates_published")) is not bool
        or agreement["exact_character_aggregates_published"] != exact
        or agreement.get("normalized_character_similarity_is_character_micro_weighted")
        is not True
        or not _optional_nonnegative_int(
            agreement.get("normalized_character_denominator"), published=exact
        )
        or not _optional_fraction(
            agreement.get("normalized_character_similarity"), published=exact
        )
        or not _optional_fraction(
            agreement.get("mean_length_agreement"), published=exact
        )
        or not _optional_nonnegative_int(
            agreement.get("exact_match_count"), published=exact, maximum=comparable
        )
        or not _optional_nonnegative_int(
            agreement.get("one_empty_disagreement_count"),
            published=exact,
            maximum=comparable,
        )
        or type(agreement.get("any_exact_match")) is not bool
        or type(agreement.get("all_comparable_exact_matches")) is not bool
        or type(agreement.get("any_one_empty_disagreement")) is not bool
        or (comparable == 0 and (normalized_bucket is not None or length_bucket is not None))
        or (comparable > 0 and not _valid_bucket(normalized_bucket, 5))
        or (comparable > 0 and not _valid_bucket(length_bucket, 5))
    ):
        return False
    if exact:
        exact_matches = agreement["exact_match_count"]
        one_empty = agreement["one_empty_disagreement_count"]
        if (
            agreement["any_exact_match"] != (exact_matches > 0)
            or agreement["all_comparable_exact_matches"]
            != (comparable > 0 and exact_matches == comparable)
            or agreement["any_one_empty_disagreement"] != (one_empty > 0)
            or exact_matches + one_empty > comparable
            or normalized_bucket
            != _fraction_bucket(agreement["normalized_character_similarity"])
            or length_bucket != _fraction_bucket(agreement["mean_length_agreement"])
        ):
            return False
    elif comparable == 0 and any(
        agreement[key]
        for key in (
            "any_exact_match",
            "all_comparable_exact_matches",
            "any_one_empty_disagreement",
        )
    ):
        return False
    if agreement["any_one_empty_disagreement"] and length_bucket == 5:
        return False
    if (
        agreement["any_one_empty_disagreement"]
        and left_empty_successes + right_empty_successes == 0
    ):
        return False
    if (
        agreement["all_comparable_exact_matches"]
        and (
            not agreement["any_exact_match"]
            or agreement["any_one_empty_disagreement"]
            or normalized_bucket != 5
            or length_bucket != 5
            or (
                exact
                and (
                    agreement["normalized_character_similarity"] != 1.0
                    or agreement["mean_length_agreement"] != 1.0
                )
            )
        )
    ):
        return False
    if normalized_bucket == 5 and not agreement["all_comparable_exact_matches"]:
        return False
    if (
        comparable == 1
        and agreement["any_exact_match"]
        != agreement["all_comparable_exact_matches"]
    ):
        return False
    if exact:
        normalized_denominator = agreement["normalized_character_denominator"]
        normalized_similarity = agreement["normalized_character_similarity"]
        minimum_denominator = 0
        if comparable == left_successful_sample_count:
            if left_total_characters is None:
                return False
            minimum_denominator = max(minimum_denominator, left_total_characters)
        if comparable == right_successful_sample_count:
            if right_total_characters is None:
                return False
            minimum_denominator = max(minimum_denominator, right_total_characters)
        if left_total_characters is None or right_total_characters is None:
            return False
        edit_distance = _integer_total_from_product(
            1.0 - normalized_similarity,
            normalized_denominator,
        )
        exact_matches = agreement["exact_match_count"]
        one_empty = agreement["one_empty_disagreement_count"]
        mean_length = agreement["mean_length_agreement"]
        if (
            not minimum_denominator
            <= normalized_denominator
            <= left_total_characters + right_total_characters
            or normalized_denominator < one_empty
            or edit_distance is None
            or edit_distance > normalized_denominator
            or edit_distance < comparable - exact_matches
            or (
                normalized_denominator == 0
                and (
                    not agreement["all_comparable_exact_matches"]
                    or normalized_similarity != 1.0
                    or mean_length != 1.0
                )
            )
            or one_empty > left_empty_successes + right_empty_successes
            or mean_length + 1e-12 < exact_matches / comparable
            or mean_length > 1.0 - one_empty / comparable + 1e-12
            or (
                one_empty > 0
                and (
                    mean_length >= 1.0
                    or edit_distance == 0
                )
            )
        ):
            return False
    return True


def _validate_sources(
    sources: list[object],
    *,
    metrics: dict,
    source_matches_registry: Callable[[dict], bool],
) -> bool:
    identities: set[tuple[str, int]] = set()
    ordered_identities: list[tuple[str, int]] = []
    candidate_metrics = metrics["candidates"]
    for expected_id, source in enumerate(sources, start=1):
        if not isinstance(source, dict) or set(source) != _SOURCE_FIELDS:
            return False
        candidate_id = source.get("candidate_id")
        config_index = source.get("config_index")
        status = source.get("status")
        availability = candidate_metrics[expected_id - 1]["availability"]
        succeeded = availability["successful_sample_count"]
        sample_count = availability["attempted_sample_count"]
        expected_status = (
            "succeeded"
            if succeeded == sample_count
            else "all_failed" if succeeded == 0 else "partial_failure"
        )
        if (
            type(source.get("candidate_evidence_id")) is not int
            or source["candidate_evidence_id"] != expected_id
            or type(candidate_id) is not str
            or _CANDIDATE_ID.fullmatch(candidate_id) is None
            or type(config_index) is not int
            or config_index < 0
            or (candidate_id, config_index) in identities
            or type(status) is not str
            or status not in _SOURCE_STATUSES
            or status != expected_status
            or type(source.get("config_fingerprint")) is not str
            or _HEX_16.fullmatch(source["config_fingerprint"]) is None
            or not source_matches_registry(source)
        ):
            return False
        identity = (candidate_id, config_index)
        identities.add(identity)
        ordered_identities.append(identity)
    return ordered_identities == sorted(ordered_identities)


def _optional_nonnegative_number(value: object, *, published: bool) -> bool:
    if not published:
        return value is None
    return (
        type(value) in {int, float}
        and math.isfinite(value)
        and value >= 0
        and not (value == 0 and math.copysign(1.0, float(value)) < 0)
    )


def _optional_fraction(value: object, *, published: bool) -> bool:
    return _optional_nonnegative_number(value, published=published) and (
        not published or value <= 1
    )


def _optional_nonnegative_int(
    value: object,
    *,
    published: bool,
    maximum: int | None = None,
) -> bool:
    if not published:
        return value is None
    return type(value) is int and value >= 0 and (maximum is None or value <= maximum)


def _valid_bucket(value: object, maximum: int) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _character_count_bucket(count: int) -> int:
    for bucket, upper_bound in enumerate(CHARACTER_COUNT_BUCKET_UPPER_BOUNDS):
        if count <= upper_bound:
            return bucket
    return len(CHARACTER_COUNT_BUCKET_UPPER_BOUNDS)


def _bucket_can_contain_at_least(bucket: int, minimum: int) -> bool:
    upper_bound = (
        CHARACTER_COUNT_BUCKET_UPPER_BOUNDS[bucket]
        if bucket < len(CHARACTER_COUNT_BUCKET_UPPER_BOUNDS)
        else None
    )
    return upper_bound is None or minimum <= upper_bound


def _integer_total_from_mean(value: object, denominator: int) -> int | None:
    if denominator <= 0:
        return None
    return _integer_total_from_product(value, denominator)


def _integer_total_from_rate(value: object, seconds: object) -> int | None:
    if type(value) not in {int, float} or type(seconds) not in {int, float}:
        return None
    total = float(value) * (float(seconds) / 60.0)
    if not math.isfinite(total) or total < 0:
        return None
    rounded = round(total)
    tolerance = max(1e-9, 4 * math.ulp(total))
    return rounded if abs(total - rounded) <= tolerance else None


def _integer_total_from_product(value: object, multiplier: object) -> int | None:
    if (
        type(value) not in {int, float}
        or type(multiplier) not in {int, float}
        or multiplier < 0
    ):
        return None
    total = float(value) * float(multiplier)
    if not math.isfinite(total) or total < 0:
        return None
    rounded = round(total)
    tolerance = max(1e-9, 4 * math.ulp(total))
    return rounded if abs(total - rounded) <= tolerance else None


def _fraction_bucket(value: float) -> int:
    if value == 1.0:
        return len(FRACTION_BUCKET_LOWER_BOUNDS) - 1
    for bucket in range(len(FRACTION_BUCKET_LOWER_BOUNDS) - 1):
        if value < FRACTION_BUCKET_LOWER_BOUNDS[bucket + 1]:
            return bucket
    return len(FRACTION_BUCKET_LOWER_BOUNDS) - 2


def _json_values_equal(left: object, right: object) -> bool:
    try:
        return json.dumps(
            left,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ) == json.dumps(
            right,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return False
