"""Define the public anchor and private commitment for blind OCR v10."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import uuid
from datetime import datetime, timezone
from functools import lru_cache


PROTOCOL = "private-ocr-blind-v10"
PRECOMMIT_PROTOCOL = "private-ocr-blind-packet-precommit-v1"
COMMITMENT_SCHEME = "private-ocr-blind-packet-hmac-sha256-v1"
PREPARATION_EVENT_FIELDS = frozenset(
    {
        "event",
        "protocol",
        "candidate_id",
        "workload_class",
        "preparation_id",
        "mapping_protocol",
        "sample_count",
        "source_count",
        "selected_source_status_counts",
        "producer_fingerprint",
        "private_packet_commitment",
        "privacy",
        "public_event_sha256",
        "timestamp_utc",
    }
)
PREPARATION_PRIVACY = {
    "private_commitment_key_published": False,
    "private_paths_or_text_published": False,
    "private_hashes_or_attempt_ids_published": False,
}
SCORE_EVENT_FIELDS = frozenset(
    {
        "event",
        "protocol",
        "candidate_id",
        "workload_class",
        "judgment_file_count",
        "preparation_public_event_sha256",
        "source_candidates",
        "producer_fingerprint",
        "metrics",
        "interpretation",
        "public_event_sha256",
        "timestamp_utc",
    }
)
SCORE_SOURCE_FIELDS = frozenset(
    {
        "candidate_evidence_id",
        "candidate_id",
        "config_index",
        "config_fingerprint",
        "attempt_status",
        "selected_available_record_count",
        "selected_failed_record_count",
        "selected_unavailable_record_count",
    }
)
SCORE_INTERPRETATION = {
    "blind_judgment_is_not_ground_truth": True,
    "procedural_blinding_only": True,
    "public_preparation_event_verified": True,
    "prejudgment_git_chronology_machine_verified": False,
    "source_artifact_commitments_verified": True,
    "source_lifecycles_and_active_configs_verified": True,
    "source_authority_lock_honoring_writers_required": True,
    "judge_identity_uniqueness_verified": False,
    "semantic_independence_verified": False,
    "private_fingerprints_published": False,
    "private_run_identifiers_published": False,
    "raw_text_or_images_published": False,
    "private_image_hashes_published": False,
    "failed_or_unavailable_options_disclosed_to_judges": True,
    "failed_or_unavailable_record_counts_published": True,
    "winner_metrics_filter_to_fully_available_comparisons": False,
    "fully_available_comparison_counts_reported": True,
    "exact_judgment_payload_uniqueness_enforced": True,
    "public_claim_level_deduplication": True,
}
_PUBLIC_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FINGERPRINT = re.compile(r"^[0-9a-f]{16}$")
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}\+00:00$"
)


def utc_timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def packet_commitment_payload(mapping: dict, packet: dict) -> bytes:
    source_bindings = sorted(
        mapping["source_bindings"],
        key=lambda source: source["variant_id"],
    )
    payload = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "preparation_id": mapping["preparation_id"],
        "variant_ids": sorted(mapping["variant_ids"]),
        "samples": sorted(mapping["samples"], key=lambda sample: sample["sample_id"]),
        "source_bindings": source_bindings,
        "preparation_producer_sha256": {
            key: mapping["preparation_producer_sha256"][key]
            for key in sorted(mapping["preparation_producer_sha256"])
        },
        "packet_fingerprint": mapping["packet_fingerprint"],
        "packet": packet,
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def public_event_sha256(event: dict) -> str:
    body = {
        key: value
        for key, value in event.items()
        if key not in {"public_event_sha256", "timestamp_utc"}
    }
    return hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def validate_preparation_event(event: object) -> dict:
    if not isinstance(event, dict):
        raise ValueError("blind OCR preparation public event is invalid")
    counts = event.get("selected_source_status_counts")
    commitment = event.get("private_packet_commitment")
    privacy = event.get("privacy")
    timestamp_value = event.get("timestamp_utc")
    try:
        preparation_id = str(uuid.UUID(event.get("preparation_id")))
        timestamp = datetime.fromisoformat(timestamp_value)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("blind OCR preparation public event is invalid") from error
    if (
        set(event) != PREPARATION_EVENT_FIELDS
        or event.get("event") != "blind_ocr_packet_prepared"
        or event.get("protocol") != PRECOMMIT_PROTOCOL
        or event.get("candidate_id") != "private_course_blind_ocr_comparison"
        or event.get("workload_class") != "private_course"
        or preparation_id != event.get("preparation_id")
        or event.get("mapping_protocol") != PROTOCOL
        or type(event.get("sample_count")) is not int
        or not 1 <= event["sample_count"] <= 100
        or type(event.get("source_count")) is not int
        or event["source_count"] != 2
        or not isinstance(counts, dict)
        or set(counts) != {"available", "failed", "unavailable"}
        or any(
            type(counts.get(status)) is not int or counts[status] < 0
            for status in counts
        )
        or sum(counts.values()) != event["sample_count"] * event["source_count"]
        or type(event.get("producer_fingerprint")) is not str
        or re.fullmatch(r"[0-9a-f]{16}", event["producer_fingerprint"]) is None
        or not isinstance(commitment, dict)
        or set(commitment) != {"scheme", "hmac_sha256"}
        or commitment.get("scheme") != COMMITMENT_SCHEME
        or type(commitment.get("hmac_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", commitment["hmac_sha256"]) is None
        or not isinstance(privacy, dict)
        or set(privacy) != set(PREPARATION_PRIVACY)
        or any(type(value) is not bool for value in privacy.values())
        or privacy != PREPARATION_PRIVACY
        or type(timestamp_value) is not str
        or _UTC_TIMESTAMP.fullmatch(timestamp_value) is None
        or timestamp.tzinfo is None
        or timestamp.utcoffset() is None
        or timestamp.utcoffset().total_seconds() != 0
        or timestamp.isoformat(timespec="microseconds") != timestamp_value
        or type(event.get("public_event_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", event["public_event_sha256"]) is None
        or not hmac.compare_digest(
            event["public_event_sha256"],
            public_event_sha256(event),
        )
    ):
        raise ValueError("blind OCR preparation public event is invalid")
    return event


def validate_score_event(event: object) -> dict:
    """Validate the complete privacy-safe v10 score schema and arithmetic."""

    if not isinstance(event, dict) or set(event) != SCORE_EVENT_FIELDS:
        raise ValueError("verified blind OCR public score event is invalid")
    sources = event.get("source_candidates")
    metrics = event.get("metrics")
    timestamp_value = event.get("timestamp_utc")
    try:
        timestamp = datetime.fromisoformat(timestamp_value)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("verified blind OCR public score event is invalid") from error
    if (
        event.get("event") != "blind_ocr_quality_scored"
        or event.get("protocol") != PROTOCOL
        or event.get("candidate_id") != "private_course_blind_ocr_comparison"
        or event.get("workload_class") != "private_course"
        or type(event.get("judgment_file_count")) is not int
        or not 2 <= event["judgment_file_count"] <= 8
        or type(event.get("preparation_public_event_sha256")) is not str
        or _SHA256.fullmatch(event["preparation_public_event_sha256"]) is None
        or not _valid_score_sources(sources)
        or type(event.get("producer_fingerprint")) is not str
        or _FINGERPRINT.fullmatch(event["producer_fingerprint"]) is None
        or not isinstance(event.get("interpretation"), dict)
        or set(event["interpretation"]) != set(SCORE_INTERPRETATION)
        or any(type(value) is not bool for value in event["interpretation"].values())
        or event["interpretation"] != SCORE_INTERPRETATION
        or not _valid_score_metrics(
            metrics,
            judgment_file_count=event["judgment_file_count"],
            sources=sources,
        )
        or type(timestamp_value) is not str
        or _UTC_TIMESTAMP.fullmatch(timestamp_value) is None
        or timestamp.tzinfo is None
        or timestamp.utcoffset() is None
        or timestamp.utcoffset().total_seconds() != 0
        or timestamp.isoformat(timespec="microseconds") != timestamp_value
        or type(event.get("public_event_sha256")) is not str
        or _SHA256.fullmatch(event["public_event_sha256"]) is None
        or not hmac.compare_digest(
            event["public_event_sha256"],
            public_event_sha256(event),
        )
    ):
        raise ValueError("verified blind OCR public score event is invalid")
    return event


def _valid_score_sources(sources: object) -> bool:
    if not isinstance(sources, list) or len(sources) != 2:
        return False
    identities = set()
    for expected_id, source in enumerate(sources, start=1):
        if not isinstance(source, dict) or set(source) != SCORE_SOURCE_FIELDS:
            return False
        identity = (source.get("candidate_id"), source.get("config_index"))
        count_fields = (
            "selected_available_record_count",
            "selected_failed_record_count",
            "selected_unavailable_record_count",
        )
        if (
            type(source.get("candidate_evidence_id")) is not int
            or source["candidate_evidence_id"] != expected_id
            or type(source.get("candidate_id")) is not str
            or _PUBLIC_ID.fullmatch(source["candidate_id"]) is None
            or type(source.get("config_index")) is not int
            or source["config_index"] < 0
            or identity in identities
            or type(source.get("config_fingerprint")) is not str
            or _FINGERPRINT.fullmatch(source["config_fingerprint"]) is None
            or type(source.get("attempt_status")) is not str
            or source["attempt_status"] not in _SOURCE_STATUSES
            or any(
                type(source.get(field)) is not int or source[field] < 0
                for field in count_fields
            )
        ):
            return False
        identities.add(identity)
    return True


def _valid_score_metrics(
    metrics: object,
    *,
    judgment_file_count: int,
    sources: list[dict],
) -> bool:
    metric_fields = {
        "sample_count",
        "judgment_file_count",
        "vote_count",
        "tie_vote_count",
        "consensus_tie_sample_count",
        "no_strict_majority_sample_count",
        "strict_majority_sample_fraction",
        "unanimous_sample_fraction",
        "pairwise_winner_agreement_fraction",
        "comparison_sample_denominators",
        "source_record_availability",
        "candidates",
    }
    if not isinstance(metrics, dict) or set(metrics) != metric_fields:
        return False
    denominators = metrics.get("comparison_sample_denominators")
    availability = metrics.get("source_record_availability")
    candidates = metrics.get("candidates")
    denominator_fields = {
        "total_selected_sample_count",
        "individual_winner_vote_denominator",
        "consensus_winner_sample_denominator",
        "strict_majority_sample_denominator",
        "unanimous_sample_denominator",
        "pairwise_winner_agreement_denominator",
        "fully_available_comparison_count",
        "not_fully_available_comparison_count",
    }
    availability_fields = {
        "candidate_sample_count",
        "available_record_count",
        "failed_record_count",
        "unavailable_record_count",
    }
    if (
        not isinstance(denominators, dict)
        or set(denominators) != denominator_fields
        or not isinstance(availability, dict)
        or set(availability) != availability_fields
        or not isinstance(candidates, list)
        or len(candidates) != 2
    ):
        return False
    integer_values = [
        metrics.get("sample_count"),
        metrics.get("judgment_file_count"),
        metrics.get("vote_count"),
        metrics.get("tie_vote_count"),
        metrics.get("consensus_tie_sample_count"),
        metrics.get("no_strict_majority_sample_count"),
        *denominators.values(),
        *availability.values(),
    ]
    if any(type(value) is not int or value < 0 for value in integer_values):
        return False
    sample_count = metrics["sample_count"]
    vote_count = sample_count * judgment_file_count
    pairwise_count = sample_count * judgment_file_count * (judgment_file_count - 1) // 2
    source_count_totals = [
        source["selected_available_record_count"]
        + source["selected_failed_record_count"]
        + source["selected_unavailable_record_count"]
        for source in sources
    ]
    source_available = sum(source["selected_available_record_count"] for source in sources)
    source_failed = sum(source["selected_failed_record_count"] for source in sources)
    source_unavailable = sum(source["selected_unavailable_record_count"] for source in sources)
    fully_available = denominators["fully_available_comparison_count"]
    available_by_source = [
        source["selected_available_record_count"] for source in sources
    ]
    fraction_denominators = {
        "strict_majority_sample_fraction": sample_count,
        "unanimous_sample_fraction": sample_count,
        "pairwise_winner_agreement_fraction": pairwise_count,
    }
    if any(
        not _valid_float_on_integer_grid(metrics.get(field), 1, denominator)
        for field, denominator in fraction_denominators.items()
    ):
        return False
    if (
        not 1 <= sample_count <= 100
        or metrics["judgment_file_count"] != judgment_file_count
        or metrics["vote_count"] != vote_count
        or metrics["tie_vote_count"] > vote_count
        or metrics["consensus_tie_sample_count"] > sample_count
        or metrics["no_strict_majority_sample_count"] > sample_count
        or metrics["strict_majority_sample_fraction"]
        != (sample_count - metrics["no_strict_majority_sample_count"])
        / sample_count
        or metrics["unanimous_sample_fraction"]
        > metrics["pairwise_winner_agreement_fraction"]
        or metrics["unanimous_sample_fraction"]
        > metrics["strict_majority_sample_fraction"]
        or metrics["pairwise_winner_agreement_fraction"]
        > metrics["unanimous_sample_fraction"]
        + (1.0 - metrics["unanimous_sample_fraction"])
        * (judgment_file_count - 2)
        / judgment_file_count
        + 1e-12
        or denominators["total_selected_sample_count"] != sample_count
        or denominators["individual_winner_vote_denominator"] != vote_count
        or denominators["consensus_winner_sample_denominator"] != sample_count
        or denominators["strict_majority_sample_denominator"] != sample_count
        or denominators["unanimous_sample_denominator"] != sample_count
        or denominators["pairwise_winner_agreement_denominator"] != pairwise_count
        or denominators["fully_available_comparison_count"]
        + denominators["not_fully_available_comparison_count"]
        != sample_count
        or fully_available > min(available_by_source)
        or fully_available < max(0, sum(available_by_source) - sample_count)
        or source_count_totals != [sample_count, sample_count]
        or availability["candidate_sample_count"] != sample_count * 2
        or availability["available_record_count"] != source_available
        or availability["failed_record_count"] != source_failed
        or availability["unavailable_record_count"] != source_unavailable
    ):
        return False
    for source in sources:
        if (
            source["attempt_status"] == "succeeded"
            and (
                source["selected_failed_record_count"] != 0
                or source["selected_unavailable_record_count"] != 0
            )
        ) or (
            source["attempt_status"] == "all_failed"
            and source["selected_available_record_count"] != 0
        ):
            return False
    candidate_fields = {
        "candidate_evidence_id",
        "win_votes",
        "consensus_wins",
        "mean_error_severity",
        "usable_vote_fraction",
        "mean_error_severity_vote_denominator",
        "usable_vote_denominator",
    }
    for expected_id, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict) or set(candidate) != candidate_fields:
            return False
        if (
            type(candidate.get("candidate_evidence_id")) is not int
            or candidate["candidate_evidence_id"] != expected_id
            or any(
                type(candidate.get(field)) is not int or candidate[field] < 0
                for field in (
                    "win_votes",
                    "consensus_wins",
                    "mean_error_severity_vote_denominator",
                    "usable_vote_denominator",
                )
            )
            or candidate["mean_error_severity_vote_denominator"] != vote_count
            or candidate["usable_vote_denominator"] != vote_count
            or candidate["win_votes"] > vote_count
            or candidate["consensus_wins"] > sample_count
            or candidate["consensus_wins"] > candidate["win_votes"]
            or candidate["win_votes"]
            < candidate["consensus_wins"] * (judgment_file_count // 2 + 1)
            or candidate["win_votes"]
            > candidate["consensus_wins"] * judgment_file_count
            + (sample_count - candidate["consensus_wins"])
            * (judgment_file_count // 2)
        ):
            return False
        for field, maximum in (
            ("mean_error_severity", 3),
            ("usable_vote_fraction", 1),
        ):
            value = candidate.get(field)
            if not _valid_float_on_integer_grid(value, maximum, vote_count):
                return False
    unanimous_count = round(metrics["unanimous_sample_fraction"] * sample_count)
    strict_majority_count = sample_count - metrics["no_strict_majority_sample_count"]
    pairwise_agreement_count = round(
        metrics["pairwise_winner_agreement_fraction"] * pairwise_count
    )
    pairs_per_sample = judgment_file_count * (judgment_file_count - 1) // 2
    strict_votes = judgment_file_count // 2 + 1
    remaining_strict_votes = judgment_file_count - strict_votes
    strict_nonunanimous_minimum = (
        strict_votes * (strict_votes - 1) // 2
        + (remaining_strict_votes // 2)
        * (remaining_strict_votes // 2 - 1)
        // 2
        + ((remaining_strict_votes + 1) // 2)
        * ((remaining_strict_votes + 1) // 2 - 1)
        // 2
    )
    balanced_votes, extra_balanced_votes = divmod(judgment_file_count, 3)
    no_majority_minimum = (
        extra_balanced_votes * balanced_votes * (balanced_votes + 1) // 2
        + (3 - extra_balanced_votes)
        * balanced_votes
        * (balanced_votes - 1)
        // 2
    )
    minimum_pairwise_agreements = (
        unanimous_count * pairs_per_sample
        + (strict_majority_count - unanimous_count)
        * strict_nonunanimous_minimum
        + metrics["no_strict_majority_sample_count"] * no_majority_minimum
    )
    maximum_unanimous_from_votes = (
        sum(candidate["win_votes"] // judgment_file_count for candidate in candidates)
        + metrics["tie_vote_count"] // judgment_file_count
    )
    category_vote_counts = [
        *(candidate["win_votes"] for candidate in candidates),
        metrics["tie_vote_count"],
    ]
    minimum_unanimous_from_votes = sum(
        max(0, category_votes - sample_count * (judgment_file_count - 1))
        for category_votes in category_vote_counts
    )
    minimum_pairwise_from_category_totals = 0
    maximum_pairwise_from_category_totals = 0
    for category_votes in category_vote_counts:
        balanced_votes, extra_votes = divmod(category_votes, sample_count)
        minimum_pairwise_from_category_totals += (
            sample_count * balanced_votes * (balanced_votes - 1) // 2
            + extra_votes * balanced_votes
        )
        full_samples, remaining_votes = divmod(
            category_votes,
            judgment_file_count,
        )
        maximum_pairwise_from_category_totals += (
            full_samples * pairs_per_sample
            + remaining_votes * (remaining_votes - 1) // 2
        )
    no_majority_category_limit = judgment_file_count // 2
    maximum_no_majority_pairs = max(
        first_votes * (first_votes - 1) // 2
        + second_votes * (second_votes - 1) // 2
        + third_votes * (third_votes - 1) // 2
        for first_votes in range(no_majority_category_limit + 1)
        for second_votes in range(no_majority_category_limit + 1)
        for third_votes in [judgment_file_count - first_votes - second_votes]
        if 0 <= third_votes <= no_majority_category_limit
    )
    maximum_pairwise_by_majority_class = (
        unanimous_count * pairs_per_sample
        + (strict_majority_count - unanimous_count)
        * (judgment_file_count - 1)
        * (judgment_file_count - 2)
        // 2
        + metrics["no_strict_majority_sample_count"]
        * maximum_no_majority_pairs
    )
    necessary_arithmetic_is_consistent = (
        sum(candidate["win_votes"] for candidate in candidates)
        + metrics["tie_vote_count"]
        == vote_count
        and sum(candidate["consensus_wins"] for candidate in candidates)
        + metrics["consensus_tie_sample_count"]
        + metrics["no_strict_majority_sample_count"]
        == sample_count
        and metrics["tie_vote_count"]
        >= metrics["consensus_tie_sample_count"]
        * (judgment_file_count // 2 + 1)
        and metrics["tie_vote_count"]
        <= metrics["consensus_tie_sample_count"] * judgment_file_count
        + (sample_count - metrics["consensus_tie_sample_count"])
        * (judgment_file_count // 2)
        and unanimous_count >= minimum_unanimous_from_votes
        and pairwise_agreement_count
        >= max(minimum_pairwise_agreements, minimum_pairwise_from_category_totals)
        and pairwise_agreement_count
        <= min(
            maximum_pairwise_by_majority_class,
            maximum_pairwise_from_category_totals,
        )
        and unanimous_count <= maximum_unanimous_from_votes
    )
    if not necessary_arithmetic_is_consistent:
        return False
    return _exact_winner_vote_summary_is_feasible(
        sample_count=sample_count,
        judgment_count=judgment_file_count,
        category_vote_counts=tuple(
            [
                *(candidate["win_votes"] for candidate in candidates),
                metrics["tie_vote_count"],
            ]
        ),
        strict_majority_counts=tuple(
            [
                *(candidate["consensus_wins"] for candidate in candidates),
                metrics["consensus_tie_sample_count"],
            ]
        ),
        no_strict_majority_count=metrics["no_strict_majority_sample_count"],
        unanimous_count=unanimous_count,
        pairwise_agreement_count=pairwise_agreement_count,
    )


_BIT_REVERSE_TABLE = bytes(
    int(f"{value:08b}"[::-1], 2) for value in range(256)
)


def _exact_winner_vote_summary_is_feasible(
    *,
    sample_count: int,
    judgment_count: int,
    category_vote_counts: tuple[int, int, int],
    strict_majority_counts: tuple[int, int, int],
    no_strict_majority_count: int,
    unanimous_count: int,
    pairwise_agreement_count: int,
) -> bool:
    """Decide exact finite feasibility for the published winner-vote histogram."""

    return _exact_winner_vote_summary_is_feasible_cached(
        sample_count,
        judgment_count,
        category_vote_counts,
        strict_majority_counts,
        no_strict_majority_count,
        unanimous_count,
        pairwise_agreement_count,
    )


@lru_cache(maxsize=512)
def _exact_winner_vote_summary_is_feasible_cached(
    sample_count: int,
    judgment_count: int,
    category_vote_counts: tuple[int, int, int],
    strict_majority_counts: tuple[int, int, int],
    no_strict_majority_count: int,
    unanimous_count: int,
    pairwise_agreement_count: int,
) -> bool:
    class_counts = (*strict_majority_counts, no_strict_majority_count)
    strict_sample_count = sum(strict_majority_counts)
    if (
        sum(category_vote_counts) != sample_count * judgment_count
        or sum(class_counts) != sample_count
        or not 0 <= unanimous_count <= strict_sample_count
    ):
        return False

    types_by_class: dict[int, list[tuple[int, int, int, int, int]]] = {
        class_id: [] for class_id in range(4)
    }
    majority_threshold = judgment_count // 2 + 1
    for first_votes in range(judgment_count + 1):
        for second_votes in range(judgment_count - first_votes + 1):
            third_votes = judgment_count - first_votes - second_votes
            votes = (first_votes, second_votes, third_votes)
            majority_class = next(
                (
                    index
                    for index, vote_total in enumerate(votes)
                    if vote_total >= majority_threshold
                ),
                3,
            )
            pair_count = sum(value * (value - 1) // 2 for value in votes)
            unanimous = int(max(votes) == judgment_count)
            types_by_class[majority_class].append(
                (*votes, unanimous, pair_count)
            )

    pair_minimums = {
        class_id: min(item[4] for item in item_types)
        for class_id, item_types in types_by_class.items()
    }
    pair_maximums = {
        class_id: max(item[4] for item in item_types)
        for class_id, item_types in types_by_class.items()
    }
    pair_minimum = sum(
        class_counts[class_id] * pair_minimums[class_id]
        for class_id in range(4)
    )
    pair_maximum = sum(
        class_counts[class_id] * pair_maximums[class_id]
        for class_id in range(4)
    )
    if not pair_minimum <= pairwise_agreement_count <= pair_maximum:
        return False

    reduced_unanimity_target = min(
        unanimous_count,
        strict_sample_count - unanimous_count,
    )
    track_unanimous = unanimous_count <= strict_sample_count - unanimous_count
    pair_excess = pairwise_agreement_count - pair_minimum
    pair_deficit = pair_maximum - pairwise_agreement_count
    reduced_pair_target = min(pair_excess, pair_deficit)
    track_pair_excess = pair_excess <= pair_deficit

    transformed_types: dict[int, tuple[tuple[int, int, int, int, int], ...]] = {}
    maximum_pair_increment = 0
    for class_id, item_types in types_by_class.items():
        transformed = []
        for first_votes, second_votes, third_votes, unanimous, pairs in item_types:
            if track_unanimous:
                reduced_unanimity = unanimous
            else:
                reduced_unanimity = 1 - unanimous if class_id != 3 else 0
            reduced_pairs = (
                pairs - pair_minimums[class_id]
                if track_pair_excess
                else pair_maximums[class_id] - pairs
            )
            maximum_pair_increment = max(maximum_pair_increment, reduced_pairs)
            transformed.append(
                (
                    first_votes,
                    second_votes,
                    third_votes,
                    reduced_unanimity,
                    reduced_pairs,
                )
            )
        transformed_types[class_id] = tuple(transformed)

    left_counts = [count // 2 for count in class_counts]
    left_remainder = sample_count // 2 - sum(left_counts)
    for class_id in sorted(
        range(4),
        key=lambda value: (
            -(class_counts[value] % 2),
            len(transformed_types[value]),
            value,
        ),
    ):
        if left_remainder == 0:
            break
        if class_counts[class_id] % 2:
            left_counts[class_id] += 1
            left_remainder -= 1
    right_counts = [
        class_counts[class_id] - left_counts[class_id]
        for class_id in range(4)
    ]
    class_order = sorted(
        range(4),
        key=lambda value: (len(transformed_types[value]), value),
    )
    left_slots = tuple(
        class_id
        for class_id in class_order
        for _ in range(left_counts[class_id])
    )
    right_slots = tuple(
        class_id
        for class_id in class_order
        for _ in range(right_counts[class_id])
    )
    stride = reduced_pair_target + maximum_pair_increment + 1
    left_profile = _build_exact_winner_vote_half_profile(
        slots=left_slots,
        other_slots=right_slots,
        transformed_types=transformed_types,
        vote_targets=category_vote_counts,
        reduced_unanimity_target=reduced_unanimity_target,
        reduced_pair_target=reduced_pair_target,
        judgment_count=judgment_count,
        stride=stride,
    )
    if not left_profile:
        return False
    right_profile = _build_exact_winner_vote_half_profile(
        slots=right_slots,
        other_slots=left_slots,
        transformed_types=transformed_types,
        vote_targets=category_vote_counts,
        reduced_unanimity_target=reduced_unanimity_target,
        reduced_pair_target=reduced_pair_target,
        judgment_count=judgment_count,
        stride=stride,
    )
    if not right_profile:
        return False

    if len(left_profile) > len(right_profile):
        left_profile, right_profile = right_profile, left_profile
    target_packed_index = (
        category_vote_counts[1] * stride + reduced_pair_target
    )
    reversal_width = target_packed_index + 1
    for (first_votes, reduced_unanimity), packed_values in left_profile.items():
        complement = right_profile.get(
            (
                category_vote_counts[0] - first_votes,
                reduced_unanimity_target - reduced_unanimity,
            )
        )
        if complement and packed_values & _reverse_low_bits(
            complement,
            reversal_width,
        ):
            return True
    return False


def _build_exact_winner_vote_half_profile(
    *,
    slots: tuple[int, ...],
    other_slots: tuple[int, ...],
    transformed_types: dict[int, tuple[tuple[int, int, int, int, int], ...]],
    vote_targets: tuple[int, int, int],
    reduced_unanimity_target: int,
    reduced_pair_target: int,
    judgment_count: int,
    stride: int,
) -> dict[tuple[int, int], int]:
    class_minimums = {
        class_id: tuple(
            min(item[index] for item in item_types) for index in range(5)
        )
        for class_id, item_types in transformed_types.items()
    }
    class_maximums = {
        class_id: tuple(
            max(item[index] for item in item_types) for index in range(5)
        )
        for class_id, item_types in transformed_types.items()
    }
    other_minimums = tuple(
        sum(class_minimums[class_id][index] for class_id in other_slots)
        for index in range(5)
    )
    other_maximums = tuple(
        sum(class_maximums[class_id][index] for class_id in other_slots)
        for index in range(5)
    )
    remaining_minimums = [other_minimums] * (len(slots) + 1)
    remaining_maximums = [other_maximums] * (len(slots) + 1)
    for index in range(len(slots) - 1, -1, -1):
        class_id = slots[index]
        remaining_minimums[index] = tuple(
            remaining_minimums[index + 1][dimension]
            + class_minimums[class_id][dimension]
            for dimension in range(5)
        )
        remaining_maximums[index] = tuple(
            remaining_maximums[index + 1][dimension]
            + class_maximums[class_id][dimension]
            for dimension in range(5)
        )

    rectangle_masks: dict[tuple[int, int, int, int], int] = {}

    def packed_rectangle_mask(
        second_vote_low: int,
        second_vote_high: int,
        pair_low: int,
        pair_high: int,
    ) -> int:
        bounds = (second_vote_low, second_vote_high, pair_low, pair_high)
        cached = rectangle_masks.get(bounds)
        if cached is not None:
            return cached
        if second_vote_low > second_vote_high or pair_low > pair_high:
            return 0
        row = ((1 << (pair_high - pair_low + 1)) - 1) << pair_low
        mask = 0
        for second_votes in range(second_vote_low, second_vote_high + 1):
            mask |= row << (second_votes * stride)
        rectangle_masks[bounds] = mask
        return mask

    profile: dict[tuple[int, int], int] = {(0, 0): 1}
    for slot_index, class_id in enumerate(slots):
        raw_profile: dict[tuple[int, int], int] = {}
        remaining_min = remaining_minimums[slot_index + 1]
        remaining_max = remaining_maximums[slot_index + 1]
        for (first_votes, reduced_unanimity), packed_values in profile.items():
            for item in transformed_types[class_id]:
                next_first_votes = first_votes + item[0]
                next_reduced_unanimity = reduced_unanimity + item[3]
                if (
                    not vote_targets[0] - remaining_max[0]
                    <= next_first_votes
                    <= vote_targets[0] - remaining_min[0]
                    or not reduced_unanimity_target - remaining_max[3]
                    <= next_reduced_unanimity
                    <= reduced_unanimity_target - remaining_min[3]
                ):
                    continue
                key = (next_first_votes, next_reduced_unanimity)
                shifted = packed_values << (item[1] * stride + item[4])
                raw_profile[key] = raw_profile.get(key, 0) | shifted

        processed_slots = slot_index + 1
        profile = {}
        pair_low = max(0, reduced_pair_target - remaining_max[4])
        pair_high = min(
            reduced_pair_target,
            reduced_pair_target - remaining_min[4],
        )
        for (first_votes, reduced_unanimity), packed_values in raw_profile.items():
            second_vote_low = max(
                0,
                vote_targets[1] - remaining_max[1],
                processed_slots * judgment_count
                - first_votes
                - (vote_targets[2] - remaining_min[2]),
            )
            second_vote_high = min(
                vote_targets[1],
                vote_targets[1] - remaining_min[1],
                processed_slots * judgment_count
                - first_votes
                - (vote_targets[2] - remaining_max[2]),
            )
            mask = packed_rectangle_mask(
                second_vote_low,
                second_vote_high,
                pair_low,
                pair_high,
            )
            masked = packed_values & mask
            if masked:
                profile[(first_votes, reduced_unanimity)] = masked
        if not profile:
            break
    return profile


def _reverse_low_bits(value: int, width: int) -> int:
    byte_count = (width + 7) // 8
    reversed_bytes = value.to_bytes(byte_count, "little").translate(
        _BIT_REVERSE_TABLE
    )
    return int.from_bytes(reversed_bytes, "big") >> (byte_count * 8 - width)


def _valid_float_on_integer_grid(
    value: object,
    maximum: int,
    denominator: int,
) -> bool:
    if (
        type(value) is not float
        or not math.isfinite(value)
        or not 0 <= value <= maximum
        or denominator <= 0
        or (value == 0.0 and math.copysign(1.0, value) < 0)
    ):
        return False
    numerator = round(value * denominator)
    return value == numerator / denominator
