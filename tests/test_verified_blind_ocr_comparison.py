import copy
import hashlib
import itertools
import json
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from local_inference_bench.load_sustained_workload import load_sustained_workload
from local_inference_bench.event_journal import (
    _locked_journal,
    append_event,
    append_event_once,
)
from local_inference_bench.load_verified_private_ocr_source import (
    load_verified_private_ocr_source,
)
from local_inference_bench.private_records_commitment import (
    PRIVATE_RECORDS_COMMITMENT_SCHEME,
    create_private_records_commitment,
)
from local_inference_bench.verified_blind_ocr_protocol import (
    _exact_winner_vote_summary_is_feasible,
    public_event_sha256,
)
from scripts.aggregate_verified_blind_ocr_judgments import (
    _aggregate_verified_judgments,
    _append_public_event_once,
)
from scripts.prepare_verified_blind_ocr_comparison import (
    _append_preparation_event_once,
    _create_private_packet_commitment,
    _prepare_verified_blind_packet,
    _preparation_event,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


def _records(path: Path, *, prefix: str) -> list[dict]:
    values = [
        {
            "sample_id": f"frame_{index:03d}",
            "success": True,
            "lines": [{"text": f"{prefix} line {index}"}],
        }
        for index in (1, 2)
    ]
    _write_jsonl(path, values)
    return values


def _fixture(tmp_path: Path) -> dict:
    images = []
    for index in (1, 2):
        image = tmp_path / f"frame_{index:03d}.png"
        image.write_bytes(b"private-image-" + bytes([index]))
        images.append(image)
    workload_path = tmp_path / "workload.json"
    _write_json(
        workload_path,
        {
            "schema_version": 1,
            "task": "ocr",
            "workload_class": "private_course",
            "items": [
                {"id": f"frame_{index:03d}", "path": image.name}
                for index, image in enumerate(images, start=1)
            ],
        },
    )
    workload = load_sustained_workload(workload_path, expected_task="ocr")
    candidates = [
        ("candidate_one", "variant_one", {"workers": 2, "phases": ["quality"]}),
        ("candidate_two", "variant_two", {"workers": 3, "phases": ["quality"]}),
    ]
    registry_path = tmp_path / "registry.json"
    _write_json(
        registry_path,
        {
            "schema_version": 1,
            "protocol": "sustained-process-v1",
            "candidates": [
                {"id": candidate_id, "task": "ocr", "configs": [config]}
                for candidate_id, _, config in candidates
            ],
        },
    )
    record_paths = {}
    provenances = {}
    events = []
    for index, (candidate_id, variant_id, config) in enumerate(candidates, start=1):
        records_path = tmp_path / "sources" / variant_id / "private-records.jsonl"
        records = _records(records_path, prefix=variant_id)
        provenance = {
            "schema_version": 1,
            "protocol": "sustained-process-v1",
            "status": "succeeded",
            "attempt_id": str(uuid.UUID(f"00000000-0000-4000-8000-{index:012d}")),
            "attempt_key": f"{index:016x}",
            "candidate_id": candidate_id,
            "task": "ocr",
            "config": config,
            "config_index": 0,
            "phase": "quality",
            "target_wall_seconds": 60.0,
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
        _write_json(records_path.with_name("records-provenance.json"), provenance)
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
        events.extend(
            [
                {**common, "event": "sustained_attempt_started"},
                {
                    **common,
                    "event": "sustained_attempt_succeeded",
                    "private_artifact_commitment": commitment["public"],
                    "result": {
                        "candidate_id": candidate_id,
                        "task": "ocr",
                        "workload_class": "private_course",
                        "status": "complete",
                        "counts": {
                            "attempted": len(records),
                            "completed": len(records),
                            "failed": 0,
                        },
                    },
                },
            ]
        )
        record_paths[variant_id] = records_path
        provenances[variant_id] = provenance
    sustained_events_path = tmp_path / "sustained-events.jsonl"
    _write_jsonl(sustained_events_path, events)
    output_dir = tmp_path / "blind-output"
    packet_path, preparation_event = _prepare_verified_blind_packet(
        workload_path=workload_path,
        candidate_specs={
            variant_id: {
                "candidate_id": candidate_id,
                "config_index": 0,
                "path": record_paths[variant_id],
            }
            for candidate_id, variant_id, _ in candidates
        },
        sample_count=2,
        seed=285,
        output_dir=output_dir,
        sustained_events_path=sustained_events_path,
        registry_path=registry_path,
    )
    quality_events_path = tmp_path / "quality-events.jsonl"
    assert _append_preparation_event_once(quality_events_path, preparation_event)
    mapping_path = output_dir / "mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    judgment_paths = []
    for judge_index, winner in enumerate(("A", "tie"), start=1):
        judgment_path = tmp_path / f"judge-{judge_index}.json"
        _write_json(
            judgment_path,
            {
                "schema_version": 2,
                "protocol": "private-ocr-blind-v10",
                "packet_fingerprint": mapping["packet_fingerprint"],
                "samples": [
                    {
                        "sample_id": sample["sample_id"],
                        "winner": winner,
                        "a_severity": judge_index - 1,
                        "b_severity": judge_index,
                        "a_usable": True,
                        "b_usable": judge_index == 1,
                        "a_error_codes": [],
                        "b_error_codes": ["missing_text"] if judge_index == 2 else [],
                    }
                    for sample in packet["samples"]
                ],
            },
        )
        judgment_paths.append(judgment_path)
    return {
        "workload_path": workload_path,
        "images": images,
        "registry_path": registry_path,
        "record_paths": record_paths,
        "provenances": provenances,
        "sustained_events_path": sustained_events_path,
        "quality_events_path": quality_events_path,
        "output_dir": output_dir,
        "packet_path": packet_path,
        "mapping_path": mapping_path,
        "mapping": mapping,
        "packet": packet,
        "judgment_paths": judgment_paths,
        "preparation_event": preparation_event,
    }


def _aggregate(fixture: dict) -> dict:
    return _aggregate_verified_judgments(
        fixture["mapping_path"],
        fixture["judgment_paths"],
        quality_events_path=fixture["quality_events_path"],
        sustained_events_path=fixture["sustained_events_path"],
        registry_path=fixture["registry_path"],
    )


def _recommit_source(records_path: Path) -> dict:
    provenance_path = records_path.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.pop("records_sha256")
    provenance.pop("private_records_commitment")
    commitment = create_private_records_commitment(records_path, provenance)
    provenance["records_sha256"] = commitment["records_sha256"]
    provenance["private_records_commitment"] = commitment["private"]
    _write_json(provenance_path, provenance)
    return provenance


def test_verified_blind_v10_end_to_end_is_privacy_safe(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    event = _aggregate(fixture)

    assert event["protocol"] == "private-ocr-blind-v10"
    assert event["interpretation"]["public_preparation_event_verified"] is True
    assert event["interpretation"]["prejudgment_git_chronology_machine_verified"] is False
    assert event["metrics"]["sample_count"] == 2
    assert event["metrics"]["comparison_sample_denominators"] == {
        "total_selected_sample_count": 2,
        "individual_winner_vote_denominator": 4,
        "consensus_winner_sample_denominator": 2,
        "strict_majority_sample_denominator": 2,
        "unanimous_sample_denominator": 2,
        "pairwise_winner_agreement_denominator": 2,
        "fully_available_comparison_count": 2,
        "not_fully_available_comparison_count": 0,
    }
    serialized = json.dumps(event, sort_keys=True)
    for private_value in [
        str(fixture["output_dir"]),
        "variant_one",
        "variant_two",
        "line 1",
        fixture["provenances"]["variant_one"]["attempt_id"],
        fixture["mapping"]["private_packet_commitment"]["key_hex"],
        fixture["mapping"]["private_packet_commitment"]["hmac_sha256"],
    ]:
        assert private_value not in serialized
    output_journal = fixture["quality_events_path"]
    assert _append_public_event_once(output_journal, event) is True
    assert _append_public_event_once(output_journal, event) is False


def test_same_preparation_anchor_cannot_publish_contradictory_scores(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first = _aggregate(fixture)
    output_journal = fixture["quality_events_path"]
    assert _append_public_event_once(output_journal, first) is True
    for judgment_path in fixture["judgment_paths"]:
        judgment = json.loads(judgment_path.read_text(encoding="utf-8"))
        for sample in judgment["samples"]:
            sample["winner"] = "B"
        _write_json(judgment_path, judgment)

    contradictory = _aggregate(fixture)
    before = output_journal.read_bytes()
    with pytest.raises(ValueError, match="append-once identity conflicts"):
        _append_public_event_once(output_journal, contradictory)

    assert output_journal.read_bytes() == before


def test_score_append_rejects_plain_dictionary(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    event = _aggregate(fixture)
    before = fixture["quality_events_path"].read_bytes()

    with pytest.raises(ValueError, match="not authorized by aggregation"):
        _append_public_event_once(fixture["quality_events_path"], dict(event))

    assert fixture["quality_events_path"].read_bytes() == before


def test_score_append_rejects_authorized_event_mutated_after_aggregation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    event = _aggregate(fixture)
    event["producer_fingerprint"] = "0" * 16
    event["public_event_sha256"] = public_event_sha256(event)
    before = fixture["quality_events_path"].read_bytes()

    with pytest.raises(ValueError, match="not authorized by aggregation"):
        _append_public_event_once(fixture["quality_events_path"], event)

    assert fixture["quality_events_path"].read_bytes() == before


def test_score_append_rejects_source_authority_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    event = _aggregate(fixture)
    append_event(
        fixture["sustained_events_path"],
        {"event": "authority_changed_by_test"},
    )
    before = fixture["quality_events_path"].read_bytes()

    with pytest.raises(ValueError, match="authority changed during validation"):
        _append_public_event_once(fixture["quality_events_path"], event)

    assert fixture["quality_events_path"].read_bytes() == before


def test_score_append_holds_source_lock_through_quality_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    event = _aggregate(fixture)
    import scripts.aggregate_verified_blind_ocr_judgments as aggregator

    real_quality_append = aggregator.append_event_once
    executor = ThreadPoolExecutor(max_workers=1)
    writer_started = threading.Event()
    writer_future = None
    writer_was_blocked = []

    def write_sustained_event() -> None:
        writer_started.set()
        append_event(
            fixture["sustained_events_path"],
            {"event": "authority_changed_after_quality_commit"},
        )

    def observed_quality_append(path, public_event, **kwargs):
        nonlocal writer_future
        writer_future = executor.submit(write_sustained_event)
        assert writer_started.wait(timeout=2)
        time.sleep(0.1)
        writer_was_blocked.append(not writer_future.done())
        return real_quality_append(path, public_event, **kwargs)

    monkeypatch.setattr(aggregator, "append_event_once", observed_quality_append)
    try:
        assert _append_public_event_once(fixture["quality_events_path"], event) is True
    finally:
        if writer_future is not None:
            writer_future.result(timeout=5)
        executor.shutdown(wait=True)

    assert writer_was_blocked == [True]


def test_score_append_holds_registry_lock_through_quality_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    event = _aggregate(fixture)
    import scripts.aggregate_verified_blind_ocr_judgments as aggregator

    real_quality_append = aggregator.append_event_once
    executor = ThreadPoolExecutor(max_workers=1)
    writer_started = threading.Event()
    writer_future = None
    writer_was_blocked = []

    def write_registry() -> None:
        writer_started.set()
        with _locked_journal(
            fixture["registry_path"],
            exclusive=True,
            create=False,
        ) as handle:
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

    monkeypatch.setattr(aggregator, "append_event_once", observed_quality_append)
    try:
        assert _append_public_event_once(fixture["quality_events_path"], event) is True
    finally:
        if writer_future is not None:
            writer_future.result(timeout=5)
        executor.shutdown(wait=True)

    assert writer_was_blocked == [True]


def test_preparation_append_holds_registry_lock_through_quality_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    import scripts.prepare_verified_blind_ocr_comparison as preparer

    real_quality_append = preparer.append_event_once
    executor = ThreadPoolExecutor(max_workers=1)
    writer_started = threading.Event()
    writer_future = None
    writer_was_blocked = []

    def acquire_registry_writer() -> None:
        writer_started.set()
        with _locked_journal(
            fixture["registry_path"],
            exclusive=True,
            create=False,
        ):
            return

    def observed_quality_append(path, public_event, **kwargs):
        nonlocal writer_future
        writer_future = executor.submit(acquire_registry_writer)
        assert writer_started.wait(timeout=2)
        time.sleep(0.1)
        writer_was_blocked.append(not writer_future.done())
        return real_quality_append(path, public_event, **kwargs)

    monkeypatch.setattr(preparer, "append_event_once", observed_quality_append)
    try:
        assert (
            _append_preparation_event_once(
                fixture["quality_events_path"],
                fixture["preparation_event"],
            )
            is False
        )
    finally:
        if writer_future is not None:
            writer_future.result(timeout=5)
        executor.shutdown(wait=True)

    assert writer_was_blocked == [True]


def test_registry_change_between_source_loads_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    import scripts.aggregate_verified_blind_ocr_judgments as aggregator

    original_loader = aggregator.load_verified_private_ocr_source
    calls = 0

    def mutate_registry_after_first_source(*args, **kwargs):
        nonlocal calls
        source = original_loader(*args, **kwargs)
        calls += 1
        if calls == 1:
            registry = json.loads(fixture["registry_path"].read_text(encoding="utf-8"))
            registry["candidates"][0]["status"] = "retired_by_test"
            _write_json(fixture["registry_path"], registry)
        return source

    monkeypatch.setattr(
        aggregator,
        "load_verified_private_ocr_source",
        mutate_registry_after_first_source,
    )

    with pytest.raises(ValueError, match="authority changed during validation"):
        _aggregate(fixture)


def test_sustained_journal_change_between_source_loads_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    import scripts.aggregate_verified_blind_ocr_judgments as aggregator

    original_loader = aggregator.load_verified_private_ocr_source
    calls = 0

    def mutate_journal_after_first_source(*args, **kwargs):
        nonlocal calls
        source = original_loader(*args, **kwargs)
        calls += 1
        if calls == 1:
            append_event(
                fixture["sustained_events_path"],
                {
                    "event": "sustained_attempts_reclassified",
                    "reclassified_attempt_ids": [
                        fixture["provenances"]["variant_one"]["attempt_id"]
                    ],
                },
            )
        return source

    monkeypatch.setattr(
        aggregator,
        "load_verified_private_ocr_source",
        mutate_journal_after_first_source,
    )

    with pytest.raises(ValueError, match="authority changed during validation"):
        _aggregate(fixture)


def test_posthoc_identity_recommit_fails_against_public_anchor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mapping = copy.deepcopy(fixture["mapping"])
    for sample in mapping["samples"]:
        sample["identities"] = {
            "A": sample["identities"]["B"],
            "B": sample["identities"]["A"],
        }
    mapping["private_packet_commitment"] = _create_private_packet_commitment(
        mapping,
        fixture["packet"],
    )
    _write_json(fixture["mapping_path"], mapping)

    with pytest.raises(ValueError, match="public preparation anchor"):
        _aggregate(fixture)


def test_conflicting_preparation_event_same_id_is_atomic_rejection(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mapping = copy.deepcopy(fixture["mapping"])
    mapping["private_packet_commitment"] = _create_private_packet_commitment(
        mapping,
        fixture["packet"],
    )
    conflicting = _preparation_event(mapping, fixture["packet"])
    before = fixture["quality_events_path"].read_bytes()

    with pytest.raises(ValueError, match="not authorized"):
        _append_preparation_event_once(fixture["quality_events_path"], conflicting)

    assert fixture["quality_events_path"].read_bytes() == before


def test_image_snapshot_mutation_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    image_path = Path(fixture["packet"]["samples"][0]["image_path"])
    image_path.write_bytes(b"changed snapshot")

    with pytest.raises(ValueError, match="image snapshot changed"):
        _aggregate(fixture)


def test_original_sources_can_change_after_preparation_but_snapshots_remain_valid(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    for path in fixture["record_paths"].values():
        path.write_text("mutated original\n", encoding="utf-8")
        path.with_name("records-provenance.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

    assert _aggregate(fixture)["metrics"]["sample_count"] == 2


def test_snapshot_records_recommit_cannot_replace_public_terminal_tag(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    snapshot = (
        fixture["output_dir"]
        / "source-snapshots"
        / "variant_one"
        / "private-records.jsonl"
    )
    records = [json.loads(line) for line in snapshot.read_text(encoding="utf-8").splitlines()]
    records[0]["lines"] = [{"text": "forged output"}]
    _write_jsonl(snapshot, records)
    _recommit_source(snapshot)

    with pytest.raises(ValueError, match="commitment mismatch"):
        _aggregate(fixture)


def test_reversed_source_lifecycle_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    events = [
        json.loads(line)
        for line in fixture["sustained_events_path"].read_text(encoding="utf-8").splitlines()
    ]
    events[0], events[1] = events[1], events[0]
    _write_jsonl(fixture["sustained_events_path"], events)

    with pytest.raises(ValueError, match="lifecycle order"):
        _aggregate(fixture)


@pytest.mark.parametrize(
    "correction",
    [
        {
            "event": "sustained_attempts_reclassified",
            "reclassified_attempt_ids": [],
        },
        {
            "event": "sustained_config_indices_reclassified",
            "reclassified_attempt_ids": [],
        },
    ],
)
def test_active_source_correction_is_rejected(
    tmp_path: Path,
    correction: dict,
) -> None:
    fixture = _fixture(tmp_path)
    correction["reclassified_attempt_ids"] = [
        fixture["provenances"]["variant_one"]["attempt_id"]
    ]
    with fixture["sustained_events_path"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(correction) + "\n")

    with pytest.raises(ValueError, match="active correction"):
        _aggregate(fixture)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda registry: registry.update(schema_version=999),
        lambda registry: registry.update(protocol="wrong"),
        lambda registry: registry["candidates"][0].update(status=None),
        lambda registry: registry["candidates"][0].update(status="blocked"),
        lambda registry: registry["candidates"][0].update(retired_config_indices=[99]),
    ],
)
def test_registry_authority_mutations_fail_closed(
    tmp_path: Path,
    mutation,
) -> None:
    fixture = _fixture(tmp_path)
    registry = json.loads(fixture["registry_path"].read_text(encoding="utf-8"))
    mutation(registry)
    _write_json(fixture["registry_path"], registry)

    with pytest.raises(ValueError):
        _aggregate(fixture)


@pytest.mark.parametrize("private_value", [0, 1, "private path", {"text": "secret"}])
def test_preparation_public_append_rejects_private_or_type_lax_mutation(
    tmp_path: Path,
    private_value: object,
) -> None:
    fixture = _fixture(tmp_path)
    event = copy.deepcopy(fixture["preparation_event"])
    if isinstance(private_value, int):
        event["privacy"]["private_paths_or_text_published"] = private_value
    else:
        event["private_path"] = private_value
    event.pop("public_event_sha256")
    from local_inference_bench.verified_blind_ocr_protocol import public_event_sha256

    event["public_event_sha256"] = public_event_sha256(event)
    destination = tmp_path / "injected-quality-events.jsonl"

    with pytest.raises(ValueError, match="not authorized"):
        _append_preparation_event_once(destination, event)

    assert not destination.exists()


def test_loader_rejects_extra_or_ambiguous_provenance_fields(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    records_path = fixture["record_paths"]["variant_one"]
    provenance_path = records_path.with_name("records-provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["uncommitted_claim"] = "private"
    _write_json(provenance_path, provenance)

    with pytest.raises(ValueError, match="provenance is invalid"):
        load_verified_private_ocr_source(
            records_path,
            sustained_events_path=fixture["sustained_events_path"],
            registry_path=fixture["registry_path"],
        )


def test_private_ocr_record_decoder_normalizes_excessive_nesting() -> None:
    import local_inference_bench.load_verified_private_ocr_source as source_loader

    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    records = b'{"sample_id":"frame","success":true,"lines":' + nested + b"}\n"

    with pytest.raises(ValueError, match="invalid private OCR record"):
        source_loader._parse_ocr_records(records)


def test_exact_winner_vote_feasibility_matches_small_exhaustive_truth() -> None:
    generator = random.Random(285)
    for sample_count in range(1, 4):
        for judgment_count in range(2, 6):
            item_types = []
            for first_votes in range(judgment_count + 1):
                for second_votes in range(judgment_count - first_votes + 1):
                    third_votes = judgment_count - first_votes - second_votes
                    votes = (first_votes, second_votes, third_votes)
                    majority = next(
                        (
                            index
                            for index, value in enumerate(votes)
                            if value > judgment_count // 2
                        ),
                        3,
                    )
                    item_types.append(
                        (
                            votes,
                            majority,
                            int(max(votes) == judgment_count),
                            sum(value * (value - 1) // 2 for value in votes),
                        )
                    )
            truth = set()
            for sequence in itertools.product(item_types, repeat=sample_count):
                category_votes = tuple(
                    sum(item[0][index] for item in sequence) for index in range(3)
                )
                majority_counts = tuple(
                    sum(item[1] == index for item in sequence) for index in range(4)
                )
                truth.add(
                    (
                        category_votes,
                        majority_counts[:3],
                        majority_counts[3],
                        sum(item[2] for item in sequence),
                        sum(item[3] for item in sequence),
                    )
                )
            for summary in truth:
                assert _exact_winner_vote_summary_is_feasible(
                    sample_count=sample_count,
                    judgment_count=judgment_count,
                    category_vote_counts=summary[0],
                    strict_majority_counts=summary[1],
                    no_strict_majority_count=summary[2],
                    unanimous_count=summary[3],
                    pairwise_agreement_count=summary[4],
                )
            for _ in range(250):
                vote_total = sample_count * judgment_count
                first_votes = generator.randint(0, vote_total)
                second_votes = generator.randint(0, vote_total - first_votes)
                category_votes = (
                    first_votes,
                    second_votes,
                    vote_total - first_votes - second_votes,
                )
                first_majorities = generator.randint(0, sample_count)
                second_majorities = generator.randint(
                    0,
                    sample_count - first_majorities,
                )
                third_majorities = generator.randint(
                    0,
                    sample_count - first_majorities - second_majorities,
                )
                strict_majorities = (
                    first_majorities,
                    second_majorities,
                    third_majorities,
                )
                no_majority = sample_count - sum(strict_majorities)
                unanimous = generator.randint(0, sum(strict_majorities))
                pairs = generator.randint(
                    0,
                    sample_count * judgment_count * (judgment_count - 1) // 2,
                )
                summary = (
                    category_votes,
                    strict_majorities,
                    no_majority,
                    unanimous,
                    pairs,
                )
                assert _exact_winner_vote_summary_is_feasible(
                    sample_count=sample_count,
                    judgment_count=judgment_count,
                    category_vote_counts=category_votes,
                    strict_majority_counts=strict_majorities,
                    no_strict_majority_count=no_majority,
                    unanimous_count=unanimous,
                    pairwise_agreement_count=pairs,
                ) is (summary in truth)


def test_exact_winner_vote_feasibility_rejects_scaled_class_conflict() -> None:
    assert not _exact_winner_vote_summary_is_feasible(
        sample_count=100,
        judgment_count=3,
        category_vote_counts=(100, 100, 100),
        strict_majority_counts=(0, 0, 50),
        no_strict_majority_count=50,
        unanimous_count=0,
        pairwise_agreement_count=50,
    )


def test_changed_preparation_id_cannot_reuse_old_judgments(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    mapping = copy.deepcopy(fixture["mapping"])
    packet = copy.deepcopy(fixture["packet"])
    new_id = "b2d37ca7-eb41-4e6f-af0e-544167c7ea22"
    mapping["preparation_id"] = new_id
    packet["preparation_id"] = new_id
    _write_json(fixture["packet_path"], packet)
    mapping["packet_fingerprint"] = hashlib.sha256(
        fixture["packet_path"].read_bytes()
    ).hexdigest()
    mapping["private_packet_commitment"] = _create_private_packet_commitment(
        mapping,
        packet,
    )
    _write_json(fixture["mapping_path"], mapping)
    new_preparation_event = _preparation_event(mapping, packet)
    assert append_event_once(
        fixture["quality_events_path"],
        new_preparation_event,
        identity_fields=("event", "protocol", "preparation_id"),
    )

    with pytest.raises(ValueError, match="judgment protocol"):
        _aggregate(fixture)
