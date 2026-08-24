import hashlib
import json
import sys
import uuid
from pathlib import Path

import pytest

import scripts.prepare_blind_ocr_comparison as prepare_module
from local_inference_bench.load_sustained_workload import load_sustained_workload
from scripts.aggregate_blind_ocr_judgments import (
    _validated_source_attempts,
    aggregate_judgments,
)
from scripts.prepare_blind_ocr_comparison import (
    _parse_candidate_specs,
    _require_new_output_directory,
    _seal_blind_packet_mapping,
    _validate_full_workload_record_status,
    _validate_variant_provenance,
    _verify_records_provenance,
    build_blind_packet,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_attempt(
    variant_id: str,
    index: int,
    *,
    candidate_id: str | None = None,
    config_index: int | None = None,
    attempt_status: str = "succeeded",
) -> dict:
    return {
        "variant_id": variant_id,
        "candidate_id": candidate_id or variant_id,
        "attempt_id": str(uuid.UUID(f"00000000-0000-4000-8000-{index:012d}")),
        "attempt_key": f"{index:016x}",
        "config_index": index if config_index is None else config_index,
        "config_fingerprint": f"{index + 100:016x}",
        "trial_index": 0,
        "attempt_status": attempt_status,
    }


def _write_sealed_packet_and_mapping(
    tmp_path: Path,
    *,
    source_attempts: list[dict] | None = None,
    records: dict[str, dict[str, dict]] | None = None,
) -> tuple[Path, Path, str]:
    image_path = (tmp_path / "ignored-frame.png").resolve()
    image_path.write_bytes(b"private image bytes")
    packet, mapping = build_blind_packet(
        {
            "task": "ocr",
            "workload_class": "private_course",
            "items": [{"id": "frame_001", "path": str(image_path)}],
        },
        records
        or {
            "one": {
                "frame_001": {
                    "success": True,
                    "lines": [{"text": "first"}],
                }
            },
            "two": {
                "frame_001": {
                    "success": True,
                    "lines": [{"text": "second"}],
                }
            },
        },
        sample_count=1,
        seed=7,
    )
    mapping["source_attempts"] = source_attempts or [
        _source_attempt("one", 1),
        _source_attempt("two", 2),
    ]
    _seal_blind_packet_mapping(packet, mapping, nonce="1" * 64)
    packet_path = tmp_path / "packet.json"
    mapping_path = tmp_path / "mapping.json"
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    packet_fingerprint = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    mapping["packet_fingerprint"] = packet_fingerprint
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return packet_path, mapping_path, packet_fingerprint


def _write_judgment(
    path: Path,
    *,
    packet_fingerprint: str,
    winner: str,
    a_severity: object = 0,
    indent: int | None = None,
    reverse_error_codes: bool = False,
) -> None:
    error_codes = ["missing_text", "small_text"]
    if reverse_error_codes:
        error_codes.reverse()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": "private-ocr-blind-v6",
                "packet_fingerprint": packet_fingerprint,
                "samples": [
                    {
                        "sample_id": "blind_001",
                        "winner": winner,
                        "a_severity": a_severity,
                        "b_severity": 1,
                        "a_usable": True,
                        "b_usable": True,
                        "a_error_codes": [],
                        "b_error_codes": error_codes,
                    }
                ],
            },
            indent=indent,
        ),
        encoding="utf-8",
    )


def _write_distinct_judgments(
    tmp_path: Path,
    packet_fingerprint: str,
) -> list[Path]:
    first = tmp_path / "judge_a.json"
    second = tmp_path / "judge_b.json"
    _write_judgment(
        first,
        packet_fingerprint=packet_fingerprint,
        winner="A",
    )
    _write_judgment(
        second,
        packet_fingerprint=packet_fingerprint,
        winner="B",
    )
    return [first, second]


def test_packet_blinds_candidate_identity_and_uses_private_frames(tmp_path: Path):
    first_image = (tmp_path / "one.png").resolve()
    second_image = (tmp_path / "two.png").resolve()
    first_image.write_bytes(b"first private image")
    second_image.write_bytes(b"second private image")
    workload = {
        "task": "ocr",
        "workload_class": "private_course",
        "items": [
            {"id": "frame_001", "path": str(first_image)},
            {"id": "frame_002", "path": str(second_image)},
        ],
    }
    records = {
        "candidate_one": {
            "frame_001": {"success": True, "lines": [{"text": "one"}]},
            "frame_002": {"success": True, "lines": [{"text": "two"}]},
        },
        "candidate_two": {
            "frame_001": {"success": True, "lines": [{"text": "1"}]},
            "frame_002": {"success": True, "lines": [{"text": "2"}]},
        },
    }

    packet, mapping = build_blind_packet(workload, records, sample_count=2, seed=7)
    mapping["source_attempts"] = [
        _source_attempt("candidate_one", 1),
        _source_attempt("candidate_two", 2),
    ]
    _seal_blind_packet_mapping(packet, mapping, nonce="2" * 64)

    assert packet["protocol"] == "private-ocr-blind-v6"
    assert "candidate_one" not in json.dumps(packet)
    assert mapping["mapping_commitment"] == packet["mapping_commitment"]
    project_root = Path(__file__).resolve().parents[1]
    assert mapping["preparation_producer_sha256"] == {
        relative_path: _sha256(project_root / relative_path)
        for relative_path in (
            "scripts/prepare_blind_ocr_comparison.py",
            "src/local_inference_bench/fingerprint.py",
            "src/local_inference_bench/load_sustained_workload.py",
        )
    }
    assert "mapping_commitment_nonce" not in packet
    assert set(mapping["variant_ids"]) == {"candidate_one", "candidate_two"}
    assert {sample["sample_id"] for sample in packet["samples"]} == {
        "blind_001",
        "blind_002",
    }
    assert all(
        set(sample["option_statuses"].values()) == {"available"}
        for sample in packet["samples"]
    )
    assert {
        sample["image_sha256"] for sample in mapping["samples"]
    } == {_sha256(first_image), _sha256(second_image)}
    assert all(
        sample["image_sha256"] not in json.dumps(packet)
        for sample in mapping["samples"]
    )


def test_relative_workload_paths_are_resolved_once_before_packet_build(
    tmp_path: Path,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    image_path = media_root / "frame.png"
    image_path.write_bytes(b"relative private image")
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "private_course",
                "items": [{"id": "frame_001", "path": "media/frame.png"}],
            }
        ),
        encoding="utf-8",
    )
    snapshot = load_sustained_workload(workload_path, expected_task="ocr")
    workload_path.write_text("not JSON anymore", encoding="utf-8")

    packet, mapping = build_blind_packet(
        snapshot,
        {
            "one": {
                "frame_001": {"success": True, "lines": [{"text": "one"}]}
            },
            "two": {
                "frame_001": {"success": True, "lines": [{"text": "two"}]}
            },
        },
        sample_count=1,
        seed=7,
    )

    assert Path(packet["samples"][0]["image_path"]).resolve() == image_path.resolve()
    assert mapping["samples"][0]["image_sha256"] == _sha256(image_path)


def test_main_uses_one_validated_workload_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    media_root = tmp_path / "media"
    media_root.mkdir()
    image_path = media_root / "frame.png"
    image_path.write_bytes(b"main relative private image")
    workload_path = tmp_path / "workload.json"
    workload_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "private_course",
                "items": [{"id": "frame_001", "path": "media/frame.png"}],
            }
        ),
        encoding="utf-8",
    )
    snapshot = load_sustained_workload(workload_path, expected_task="ocr")
    candidate_arguments = []
    for index, variant_id in enumerate(("one", "two"), start=1):
        records_root = tmp_path / variant_id
        records_root.mkdir()
        records_path = records_root / "private-records.jsonl"
        records_path.write_text(
            json.dumps(
                {
                    "sample_id": "frame_001",
                    "success": True,
                    "lines": [{"text": variant_id}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        provenance = {
            "schema_version": 1,
            "protocol": "sustained-process-v1",
            "status": "succeeded",
            "attempt_id": str(
                uuid.UUID(f"00000000-0000-4000-8000-{index:012d}")
            ),
            "attempt_key": f"{index:016x}",
            "candidate_id": f"candidate_{variant_id}",
            "task": "ocr",
            "config": {"processes": index},
            "config_index": index,
            "phase": "quality",
            "trial_index": 0,
            "workload_class": "private_course",
            "workload_fingerprint": snapshot["fingerprint"],
            "code_fingerprint": "a" * 16,
            "environment_fingerprint": "b" * 16,
            "records_sha256": _sha256(records_path),
        }
        records_path.with_name("records-provenance.json").write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )
        candidate_arguments.extend(
            [
                "--candidate",
                (
                    f"{variant_id}@candidate_{variant_id}#{index}="
                    f"{records_path}"
                ),
            ]
        )

    original_loader = prepare_module.load_sustained_workload
    load_count = 0

    def load_once(path: Path, *, expected_task: str) -> dict:
        nonlocal load_count
        load_count += 1
        loaded = original_loader(path, expected_task=expected_task)
        workload_path.write_text("invalid after validated snapshot", encoding="utf-8")
        return loaded

    output_dir = tmp_path / "private-output"
    monkeypatch.setattr(prepare_module, "load_sustained_workload", load_once)
    monkeypatch.setattr(prepare_module, "_PROJECT_ROOT", tmp_path / "other-project")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_blind_ocr_comparison.py",
            "--workload",
            str(workload_path),
            *candidate_arguments,
            "--sample-count",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )

    prepare_module.main()

    assert load_count == 1
    packet = json.loads((output_dir / "packet.json").read_text(encoding="utf-8"))
    mapping = json.loads((output_dir / "mapping.json").read_text(encoding="utf-8"))
    assert Path(packet["samples"][0]["image_path"]).resolve() == image_path.resolve()
    assert mapping["samples"][0]["image_sha256"] == _sha256(image_path)


def test_packet_producer_rejects_more_than_one_hundred_samples(
    tmp_path: Path,
) -> None:
    image_path = (tmp_path / "frame.png").resolve()
    image_path.write_bytes(b"bounded private image")
    workload = {
        "task": "ocr",
        "workload_class": "private_course",
        "items": [
            {"id": f"frame_{index:03d}", "path": str(image_path)}
            for index in range(101)
        ],
    }

    with pytest.raises(ValueError, match="sample_count"):
        build_blind_packet(
            workload,
            {"one": {}, "two": {}},
            sample_count=101,
            seed=7,
        )


def test_full_workload_record_status_uses_every_workload_item() -> None:
    workload_items = [
        {"id": "frame_001"},
        {"id": "frame_002"},
        {"id": "frame_003"},
    ]

    assert _validate_full_workload_record_status(
        {
            "frame_001": {"success": True},
            "frame_002": {"success": True},
            "frame_003": {"success": True},
        },
        workload_items,
        "succeeded",
    ) == {"total": 3, "available": 3, "failed": 0, "unavailable": 0}
    assert _validate_full_workload_record_status(
        {
            "frame_001": {"success": True},
            "frame_002": {"success": False},
        },
        workload_items,
        "partial_failure",
    ) == {"total": 3, "available": 1, "failed": 1, "unavailable": 1}
    assert _validate_full_workload_record_status(
        {
            "frame_001": {"success": False},
            "frame_002": {"success": False},
        },
        workload_items,
        "all_failed",
    ) == {"total": 3, "available": 0, "failed": 2, "unavailable": 1}


@pytest.mark.parametrize(
    ("records", "attempt_status"),
    [
        ({"frame_001": {"success": True}}, "succeeded"),
        (
            {
                "frame_001": {"success": True},
                "frame_002": {"success": False},
            },
            "succeeded",
        ),
        (
            {
                "frame_001": {"success": True},
                "frame_002": {"success": True},
            },
            "partial_failure",
        ),
        ({"frame_001": {"success": False}}, "partial_failure"),
        (
            {
                "frame_001": {"success": True},
                "frame_002": {"success": False},
            },
            "all_failed",
        ),
    ],
)
def test_full_workload_record_status_rejects_inconsistent_provenance(
    records: dict[str, dict],
    attempt_status: str,
) -> None:
    with pytest.raises(ValueError, match="status does not match workload records"):
        _validate_full_workload_record_status(
            records,
            [{"id": "frame_001"}, {"id": "frame_002"}],
            attempt_status,
        )


def test_full_workload_record_status_rejects_records_outside_workload() -> None:
    with pytest.raises(ValueError, match="records do not match the workload"):
        _validate_full_workload_record_status(
            {
                "frame_001": {"success": True},
                "not_in_workload": {"success": True},
            },
            [{"id": "frame_001"}],
            "succeeded",
        )


def test_packet_sealing_rejects_succeeded_attempt_with_unavailable_selection(
    tmp_path: Path,
) -> None:
    image_path = (tmp_path / "frame.png").resolve()
    image_path.write_bytes(b"private image")
    packet, mapping = build_blind_packet(
        {
            "task": "ocr",
            "workload_class": "private_course",
            "items": [{"id": "frame_001", "path": str(image_path)}],
        },
        {
            "one": {},
            "two": {
                "frame_001": {
                    "success": True,
                    "lines": [{"text": "available"}],
                }
            },
        },
        sample_count=1,
        seed=7,
    )
    mapping["source_attempts"] = [
        _source_attempt("one", 1, attempt_status="succeeded"),
        _source_attempt("two", 2, attempt_status="succeeded"),
    ]

    with pytest.raises(ValueError, match="contradict attempt status"):
        _seal_blind_packet_mapping(packet, mapping, nonce="2" * 64)


def test_source_attempts_accept_same_candidate_with_distinct_configs() -> None:
    attempts = [
        _source_attempt(
            "rapid_full",
            1,
            candidate_id="rapidocr_cpu",
            config_index=17,
        ),
        _source_attempt(
            "rapid_1280",
            2,
            candidate_id="rapidocr_cpu",
            config_index=14,
        ),
    ]

    validated = _validated_source_attempts(
        attempts,
        ["rapid_1280", "rapid_full"],
    )
    assert {item["candidate_id"] for item in validated} == {"rapidocr_cpu"}
    assert {item["config_index"] for item in validated} == {14, 17}

    attempts[0]["attempt_status"] = "complete"
    with pytest.raises(ValueError, match="source attempts are invalid"):
        _validated_source_attempts(attempts, ["rapid_1280", "rapid_full"])


def test_aggregate_publishes_unambiguous_same_candidate_variant_attribution(
    tmp_path: Path,
) -> None:
    attempts = [
        _source_attempt("one", 1, candidate_id="rapidocr_cpu", config_index=17),
        _source_attempt("two", 2, candidate_id="rapidocr_cpu", config_index=14),
    ]
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(
        tmp_path,
        source_attempts=attempts,
    )
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)

    event = aggregate_judgments(mapping_path, judgments)

    assert event["source_variants"] == [
        {
            "variant_id": "one",
            "candidate_id": "rapidocr_cpu",
            "config_index": 17,
            "config_fingerprint": f"{101:016x}",
            "attempt_status": "succeeded",
            "selected_available_record_count": 1,
            "selected_failed_record_count": 0,
            "selected_unavailable_record_count": 0,
        },
        {
            "variant_id": "two",
            "candidate_id": "rapidocr_cpu",
            "config_index": 14,
            "config_fingerprint": f"{102:016x}",
            "attempt_status": "succeeded",
            "selected_available_record_count": 1,
            "selected_failed_record_count": 0,
            "selected_unavailable_record_count": 0,
        },
    ]


def test_failed_and_unavailable_records_are_disclosed_and_counted(
    tmp_path: Path,
) -> None:
    attempts = [
        _source_attempt("one", 1, attempt_status="partial_failure"),
        _source_attempt("two", 2, attempt_status="all_failed"),
    ]
    packet_path, mapping_path, packet_fingerprint = (
        _write_sealed_packet_and_mapping(
            tmp_path,
            source_attempts=attempts,
            records={
                "one": {"frame_001": {"success": False}},
                "two": {},
            },
        )
    )
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)

    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert set(packet["samples"][0]["option_statuses"].values()) == {
        "failed",
        "unavailable",
    }
    event = aggregate_judgments(mapping_path, judgments)
    variants = {item["variant_id"]: item for item in event["source_variants"]}

    assert variants["one"]["attempt_status"] == "partial_failure"
    assert variants["one"]["selected_failed_record_count"] == 1
    assert variants["one"]["selected_unavailable_record_count"] == 0
    assert variants["two"]["attempt_status"] == "all_failed"
    assert variants["two"]["selected_failed_record_count"] == 0
    assert variants["two"]["selected_unavailable_record_count"] == 1
    assert event["metrics"]["source_record_availability"] == {
        "variant_sample_count": 2,
        "available_record_count": 0,
        "failed_record_count": 1,
        "unavailable_record_count": 1,
    }
    for variant_metrics in event["metrics"]["variants"].values():
        assert variant_metrics["mean_error_severity_vote_denominator"] == 2
        assert variant_metrics["usable_vote_denominator"] == 2
    assert event["metrics"]["comparison_sample_denominators"] == {
        "total_selected_sample_count": 1,
        "individual_winner_vote_denominator": 2,
        "consensus_winner_sample_denominator": 1,
        "strict_majority_sample_denominator": 1,
        "unanimous_sample_denominator": 1,
        "pairwise_winner_agreement_denominator": 1,
        "fully_available_comparison_count": 0,
        "not_fully_available_comparison_count": 1,
    }
    assert event["interpretation"][
        "failed_or_unavailable_options_disclosed_to_judges"
    ] is True

    failed_label = next(
        label
        for label, status in packet["samples"][0]["option_statuses"].items()
        if status == "failed"
    )
    packet["samples"][0]["options"][failed_label] = ["misleading output"]
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["packet_fingerprint"] = _sha256(packet_path)
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    judgments = _write_distinct_judgments(
        tmp_path,
        mapping["packet_fingerprint"],
    )
    with pytest.raises(ValueError, match="must be empty"):
        aggregate_judgments(mapping_path, judgments)


def test_aggregation_rejects_recommitted_impossible_selected_status(
    tmp_path: Path,
) -> None:
    packet_path, mapping_path, _ = _write_sealed_packet_and_mapping(
        tmp_path,
        source_attempts=[
            _source_attempt("one", 1, attempt_status="partial_failure"),
            _source_attempt("two", 2, attempt_status="succeeded"),
        ],
        records={
            "one": {"frame_001": {"success": False}},
            "two": {
                "frame_001": {
                    "success": True,
                    "lines": [{"text": "available"}],
                }
            },
        },
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_one = next(
        source for source in mapping["source_attempts"] if source["variant_id"] == "one"
    )
    source_one["attempt_status"] = "succeeded"
    commitment = prepare_module._mapping_commitment(
        mapping,
        mapping["mapping_commitment_nonce"],
    )
    mapping["mapping_commitment"] = commitment
    packet["mapping_commitment"] = commitment
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    mapping["packet_fingerprint"] = _sha256(packet_path)
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    judgments = _write_distinct_judgments(
        tmp_path,
        mapping["packet_fingerprint"],
    )

    with pytest.raises(ValueError, match="contradict attempt status"):
        aggregate_judgments(mapping_path, judgments)


def test_variant_spec_binds_candidate_and_config_provenance() -> None:
    specs = _parse_candidate_specs(
        [
            "rapid_full@rapidocr_cpu#17=C:/full/records.jsonl",
            "rapid_1280@rapidocr_cpu#14=C:/small/records.jsonl",
        ]
    )
    provenance = {"candidate_id": "rapidocr_cpu", "config_index": 17}
    _validate_variant_provenance("rapid_full", specs["rapid_full"], provenance)

    with pytest.raises(ValueError, match="candidate/config"):
        _validate_variant_provenance(
            "rapid_full",
            specs["rapid_full"],
            {"candidate_id": "ppocrv6_tiny_cpu", "config_index": 17},
        )
    with pytest.raises(ValueError, match="candidate/config"):
        _validate_variant_provenance(
            "rapid_full",
            specs["rapid_full"],
            {"candidate_id": "rapidocr_cpu", "config_index": 14},
        )


def test_aggregation_resolves_strict_majority_without_private_fingerprint(
    tmp_path: Path,
) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = []
    for index, winner in enumerate(("A", "A", "tie")):
        path = tmp_path / f"judge_{index}.json"
        _write_judgment(
            path,
            packet_fingerprint=packet_fingerprint,
            winner=winner,
            a_severity=index,
        )
        judgments.append(path)

    event = aggregate_judgments(mapping_path, judgments)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    identities = mapping["samples"][0]["identities"]
    private_image_sha256 = mapping["samples"][0]["image_sha256"]

    assert event["protocol"] == "private-ocr-blind-v8"
    assert event["metrics"]["variants"][identities["A"]]["consensus_wins"] == 1
    assert event["metrics"]["variants"][identities["B"]]["consensus_wins"] == 0
    for variant_metrics in event["metrics"]["variants"].values():
        assert variant_metrics["mean_error_severity_vote_denominator"] == 3
        assert variant_metrics["usable_vote_denominator"] == 3
    assert event["metrics"]["strict_majority_sample_fraction"] == 1.0
    assert event["metrics"]["unanimous_sample_fraction"] == 0.0
    assert event["metrics"]["pairwise_winner_agreement_fraction"] == pytest.approx(
        1 / 3
    )
    assert "judgment_set_fingerprint" not in event
    assert event["interpretation"]["private_fingerprints_published"] is False
    assert event["interpretation"]["private_run_identifiers_published"] is False
    assert event["interpretation"]["mapping_commitment_verified"] is True
    assert event["interpretation"]["semantic_duplicate_guard"] is True
    assert event["interpretation"]["procedural_blinding_only"] is True
    assert event["interpretation"]["judge_independence_verified"] is False
    assert event["interpretation"]["private_image_hashes_published"] is False
    assert private_image_sha256 not in json.dumps(event)
    assert event["metrics"]["source_record_availability"] == {
        "variant_sample_count": 2,
        "available_record_count": 2,
        "failed_record_count": 0,
        "unavailable_record_count": 0,
    }
    assert event["metrics"]["comparison_sample_denominators"] == {
        "total_selected_sample_count": 1,
        "individual_winner_vote_denominator": 3,
        "consensus_winner_sample_denominator": 1,
        "strict_majority_sample_denominator": 1,
        "unanimous_sample_denominator": 1,
        "pairwise_winner_agreement_denominator": 3,
        "fully_available_comparison_count": 1,
        "not_fully_available_comparison_count": 0,
    }
    assert event["source_variants"] == [
        {
            "variant_id": "one",
            "candidate_id": "one",
            "config_index": 1,
            "config_fingerprint": f"{101:016x}",
            "attempt_status": "succeeded",
            "selected_available_record_count": 1,
            "selected_failed_record_count": 0,
            "selected_unavailable_record_count": 0,
        },
        {
            "variant_id": "two",
            "candidate_id": "two",
            "config_index": 2,
            "config_fingerprint": f"{102:016x}",
            "attempt_status": "succeeded",
            "selected_available_record_count": 1,
            "selected_failed_record_count": 0,
            "selected_unavailable_record_count": 0,
        },
    ]
    assert set(event["producer_sha256"]) == {
        "scripts/aggregate_blind_ocr_judgments.py",
        "scripts/prepare_blind_ocr_comparison.py",
        "src/local_inference_bench/fingerprint.py",
        "src/local_inference_bench/load_sustained_workload.py",
        "src/local_inference_bench/validate_public_summary.py",
    }
    for relative_path, digest in mapping["preparation_producer_sha256"].items():
        assert event["producer_sha256"][relative_path] == digest
    project_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "scripts/aggregate_blind_ocr_judgments.py",
        "src/local_inference_bench/validate_public_summary.py",
    ):
        assert event["producer_sha256"][relative_path] == _sha256(
            project_root / relative_path
        )
    assert "source_attempts" not in event


def test_four_judges_require_more_than_two_matching_votes(tmp_path: Path) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = []
    for index, winner in enumerate(("A", "A", "B", "tie")):
        path = tmp_path / f"judge_{index}.json"
        _write_judgment(
            path,
            packet_fingerprint=packet_fingerprint,
            winner=winner,
            a_severity=index % 4,
        )
        judgments.append(path)

    event = aggregate_judgments(mapping_path, judgments)

    assert event["metrics"]["strict_majority_sample_fraction"] == 0.0
    assert event["metrics"]["unanimous_sample_fraction"] == 0.0
    assert event["metrics"]["pairwise_winner_agreement_fraction"] == pytest.approx(
        1 / 6
    )
    assert event["metrics"]["comparison_sample_denominators"][
        "pairwise_winner_agreement_denominator"
    ] == 6
    for variant_metrics in event["metrics"]["variants"].values():
        assert variant_metrics["mean_error_severity_vote_denominator"] == 4
        assert variant_metrics["usable_vote_denominator"] == 4
    assert event["metrics"]["variants"]["one"]["consensus_wins"] == 0


def test_rejects_duplicate_judgment_paths_and_boolean_severity(tmp_path: Path) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgment = tmp_path / "judge.json"
    _write_judgment(
        judgment,
        packet_fingerprint=packet_fingerprint,
        winner="A",
    )

    with pytest.raises(ValueError, match="distinct judgment"):
        aggregate_judgments(mapping_path, [judgment, judgment])

    reformatted = tmp_path / "reformatted.json"
    _write_judgment(
        reformatted,
        packet_fingerprint=packet_fingerprint,
        winner="A",
        indent=4,
        reverse_error_codes=True,
    )
    assert judgment.read_bytes() != reformatted.read_bytes()
    with pytest.raises(ValueError, match="duplicate semantic votes"):
        aggregate_judgments(mapping_path, [judgment, reformatted])

    second = tmp_path / "judge_2.json"
    _write_judgment(
        judgment,
        packet_fingerprint=packet_fingerprint,
        winner="A",
        a_severity=True,
    )
    _write_judgment(
        second,
        packet_fingerprint=packet_fingerprint,
        winner="B",
    )
    with pytest.raises(ValueError, match="severity"):
        aggregate_judgments(mapping_path, [judgment, second])


def test_mapping_commitment_rejects_identity_swap(tmp_path: Path) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    identities = mapping["samples"][0]["identities"]
    identities["A"], identities["B"] = identities["B"], identities["A"]
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="mapping commitment does not match"):
        aggregate_judgments(mapping_path, judgments)


def test_mapping_commitment_rejects_private_image_hash_replacement(
    tmp_path: Path,
) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["samples"][0]["image_sha256"] = "f" * 64
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="mapping commitment does not match"):
        aggregate_judgments(mapping_path, judgments)


def test_mapping_commitment_rejects_preparation_producer_hash_replacement(
    tmp_path: Path,
) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["preparation_producer_sha256"][
        "scripts/prepare_blind_ocr_comparison.py"
    ] = "f" * 64
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="mapping commitment does not match"):
        aggregate_judgments(mapping_path, judgments)


def test_recommitted_historical_preparation_hash_is_not_recomputed(
    tmp_path: Path,
) -> None:
    packet_path, mapping_path, _ = _write_sealed_packet_and_mapping(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    preparation_path = "scripts/prepare_blind_ocr_comparison.py"
    historical_digest = "f" * 64
    mapping["preparation_producer_sha256"][preparation_path] = historical_digest
    commitment = prepare_module._mapping_commitment(
        mapping,
        mapping["mapping_commitment_nonce"],
    )
    mapping["mapping_commitment"] = commitment
    packet["mapping_commitment"] = commitment
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    mapping["packet_fingerprint"] = _sha256(packet_path)
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    judgments = _write_distinct_judgments(
        tmp_path,
        mapping["packet_fingerprint"],
    )

    event = aggregate_judgments(mapping_path, judgments)

    current_digest = _sha256(Path(__file__).resolve().parents[1] / preparation_path)
    assert historical_digest != current_digest
    assert event["producer_sha256"][preparation_path] == historical_digest


@pytest.mark.parametrize(
    "mutation",
    [
        "not_a_mapping",
        "missing_path",
        "extra_path",
        "uppercase_digest",
    ],
)
def test_preparation_producer_hashes_require_exact_shape(
    tmp_path: Path,
    mutation: str,
) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    hashes = mapping["preparation_producer_sha256"]
    if mutation == "not_a_mapping":
        mapping["preparation_producer_sha256"] = []
    elif mutation == "missing_path":
        hashes.pop("src/local_inference_bench/fingerprint.py")
    elif mutation == "extra_path":
        hashes["scripts/unrelated.py"] = "a" * 64
    else:
        hashes["scripts/prepare_blind_ocr_comparison.py"] = "A" * 64
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="preparation producer hashes are invalid"):
        aggregate_judgments(mapping_path, judgments)


def test_aggregation_rejects_image_mutation_after_packet_preparation(
    tmp_path: Path,
) -> None:
    packet_path, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(
        tmp_path
    )
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    Path(packet["samples"][0]["image_path"]).write_bytes(b"mutated private image")

    with pytest.raises(ValueError, match="changed after packet preparation"):
        aggregate_judgments(mapping_path, judgments)


def test_packet_bytes_and_embedded_commitment_are_verified(tmp_path: Path) -> None:
    packet_path, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(
        tmp_path
    )
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["samples"][0]["options"]["A"].append("changed")
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    with pytest.raises(ValueError, match="packet fingerprint does not match"):
        aggregate_judgments(mapping_path, judgments)

    packet_path, mapping_path, _ = _write_sealed_packet_and_mapping(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    packet["mapping_commitment"] = "f" * 64
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    mapping["packet_fingerprint"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    judgments = _write_distinct_judgments(
        tmp_path,
        mapping["packet_fingerprint"],
    )

    with pytest.raises(ValueError, match="packet mapping commitment does not match"):
        aggregate_judgments(mapping_path, judgments)


def test_packet_mapping_sample_sets_and_exact_judgment_schema_are_required(
    tmp_path: Path,
) -> None:
    packet_path, mapping_path, _ = _write_sealed_packet_and_mapping(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    packet["samples"][0]["sample_id"] = "blind_002"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    mapping["packet_fingerprint"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    judgments = _write_distinct_judgments(tmp_path, mapping["packet_fingerprint"])

    with pytest.raises(ValueError, match="packet and mapping samples"):
        aggregate_judgments(mapping_path, judgments)

    packet_path, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(
        tmp_path
    )
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    invalid = json.loads(judgments[0].read_text(encoding="utf-8"))
    invalid["samples"][0]["notes"] = "irrelevant duplicate bypass"
    judgments[0].write_text(json.dumps(invalid), encoding="utf-8")

    with pytest.raises(ValueError, match="judgment sample is invalid"):
        aggregate_judgments(mapping_path, judgments)


def test_packet_and_mapping_root_schemas_reject_extra_fields(tmp_path: Path) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping["extra"] = True
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported blind mapping protocol"):
        aggregate_judgments(mapping_path, judgments)

    packet_path, mapping_path, _ = _write_sealed_packet_and_mapping(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    packet["extra"] = True
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    mapping["packet_fingerprint"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    judgments = _write_distinct_judgments(tmp_path, mapping["packet_fingerprint"])

    with pytest.raises(ValueError, match="unsupported blind packet protocol"):
        aggregate_judgments(mapping_path, judgments)


@pytest.mark.parametrize(
    "field",
    ["mapping_commitment", "mapping_commitment_nonce"],
)
def test_mapping_commitment_fields_require_lowercase_sha256_shape(
    tmp_path: Path,
    field: str,
) -> None:
    _, mapping_path, packet_fingerprint = _write_sealed_packet_and_mapping(tmp_path)
    judgments = _write_distinct_judgments(tmp_path, packet_fingerprint)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    mapping[field] = "A" * 64
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="mapping commitment is invalid"):
        aggregate_judgments(mapping_path, judgments)


def test_private_packet_cannot_be_written_to_publishable_repo_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare_module, "_PROJECT_ROOT", tmp_path)

    class Result:
        returncode = 1

    monkeypatch.setattr(prepare_module.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(ValueError, match="must be ignored"):
        prepare_module._require_ignored_output_directory(tmp_path / "publishable")

    Result.returncode = 0
    prepare_module._require_ignored_output_directory(tmp_path / "ignored")


def test_existing_blind_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "packet.json"
    sentinel.write_text("valuable prior evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        _require_new_output_directory(output_dir)
    assert sentinel.read_text(encoding="utf-8") == "valuable prior evidence"


def test_record_provenance_binds_hash_and_workload(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    workload_path = tmp_path / "workload.json"
    records_path = tmp_path / "private-records.jsonl"
    image_path.write_bytes(b"image")
    workload_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "private_course",
                "items": [{"id": "frame_001", "path": image_path.name}],
            }
        ),
        encoding="utf-8",
    )
    records_path.write_text(
        json.dumps(
            {
                "sample_id": "frame_001",
                "success": True,
                "lines": [{"text": "ignored"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    workload = load_sustained_workload(workload_path, expected_task="ocr")
    provenance = {
        "schema_version": 1,
        "protocol": "sustained-process-v1",
        "status": "succeeded",
        "attempt_id": "00000000-0000-4000-8000-000000000001",
        "attempt_key": "1" * 16,
        "candidate_id": "rapidocr_cpu",
        "task": "ocr",
        "config": {"processes": 1},
        "config_index": 1,
        "phase": "quality",
        "trial_index": 0,
        "workload_class": "private_course",
        "workload_fingerprint": workload["fingerprint"],
        "code_fingerprint": "a" * 16,
        "environment_fingerprint": "b" * 16,
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
    }
    records_path.with_name("records-provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )

    verified = _verify_records_provenance(
        records_path,
        workload_fingerprint=workload["fingerprint"],
    )
    assert verified["attempt_id"] == provenance["attempt_id"]

    for status in ("succeeded", "partial_failure", "all_failed"):
        provenance["status"] = status
        records_path.with_name("records-provenance.json").write_text(
            json.dumps(provenance),
            encoding="utf-8",
        )
        assert _verify_records_provenance(
            records_path,
            workload_fingerprint=workload["fingerprint"],
        )["status"] == status

    provenance["status"] = "complete"
    records_path.with_name("records-provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance is invalid"):
        _verify_records_provenance(
            records_path,
            workload_fingerprint=workload["fingerprint"],
        )

    provenance["status"] = "succeeded"
    records_path.with_name("records-provenance.json").write_text(
        json.dumps(provenance),
        encoding="utf-8",
    )
    records_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance is invalid"):
        _verify_records_provenance(
            records_path,
            workload_fingerprint=workload["fingerprint"],
        )
