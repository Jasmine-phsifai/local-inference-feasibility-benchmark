import json

from scripts.aggregate_blind_ocr_judgments import aggregate_judgments
from scripts.prepare_blind_ocr_comparison import build_blind_packet


def test_packet_blinds_candidate_identity_and_uses_private_frames():
    workload = {
        "task": "ocr",
        "workload_class": "private_course",
        "items": [
            {"id": "frame_001", "path": "one.png"},
            {"id": "frame_002", "path": "two.png"},
        ],
    }
    records = {
        "candidate_one": {
            "frame_001": {"lines": [{"text": "one"}]},
            "frame_002": {"lines": [{"text": "two"}]},
        },
        "candidate_two": {
            "frame_001": {"lines": [{"text": "1"}]},
            "frame_002": {"lines": [{"text": "2"}]},
        },
    }

    packet, mapping = build_blind_packet(workload, records, sample_count=2, seed=7)

    assert packet["protocol"] == "private-ocr-blind-v1"
    assert "candidate_one" not in json.dumps(packet)
    assert set(mapping["candidate_ids"]) == {"candidate_one", "candidate_two"}
    assert {sample["sample_id"] for sample in packet["samples"]} == {
        "blind_001",
        "blind_002",
    }


def test_aggregation_resolves_blind_votes_without_private_content(tmp_path):
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "protocol": "private-ocr-blind-v1",
                "candidate_ids": ["one", "two"],
                "packet_fingerprint": "abc",
                "samples": [
                    {
                        "sample_id": "blind_001",
                        "identities": {"A": "one", "B": "two"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    judgments = []
    for index, winner in enumerate(("A", "A", "tie")):
        path = tmp_path / f"judge_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "protocol": "private-ocr-blind-v1",
                    "packet_fingerprint": "ABC" if index == 0 else "abc",
                    "samples": [
                        {
                            "sample_id": "blind_001",
                            "winner": winner,
                            "a_severity": 0,
                            "b_severity": 1,
                            "a_usable": True,
                            "b_usable": True,
                            "a_error_codes": [],
                            "b_error_codes": ["missing_text"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        judgments.append(path)

    event = aggregate_judgments(mapping_path, judgments)

    assert event["metrics"]["candidates"]["one"]["consensus_wins"] == 1
    assert event["metrics"]["candidates"]["two"]["consensus_wins"] == 0
    assert event["metrics"]["agreement_sample_fraction"] == 1.0
