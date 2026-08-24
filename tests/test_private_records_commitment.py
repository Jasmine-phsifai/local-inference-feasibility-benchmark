import json
from pathlib import Path

import pytest

from local_inference_bench.private_records_commitment import (
    PRIVATE_RECORDS_COMMITMENT_SCHEME,
    create_private_records_commitment,
    decode_private_records_provenance_bytes,
    verify_private_records_bytes_commitment,
    verify_private_records_commitment,
)


def _identity() -> dict:
    return {
        "protocol": "sustained-process-v1",
        "status": "succeeded",
        "attempt_id": "11111111-1111-4111-8111-111111111111",
        "attempt_key": "a" * 16,
        "candidate_id": "candidate",
        "task": "asr",
        "config": {"processes": 1},
        "config_index": 0,
        "phase": "quality",
        "target_wall_seconds": 1.0,
        "trial_index": 0,
        "workload_class": "private_course",
        "workload_fingerprint": "b" * 64,
        "code_fingerprint": "c" * 16,
        "environment_fingerprint": "d" * 16,
        "controller_environment_fingerprint": "e" * 16,
        "execution_policy_fingerprint": "f" * 16,
    }


def _verify(path: Path, identity: dict, commitment: dict) -> None:
    verify_private_records_commitment(
        path,
        identity,
        records_sha256=commitment["records_sha256"],
        private_commitment=commitment["private"],
        public_commitment=commitment["public"],
    )


def test_provenance_decoder_normalizes_excessive_json_nesting() -> None:
    nested = b"[" * 10_000 + b"0" + b"]" * 10_000
    raw = b'{"config":' + nested + b"}"

    with pytest.raises(ValueError, match="provenance is invalid"):
        decode_private_records_provenance_bytes(raw)


def test_private_records_commitment_binds_exact_bytes_and_identity(
    tmp_path: Path,
) -> None:
    records = tmp_path / "private-records.jsonl"
    records.write_text('{"prediction":"private"}\n', encoding="utf-8")
    identity = _identity()
    commitment = create_private_records_commitment(records, identity)

    _verify(records, identity, commitment)
    assert commitment["public"] == {
        "scheme": PRIVATE_RECORDS_COMMITMENT_SCHEME,
        "hmac_sha256": commitment["private"]["hmac_sha256"],
    }
    public_json = json.dumps(commitment["public"])
    assert commitment["private"]["key_hex"] not in public_json
    assert commitment["records_sha256"] not in public_json

    records.write_text('{"prediction":"rewritten"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="commitment is invalid"):
        _verify(records, identity, commitment)


def test_commitment_rejects_cross_attempt_replay(tmp_path: Path) -> None:
    records = tmp_path / "private-records.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    identity = _identity()
    commitment = create_private_records_commitment(records, identity)
    replayed_identity = {**identity, "attempt_id": "22222222-2222-4222-8222-222222222222"}

    with pytest.raises(ValueError, match="commitment is invalid"):
        _verify(records, replayed_identity, commitment)


def test_immutable_records_snapshot_uses_the_same_commitment_contract(
    tmp_path: Path,
) -> None:
    records = tmp_path / "private-records.jsonl"
    records.write_text('{"prediction":"bound"}\n', encoding="utf-8")
    identity = _identity()
    commitment = create_private_records_commitment(records, identity)
    snapshot = records.read_bytes()

    verify_private_records_bytes_commitment(
        snapshot,
        identity,
        records_sha256=commitment["records_sha256"],
        private_commitment=commitment["private"],
        public_commitment=commitment["public"],
    )
    with pytest.raises(ValueError, match="commitment is invalid"):
        verify_private_records_bytes_commitment(
            snapshot + b"forged",
            identity,
            records_sha256=commitment["records_sha256"],
            private_commitment=commitment["private"],
            public_commitment=commitment["public"],
        )


def test_same_records_use_independent_random_commitment_keys(tmp_path: Path) -> None:
    records = tmp_path / "private-records.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    first = create_private_records_commitment(records, _identity())
    second = create_private_records_commitment(records, _identity())

    assert first["private"]["key_hex"] != second["private"]["key_hex"]
    assert first["public"]["hmac_sha256"] != second["public"]["hmac_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("key_hex", "0" * 62),
        ("key_hex", "G" * 64),
        ("hmac_sha256", "0" * 63),
    ],
)
def test_malformed_private_commitment_rejects(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    records = tmp_path / "private-records.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    identity = _identity()
    commitment = create_private_records_commitment(records, identity)
    commitment["private"][field] = value

    with pytest.raises(ValueError, match="commitment is invalid"):
        _verify(records, identity, commitment)
