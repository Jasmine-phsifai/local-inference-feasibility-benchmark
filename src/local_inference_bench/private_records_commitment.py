"""Bind ignored prediction records to a privacy-safe public journal tag."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import struct
from pathlib import Path


PRIVATE_RECORDS_COMMITMENT_SCHEME = "private-records-hmac-sha256-v1"
_COMMITMENT_DOMAIN = b"local-inference-private-records-commitment-v1\0"
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTITY_FIELDS = (
    "protocol",
    "status",
    "attempt_id",
    "attempt_key",
    "candidate_id",
    "task",
    "config",
    "config_index",
    "phase",
    "target_wall_seconds",
    "trial_index",
    "workload_class",
    "workload_fingerprint",
    "code_fingerprint",
    "environment_fingerprint",
    "controller_environment_fingerprint",
    "execution_policy_fingerprint",
)
PRIVATE_RECORDS_PROVENANCE_FIELDS = frozenset(
    {"schema_version", *_IDENTITY_FIELDS, "records_sha256", "private_records_commitment"}
)


class _DuplicateJSONKeyError(ValueError):
    pass


def create_private_records_commitment(
    records_path: Path,
    provenance_identity: dict,
) -> dict:
    """Return private provenance fields and the public opaque commitment."""

    key = secrets.token_bytes(32)
    records_sha256, tag = _stream_commitment(
        records_path,
        provenance_identity,
        key=key,
    )
    return {
        "records_sha256": records_sha256,
        "private": {
            "scheme": PRIVATE_RECORDS_COMMITMENT_SCHEME,
            "key_hex": key.hex(),
            "hmac_sha256": tag,
        },
        "public": {
            "scheme": PRIVATE_RECORDS_COMMITMENT_SCHEME,
            "hmac_sha256": tag,
        },
    }


def create_private_records_bytes_commitment(
    records_bytes: bytes,
    provenance_identity: dict,
) -> dict:
    """Commit the exact immutable byte snapshot already validated by the runner."""

    if type(records_bytes) is not bytes:
        raise ValueError("private records snapshot is invalid")
    key = secrets.token_bytes(32)
    records_sha256, tag = _bytes_commitment(
        records_bytes,
        provenance_identity,
        key=key,
    )
    return {
        "records_sha256": records_sha256,
        "private": {
            "scheme": PRIVATE_RECORDS_COMMITMENT_SCHEME,
            "key_hex": key.hex(),
            "hmac_sha256": tag,
        },
        "public": {
            "scheme": PRIVATE_RECORDS_COMMITMENT_SCHEME,
            "hmac_sha256": tag,
        },
    }


def decode_private_records_provenance_bytes(
    raw: bytes,
    *,
    maximum_bytes: int = 65_536,
) -> dict:
    """Decode one exact-field, finite JSON provenance sidecar snapshot."""

    if type(raw) is not bytes or len(raw) > maximum_bytes:
        raise ValueError("private records provenance is invalid")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJSONKeyError(key)
            result[key] = value
        return result

    def reject_constant(_constant: str):
        raise ValueError("private records provenance is invalid")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("private records provenance is invalid")
        return parsed

    try:
        provenance = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
            parse_float=parse_finite_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError("private records provenance is invalid") from error
    if (
        not isinstance(provenance, dict)
        or set(provenance) != PRIVATE_RECORDS_PROVENANCE_FIELDS
    ):
        raise ValueError("private records provenance is invalid")
    return provenance


def verify_private_records_commitment(
    records_path: Path,
    provenance_identity: dict,
    *,
    records_sha256: object,
    private_commitment: object,
    public_commitment: object,
) -> None:
    """Fail unless private records match the journal-anchored HMAC tag."""

    key, expected_tag = _validated_commitment_material(
        records_sha256=records_sha256,
        private_commitment=private_commitment,
        public_commitment=public_commitment,
    )
    actual_records_sha256, actual_tag = _stream_commitment(
        records_path,
        provenance_identity,
        key=key,
    )
    _require_matching_commitment(
        records_sha256=records_sha256,
        expected_tag=expected_tag,
        actual_records_sha256=actual_records_sha256,
        actual_tag=actual_tag,
    )


def verify_private_records_bytes_commitment(
    records_bytes: bytes,
    provenance_identity: dict,
    *,
    records_sha256: object,
    private_commitment: object,
    public_commitment: object,
) -> None:
    """Verify the exact immutable bytes that a scorer parses."""

    if type(records_bytes) is not bytes:
        raise ValueError("private records snapshot is invalid")
    key, expected_tag = _validated_commitment_material(
        records_sha256=records_sha256,
        private_commitment=private_commitment,
        public_commitment=public_commitment,
    )
    actual_records_sha256, actual_tag = _bytes_commitment(
        records_bytes,
        provenance_identity,
        key=key,
    )
    _require_matching_commitment(
        records_sha256=records_sha256,
        expected_tag=expected_tag,
        actual_records_sha256=actual_records_sha256,
        actual_tag=actual_tag,
    )


def _validated_commitment_material(
    *,
    records_sha256: object,
    private_commitment: object,
    public_commitment: object,
) -> tuple[bytes, str]:
    if (
        type(records_sha256) is not str
        or _LOWER_SHA256.fullmatch(records_sha256) is None
        or not isinstance(private_commitment, dict)
        or set(private_commitment) != {"scheme", "key_hex", "hmac_sha256"}
        or private_commitment.get("scheme") != PRIVATE_RECORDS_COMMITMENT_SCHEME
        or type(private_commitment.get("key_hex")) is not str
        or _LOWER_SHA256.fullmatch(private_commitment["key_hex"]) is None
        or type(private_commitment.get("hmac_sha256")) is not str
        or _LOWER_SHA256.fullmatch(private_commitment["hmac_sha256"]) is None
        or not isinstance(public_commitment, dict)
        or set(public_commitment) != {"scheme", "hmac_sha256"}
        or public_commitment.get("scheme") != PRIVATE_RECORDS_COMMITMENT_SCHEME
        or type(public_commitment.get("hmac_sha256")) is not str
        or _LOWER_SHA256.fullmatch(public_commitment["hmac_sha256"]) is None
    ):
        raise ValueError("private records commitment is invalid")
    if not hmac.compare_digest(
        private_commitment["hmac_sha256"],
        public_commitment["hmac_sha256"],
    ):
        raise ValueError("private records commitment is invalid")
    return (
        bytes.fromhex(private_commitment["key_hex"]),
        private_commitment["hmac_sha256"],
    )


def _require_matching_commitment(
    *,
    records_sha256: str,
    expected_tag: str,
    actual_records_sha256: str,
    actual_tag: str,
) -> None:
    records_match = hmac.compare_digest(records_sha256, actual_records_sha256)
    tag_matches = hmac.compare_digest(expected_tag, actual_tag)
    if not records_match or not tag_matches:
        raise ValueError("private records commitment is invalid")


def private_records_commitment_identity(provenance: dict) -> dict:
    """Project the closed, versioned attempt identity committed by the HMAC."""

    if not isinstance(provenance, dict) or any(
        field not in provenance for field in _IDENTITY_FIELDS
    ):
        raise ValueError("private records commitment identity is invalid")
    identity = {
        "scheme": PRIVATE_RECORDS_COMMITMENT_SCHEME,
        **{field: provenance[field] for field in _IDENTITY_FIELDS},
    }
    try:
        json.dumps(
            identity,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("private records commitment identity is invalid") from error
    return identity


def _stream_commitment(
    records_path: Path,
    provenance_identity: dict,
    *,
    key: bytes,
) -> tuple[str, str]:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("private records commitment key is invalid")
    identity = private_records_commitment_identity(provenance_identity)
    identity_bytes = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256()
    commitment = hmac.new(key, digestmod=hashlib.sha256)
    commitment.update(_COMMITMENT_DOMAIN)
    commitment.update(struct.pack(">Q", len(identity_bytes)))
    commitment.update(identity_bytes)
    with records_path.open("rb") as handle:
        initial = os.fstat(handle.fileno())
        commitment.update(struct.pack(">Q", initial.st_size))
        consumed = 0
        while chunk := handle.read(1024 * 1024):
            consumed += len(chunk)
            digest.update(chunk)
            commitment.update(chunk)
        final = os.fstat(handle.fileno())
    if (
        consumed != initial.st_size
        or final.st_size != initial.st_size
        or final.st_mtime_ns != initial.st_mtime_ns
    ):
        raise OSError("private records changed while commitment was created")
    return digest.hexdigest(), commitment.hexdigest()


def _bytes_commitment(
    records_bytes: bytes,
    provenance_identity: dict,
    *,
    key: bytes,
) -> tuple[str, str]:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("private records commitment key is invalid")
    identity = private_records_commitment_identity(provenance_identity)
    identity_bytes = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(records_bytes)
    commitment = hmac.new(key, digestmod=hashlib.sha256)
    commitment.update(_COMMITMENT_DOMAIN)
    commitment.update(struct.pack(">Q", len(identity_bytes)))
    commitment.update(identity_bytes)
    commitment.update(struct.pack(">Q", len(records_bytes)))
    commitment.update(records_bytes)
    return digest.hexdigest(), commitment.hexdigest()
