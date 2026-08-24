"""Prepare a HMAC-anchored, ignored blind OCR judging packet."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import secrets
import uuid
from pathlib import Path

from local_inference_bench.event_journal import (
    append_event_once,
    locked_file_bytes,
    locked_journal_bytes,
)
from local_inference_bench.fingerprint import fingerprint_json
from local_inference_bench.load_sustained_workload import load_sustained_workload
from local_inference_bench.load_verified_private_ocr_source import (
    capture_verified_private_ocr_authority,
    load_verified_private_ocr_source,
    VerifiedPrivateOcrAuthoritySnapshot,
    verify_private_ocr_authority_is_current,
)
from local_inference_bench.project_paths import (
    QUALITY_EVENTS_PATH,
    SUSTAINED_EVENTS_PATH,
    SUSTAINED_REGISTRY_PATH,
)
from local_inference_bench.verified_blind_ocr_protocol import (
    COMMITMENT_SCHEME,
    PRECOMMIT_PROTOCOL,
    PREPARATION_PRIVACY,
    PROTOCOL,
    packet_commitment_payload,
    public_event_sha256,
    utc_timestamp_now,
    validate_preparation_event,
)
from scripts.prepare_blind_ocr_comparison import (
    _parse_candidate_specs,
    _require_ignored_output_directory,
    _require_new_output_directory,
    _validate_full_workload_record_status,
    build_blind_packet,
)


_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 512 * 1024 * 1024
_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_APPEND_TICKET_KEY = secrets.token_bytes(32)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_PATHS = (
    "registries/sustained_candidates.json",
    "scripts/prepare_verified_blind_ocr_comparison.py",
    "scripts/prepare_blind_ocr_comparison.py",
    "src/local_inference_bench/event_journal.py",
    "src/local_inference_bench/fingerprint.py",
    "src/local_inference_bench/journal_integrity.py",
    "src/local_inference_bench/load_registry.py",
    "src/local_inference_bench/load_sustained_workload.py",
    "src/local_inference_bench/load_verified_private_ocr_source.py",
    "src/local_inference_bench/private_records_commitment.py",
    "src/local_inference_bench/project_paths.py",
    "src/local_inference_bench/verified_blind_ocr_protocol.py",
)


class _AuthorizedBlindOcrPreparationEvent(dict):
    """Carry non-serialized packet authority from preparation to publication."""

    def __init__(
        self,
        event: dict,
        *,
        authority_snapshot: VerifiedPrivateOcrAuthoritySnapshot,
        append_ticket: str,
    ):
        super().__init__(event)
        self.authority_snapshot = authority_snapshot
        self.append_ticket = append_ticket


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        help=(
            "Exactly two VARIANT@CANDIDATE_ID#CONFIG_INDEX="
            "private-records.jsonl values."
        ),
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=285)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--append-journal", required=True, type=Path)
    args = parser.parse_args()

    if args.append_journal.resolve() != QUALITY_EVENTS_PATH.resolve():
        raise ValueError("verified blind OCR preparation requires the canonical quality journal")
    packet_path, preparation_event = prepare_verified_blind_packet(
        workload_path=args.workload,
        candidate_specs=_parse_candidate_specs(args.candidate),
        sample_count=args.sample_count,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    _append_preparation_event_once(args.append_journal, preparation_event)
    print(packet_path)


def prepare_verified_blind_packet(
    *,
    workload_path: Path,
    candidate_specs: dict[str, dict],
    sample_count: int,
    seed: int,
    output_dir: Path,
) -> tuple[Path, dict]:
    """Prepare against the repository's canonical public authorities."""

    return _prepare_verified_blind_packet(
        workload_path=workload_path,
        candidate_specs=candidate_specs,
        sample_count=sample_count,
        seed=seed,
        output_dir=output_dir,
        sustained_events_path=SUSTAINED_EVENTS_PATH,
        registry_path=SUSTAINED_REGISTRY_PATH,
    )


def _prepare_verified_blind_packet(
    *,
    workload_path: Path,
    candidate_specs: dict[str, dict],
    sample_count: int,
    seed: int,
    output_dir: Path,
    sustained_events_path: Path,
    registry_path: Path,
) -> tuple[Path, dict]:
    """Build the packet and return its privacy-safe public preparation event."""

    _require_ignored_output_directory(output_dir)
    _require_new_output_directory(output_dir)
    if (
        type(sample_count) is not int
        or not 1 <= sample_count <= 100
        or type(seed) is not int
        or not 0 <= seed <= 2**63 - 1
    ):
        raise ValueError("verified blind OCR sample count or seed is invalid")
    resolved_workload_path = workload_path.resolve(strict=True)
    workload = load_sustained_workload(resolved_workload_path, expected_task="ocr")
    if workload["workload_class"] != "private_course":
        raise ValueError("verified blind OCR requires a private_course workload")
    image_snapshots = _load_bound_image_snapshots(workload, output_dir)
    authority_snapshot = capture_verified_private_ocr_authority(
        sustained_events_path=sustained_events_path,
        registry_path=registry_path,
    )

    sources = {}
    resolved_record_paths = set()
    source_identities = set()
    attempt_ids = set()
    for variant_id, spec in candidate_specs.items():
        verified = load_verified_private_ocr_source(
            spec["path"],
            expected_workload_fingerprint=workload["fingerprint"],
            expected_workload_summary=workload["public_summary"],
            sustained_events_path=sustained_events_path,
            registry_path=registry_path,
            authority_snapshot=authority_snapshot,
        )
        provenance = verified["provenance"]
        registered = verified["registered_source"]
        if (
            registered["candidate_id"] != spec["candidate_id"]
            or registered["config_index"] != spec["config_index"]
        ):
            raise ValueError("verified blind source does not match its CLI identity")
        resolved_path = verified["records_path"]
        source_identity = (registered["candidate_id"], registered["config_index"])
        if (
            resolved_path in resolved_record_paths
            or source_identity in source_identities
            or provenance["attempt_id"] in attempt_ids
        ):
            raise ValueError("verified blind OCR sources must be distinct")
        resolved_record_paths.add(resolved_path)
        source_identities.add(source_identity)
        attempt_ids.add(provenance["attempt_id"])
        _validate_full_workload_record_status(
            verified["records"],
            workload["items"],
            provenance["status"],
        )
        sources[variant_id] = verified

    packet, mapping = build_blind_packet(
        workload,
        {variant_id: source["records"] for variant_id, source in sources.items()},
        sample_count=sample_count,
        seed=seed,
        verified_image_bindings={
            sample_id: {
                "path": str(snapshot["path"]),
                "content_sha256": snapshot["content_sha256"],
            }
            for sample_id, snapshot in image_snapshots.items()
        },
    )
    preparation_id = str(uuid.uuid4())
    packet["schema_version"] = 2
    packet["protocol"] = PROTOCOL
    packet["preparation_id"] = preparation_id
    mapping["schema_version"] = 2
    mapping["protocol"] = PROTOCOL
    mapping["preparation_id"] = preparation_id
    mapping["source_bindings"] = [
        {
            "variant_id": variant_id,
            "attempt_id": source["provenance"]["attempt_id"],
            "private_artifact_commitment": {
                "scheme": source["provenance"]["private_records_commitment"][
                    "scheme"
                ],
                "hmac_sha256": source["provenance"]["private_records_commitment"][
                    "hmac_sha256"
                ],
            },
        }
        for variant_id, source in sorted(sources.items())
    ]
    mapping["preparation_producer_sha256"] = _producer_sha256(
        registry_bytes=authority_snapshot.registry_bytes
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    for sample in mapping["samples"]:
        snapshot = image_snapshots[sample["source_id"]]
        snapshot["path"].parent.mkdir(parents=True, exist_ok=True)
        snapshot["path"].write_bytes(snapshot["bytes"])
    snapshots_root = output_dir / "source-snapshots"
    for variant_id, source in sorted(sources.items()):
        snapshot_dir = snapshots_root / variant_id
        snapshot_dir.mkdir(parents=True)
        (snapshot_dir / "private-records.jsonl").write_bytes(source["records_bytes"])
        (snapshot_dir / "records-provenance.json").write_bytes(
            source["provenance_bytes"]
        )

    packet_path = output_dir / "packet.json"
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    mapping["packet_fingerprint"] = _sha256(packet_path)
    private_commitment = _create_private_packet_commitment(mapping, packet)
    mapping["private_packet_commitment"] = private_commitment
    mapping_path = output_dir / "mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_private_ocr_authority_is_current(authority_snapshot)
    preparation_event = _preparation_event(mapping, packet)
    return packet_path, _AuthorizedBlindOcrPreparationEvent(
        preparation_event,
        authority_snapshot=authority_snapshot,
        append_ticket=_preparation_append_ticket(
            preparation_event,
            authority_snapshot,
        ),
    )


def _create_private_packet_commitment(mapping: dict, packet: dict) -> dict:
    key = secrets.token_bytes(32)
    payload = packet_commitment_payload(mapping, packet)
    tag = hmac.new(
        key,
        b"private-ocr-blind-packet-v10\0" + payload,
        hashlib.sha256,
    ).hexdigest()
    return {
        "scheme": COMMITMENT_SCHEME,
        "key_hex": key.hex(),
        "hmac_sha256": tag,
    }


def _preparation_event(mapping: dict, packet: dict) -> dict:
    status_counts = {"available": 0, "failed": 0, "unavailable": 0}
    for sample in mapping["samples"]:
        for status in sample["record_statuses"].values():
            status_counts[status] += 1
    private_commitment = mapping["private_packet_commitment"]
    event = {
        "event": "blind_ocr_packet_prepared",
        "protocol": PRECOMMIT_PROTOCOL,
        "candidate_id": "private_course_blind_ocr_comparison",
        "workload_class": "private_course",
        "preparation_id": mapping["preparation_id"],
        "mapping_protocol": PROTOCOL,
        "sample_count": len(packet["samples"]),
        "source_count": len(mapping["source_bindings"]),
        "selected_source_status_counts": status_counts,
        "producer_fingerprint": fingerprint_json(
            mapping["preparation_producer_sha256"]
        ),
        "private_packet_commitment": {
            "scheme": COMMITMENT_SCHEME,
            "hmac_sha256": private_commitment["hmac_sha256"],
        },
        "privacy": dict(PREPARATION_PRIVACY),
    }
    event["public_event_sha256"] = public_event_sha256(event)
    return {**event, "timestamp_utc": utc_timestamp_now()}


def _append_preparation_event_once(path: Path, event: dict) -> bool:
    if (
        not isinstance(event, _AuthorizedBlindOcrPreparationEvent)
        or type(getattr(event, "append_ticket", None)) is not str
        or not hmac.compare_digest(
            event.append_ticket,
            _preparation_append_ticket(event, event.authority_snapshot),
        )
    ):
        raise ValueError("verified blind OCR preparation was not authorized")
    with locked_file_bytes(event.authority_snapshot.registry_path) as registry_bytes:
        with locked_journal_bytes(
            event.authority_snapshot.sustained_events_path
        ) as sustained_events_bytes:
            validate_preparation_event(event)
            verify_private_ocr_authority_is_current(
                event.authority_snapshot,
                sustained_events_bytes=sustained_events_bytes,
                registry_bytes=registry_bytes,
            )
            return append_event_once(
                path,
                event,
                identity_fields=("event", "protocol", "preparation_id"),
            )


def _preparation_append_ticket(
    event: dict,
    authority_snapshot: VerifiedPrivateOcrAuthoritySnapshot,
) -> str:
    payload = {
        "event": event,
        "registry_sha256": hashlib.sha256(authority_snapshot.registry_bytes).hexdigest(),
        "sustained_events_sha256": authority_snapshot.sustained_events_sha256,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    return hmac.new(_APPEND_TICKET_KEY, canonical, hashlib.sha256).hexdigest()


def _load_bound_image_snapshots(workload: dict, output_dir: Path) -> dict[str, dict]:
    bindings = workload.get("item_content_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("verified blind OCR workload bindings are invalid")
    resolved_output_dir = output_dir.resolve()
    snapshots = {}
    total_bytes = 0
    for item in workload["items"]:
        sample_id = item["id"]
        binding = bindings.get(sample_id)
        size_bytes = binding.get("size_bytes") if isinstance(binding, dict) else None
        content_sha256 = (
            binding.get("content_sha256") if isinstance(binding, dict) else None
        )
        if (
            type(size_bytes) is not int
            or not 0 < size_bytes <= _MAX_IMAGE_BYTES
            or type(content_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
        ):
            raise ValueError("verified blind OCR workload image binding is invalid")
        try:
            with Path(item["path"]).open("rb") as handle:
                image_bytes = handle.read(size_bytes + 1)
        except OSError as error:
            raise ValueError("verified blind OCR workload image is unavailable") from error
        if (
            len(image_bytes) != size_bytes
            or not hmac.compare_digest(
                hashlib.sha256(image_bytes).hexdigest(),
                content_sha256,
            )
        ):
            raise ValueError("verified blind OCR workload image changed after binding")
        total_bytes += len(image_bytes)
        if total_bytes > _MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("verified blind OCR workload image byte budget exceeded")
        suffix = Path(item["path"]).suffix.casefold()
        if suffix not in _IMAGE_SUFFIXES:
            suffix = ".image"
        snapshots[sample_id] = {
            "bytes": image_bytes,
            "content_sha256": content_sha256,
            "path": (resolved_output_dir / "image-snapshots" / f"{sample_id}{suffix}"),
        }
    return snapshots


def _producer_sha256(*, registry_bytes: bytes | None = None) -> dict[str, str]:
    result = {path: _sha256(PROJECT_ROOT / path) for path in PRODUCER_PATHS}
    if registry_bytes is not None:
        result["registries/sustained_candidates.json"] = hashlib.sha256(
            registry_bytes
        ).hexdigest()
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
