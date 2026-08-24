"""Create an ignored, candidate-blinded OCR judging packet."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import random
import re
import secrets
import subprocess
import uuid
from pathlib import Path

from local_inference_bench.fingerprint import fingerprint_json
from local_inference_bench.load_sustained_workload import load_sustained_workload


_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_LINE_CHARACTERS = 10_000
_MAX_RECORD_CHARACTERS = 1_000_000
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = "private-ocr-blind-v6"
_PREPARATION_PRODUCER_PATHS = (
    "scripts/prepare_blind_ocr_comparison.py",
    "src/local_inference_bench/fingerprint.py",
    "src/local_inference_bench/load_sustained_workload.py",
)
_SOURCE_STATUSES = {"succeeded", "partial_failure", "all_failed"}
_VARIANT_SPEC = re.compile(
    r"^(?P<variant>[a-z0-9][a-z0-9_-]{0,63})@"
    r"(?P<candidate>[a-z0-9][a-z0-9_-]{0,63})#"
    r"(?P<config>0|[1-9][0-9]*)$"
)


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
    args = parser.parse_args()

    _require_ignored_output_directory(args.output_dir)
    _require_new_output_directory(args.output_dir)
    variant_specs = _parse_candidate_specs(args.candidate)
    workload_path = args.workload.resolve(strict=True)
    workload = load_sustained_workload(workload_path, expected_task="ocr")
    records = {}
    provenances = {}
    resolved_paths = set()
    attempt_ids = set()
    attempt_keys = set()
    for variant_id, spec in variant_specs.items():
        path = spec["path"]
        resolved_path = path.resolve(strict=True)
        if resolved_path in resolved_paths:
            raise ValueError("blind comparison requires distinct record files")
        resolved_paths.add(resolved_path)
        provenance = _verify_records_provenance(
            resolved_path,
            workload_fingerprint=workload["fingerprint"],
        )
        _validate_variant_provenance(variant_id, spec, provenance)
        if provenance["attempt_id"] in attempt_ids:
            raise ValueError("blind comparison requires distinct attempts")
        if provenance["attempt_key"] in attempt_keys:
            raise ValueError("blind comparison requires distinct attempt keys")
        attempt_ids.add(provenance["attempt_id"])
        attempt_keys.add(provenance["attempt_key"])
        provenances[variant_id] = provenance
        variant_records = _read_records(resolved_path)
        _validate_full_workload_record_status(
            variant_records,
            workload["items"],
            provenance["status"],
        )
        records[variant_id] = variant_records
    packet, mapping = build_blind_packet(
        workload,
        records,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    mapping["source_attempts"] = [
        {
            "variant_id": variant_id,
            "candidate_id": provenances[variant_id]["candidate_id"],
            "attempt_id": provenances[variant_id]["attempt_id"],
            "attempt_key": provenances[variant_id]["attempt_key"],
            "config_index": provenances[variant_id]["config_index"],
            "config_fingerprint": fingerprint_json(
                provenances[variant_id]["config"]
            ),
            "trial_index": provenances[variant_id]["trial_index"],
            "attempt_status": provenances[variant_id]["status"],
        }
        for variant_id in sorted(provenances)
    ]
    _seal_blind_packet_mapping(packet, mapping)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    packet_path = args.output_dir / "packet.json"
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    mapping["packet_fingerprint"] = _sha256(packet_path)
    (args.output_dir / "mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(packet_path)


def build_blind_packet(
    workload: dict,
    records: dict[str, dict[str, dict]],
    *,
    sample_count: int,
    seed: int,
    verified_image_bindings: dict[str, dict[str, str]] | None = None,
) -> tuple[dict, dict]:
    if workload.get("task") != "ocr":
        raise ValueError("blind comparison requires an OCR workload")
    if workload.get("workload_class") != "private_course":
        raise ValueError("blind comparison is restricted to ignored private-course data")
    if len(records) != 2:
        raise ValueError("blind comparison requires exactly two variants")
    if any(_CANDIDATE_ID.fullmatch(variant_id) is None for variant_id in records):
        raise ValueError("blind comparison variant IDs must be public identifiers")
    items = workload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("workload requires nonempty items")
    if not 1 <= sample_count <= min(len(items), 100):
        raise ValueError("sample_count is outside the workload")

    variant_ids = sorted(records)
    rng = random.Random(seed)
    selected = rng.sample(items, sample_count)
    selected_ids = [item.get("id") for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("blind comparison workload IDs must be unique")
    packet_samples = []
    mapping_samples = []
    for index, item in enumerate(selected, start=1):
        sample_id = item.get("id")
        image_path = item.get("path")
        if not isinstance(sample_id, str) or not isinstance(image_path, str):
            raise ValueError("workload item is missing an opaque ID or path")
        if verified_image_bindings is None:
            resolved_image_path = Path(image_path)
            if not resolved_image_path.is_absolute() or not resolved_image_path.is_file():
                raise ValueError("blind comparison requires resolved workload images")
            image_sha256 = _sha256(resolved_image_path)
        else:
            binding = verified_image_bindings.get(sample_id)
            if (
                not isinstance(binding, dict)
                or set(binding) != {"path", "content_sha256"}
                or type(binding.get("path")) is not str
                or not binding["path"]
                or not Path(binding["path"]).is_absolute()
                or type(binding.get("content_sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", binding["content_sha256"])
                is None
            ):
                raise ValueError("verified blind image binding is invalid")
            resolved_image_path = Path(binding["path"])
            image_sha256 = binding["content_sha256"]
        labels = ["A", "B"]
        randomized_variants = list(variant_ids)
        rng.shuffle(randomized_variants)
        blind_id = f"blind_{index:03d}"
        options = {}
        identities = {}
        record_statuses = {}
        for label, variant_id in zip(labels, randomized_variants, strict=True):
            record = records[variant_id].get(sample_id)
            if record is None:
                record_status = "unavailable"
                lines = []
            elif type(record.get("success")) is not bool:
                raise ValueError("blind comparison record success is invalid")
            elif record["success"]:
                record_status = "available"
                lines = record.get("lines")
                if not isinstance(lines, list):
                    raise ValueError("blind comparison successful record has no lines")
            else:
                record_status = "failed"
                lines = []
            options[label] = [str(line.get("text", "")) for line in lines]
            identities[label] = variant_id
            record_statuses[label] = record_status
        packet_samples.append(
            {
                "sample_id": blind_id,
                "image_path": str(resolved_image_path),
                "options": options,
                "option_statuses": dict(record_statuses),
            }
        )
        mapping_samples.append(
            {
                "sample_id": blind_id,
                "source_id": sample_id,
                "identities": identities,
                "record_statuses": dict(record_statuses),
                "image_sha256": image_sha256,
            }
        )
    return (
        {
            "schema_version": 1,
            "protocol": _PROTOCOL,
            "instructions": {
                "winner_values": ["A", "B", "tie"],
                "severity_scale": [0, 1, 2, 3],
                "allowed_error_codes": sorted(_ALLOWED_ERROR_CODES),
                "option_status_values": ["available", "failed", "unavailable"],
            },
            "samples": packet_samples,
        },
        {
            "schema_version": 1,
            "protocol": _PROTOCOL,
            "variant_ids": variant_ids,
            "samples": mapping_samples,
        },
    )


def _seal_blind_packet_mapping(
    packet: dict,
    mapping: dict,
    *,
    nonce: str | None = None,
) -> None:
    """Bind the private identity mapping to the exact packet shown to judges."""

    resolved_nonce = secrets.token_hex(32) if nonce is None else nonce
    if (
        type(resolved_nonce) is not str
        or re.fullmatch(r"[0-9a-f]{64}", resolved_nonce) is None
    ):
        raise ValueError("blind mapping commitment nonce is invalid")
    mapping["preparation_producer_sha256"] = _preparation_producer_sha256()
    _validate_selected_record_statuses_against_attempts(mapping)
    commitment = _mapping_commitment(mapping, resolved_nonce)
    mapping["mapping_commitment_nonce"] = resolved_nonce
    mapping["mapping_commitment"] = commitment
    packet["mapping_commitment"] = commitment


def _validate_selected_record_statuses_against_attempts(mapping: dict) -> None:
    """Reject selected records that contradict their source attempt status."""

    payload = _mapping_commitment_payload(mapping)
    statuses_by_variant = {variant_id: [] for variant_id in payload["variant_ids"]}
    for sample in payload["samples"]:
        for label, variant_id in sample["identities"].items():
            statuses_by_variant[variant_id].append(sample["record_statuses"][label])
    for source in payload["source_attempts"]:
        statuses = statuses_by_variant.get(source["variant_id"], [])
        if source["attempt_status"] == "succeeded" and any(
            status != "available" for status in statuses
        ):
            raise ValueError(
                "blind comparison selected records contradict attempt status"
            )
        if source["attempt_status"] == "all_failed" and any(
            status == "available" for status in statuses
        ):
            raise ValueError(
                "blind comparison selected records contradict attempt status"
            )


def _mapping_commitment(mapping: dict, nonce: str) -> str:
    canonical = json.dumps(
        _mapping_commitment_payload(mapping),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        bytes.fromhex(nonce),
        b"private-ocr-blind-v6-mapping\0" + canonical,
        hashlib.sha256,
    ).hexdigest()


def _mapping_commitment_payload(mapping: dict) -> dict:
    try:
        variant_ids = sorted(mapping["variant_ids"])
        samples = sorted(
            (
                {
                    "sample_id": sample["sample_id"],
                    "source_id": sample["source_id"],
                    "identities": {
                        "A": sample["identities"]["A"],
                        "B": sample["identities"]["B"],
                    },
                    "record_statuses": {
                        "A": sample["record_statuses"]["A"],
                        "B": sample["record_statuses"]["B"],
                    },
                    "image_sha256": sample["image_sha256"],
                }
                for sample in mapping["samples"]
            ),
            key=lambda sample: sample["sample_id"],
        )
        source_attempts = sorted(
            (
                {
                    "variant_id": source["variant_id"],
                    "candidate_id": source["candidate_id"],
                    "attempt_id": source["attempt_id"],
                    "attempt_key": source["attempt_key"],
                    "config_index": source["config_index"],
                    "config_fingerprint": source["config_fingerprint"],
                    "trial_index": source["trial_index"],
                    "attempt_status": source["attempt_status"],
                }
                for source in mapping["source_attempts"]
            ),
            key=lambda source: source["variant_id"],
        )
        preparation_producer_sha256 = _validated_preparation_producer_sha256(
            mapping["preparation_producer_sha256"]
        )
    except (KeyError, TypeError) as error:
        raise ValueError("blind mapping cannot be committed") from error
    return {
        "protocol": _PROTOCOL,
        "variant_ids": variant_ids,
        "samples": samples,
        "source_attempts": source_attempts,
        "preparation_producer_sha256": preparation_producer_sha256,
    }


def _preparation_producer_sha256() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[1]
    return {
        relative_path: _sha256(project_root / relative_path)
        for relative_path in _PREPARATION_PRODUCER_PATHS
    }


def _validated_preparation_producer_sha256(value: object) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != set(_PREPARATION_PRODUCER_PATHS)
        or any(
            type(digest) is not str
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for digest in value.values()
        )
    ):
        raise ValueError("blind preparation producer hashes are invalid")
    return {key: value[key] for key in sorted(value)}


def _parse_candidate_specs(values: list[str]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for value in values:
        identity, separator, raw_path = value.partition("=")
        match = _VARIANT_SPEC.fullmatch(identity)
        if not separator or match is None or not raw_path:
            raise ValueError(
                "candidate values must use VARIANT@CANDIDATE_ID#CONFIG_INDEX=PATH"
            )
        variant_id = match.group("variant")
        if variant_id in result:
            raise ValueError("blind comparison variant IDs must be unique")
        result[variant_id] = {
            "candidate_id": match.group("candidate"),
            "config_index": int(match.group("config")),
            "path": Path(raw_path),
        }
    if len(result) != 2:
        raise ValueError("exactly two candidate values are required")
    return result


def _validate_variant_provenance(
    variant_id: str,
    spec: dict,
    provenance: dict,
) -> None:
    if (
        _CANDIDATE_ID.fullmatch(variant_id) is None
        or provenance.get("candidate_id") != spec.get("candidate_id")
        or provenance.get("config_index") != spec.get("config_index")
    ):
        raise ValueError(
            "blind comparison variant does not match provenance candidate/config"
        )


def _require_new_output_directory(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError("blind comparison output directory already exists")


def _require_ignored_output_directory(output_dir: Path) -> None:
    resolved_root = _PROJECT_ROOT.resolve()
    resolved_output = output_dir.resolve()
    try:
        relative_output = resolved_output.relative_to(resolved_root)
    except ValueError:
        return
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(resolved_root),
            "check-ignore",
            "--quiet",
            "--",
            relative_output.as_posix(),
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("blind comparison output inside the repository must be ignored")


def _read_records(path: Path) -> dict[str, dict]:
    records = {}
    total_characters = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"invalid record at line {line_number}")
            sample_id = record.get("sample_id")
            success = record.get("success")
            lines = record.get("lines")
            if (
                not isinstance(sample_id, str)
                or sample_id in records
                or type(success) is not bool
                or (success and not isinstance(lines, list))
                or (lines is not None and not isinstance(lines, list))
                or len(lines or []) > 10_000
            ):
                raise ValueError(f"invalid record at line {line_number}")
            for output_line in lines or []:
                text = output_line.get("text") if isinstance(output_line, dict) else None
                if not isinstance(text, str) or len(text) > _MAX_LINE_CHARACTERS:
                    raise ValueError(f"invalid record at line {line_number}")
                total_characters += len(text)
                if total_characters > _MAX_RECORD_CHARACTERS:
                    raise ValueError("blind comparison record text budget exceeded")
            records[sample_id] = record
    return records


def _validate_full_workload_record_status(
    records: dict[str, dict],
    workload_items: list[dict],
    attempt_status: str,
) -> dict[str, int]:
    """Require runner provenance status to match every workload record."""

    workload_ids = [item.get("id") for item in workload_items]
    if (
        not workload_ids
        or any(type(sample_id) is not str for sample_id in workload_ids)
        or len(workload_ids) != len(set(workload_ids))
        or set(records) - set(workload_ids)
    ):
        raise ValueError("blind comparison records do not match the workload")
    available = sum(
        records.get(sample_id, {}).get("success") is True
        for sample_id in workload_ids
    )
    failed = sum(
        sample_id in records and records[sample_id].get("success") is False
        for sample_id in workload_ids
    )
    unavailable = len(workload_ids) - available - failed
    derived_status = (
        "succeeded"
        if available == len(workload_ids)
        else "all_failed" if available == 0 else "partial_failure"
    )
    if attempt_status != derived_status:
        raise ValueError(
            "blind comparison provenance status does not match workload records"
        )
    return {
        "total": len(workload_ids),
        "available": available,
        "failed": failed,
        "unavailable": unavailable,
    }


def _verify_records_provenance(
    records_path: Path,
    *,
    workload_fingerprint: str,
) -> dict:
    provenance = json.loads(
        records_path.with_name("records-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    candidate_id = provenance.get("candidate_id")
    if (
        provenance.get("schema_version") != 1
        or provenance.get("protocol") != "sustained-process-v1"
        or type(provenance.get("status")) is not str
        or provenance["status"] not in _SOURCE_STATUSES
        or provenance.get("task") != "ocr"
        or provenance.get("phase") not in {"quality", "compatibility"}
        or provenance.get("workload_class") != "private_course"
        or provenance.get("workload_fingerprint") != workload_fingerprint
        or provenance.get("records_sha256") != _sha256(records_path)
        or type(candidate_id) is not str
        or _CANDIDATE_ID.fullmatch(candidate_id) is None
        or not isinstance(provenance.get("config"), dict)
    ):
        raise ValueError("blind comparison records provenance is invalid")
    try:
        uuid.UUID(provenance.get("attempt_id", ""))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("blind comparison records provenance is invalid") from error
    for key in ("attempt_key", "code_fingerprint", "environment_fingerprint"):
        if (
            type(provenance.get(key)) is not str
            or re.fullmatch(r"[0-9a-f]{16}", provenance[key]) is None
        ):
            raise ValueError("blind comparison records provenance is invalid")
    for key in ("config_index", "trial_index"):
        if type(provenance.get(key)) is not int or provenance[key] < 0:
            raise ValueError("blind comparison records provenance is invalid")
    return provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ALLOWED_ERROR_CODES = {
    "false_positive",
    "formula_or_code",
    "garble",
    "missing_text",
    "reading_order",
    "small_text",
}


if __name__ == "__main__":
    main()
