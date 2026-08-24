"""Verify one private b10598 run and emit an aggregate-only public v3 event."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from local_inference_bench.bounded_vlm_assets import (  # noqa: E402
    load_and_verify_candidate_assets,
    sha256_file,
)
from local_inference_bench.load_sustained_workload import (  # noqa: E402
    load_sustained_workload,
)
from local_inference_bench.score_document_fidelity import (  # noqa: E402
    _canonical_markdown,
    _score_sample as score_document_sample,
)
from local_inference_bench.score_ocr_quality import (  # noqa: E402
    _aggregate_scores as aggregate_ocr_scores,
    _score_sample as score_ocr_sample,
)
from local_inference_bench.validate_public_summary import (  # noqa: E402
    validate_public_string_privacy,
)
from scripts.run_bounded_vlm_b10598_quality import (  # noqa: E402
    CANDIDATES,
    PROTOCOL as RUN_PROTOCOL,
    _controller_environment_fingerprint,
    _producer_hashes,
    _run_evidence_hashes,
    _verify_controller_environment,
)


EVENT_PROTOCOL = "bounded-community-screen-v4"
EVENT_NAME = "bounded_candidate_quality_verified"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_REVISION = "b10598-56db501e73cfb10c8fcce61be708f5c3ee749271"
_FORBIDDEN_PUBLIC_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "credential",
    "email",
    "password",
    "path",
    "paths",
    "phone",
    "prediction",
    "predictions",
    "raw_output",
    "raw_response",
    "secret",
    "transcript",
    "username",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=tuple(CANDIDATES))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    event = build_bounded_vlm_v3_event(
        candidate_id=args.candidate,
        run_dir=args.run_dir,
    )
    if args.output.exists():
        raise FileExistsError("v3 event builder refuses to overwrite output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(event, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(event, indent=2, sort_keys=True))


def build_bounded_vlm_v3_event(
    *,
    candidate_id: str,
    run_dir: Path,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    """Recompute the fixed quality gate after checking every bound identity."""

    root = project_root.resolve()
    artifact_root = run_dir.resolve()
    if not artifact_root.is_relative_to((root / "results" / "artifacts").resolve()):
        raise ValueError("bounded VLM run must be under ignored artifacts")
    _verify_controller_environment()
    assets = load_and_verify_candidate_assets(
        project_root=root,
        candidate_id=candidate_id,
    )
    paths = {
        "request": artifact_root / "request.json",
        "response": artifact_root / "response.json",
        "records": artifact_root / "private-records.jsonl",
        "monitor_summary": artifact_root / "monitor-summary.json",
        "provenance": artifact_root / "run-provenance.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("bounded VLM run is incomplete")

    request = _read_json_object(paths["request"], "request")
    response = _read_json_object(paths["response"], "response")
    monitor = _read_json_object(paths["monitor_summary"], "monitor summary")
    provenance = _read_json_object(paths["provenance"], "run provenance")
    records = _read_records(paths["records"])
    workload = load_sustained_workload(
        assets["fixtures"]["manifest"]["path"],
        expected_task="ocr",
    )
    _validate_request(
        request,
        candidate_id=candidate_id,
        artifact_root=artifact_root,
        workload=workload,
        assets=assets,
    )
    summary = _validate_response(
        response,
        candidate_id=candidate_id,
        expected_count=len(workload["items"]),
        records=records,
    )
    _validate_monitor(monitor)
    _validate_provenance(
        provenance,
        candidate_id=candidate_id,
        assets=assets,
        paths=paths,
        project_root=root,
    )
    expected_ids = assets["candidate"]["sample_ids"]
    if list(records) != expected_ids:
        raise ValueError("bounded VLM records changed item order or identity")

    if candidate_id == "ovisocr2_q8_cpu":
        fidelity_metrics = _score_ovis(records, assets=assets)
    elif candidate_id == "hunyuanocr_1_5_gguf_cpu":
        fidelity_metrics = _score_hunyuan(records, assets=assets)
    else:  # pragma: no cover - argparse and the asset registry both reject this
        raise ValueError("unsupported bounded VLM candidate")
    quality_gate_pass = _quality_gate_passes(
        metrics=fidelity_metrics,
        gate=assets["candidate"]["quality_gate"],
    )
    outcome_kind = (
        "quality_gate_passed"
        if quality_gate_pass
        else assets["candidate"]["failed_outcome_kind"]
    )

    event = {
        "event": EVENT_NAME,
        "protocol": EVENT_PROTOCOL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "workload_class": "generated_quality_control",
        "fixture_disclosure": (
            "Tracked deterministic public controls contain invented benchmark "
            "content only; they are not lecture-derived."
        ),
        "model": _public_model_identity(candidate_id, assets=assets),
        "provenance": _public_provenance(
            candidate_id=candidate_id,
            provenance=provenance,
            assets=assets,
            project_root=root,
        ),
        "result": {
            "outcome_kind": outcome_kind,
            "quality_gate_pass": quality_gate_pass,
            "runtime_completed_without_implementation_failure": (
                summary["counts"]["completed"] == len(expected_ids)
                and summary["counts"]["failed"] == 0
                and fidelity_metrics["token_cap_hit_count"] == 0
            ),
            "counts": dict(summary["counts"]),
            "timing": _finite_allowlist(
                summary.get("timing", {}),
                (
                    "load_seconds_mean",
                    "warmup_seconds_mean",
                    "steady_wall_seconds",
                    "latency_seconds_p50",
                    "latency_seconds_p95",
                    "latency_seconds_max",
                    "load_seconds",
                    "prompt_eval_seconds",
                    "generation_seconds",
                    "total_seconds",
                    "image_encode_seconds",
                    "image_decode_seconds",
                ),
            ),
            "throughput_images_per_hour": _throughput(summary),
            "generation": _public_generation(summary),
            "resources": _public_resources(monitor),
            "fidelity": fidelity_metrics,
        },
    }
    validate_public_v3_event(event, project_root=root)
    return event


def validate_public_v3_event(event: object, *, project_root: Path) -> None:
    """Reject accidental local paths, raw output, nonfinite numbers, or short hashes."""

    if not isinstance(event, dict):
        raise ValueError("bounded VLM public event must be an object")

    def visit(
        value: object,
        key: str | None = None,
        *,
        hash_context: bool = False,
    ) -> None:
        if key is not None and key.casefold() in _FORBIDDEN_PUBLIC_KEYS:
            raise ValueError(f"bounded VLM public event exposes forbidden key: {key}")
        child_hash_context = hash_context or (
            key is not None and key.endswith("sha256")
        )
        if isinstance(value, dict):
            for child_key, child in value.items():
                if (
                    type(child_key) is not str
                    or not child_key
                    or len(child_key) > 256
                    or "\r" in child_key
                    or "\n" in child_key
                ):
                    raise ValueError("bounded VLM public event keys must be bounded strings")
                if not child_hash_context:
                    validate_public_string_privacy(child_key, key="field_name")
                visit(child, child_key, hash_context=child_hash_context)
        elif isinstance(value, list):
            for child in value:
                visit(child, key, hash_context=child_hash_context)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("bounded VLM public event contains nonfinite metrics")
        elif isinstance(value, str):
            if len(value) > 512 or "\r" in value or "\n" in value:
                raise ValueError("bounded VLM public event contains an unbounded string")
            validate_public_string_privacy(value, key=key or "sequence_value")
            if child_hash_context and _SHA256.fullmatch(value) is None:
                raise ValueError("bounded VLM public event contains a non-full hash")

    visit(event)


def _score_ovis(records: dict[str, dict], *, assets: dict) -> dict:
    manifest = _read_json_object(
        assets["fixtures"]["manifest"]["path"], "Ovis fixture manifest"
    )
    sample_id = assets["candidate"]["sample_ids"][0]
    reference = manifest.get("references", {}).get(sample_id)
    if not isinstance(reference, dict):
        raise ValueError("Ovis fixture reference is missing")
    _verify_reference_media_hash(reference, sample_id=sample_id, assets=assets)
    record = records[sample_id]
    prediction = record["prediction"] if record["success"] else ""
    score = score_document_sample(
        reference,
        _canonical_markdown(prediction),
        record,
        edit_budget=[5_000_000],
    )
    return {
        "sample_count": 1,
        "failure_count": int(score["failed"]),
        "token_cap_hit_count": int(score["token_cap_hit"]),
        "reading_order_pair_accuracy": _ratio(score["anchor_hits"], score["anchor_pairs"]),
        "protected_span_recall": _ratio(
            score["protected_hits"], score["protected_total"]
        ),
        "lexical_recall": _ratio(score["lexical_hits"], score["lexical_expected"]),
        "lexical_precision": _ratio(
            score["lexical_hits"], score["lexical_predicted"]
        ),
        "table_cell_exact_fraction": _ratio(
            score["table_cell_hits"], score["table_cell_total"]
        ),
        "complete_html_table_count": int(record.get("complete_html_table") is True),
    }


def _score_hunyuan(records: dict[str, dict], *, assets: dict) -> dict:
    manifest = _read_json_object(
        assets["fixtures"]["manifest"]["path"], "Hunyuan fixture manifest"
    )
    references = manifest.get("references")
    if not isinstance(references, dict) or set(references) != set(records):
        raise ValueError("Hunyuan fixture references do not match records")
    edit_budget = [5_000_000]
    scores = []
    for sample_id, record in records.items():
        reference = references[sample_id]
        _verify_reference_media_hash(reference, sample_id=sample_id, assets=assets)
        lines = record.get("lines", []) if record["success"] else []
        scores.append(
            score_ocr_sample(
                category=reference["category"],
                reference_lines=reference["lines"],
                required_tokens=reference["required_tokens"],
                predicted_lines=[line["text"] for line in lines],
                confidences=[],
                failed=not record["success"],
                edit_budget=edit_budget,
            )
        )
    aggregate = aggregate_ocr_scores(scores)
    overall = aggregate["overall"]
    negative = aggregate["categories"].get("negative", {})
    return {
        "sample_count": overall["sample_count"],
        "failure_count": overall["failure_count"],
        "token_cap_hit_count": sum(
            int(record["token_cap_hit"]) for record in records.values()
        ),
        "normalized_character_error_rate": overall[
            "normalized_character_error_rate"
        ],
        "required_token_recall": overall["required_token_recall"],
        "mean_absolute_line_count_error": overall[
            "mean_absolute_line_count_error"
        ],
        "negative_false_positive_characters": negative.get(
            "false_positive_characters", 0
        ),
    }


def _quality_gate_passes(*, metrics: dict, gate: dict) -> bool:
    if metrics["failure_count"] > gate["maximum_failure_count"]:
        return False
    if metrics["token_cap_hit_count"] > gate["maximum_token_cap_hit_count"]:
        return False
    rules = {
        "minimum_reading_order_pair_accuracy": (">=", "reading_order_pair_accuracy"),
        "minimum_protected_span_recall": (">=", "protected_span_recall"),
        "minimum_lexical_recall": (">=", "lexical_recall"),
        "minimum_lexical_precision": (">=", "lexical_precision"),
        "minimum_table_cell_exact_fraction": (">=", "table_cell_exact_fraction"),
        "maximum_normalized_character_error_rate": (
            "<=",
            "normalized_character_error_rate",
        ),
        "minimum_required_token_recall": (">=", "required_token_recall"),
        "maximum_negative_false_positive_characters": (
            "<=",
            "negative_false_positive_characters",
        ),
    }
    for gate_key, (operation, metric_key) in rules.items():
        if gate_key not in gate:
            continue
        if operation == ">=" and metrics[metric_key] < gate[gate_key]:
            return False
        if operation == "<=" and metrics[metric_key] > gate[gate_key]:
            return False
    return True


def _validate_request(
    request: dict,
    *,
    candidate_id: str,
    artifact_root: Path,
    workload: dict,
    assets: dict,
) -> None:
    spec = CANDIDATES[candidate_id]
    if (
        request.get("protocol") != RUN_PROTOCOL
        or request.get("candidate_id") != candidate_id
        or request.get("task") != "ocr"
        or request.get("phase") != "quality"
        or request.get("capture_predictions") is not True
        or request.get("config") != spec["config"]
        or request.get("timeout_seconds") != spec["worker_timeout_seconds"]
    ):
        raise ValueError("bounded VLM request identity is invalid")
    request_workload = request.get("workload")
    if (
        not isinstance(request_workload, dict)
        or request_workload.get("workload_class") != "generated_quality_control"
        or request_workload.get("fingerprint") != workload["fingerprint"]
        or [item.get("id") for item in request_workload.get("items", [])]
        != assets["candidate"]["sample_ids"]
    ):
        raise ValueError("bounded VLM request workload identity is invalid")
    if Path(request.get("response_path", "")).resolve() != artifact_root / "response.json":
        raise ValueError("bounded VLM response path binding is invalid")
    if (
        Path(request.get("private_records_path", "")).resolve()
        != artifact_root / "private-records.jsonl"
    ):
        raise ValueError("bounded VLM records path binding is invalid")


def _validate_response(
    response: dict,
    *,
    candidate_id: str,
    expected_count: int,
    records: dict[str, dict],
) -> dict:
    summary = response.get("public_summary")
    if not isinstance(summary, dict):
        raise ValueError("bounded VLM response summary is missing")
    counts = summary.get("counts")
    if (
        summary.get("candidate_id") != candidate_id
        or summary.get("task") != "ocr"
        or summary.get("runtime_version") != _RUNTIME_REVISION
        or summary.get("workload_class") != "generated_quality_control"
        or not isinstance(counts, dict)
    ):
        raise ValueError("bounded VLM response identity is invalid")
    completed = sum(record["success"] for record in records.values())
    if counts != {
        "attempted": expected_count,
        "completed": completed,
        "failed": expected_count - completed,
    }:
        raise ValueError("bounded VLM response counts do not match records")
    _require_finite_tree(summary)
    return summary


def _validate_monitor(monitor: dict) -> None:
    host = monitor.get("host_telemetry")
    process = monitor.get("process_resources")
    if (
        monitor.get("exit_code") != 0
        or monitor.get("failure_kind") is not None
        or not isinstance(host, dict)
        or not isinstance(process, dict)
        or host.get("status") != "observed"
        or host.get("monitor_partial") is not False
        or host.get("sample_count", 0) < 2
        or process.get("sample_count", 0) < 2
    ):
        raise ValueError("bounded VLM monitoring is incomplete")
    _require_finite_tree(monitor)


def _validate_provenance(
    provenance: dict,
    *,
    candidate_id: str,
    assets: dict,
    paths: dict[str, Path],
    project_root: Path,
) -> None:
    if (
        provenance.get("schema_version") != 2
        or provenance.get("protocol") != RUN_PROTOCOL
        or provenance.get("candidate_id") != candidate_id
        or provenance.get("status") != "succeeded"
    ):
        raise ValueError("bounded VLM run provenance identity is invalid")
    expected = {
        "asset_registry_sha256": sha256_file(assets["registry_path"]),
        "runtime": {
            "archive_sha256": assets["runtime"]["archive"]["sha256"],
            "tree_sha256": assets["runtime"]["tree_fingerprint"]["sha256"],
            "entrypoint_sha256": {
                name: value["sha256"]
                for name, value in sorted(assets["runtime"]["entrypoints"].items())
            },
        },
        "candidate_artifact_sha256": {
            name: value["sha256"]
            for name, value in sorted(assets["artifacts"].items())
        },
        "input": {
            "manifest_sha256": assets["fixtures"]["manifest"]["sha256"],
            "image_sha256": {
                name: value["sha256"]
                for name, value in sorted(assets["fixtures"]["images"].items())
            },
        },
        "producer_sha256": _producer_hashes(candidate_id),
        "controller_environment_fingerprint": (
            _controller_environment_fingerprint()
        ),
        "run_artifact_sha256": _run_evidence_hashes(
            candidate_id=candidate_id,
            artifact_root=paths["request"].parent,
        ),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"bounded VLM provenance mismatch: {key}")
    validate_public_v3_event(expected, project_root=project_root)


def _read_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    total_characters = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid VLM record at line {line_number}")
            sample_id = value.get("sample_id")
            prediction = value.get("prediction")
            lines = value.get("lines")
            if (
                type(sample_id) is not str
                or sample_id in records
                or type(value.get("success")) is not bool
                or type(value.get("token_cap_hit")) is not bool
                or type(value.get("stop_finish")) is not bool
                or type(prediction) is not str
                or not isinstance(lines, list)
            ):
                raise ValueError(f"invalid VLM record at line {line_number}")
            if value["success"] and not prediction.strip():
                raise ValueError("successful VLM record cannot have empty output")
            for output_line in lines:
                if not isinstance(output_line, dict) or type(output_line.get("text")) is not str:
                    raise ValueError(f"invalid VLM line at record {line_number}")
            total_characters += len(prediction)
            if total_characters > 1_000_000:
                raise ValueError("bounded VLM record character budget exceeded")
            records[sample_id] = value
    if not records:
        raise ValueError("bounded VLM records are empty")
    return records


def _public_model_identity(candidate_id: str, *, assets: dict) -> dict:
    candidate = assets["candidate"]
    return {
        "backend": "llama.cpp-cpu",
        "runtime_revision": _RUNTIME_REVISION,
        "upstream_revision": candidate["upstream"]["revision"],
        "artifact_revision": (
            candidate.get("artifact_repository", {}).get("revision")
            if candidate_id == "ovisocr2_q8_cpu"
            else candidate["upstream"]["revision"]
        ),
        "compute_type": candidate["compute_type"],
        "threads": 24,
    }


def _public_provenance(
    *,
    candidate_id: str,
    provenance: dict,
    assets: dict,
    project_root: Path,
) -> dict:
    value = {
        "asset_registry_sha256": provenance["asset_registry_sha256"],
        "runtime_archive_sha256": provenance["runtime"]["archive_sha256"],
        "runtime_tree_sha256": provenance["runtime"]["tree_sha256"],
        "runtime_entrypoint_sha256": provenance["runtime"]["entrypoint_sha256"],
        "candidate_artifact_sha256": provenance["candidate_artifact_sha256"],
        "candidate_lineage_sha256": {
            name: item["sha256"]
            for name, item in sorted(assets["lineage_files"].items())
        },
        "input_manifest_sha256": provenance["input"]["manifest_sha256"],
        "input_image_sha256": sorted(provenance["input"]["image_sha256"].values()),
        "producer_sha256": provenance["producer_sha256"],
        "controller_environment_fingerprint": provenance[
            "controller_environment_fingerprint"
        ],
        "run_artifact_sha256": provenance["run_artifact_sha256"],
        "records_sha256": provenance["run_artifact_sha256"][
            "private-records.jsonl"
        ],
        "public_builder_sha256": sha256_file(Path(__file__).resolve()),
    }
    validate_public_v3_event(value, project_root=project_root)
    return value


def _public_generation(summary: dict) -> dict:
    generation = summary.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("bounded VLM response generation summary is missing")
    return {
        key: value
        for key, value in generation.items()
        if key
        in {
            "max_new_tokens",
            "completion_tokens_total",
            "completion_tokens_max",
            "token_cap_hit_count",
            "length_finish_count",
            "stop_finish_count",
            "latex_marker_count",
            "complete_html_table_count",
            "mean_generated_tokens_per_second",
        }
    }


def _public_resources(monitor: dict) -> dict:
    process = monitor["process_resources"]
    host = monitor["host_telemetry"]
    return {
        **_finite_allowlist(
            process,
            (
                "sample_count",
                "sampled_seconds",
                "peak_rss_bytes",
                "peak_threads",
                "peak_processes",
                "mean_cpu_percent_of_host",
                "p95_cpu_percent_of_host",
                "peak_cpu_percent_of_host",
            ),
        ),
        **_finite_allowlist(
            host,
            (
                "mean_cpu_utility_percent",
                "p95_cpu_utility_percent",
                "mean_rapl_package_power_watts",
                "p95_rapl_package_power_watts",
                "minimum_available_memory_mib",
                "maximum_committed_memory_percent",
                "maximum_performance_limit_flags",
                "maximum_thermal_throttle_reasons",
                "minimum_thermal_passive_limit_percent",
            ),
        ),
        "package_temperature_available": host.get(
            "package_temperature_available", False
        ),
    }


def _throughput(summary: dict) -> float:
    timing = summary.get("timing")
    counts = summary.get("counts")
    if not isinstance(timing, dict) or not isinstance(counts, dict):
        raise ValueError("bounded VLM throughput cannot be derived")
    wall = timing.get("steady_wall_seconds")
    completed = counts.get("completed")
    if (
        type(completed) is not int
        or completed < 0
        or not _is_finite_number(wall)
        or float(wall) <= 0
    ):
        raise ValueError("bounded VLM throughput cannot be derived")
    return completed / float(wall) * 3600.0


def _finite_allowlist(value: object, keys: tuple[str, ...]) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        key: item
        for key in keys
        if (item := value.get(key)) is not None and _is_finite_number(item)
    }


def _require_finite_tree(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _require_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _require_finite_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("bounded VLM evidence contains nonfinite numbers")


def _read_json_object(path: Path, label: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"bounded VLM {label} must be an object")
    return value


def _verify_reference_media_hash(
    reference: object,
    *,
    sample_id: str,
    assets: dict,
) -> None:
    if not isinstance(reference, dict):
        raise ValueError("bounded VLM fixture reference is invalid")
    declared = reference.get("image_sha256")
    image = assets.get("fixtures", {}).get("images", {}).get(sample_id)
    if (
        type(declared) is not str
        or _SHA256.fullmatch(declared) is None
        or not isinstance(image, dict)
        or declared != image.get("sha256")
    ):
        raise ValueError("bounded VLM fixture reference media hash is invalid")


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 1.0


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


if __name__ == "__main__":
    main()
