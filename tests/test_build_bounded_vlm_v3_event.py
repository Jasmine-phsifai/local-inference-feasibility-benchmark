import hashlib
import json
from pathlib import Path

import pytest

import scripts.build_bounded_vlm_v3_event as builder
import scripts.run_bounded_vlm_b10598_quality as runner
from local_inference_bench.bounded_vlm_assets import sha256_file
from local_inference_bench.load_sustained_workload import load_sustained_workload
from scripts.run_bounded_vlm_b10598_quality import build_request


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _ovis_run_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    project_root = tmp_path / "repo"
    fixture_root = project_root / "fixtures"
    fixture_root.mkdir(parents=True)
    image_path = fixture_root / "control.png"
    image_path.write_bytes(b"generated public image")
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    sample_id = "page_008_table_columns"
    expected = "<!-- meta:page number=8 -->\n## Runtime comparison\nA-01 CPU 0.031"
    manifest = {
        "schema_version": 1,
        "task": "ocr",
        "workload_class": "generated_quality_control",
        "warmup_item_id": sample_id,
        "items": [
            {
                "id": sample_id,
                "path": "control.png",
                "expected_text": True,
                "output_marker": "<!-- meta:page number=8 -->",
            }
        ],
        "references": {
            sample_id: {
                "image_sha256": image_sha,
                "marker": "<!-- meta:page number=8 -->",
                "expected_markdown": expected,
                "expected_visible_text": "Runtime comparison\nA-01 CPU 0.031",
                "headings": ["## Runtime comparison"],
                "formulas": [],
                "code_blocks": [],
                "tables": [],
                "ordered_anchors": ["Runtime comparison", "A-01", "0.031"],
                "protected_spans": ["A-01", "CPU", "0.031"],
                "forbidden_spans": ["invented"],
            }
        },
    }
    manifest_path = fixture_root / "manifest.json"
    _write_json(manifest_path, manifest)
    registry_path = project_root / "registry.json"
    _write_json(registry_path, {"fixture": True})
    assets = {
        "registry_path": registry_path,
        "runtime": {
            "archive": {"sha256": "1" * 64},
            "tree_fingerprint": {"sha256": "2" * 64},
            "entrypoints": {
                "llama_mtmd_cli": {"sha256": "3" * 64},
                "llama_server": {"sha256": "4" * 64},
            },
        },
        "artifacts": {
            "model": {"sha256": "5" * 64},
            "projector": {"sha256": "6" * 64},
        },
        "lineage_files": {},
        "fixtures": {
            "manifest": {
                "path": manifest_path,
                "sha256": sha256_file(manifest_path),
            },
            "images": {
                sample_id: {"path": image_path, "sha256": image_sha}
            },
        },
        "candidate": {
            "upstream": {"revision": "upstream-revision"},
            "artifact_repository": {"revision": "artifact-revision"},
            "compute_type": "q8_0_text_bf16_projector",
            "sample_ids": [sample_id],
            "quality_gate": {
                "maximum_failure_count": 0,
                "maximum_token_cap_hit_count": 0,
                "minimum_reading_order_pair_accuracy": 1.0,
                "minimum_protected_span_recall": 1.0,
                "minimum_structure_visible_lexical_recall": 0.9,
                "minimum_structure_visible_lexical_precision": 0.9,
                "minimum_semantic_table_cell_exact_fraction": 0.95,
            },
            "failed_outcome_kind": "experimental_quality_gate_failed",
        },
    }
    monkeypatch.setattr(
        builder,
        "load_and_verify_candidate_assets",
        lambda **_: assets,
    )
    monkeypatch.setattr(builder, "_producer_hashes", lambda _: {"producer": "7" * 64})
    monkeypatch.setattr(builder, "_verify_controller_environment", lambda: None)
    monkeypatch.setattr(
        builder,
        "_controller_environment_fingerprint",
        lambda: "8" * 16,
    )

    run_root = project_root / "results" / "artifacts" / "run-v3"
    run_root.mkdir(parents=True)
    request = build_request(
        candidate_id="ovisocr2_q8_cpu",
        assets=assets,
        response_path=run_root / "response.json",
        records_path=run_root / "private-records.jsonl",
    )
    _write_json(run_root / "request.json", request)
    prediction = "<!-- meta:page number=8 -->\n## Runtime comparison\nA-01"
    record = {
        "sample_id": sample_id,
        "success": True,
        "prediction": prediction,
        "lines": [{"text": line} for line in prediction.splitlines()],
        "token_cap_hit": False,
        "stop_finish": True,
        "complete_html_table": False,
    }
    (run_root / "private-records.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    response = {
        "public_summary": {
            "candidate_id": "ovisocr2_q8_cpu",
            "task": "ocr",
            "runtime_version": builder._RUNTIME_REVISION,
            "workload_class": "generated_quality_control",
            "counts": {"attempted": 1, "completed": 1, "failed": 0},
            "timing": {"steady_wall_seconds": 10.0, "load_seconds": 1.0},
            "generation": {
                "max_new_tokens": 4096,
                "completion_tokens_total": 20,
                "token_cap_hit_count": 0,
                "stop_finish_count": 1,
            },
        }
    }
    _write_json(run_root / "response.json", response)
    monitor = {
        "exit_code": 0,
        "failure_kind": None,
        "wall_seconds": 10.5,
        "process_resources": {
            "sample_count": 20,
            "sampled_seconds": 10.0,
            "peak_rss_bytes": 1234,
            "peak_threads": 24,
            "peak_processes": 2,
            "mean_cpu_percent_of_host": 70.0,
            "p95_cpu_percent_of_host": 80.0,
            "peak_cpu_percent_of_host": 85.0,
        },
        "host_telemetry": {
            "status": "observed",
            "monitor_partial": False,
            "sample_count": 5,
            "package_temperature_available": False,
            "maximum_performance_limit_flags": 0.0,
            "maximum_thermal_throttle_reasons": 0.0,
            "minimum_thermal_passive_limit_percent": 100.0,
        },
    }
    _write_json(run_root / "monitor-summary.json", monitor)
    for name in (
        "host-telemetry.jsonl",
        "process-resources.jsonl",
        "worker.stderr.txt",
        "worker.stdout.txt",
        "ovis.command.json",
        "ovis.llama.log",
        "ovis.stderr.txt",
        "ovis.stdout.txt",
    ):
        (run_root / name).write_text(f"bounded fixture: {name}\n", encoding="utf-8")
    run_artifact_sha256 = builder._run_evidence_hashes(
        candidate_id="ovisocr2_q8_cpu",
        artifact_root=run_root,
    )
    provenance = {
        "schema_version": 2,
        "protocol": builder.RUN_PROTOCOL,
        "candidate_id": "ovisocr2_q8_cpu",
        "status": "succeeded",
        "asset_registry_sha256": sha256_file(registry_path),
        "runtime": {
            "archive_sha256": "1" * 64,
            "tree_sha256": "2" * 64,
            "entrypoint_sha256": {
                "llama_mtmd_cli": "3" * 64,
                "llama_server": "4" * 64,
            },
        },
        "candidate_artifact_sha256": {"model": "5" * 64, "projector": "6" * 64},
        "input": {
            "manifest_sha256": sha256_file(manifest_path),
            "image_sha256": {sample_id: image_sha},
        },
        "producer_sha256": {"producer": "7" * 64},
        "controller_environment_fingerprint": "8" * 16,
        "run_artifact_sha256": run_artifact_sha256,
    }
    _write_json(run_root / "run-provenance.json", provenance)
    return project_root, run_root


def test_v3_builder_emits_only_bound_aggregate_evidence(tmp_path, monkeypatch):
    project_root, run_root = _ovis_run_fixture(tmp_path, monkeypatch)

    event = builder.build_bounded_vlm_v3_event(
        candidate_id="ovisocr2_q8_cpu",
        run_dir=run_root,
        project_root=project_root,
    )

    assert event["result"]["outcome_kind"] == "experimental_quality_gate_failed"
    assert event["result"]["runtime_completed_without_implementation_failure"] is True
    assert "raw_lexical_recall" in event["result"]["fidelity"]
    assert "structure_visible_lexical_recall" in event["result"]["fidelity"]
    assert len(event["provenance"]["records_sha256"]) == 64
    serialized = json.dumps(event)
    assert str(project_root) not in serialized
    assert "A-01" not in serialized
    assert "prediction" not in serialized


def test_v3_builder_rejects_records_changed_after_provenance(tmp_path, monkeypatch):
    project_root, run_root = _ovis_run_fixture(tmp_path, monkeypatch)
    with (run_root / "private-records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="run_artifact_sha256"):
        builder.build_bounded_vlm_v3_event(
            candidate_id="ovisocr2_q8_cpu",
            run_dir=run_root,
            project_root=project_root,
        )


def test_public_validator_rejects_local_paths_raw_output_and_short_hash(tmp_path):
    with pytest.raises(ValueError, match="local path"):
        builder.validate_public_v3_event(
            {"value": r"D:\private\course.png"}, project_root=tmp_path
        )
    with pytest.raises(ValueError, match="forbidden key"):
        builder.validate_public_v3_event(
            {"prediction": "raw text"}, project_root=tmp_path
        )
    with pytest.raises(ValueError, match="non-full hash"):
        builder.validate_public_v3_event(
            {"records_sha256": "abc"}, project_root=tmp_path
        )


@pytest.mark.parametrize(
    "value",
    [
        "/home/alice/private/model.gguf",
        "../private/model.gguf",
        "models/private/model.gguf",
        "alice@example.edu",
        "13800138000",
        "sk-proj-0123456789abcdef",
        "Abcdefghijklmnopqrstuvwxyz0123456789",
        "Authorization: Bearer abcdefghijklmnop",
        "api_key=abcdefghijklmnop",
    ],
)
def test_public_validator_rejects_posix_paths_pii_and_secrets(tmp_path, value):
    with pytest.raises(ValueError):
        builder.validate_public_v3_event({"value": value}, project_root=tmp_path)


def test_public_event_throughput_ignores_contradictory_worker_claim() -> None:
    summary = {
        "counts": {"completed": 2},
        "timing": {"steady_wall_seconds": 10.0},
        "throughput": {"value": 999999.0, "unit": "images_per_hour"},
    }

    assert builder._throughput(summary) == 720.0


def test_public_validator_rejects_secret_bearing_field_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        builder.validate_public_v3_event({"api_key": 123}, project_root=tmp_path)


@pytest.mark.parametrize("wall", [0, -1, float("nan"), float("inf")])
def test_public_event_throughput_requires_positive_finite_wall(wall) -> None:
    with pytest.raises(ValueError, match="cannot be derived"):
        builder._throughput(
            {"counts": {"completed": 1}, "timing": {"steady_wall_seconds": wall}}
        )


def test_semantic_html_table_score_ignores_gfm_separator_row() -> None:
    expected = [
        {
            "rows": [
                ["ID", "Device", "Error"],
                ["---", ":---:", "---:"],
                ["A-01", "CPU", "0.031"],
            ]
        }
    ]
    predicted = [
        [["ID", "Device", "Error"], ["A-01", "CPU", "0.031"]]
    ]

    assert builder._score_semantic_html_tables(
        expected, predicted
    )["cell_exact_fraction"] == 1.0


def test_structure_visible_quality_gate_uses_explicit_metrics() -> None:
    metrics = {
        "failure_count": 0,
        "token_cap_hit_count": 0,
        "structure_visible_lexical_recall": 0.89,
        "structure_visible_lexical_precision": 1.0,
        "semantic_table_cell_exact_fraction": 1.0,
    }
    gate = {
        "maximum_failure_count": 0,
        "maximum_token_cap_hit_count": 0,
        "minimum_structure_visible_lexical_recall": 0.9,
        "minimum_structure_visible_lexical_precision": 0.9,
        "minimum_semantic_table_cell_exact_fraction": 0.95,
    }

    assert builder._quality_gate_passes(metrics=metrics, gate=gate) is False


def test_like_for_like_visible_lexical_score_can_pass_for_html() -> None:
    expected = "Runtime comparison\nID Device Error\nA-01 CPU 0.031"
    predicted = (
        "<h2>Runtime comparison</h2><table><tr><th>ID</th><th>Device</th>"
        "<th>Error</th></tr><tr><td>A-01</td><td>CPU</td>"
        "<td>0.031</td></tr></table>"
    )

    score = builder._lexical_overlap(
        expected,
        builder.project_html_visible_text(predicted),
    )

    assert score == {"recall": 1.0, "precision": 1.0}


def test_raw_negative_false_positives_remain_a_hard_gate() -> None:
    metrics = {
        "failure_count": 0,
        "token_cap_hit_count": 0,
        "structure_visible_normalized_character_error_rate": 0.0,
        "structure_visible_required_token_recall": 1.0,
        "structure_visible_negative_false_positive_characters": 0,
        "raw_negative_false_positive_characters": 1,
    }
    gate = {
        "maximum_failure_count": 0,
        "maximum_token_cap_hit_count": 0,
        "maximum_structure_visible_normalized_character_error_rate": 0.25,
        "minimum_structure_visible_required_token_recall": 0.8,
        "maximum_structure_visible_negative_false_positive_characters": 0,
        "maximum_raw_negative_false_positive_characters": 0,
    }

    assert builder._quality_gate_passes(metrics=metrics, gate=gate) is False


def test_record_reader_rejects_lines_that_differ_from_prediction(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "success": True,
                "prediction": "trusted output",
                "lines": [{"text": "different output"}],
                "token_cap_hit": False,
                "stop_finish": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="lines do not match"):
        builder._read_records(path)


def test_hunyuan_registry_names_failed_runtime_gate_as_quality_blocker():
    registry = json.loads(
        Path("registries/bounded_vlm_b10598_assets.json").read_text(encoding="utf-8")
    )
    candidate = registry["candidates"]["hunyuanocr_1_5_gguf_cpu"]

    assert candidate["failed_outcome_kind"] == "quality_blocker"
    assert "implementation" not in candidate["failed_outcome_kind"]


def test_reference_media_hash_must_match_verified_fixture_asset() -> None:
    assets = {
        "fixtures": {
            "images": {"sample": {"sha256": "a" * 64}},
        }
    }
    builder._verify_reference_media_hash(
        {"image_sha256": "a" * 64},
        sample_id="sample",
        assets=assets,
    )
    with pytest.raises(ValueError, match="reference media hash"):
        builder._verify_reference_media_hash(
            {"image_sha256": "b" * 64},
            sample_id="sample",
            assets=assets,
        )


@pytest.mark.parametrize(
    "dependency",
    (
        "src/local_inference_bench/load_sustained_workload.py",
        "src/local_inference_bench/html_output_projection.py",
        "src/local_inference_bench/score_document_fidelity.py",
        "src/local_inference_bench/score_ocr_quality.py",
        "src/local_inference_bench/validate_public_summary.py",
        "src/local_inference_bench/windows_kill_on_close_job.py",
    ),
)
def test_v3_producer_hashes_change_with_each_scoring_dependency(
    tmp_path: Path,
    monkeypatch,
    dependency: str,
) -> None:
    producer_paths = {
        "scripts/run_bounded_vlm_b10598_quality.py",
        "src/local_inference_bench/bounded_vlm_assets.py",
        "src/local_inference_bench/html_output_projection.py",
        "src/local_inference_bench/load_sustained_workload.py",
        "src/local_inference_bench/resource_monitor.py",
        "src/local_inference_bench/score_document_fidelity.py",
        "src/local_inference_bench/score_ocr_quality.py",
        "src/local_inference_bench/terminate_process_tree.py",
        "src/local_inference_bench/validate_public_summary.py",
        "src/local_inference_bench/windows_host_monitor.py",
        "src/local_inference_bench/windows_kill_on_close_job.py",
        "src/local_inference_bench/fingerprint.py",
        "src/local_inference_bench/verify_locked_environment.py",
        "environments/control/environment.yml",
        "environments/control/requirements.lock.txt",
        "scripts/capture_environment_identity.py",
        "scripts/verify_control_environment.py",
        runner.CANDIDATES["ovisocr2_q8_cpu"]["worker"],
    }
    for relative_path in producer_paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"baseline: {relative_path}\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    baseline = runner._producer_hashes("ovisocr2_q8_cpu")

    changed_path = tmp_path / dependency
    changed_path.write_text("mutated dependency\n", encoding="utf-8")
    changed = runner._producer_hashes("ovisocr2_q8_cpu")

    assert changed[dependency] != baseline[dependency]
    assert {
        path: digest for path, digest in changed.items() if path != dependency
    } == {
        path: digest for path, digest in baseline.items() if path != dependency
    }
