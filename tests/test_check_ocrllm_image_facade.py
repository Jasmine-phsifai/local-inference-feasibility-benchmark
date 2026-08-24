import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.check_ocrllm_image_facade import (
    PROJECT_ROOT,
    _build_compatibility_event,
    _build_public_summary,
    _load_code_formula_reference,
    _normalize_visible,
    _producer_sha256,
    _require_safe_private_record_path,
    _sha256,
)
from scripts.ocrllm_compatibility_provenance import (
    EXPECTED_RUNTIME_VERSIONS,
    _canonical_distribution_name,
    _file_url_to_path,
    _installed_runtime_components,
    _python_file_hashes,
)


def _write_manifest(tmp_path: Path, image: Path) -> Path:
    image_sha256 = hashlib.sha256(image.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "generated_quality_control",
                "items": [
                    {
                        "id": "code_formula",
                        "path": image.name,
                        "expected_text": True,
                    }
                ],
                "references": {
                    "code_formula": {
                        "category": "code_formula",
                        "required_tokens": ["alpha  beta", "formula"],
                        "image_sha256": image_sha256,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _provenance() -> dict:
    return {
        "reviewed_baseline_ancestor": True,
        "snapshot_clean": True,
        "installed_noneditable": True,
        "installed_from_pinned_snapshot": True,
        "installed_source_matches_snapshot": True,
        "python_file_count": 123,
        "runtime_component_count": 7,
        "rapidocr_model_file_count": 3,
        "experimental_direct_short_mp3_public_facade": True,
        "local_asr_public_symbol_count": 0,
        "local_asr_facade_available": False,
        "filetrans_public_symbol_count": 0,
        "filetrans_facade_available": False,
        "configured_direct_audio_limit_seconds": 300.0,
        "long_audio_facade_available": False,
        "direct_short_inmemory_options_accepted": True,
        "registered_audio_available_capability_count": 0,
        "audio_nonmemory_option_rejection_count": 3,
        "audio_persistence_available": False,
        "audio_resume_available": False,
        "audio_worker_command_count": 0,
        "audio_worker_support_available": False,
        "benchmark_owned_asr_adapters_required": True,
    }


def test_manifest_reference_binds_image_hash_and_required_tokens(tmp_path):
    image = tmp_path / "code_formula.png"
    image.write_bytes(b"generated-image")
    manifest = _write_manifest(tmp_path, image)

    reference = _load_code_formula_reference(manifest, image)

    assert reference["required_tokens"] == ["alpha  beta", "formula"]
    assert reference["image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert len(reference["manifest_sha256"]) == 64
    assert len(reference["required_tokens_sha256"]) == 64


def test_manifest_reference_rejects_image_drift(tmp_path):
    image = tmp_path / "code_formula.png"
    image.write_bytes(b"generated-image")
    manifest = _write_manifest(tmp_path, image)
    image.write_bytes(b"changed-image")

    with pytest.raises(ValueError, match="fingerprint"):
        _load_code_formula_reference(manifest, image)


def test_private_record_must_be_outside_repo_or_git_ignored(tmp_path):
    _require_safe_private_record_path(
        Path("results/artifacts/private_ocrllm_test/output.json")
    )
    _require_safe_private_record_path(tmp_path / "outside-repo.json")
    with pytest.raises(ValueError, match="must be ignored"):
        _require_safe_private_record_path(Path("reports/raw-ocrllm-output.json"))


def test_public_summary_is_aggregate_only_and_preserves_strict_recall():
    secret_raw_ocr = "alpha beta\nformula\nsecret-raw-ocr"
    result = SimpleNamespace(
        markdown=secret_raw_ocr,
        status="complete",
        output_path=None,
        warnings=("aggregate-only",),
        metadata={
            "network_call_count": 0,
            "detected_line_count": 3,
            "retained_line_count": 3,
            "mean_confidence": 0.875,
        },
    )
    summary = _build_public_summary(
        runtime_version="0.1.0",
        result=result,
        result_fields={"markdown", "metadata"},
        required_tokens=["alpha  beta", "formula"],
        elapsed_seconds=1.25,
        network_attempt_count=0,
        provenance=_provenance(),
    )

    compatibility = summary["compatibility"]
    assert compatibility["required_token_count"] == 2
    assert compatibility["exact_required_token_hit_count"] == 1
    assert compatibility["exact_required_token_recall"] == 0.5
    assert compatibility["whitespace_insensitive_required_token_hit_count"] == 2
    assert compatibility["whitespace_insensitive_required_token_recall"] == 1.0
    assert compatibility["mean_reported_confidence"] == 0.875
    assert "secret-raw-ocr" not in json.dumps(summary)
    boundary = summary["authority_boundary"]
    assert boundary["experimental_direct_short_mp3_public_facade"] is True
    assert boundary["local_asr_facade_available"] is False
    assert boundary["filetrans_facade_available"] is False
    assert boundary["configured_direct_audio_limit_seconds"] == 300.0
    assert boundary["long_audio_facade_available"] is False
    assert boundary["audio_worker_support_available"] is False
    assert set(compatibility) == {
        "recognition_succeeded",
        "elapsed_seconds",
        "network_attempt_count",
        "reported_network_call_count",
        "memory_only_output",
        "warning_count",
        "detected_line_count",
        "facade_line_count",
        "output_character_count",
        "required_token_count",
        "exact_required_token_hit_count",
        "exact_required_token_recall",
        "whitespace_insensitive_required_token_hit_count",
        "whitespace_insensitive_required_token_recall",
        "mean_reported_confidence",
        "latex_marker_count",
        "facade_exposes_boxes",
        "facade_exposes_line_confidences",
    }


def test_invalid_mean_confidence_is_rejected():
    result = SimpleNamespace(
        markdown="formula",
        status="complete",
        output_path=None,
        warnings=(),
        metadata={"mean_confidence": float("nan")},
    )
    with pytest.raises(ValueError, match="mean confidence"):
        _build_public_summary(
            runtime_version="0.1.0",
            result=result,
            result_fields=set(),
            required_tokens=["formula"],
            elapsed_seconds=1.0,
            network_attempt_count=0,
            provenance=_provenance(),
        )


@pytest.mark.parametrize("network_call_count", [None, -1, 0.0, False, "0"])
def test_reported_network_call_count_must_be_an_actual_nonnegative_integer(
    network_call_count,
):
    result = SimpleNamespace(
        markdown="formula",
        status="complete",
        output_path=None,
        warnings=(),
        metadata={
            "mean_confidence": 0.9,
            "network_call_count": network_call_count,
        },
    )
    with pytest.raises(ValueError, match="network call count"):
        _build_public_summary(
            runtime_version="0.1.0",
            result=result,
            result_fields=set(),
            required_tokens=["formula"],
            elapsed_seconds=1.0,
            network_attempt_count=0,
            provenance=_provenance(),
        )


def test_compatibility_event_has_exact_attribution_schema_and_producers():
    provenance = {
        "revision": "a" * 40,
        "source_tree_fingerprint": "b" * 40,
        "python_source_fingerprint": "c" * 64,
        "runtime_environment_fingerprint": "d" * 64,
        "rapidocr_model_fingerprint": "e" * 64,
    }
    event = _build_compatibility_event(
        reference={
            "image_sha256": "f" * 64,
            "manifest_sha256": "1" * 64,
            "required_tokens_sha256": "2" * 64,
        },
        provenance=provenance,
        public_summary={"candidate_id": "ocrllm_active_image_facade"},
        timestamp_utc="2026-08-24T00:00:00+00:00",
    )

    assert set(event) == {
        "event",
        "protocol",
        "timestamp_utc",
        "candidate_id",
        "dataset_fingerprint",
        "workload_manifest_fingerprint",
        "exact_reference_fingerprint",
        "source_revision",
        "source_tree_fingerprint",
        "installed_python_source_fingerprint",
        "runtime_environment_fingerprint",
        "rapidocr_model_fingerprint",
        "producer_sha256",
        "result",
    }
    assert event["candidate_id"] == "ocrllm_active_image_facade"
    assert event["protocol"] == "ocrllm-image-compatibility-v3"
    assert set(event["producer_sha256"]) == {
        "scripts/check_ocrllm_image_facade.py",
        "scripts/ocrllm_compatibility_provenance.py",
        "src/local_inference_bench/validate_public_summary.py",
        "src/local_inference_bench/event_journal.py",
    }
    for relative_path, digest in event["producer_sha256"].items():
        assert len(digest) == 64
        assert digest == _sha256(PROJECT_ROOT / relative_path)


def test_compatibility_event_rejects_result_candidate_misattribution():
    with pytest.raises(ValueError, match="candidate identity"):
        _build_compatibility_event(
            reference={},
            provenance={},
            public_summary={"candidate_id": "different_candidate"},
            timestamp_utc="2026-08-24T00:00:00+00:00",
        )


def test_provenance_helpers_normalize_file_url_and_hash_python_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "ignored.txt").write_text("not Python", encoding="utf-8")

    assert _file_url_to_path(source.as_uri()) == source.resolve()
    assert list(_python_file_hashes(source)) == ["one.py"]


def test_runtime_version_checks_match_the_environment_manifest():
    requirements = Path(
        "environments/ocrllm_compatibility/requirements.lock.txt"
    )
    pinned = dict(
        (name.casefold(), version)
        for name, version in (
            line.split("==", maxsplit=1)
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        )
    )
    for name, version in EXPECTED_RUNTIME_VERSIONS.items():
        assert pinned[name] == version


def test_distribution_names_are_canonicalized_for_stable_fingerprints():
    assert _canonical_distribution_name("opencv_python") == "opencv-python"
    assert _canonical_distribution_name("Pillow") == "pillow"


def test_runtime_fingerprint_ignores_distribution_metadata_on_project_path(tmp_path):
    before = _installed_runtime_components()
    fake_metadata = tmp_path / "unrelated-99.0.dist-info"
    fake_metadata.mkdir()
    (fake_metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unrelated\nVersion: 99.0\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        after = _installed_runtime_components()
    finally:
        sys.path.remove(str(tmp_path))

    assert after == before
    assert "distribution:unrelated" not in after


def test_normalization_is_unicode_and_whitespace_insensitive():
    assert _normalize_visible(" CPU\n二十四 ") == "cpu二十四"
