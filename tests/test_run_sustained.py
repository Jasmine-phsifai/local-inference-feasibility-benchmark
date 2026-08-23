import json
import subprocess
from pathlib import Path

from local_inference_bench.event_journal import read_events
from local_inference_bench import run_sustained


def _common():
    return {
        "protocol": "sustained-process-v1",
        "candidate_id": "candidate",
        "task": "asr",
        "attempt_id": "attempt",
        "attempt_key": "key",
        "code_fingerprint": "code",
        "environment_fingerprint": "environment",
        "config": {"processes": 1},
        "config_index": 0,
        "phase": "screen",
        "target_wall_seconds": 10,
        "trial_index": 0,
        "workload": {
            "workload_class": "private_course",
            "item_count": 1,
            "total_duration_seconds": 60,
        },
    }


def test_outcome_journal_uses_only_validated_public_summary(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps(
            {
                "public_summary": {
                    "candidate_id": "candidate",
                    "task": "asr",
                    "runtime_name": "runtime",
                    "runtime_version": "1",
                    "load_semantics": "resident_model",
                    "workload_class": "private_course",
                    "counts": {"completed": 1, "failed": 0, "attempted": 1},
                    "throughput": {
                        "value": 1.0,
                        "unit": "audio_hours_per_wall_hour",
                    },
                    "timing": {
                        "steady_wall_seconds": 1.0,
                        "target_wall_seconds": 10.0,
                    },
                },
                "private_records": {
                    "transcript": "must not be copied",
                    "path": "D:\\private\\lecture.wav",
                },
            }
        ),
        encoding="utf-8",
    )
    private_records_path = tmp_path / "private-records.jsonl"
    private_records_path.write_text("{}\n", encoding="utf-8")

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
        private_records_path=private_records_path,
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={"sample_count": 1},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    serialized = json.dumps(event)
    assert event["event"] == "sustained_attempt_succeeded"
    assert "transcript" not in serialized
    assert "lecture.wav" not in serialized
    provenance = json.loads(
        (tmp_path / "records-provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["attempt_key"] == "key"
    assert provenance["records_sha256"] == run_sustained._sha256(
        private_records_path
    )


def test_invalid_preview_field_becomes_generic_failure(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    response_path = tmp_path / "response.json"
    response_path.write_text(
        json.dumps({"public_summary": {"output_preview": "private"}}),
        encoding="utf-8",
    )

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
        private_records_path=tmp_path / "missing-records.jsonl",
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    assert event["event"] == "sustained_attempt_failed"
    assert event["failure_kind"] == "invalid_response"


def test_empty_summary_cannot_be_journaled_as_success(tmp_path, monkeypatch):
    events_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(run_sustained, "SUSTAINED_EVENTS_PATH", events_path)
    response_path = tmp_path / "response.json"
    response_path.write_text('{"public_summary": {}}', encoding="utf-8")

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
        private_records_path=tmp_path / "missing-records.jsonl",
        workload_fingerprint="c" * 64,
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    assert event["event"] == "sustained_attempt_failed"
    assert event["failure_kind"] == "invalid_response"


def test_phase_scoped_configs_are_selected_without_crossing_safety_gate():
    candidate = {
        "configs": [
            {"phases": ["compatibility"]},
            {"phases": ["quality"]},
            {"processes": 1},
        ]
    }

    assert run_sustained._select_config_indices(
        candidate,
        "compatibility",
        None,
    ) == (0, 2)
    assert run_sustained._select_config_indices(candidate, "quality", None) == (1, 2)


def test_explicit_phase_mismatch_is_rejected():
    candidate = {"configs": [{"phases": ["compatibility"]}]}

    try:
        run_sustained._select_config_indices(candidate, "quality", (0,))
    except ValueError as error:
        assert "does not support" in str(error)
    else:
        raise AssertionError("phase-mismatched config was accepted")


def test_environment_identity_capture_is_bounded(monkeypatch):
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(run_sustained.subprocess, "run", timeout)

    try:
        run_sustained._capture_environment_fingerprint(
            Path("python.exe"),
            Path("identity.py"),
        )
    except RuntimeError as error:
        assert "timed out" in str(error)
    else:
        raise AssertionError("unbounded environment identity capture was accepted")


def test_trials_alternate_config_order() -> None:
    selected = (15, 16)

    assert run_sustained._ordered_config_indices(selected, 0) == (15, 16)
    assert run_sustained._ordered_config_indices(selected, 1) == (16, 15)
    assert run_sustained._ordered_config_indices(selected, 2) == (15, 16)
