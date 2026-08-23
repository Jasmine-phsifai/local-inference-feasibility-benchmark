import json

from local_inference_bench.event_journal import read_events
from local_inference_bench import run_sustained


def _common():
    return {
        "protocol": "sustained-process-v1",
        "candidate_id": "candidate",
        "attempt_id": "attempt",
        "attempt_key": "key",
        "code_fingerprint": "code",
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
                    "workload_class": "private_course",
                    "counts": {"completed": 1, "failed": 0},
                },
                "private_records": {
                    "transcript": "must not be copied",
                    "path": "D:\\private\\lecture.wav",
                },
            }
        ),
        encoding="utf-8",
    )

    run_sustained._record_attempt_outcome(
        common=_common(),
        response_path=response_path,
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
        exit_code=0,
        failure_kind=None,
        wall_seconds=1.0,
        process_resources={},
        host_telemetry={"status": "observed"},
    )

    event = read_events(events_path)[0]
    assert event["event"] == "sustained_attempt_failed"
    assert event["failure_kind"] == "invalid_response"
