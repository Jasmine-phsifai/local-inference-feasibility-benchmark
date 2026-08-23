import json

from local_inference_bench.load_sustained_workload import load_sustained_workload


def _write_manifest(tmp_path, audio_path):
    manifest_path = tmp_path / "local_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "asr",
                "workload_class": "private_course",
                "warmup_item_id": "sample-001",
                "items": [
                    {
                        "id": "sample-001",
                        "path": audio_path.name,
                        "duration_seconds": 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_public_summary_omits_paths_and_content_hashes(tmp_path):
    audio_path = tmp_path / "private.wav"
    audio_path.write_bytes(b"private bytes")
    workload = load_sustained_workload(
        _write_manifest(tmp_path, audio_path),
        expected_task="asr",
    )

    assert workload["public_summary"] == {
        "workload_class": "private_course",
        "item_count": 1,
        "total_duration_seconds": 60.0,
    }
    assert "path" not in json.dumps(workload["public_summary"])
    assert workload["fingerprint"] not in json.dumps(workload["public_summary"])


def test_input_byte_change_changes_opaque_fingerprint(tmp_path):
    audio_path = tmp_path / "private.wav"
    audio_path.write_bytes(b"first")
    manifest_path = _write_manifest(tmp_path, audio_path)
    first = load_sustained_workload(manifest_path, expected_task="asr")
    audio_path.write_bytes(b"second")
    second = load_sustained_workload(manifest_path, expected_task="asr")

    assert first["fingerprint"] != second["fingerprint"]
