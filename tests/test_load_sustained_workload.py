import hashlib
import json
from pathlib import Path

import pytest

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


def test_workload_class_changes_opaque_fingerprint(tmp_path):
    audio_path = tmp_path / "public.wav"
    audio_path.write_bytes(b"same bytes")
    manifest_path = _write_manifest(tmp_path, audio_path)
    private = load_sustained_workload(manifest_path, expected_task="asr")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["workload_class"] = "public_course"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    public = load_sustained_workload(manifest_path, expected_task="asr")

    assert private["fingerprint"] != public["fingerprint"]


def test_reused_item_warmup_selection_changes_opaque_fingerprint(tmp_path):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first")
    second_audio.write_bytes(b"second")
    manifest_path = tmp_path / "local_manifest.json"
    document = {
        "schema_version": 1,
        "task": "asr",
        "workload_class": "private_course",
        "warmup_item_id": "sample-001",
        "items": [
            {
                "id": "sample-001",
                "path": first_audio.name,
                "duration_seconds": 60,
            },
            {
                "id": "sample-002",
                "path": second_audio.name,
                "duration_seconds": 60,
            },
        ],
    }
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    first_warmup = load_sustained_workload(manifest_path, expected_task="asr")
    document["warmup_item_id"] = "sample-002"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    second_warmup = load_sustained_workload(manifest_path, expected_task="asr")

    assert first_warmup["fingerprint"] != second_warmup["fingerprint"]


def test_item_order_changes_opaque_fingerprint(tmp_path):
    first_audio = tmp_path / "first.wav"
    second_audio = tmp_path / "second.wav"
    first_audio.write_bytes(b"first")
    second_audio.write_bytes(b"second")
    manifest_path = tmp_path / "local_manifest.json"
    document = {
        "schema_version": 1,
        "task": "asr",
        "workload_class": "private_course",
        "warmup_item_id": "sample-001",
        "items": [
            {
                "id": "sample-001",
                "path": first_audio.name,
                "duration_seconds": 60,
            },
            {
                "id": "sample-002",
                "path": second_audio.name,
                "duration_seconds": 60,
            },
        ],
    }
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    first_order = load_sustained_workload(manifest_path, expected_task="asr")
    document["items"].reverse()
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    second_order = load_sustained_workload(manifest_path, expected_task="asr")

    assert first_order["fingerprint"] != second_order["fingerprint"]


def test_framed_fingerprint_breaks_legacy_structural_collision(tmp_path):
    warmup = tmp_path / "warmup.wav"
    first_a = tmp_path / "first-a.wav"
    first_b = tmp_path / "first-b.wav"
    second_a = tmp_path / "second-a.wav"
    second_b = tmp_path / "second-b.wav"
    warmup.write_bytes(b"W")
    first_a.write_bytes(b"XYcccccccccc")
    first_b.write_bytes(b"XY")
    second_a.write_bytes(b"Z")
    second_b.write_bytes(b"Z")
    manifest_path = tmp_path / "local_manifest.json"

    def load(items):
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": "asr",
                    "workload_class": "private_course",
                    "warmup_item_id": "warmup",
                    "items": [
                        *items,
                        {
                            "id": "warmup",
                            "path": warmup.name,
                            "duration_seconds": 3,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return load_sustained_workload(manifest_path, expected_task="asr")

    legacy_collision_a = load(
        [
            {"id": "a", "path": first_a.name, "duration_seconds": 3},
            {"id": "b", "path": second_a.name, "duration_seconds": 3},
        ]
    )
    legacy_collision_b = load(
        [
            {"id": "a1", "path": first_b.name, "duration_seconds": 3},
            {
                "id": "ccccccccccb",
                "path": second_b.name,
                "duration_seconds": 3,
            },
        ]
    )

    assert _legacy_v2_fingerprint(legacy_collision_a) == _legacy_v2_fingerprint(
        legacy_collision_b
    )
    assert legacy_collision_a["fingerprint"] != legacy_collision_b["fingerprint"]


def test_default_and_explicit_same_warmup_have_same_fingerprint(tmp_path):
    audio_path = tmp_path / "private.wav"
    audio_path.write_bytes(b"same bytes")
    manifest_path = _write_manifest(tmp_path, audio_path)
    explicit = load_sustained_workload(manifest_path, expected_task="asr")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    del document["warmup_item_id"]
    manifest_path.write_text(json.dumps(document), encoding="utf-8")

    implicit = load_sustained_workload(manifest_path, expected_task="asr")

    assert explicit["fingerprint"] == implicit["fingerprint"]


def test_separate_warmup_is_loaded_but_not_counted_as_measured_work(tmp_path):
    measured = tmp_path / "measured.wav"
    warmup = tmp_path / "warmup.wav"
    measured.write_bytes(b"measured")
    warmup.write_bytes(b"warmup")
    manifest_path = tmp_path / "local_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "asr",
                "workload_class": "private_course",
                "items": [
                    {
                        "id": "audio_001",
                        "path": measured.name,
                        "duration_seconds": 900,
                    }
                ],
                "warmup": {
                    "id": "warmup",
                    "path": warmup.name,
                    "duration_seconds": 30,
                },
            }
        ),
        encoding="utf-8",
    )

    workload = load_sustained_workload(manifest_path, expected_task="asr")

    assert workload["warmup_item"]["id"] == "warmup"
    assert workload["public_summary"]["item_count"] == 1
    assert workload["public_summary"]["total_duration_seconds"] == 900.0


def test_generated_quality_control_is_an_allowed_public_class(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "generated_quality_control",
                "items": [{"id": "image_001", "path": image.name}],
            }
        ),
        encoding="utf-8",
    )

    workload = load_sustained_workload(manifest_path, expected_task="ocr")

    assert workload["workload_class"] == "generated_quality_control"


def test_ocr_output_marker_is_private_and_binds_fingerprint(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    manifest_path = tmp_path / "manifest.json"

    def load(marker: str):
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": "ocr",
                    "workload_class": "generated_quality_control",
                    "items": [
                        {
                            "id": "image_001",
                            "path": image.name,
                            "output_marker": marker,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return load_sustained_workload(manifest_path, expected_task="ocr")

    first = load("<!-- meta:page number=7 -->")
    second = load("<!-- meta:page number=8 -->")

    assert first["items"][0]["output_marker"] == "<!-- meta:page number=7 -->"
    assert "output_marker" not in json.dumps(first["public_summary"])
    assert first["fingerprint"] != second["fingerprint"]


def test_ocr_output_marker_rejects_unbounded_content(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"png")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "ocr",
                "workload_class": "generated_quality_control",
                "items": [
                    {
                        "id": "image_001",
                        "path": image.name,
                        "output_marker": "<!-- meta:page number=7 --> private text",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output_marker"):
        load_sustained_workload(manifest_path, expected_task="ocr")


def _legacy_v2_fingerprint(workload: dict) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "protocol": "sustained-workload-v2",
                "task": workload["task"],
                "warmup_item_id": workload["warmup_item"]["id"],
                "workload_class": workload["workload_class"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    for item in workload["items"]:
        path = Path(item["path"])
        digest.update(item["id"].encode("ascii"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(repr(item["duration_seconds"]).encode("ascii"))
        digest.update(str(item["expected_speech"]).encode("ascii"))
        digest.update(path.read_bytes())
    return digest.hexdigest()
