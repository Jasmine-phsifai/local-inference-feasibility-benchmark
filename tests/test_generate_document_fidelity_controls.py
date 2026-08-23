import hashlib
import json

from PIL import Image

from scripts.generate_document_fidelity_controls import (
    generate_document_fidelity_controls,
)


def test_generator_writes_three_deterministic_1080p_controls(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_manifest_path = generate_document_fidelity_controls(first_root)
    second_manifest_path = generate_document_fidelity_controls(second_root)
    first = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    second = json.loads(second_manifest_path.read_text(encoding="utf-8"))

    assert first_manifest_path.read_bytes() == second_manifest_path.read_bytes()
    assert len(first["items"]) == 3
    assert set(first["references"]) == {item["id"] for item in first["items"]}
    assert [item["output_marker"] for item in first["items"]] == [
        "<!-- meta:page number=7 -->",
        "<!-- meta:frame id=frame_012_420s -->",
        "<!-- meta:page number=8 -->",
    ]

    for item in first["items"]:
        first_image = first_root / item["path"]
        second_image = second_root / item["path"]
        assert first_image.read_bytes() == second_image.read_bytes()
        assert hashlib.sha256(first_image.read_bytes()).hexdigest() == first[
            "references"
        ][item["id"]]["image_sha256"]
        with Image.open(first_image) as image:
            assert image.mode == "RGB"
            assert image.size == (1920, 1080)


def test_every_fixture_declares_source_fidelity_contract(tmp_path):
    manifest_path = generate_document_fidelity_controls(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert [reference.get("page_number") for reference in manifest["references"].values()] == [
        7,
        None,
        8,
    ]
    for reference in manifest["references"].values():
        assert reference["expected_markdown"].splitlines()[0] == reference["marker"]
        assert reference["ordered_anchors"]
        assert reference["protected_spans"]
        assert isinstance(reference["forbidden_spans"], list)
