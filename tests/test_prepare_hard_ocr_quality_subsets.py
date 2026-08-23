from scripts.prepare_hard_ocr_quality_subsets import build_subset


def test_subset_keeps_only_selected_items_and_references():
    source = {
        "schema_version": 1,
        "task": "ocr",
        "workload_class": "generated_quality_control",
        "disclosure": "generated",
        "warmup": {"id": "warmup", "path": "warmup.png"},
        "items": [
            {"id": "one", "path": "one.png"},
            {"id": "two", "path": "two.png"},
        ],
        "references": {
            "one": {"lines": ["one"]},
            "two": {"lines": ["two"]},
        },
    }

    subset = build_subset(source, ("two",))

    assert [item["id"] for item in subset["items"]] == ["two"]
    assert list(subset["references"]) == ["two"]
    assert subset["subset"]["sample_count"] == 1
