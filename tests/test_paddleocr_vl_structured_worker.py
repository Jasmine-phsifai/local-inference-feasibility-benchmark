from workers.paddleocr_vl_structured_worker import (
    _blocks_to_lines,
    _extract_blocks,
)


def test_structured_blocks_preserve_reading_order_and_semantics():
    payloads = [
        {
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": "formula",
                        "block_content": "$x^2$",
                        "block_bbox": [10, 30, 80, 60],
                        "block_id": 2,
                        "block_order": 2,
                    },
                    {
                        "block_label": "text",
                        "block_content": "heading\nbody",
                        "block_bbox": [10, 5, 80, 25],
                        "block_id": 1,
                        "block_order": 1,
                    },
                ]
            }
        }
    ]

    blocks = _extract_blocks(payloads)
    lines = _blocks_to_lines(blocks)

    assert [block["label"] for block in blocks] == ["text", "formula"]
    assert [line["text"] for line in lines] == ["heading", "body", "$x^2$"]
    assert lines[-1]["label"] == "formula"
