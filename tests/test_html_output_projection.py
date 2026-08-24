from local_inference_bench.html_output_projection import (
    extract_html_table_rows,
    project_html_visible_text,
)


def test_visible_text_excludes_tags_attributes_and_nonvisible_content() -> None:
    value = (
        '<section data-private="ignored"><h2>Runtime &amp; OCR</h2>'
        '<script>hidden()</script><p>A-01<br>CPU</p></section>'
    )

    assert project_html_visible_text(value) == "Runtime & OCR\nA-01\nCPU"


def test_self_closing_nonvoid_suppressed_tag_hides_following_text() -> None:
    assert project_html_visible_text("before<script/>after") == "before"
    assert project_html_visible_text("before<style/>after") == "before"
    assert project_html_visible_text("before<template/>after") == "before"
    assert (
        project_html_visible_text("before<script/>ignored</script>after")
        == "beforeafter"
    )


def test_hidden_attributes_and_nonrendered_containers_are_excluded() -> None:
    value = (
        "visible<head><title>head text</title></head>"
        "<title>stray title</title>"
        "<template>template text</template>"
        "<dialog>closed dialog</dialog>"
        "<details><summary>summary</summary><p>closed details</p></details>"
        "<select><option>select option</option></select>"
        "<div popover>popover text</div>"
        "<iframe>fallback frame text</iframe>"
        "<p hidden>hidden attribute</p>"
        '<p aria-hidden="true">aria hidden</p>'
        '<p style="display: none !important">css hidden</p>after'
    )

    assert project_html_visible_text(value) == "visibleafter"


def test_open_dialog_and_details_are_visible() -> None:
    value = "<dialog open>dialog</dialog><details open><summary>summary</summary>body</details>"

    assert project_html_visible_text(value) == "dialogsummarybody"


def test_plain_text_and_html_table_project_in_reading_order() -> None:
    value = (
        "Notes\n<table><thead><tr><th>ID</th><th>Device</th></tr></thead>"
        "<tbody><tr><td>A-01</td><td><b>CPU</b></td></tr></tbody></table>"
    )

    assert project_html_visible_text(value) == "Notes\nID Device\nA-01 CPU"
    assert extract_html_table_rows(value) == [
        [["ID", "Device"], ["A-01", "CPU"]]
    ]


def test_table_parser_rejects_a_truncated_final_table() -> None:
    value = "<table><tr><td>A-01</td><td>0.031"

    assert extract_html_table_rows(value) == []


def test_any_malformed_table_invalidates_table_credit_for_the_output() -> None:
    valid = "<table><tr><td>A</td></tr></table>"
    malformed = "<table><tr><td>B"
    misnested = "<table><tr><td><b>B</td></tr></table>"

    assert extract_html_table_rows(valid + malformed) == []
    assert extract_html_table_rows(malformed + valid) == []
    assert extract_html_table_rows(valid + misnested) == []
    assert extract_html_table_rows(misnested + valid) == []


def test_hidden_table_invalidates_table_credit_for_the_output() -> None:
    valid = "<table><tr><td>A</td></tr></table>"
    malformed = "<table><tr><td><b>B</td></tr></table>"
    hidden_variants = [
        f"<div hidden>{malformed}</div>",
        f"<dialog>{malformed}</dialog>",
        f"<details>{malformed}</details>",
        f'<div style="display:none">{malformed}</div>',
        malformed.replace("<table>", "<table hidden>", 1),
    ]

    assert all(
        extract_html_table_rows(valid + hidden_table) == []
        for hidden_table in hidden_variants
    )


def test_table_parser_rejects_cells_without_rows_and_nested_tables() -> None:
    assert extract_html_table_rows("<table><td>A-01</td></table>") == []
    assert extract_html_table_rows(
        "<table><tr><td>A<table><tr><td>B</td></tr></table></td></tr></table>"
    ) == []


def test_table_parser_rejects_invalid_table_children_and_section_nesting() -> None:
    invalid = [
        "<table>rogue<tr><td>ID</td></tr></table>",
        "<table><div>bad</div><tr><td>ID</td></tr></table>",
        "<table><tr><div>bad</div><td>ID</td></tr></table>",
        "<table><thead><tr><td>ID</td></tr></tbody></table>",
        "<table><thead><tbody><tr><td>ID</td></tr></tbody></thead></table>",
        "<table><tr><td><b>ID</td></tr></table>",
        "<table><tr><td><b>ID</td></b></tr></table>",
        (
            "<table><thead><tr><th>ID</th></tr></thead>"
            "<thead><tr><th>ID</th></tr></thead></table>"
        ),
        (
            "<table><tfoot><tr><td>total</td></tr></tfoot>"
            "<tbody><tr><td>ID</td></tr></tbody></table>"
        ),
        (
            "<table><tbody><tr><td>ID</td></tr></tbody>"
            "<tfoot><tr><td>total</td></tr></tfoot>"
            "<tfoot><tr><td>total</td></tr></tfoot></table>"
        ),
        "<table><col><tr><td>A</td></tr></table>",
        "<table><tr><td><caption>A</caption></td></tr></table>",
        "<table><tr><td><colgroup><col></colgroup>A</td></tr></table>",
        "<table><tr><td><col>A</td></tr></table>",
        "<table><tr><td><caption hidden>A</caption>B</td></tr></table>",
        "<table><tr><td colspan=2>A</td></tr></table>",
        "<table><tr><td rowspan=0>A</td></tr></table>",
    ]

    assert all(extract_html_table_rows(value) == [] for value in invalid)


def test_table_parser_accepts_valid_multiple_body_sections() -> None:
    value = (
        "<table><caption>Runs</caption><colgroup><col></colgroup>"
        "<thead><tr><th>ID</th></tr></thead>"
        "<tbody><tr><td>A</td></tr></tbody>"
        "<tbody><tr><td>B</td></tr></tbody>"
        "<tfoot><tr><td>2</td></tr></tfoot></table>"
    )

    assert extract_html_table_rows(value) == [
        [["ID"], ["A"], ["B"], ["2"]]
    ]


def test_table_parser_accepts_explicit_default_cell_spans() -> None:
    value = "<table><tr><td colspan=1 rowspan=1>A</td></tr></table>"

    assert extract_html_table_rows(value) == [[["A"]]]


def test_table_parser_ignores_script_and_style_cell_content() -> None:
    value = (
        "<table><tr><td>safe<script>not visible</script></td>"
        "<td><style>also hidden</style>CPU</td></tr></table>"
    )

    assert extract_html_table_rows(value) == [[["safe", "CPU"]]]
