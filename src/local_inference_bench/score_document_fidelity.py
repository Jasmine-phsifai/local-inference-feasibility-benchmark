"""Score raw OCR Markdown without erasing source structure."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .event_journal import append_event
from .fingerprint import fingerprint_files, fingerprint_json
from .load_sustained_workload import load_sustained_workload
from .score_ocr_quality import _levenshtein
from .validate_public_summary import validate_public_summary


PROTOCOL_VERSION = "source-faithful.v1"
_SCORER_PROTOCOL = "document-fidelity-v2"
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_MARKER = re.compile(
    r"^<!-- meta:(?:page number=[1-9][0-9]*|frame id=[a-z0-9][a-z0-9_-]{0,63}) -->$",
    re.MULTILINE,
)
_HEADING = re.compile(r"^(#{1,6})[ \t]+.+$", re.MULTILINE)
_PAGE_MARKER = re.compile(r"^<!-- meta:page number=([1-9][0-9]*) -->$")
_FORMULA_UNICODE = frozenset("≤≥×−∑∇ηαβ→∂")
_FENCE = chr(96) * 3
_SOURCE_STATUSES = frozenset({"succeeded", "partial_failure", "all_failed"})
_MAX_TRIAL_COUNT = 8
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_RECORD_FILE_BYTES = 2_000_000
_MAX_PROVENANCE_BYTES = 64_000
_MAX_RECORDS_PER_TRIAL = 100
_MAX_PREDICTION_CHARACTERS = 200_000
_MAX_TOTAL_PREDICTION_CHARACTERS = 1_000_000
_MAX_REFERENCE_CHARACTERS = 500_000
_MAX_TOTAL_EDIT_CELLS = 5_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, action="append", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--mode", choices=("raw",), default="raw")
    parser.add_argument("--append-journal", type=Path)
    args = parser.parse_args()
    event = score_document_fidelity(
        manifest_path=args.manifest,
        records_paths=args.records,
        candidate_id=args.candidate,
        mode=args.mode,
    )
    if args.append_journal is not None:
        append_event(args.append_journal, event)
    print(json.dumps(event, indent=2, sort_keys=True))


def score_document_fidelity(
    *,
    manifest_path: Path,
    records_paths: list[Path],
    candidate_id: str,
    mode: str,
) -> dict:
    """Return a fixed-key aggregate event for one or more complete trials."""

    if _CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ValueError("candidate_id must be a bounded public identifier")
    if mode != "raw":
        raise ValueError("adapted fidelity requires a separately bound adapter")
    if not 1 <= len(records_paths) <= _MAX_TRIAL_COUNT:
        raise ValueError(
            f"document fidelity scoring requires one to {_MAX_TRIAL_COUNT} trials"
        )
    if len({path.resolve() for path in records_paths}) != len(records_paths):
        raise ValueError("document fidelity trials require distinct record files")
    _require_bounded_file(
        manifest_path,
        maximum_bytes=_MAX_MANIFEST_BYTES,
        label="manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    references = _validate_manifest(manifest, manifest_path=manifest_path)
    workload = load_sustained_workload(manifest_path, expected_task="ocr")
    expected_ids = set(references)
    trials: list[dict[str, dict]] = []
    provenances: list[dict] = []
    prediction_budget = [_MAX_TOTAL_PREDICTION_CHARACTERS]
    for records_path in records_paths:
        records = _read_trial_records(
            records_path,
            prediction_budget=prediction_budget,
        )
        if set(records) != expected_ids:
            raise ValueError("document fidelity records must exactly match manifest IDs")
        trials.append(records)
        provenance = _read_records_provenance(
            records_path,
            candidate_id=candidate_id,
            workload_fingerprint=workload["fingerprint"],
        )
        _validate_source_status(
            provenance["status"],
            expected_ids=expected_ids,
            records=records,
        )
        provenances.append(provenance)
    _validate_trial_provenance(provenances)
    scores: list[dict] = []
    edit_budget = [_MAX_TOTAL_EDIT_CELLS]
    predictions_by_sample: dict[str, list[str]] = {
        sample_id: [] for sample_id in references
    }
    for records in trials:
        for sample_id, reference in references.items():
            record = records[sample_id]
            prediction = record.get("prediction")
            if type(prediction) is not str:
                prediction = ""
            canonical = _canonical_markdown(prediction)
            predictions_by_sample[sample_id].append(canonical)
            scores.append(
                _score_sample(
                    reference,
                    canonical,
                    record,
                    edit_budget=edit_budget,
                )
            )
    metrics = _aggregate_scores(
        scores=scores,
        references=references,
        trials=trials,
        predictions_by_sample=predictions_by_sample,
    )
    return {
        "event": "document_fidelity_scored",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "mode": mode,
        "protocol": PROTOCOL_VERSION,
        "scorer_protocol": _SCORER_PROTOCOL,
        "scorer_fingerprint": _scorer_fingerprint(),
        "workload_class": "generated_quality_control",
        "dataset_fingerprint": _sha256(manifest_path),
        "workload_fingerprint": workload["fingerprint"],
        "records_fingerprints": [_sha256(path) for path in records_paths],
        "attempt_keys": [provenance["attempt_key"] for provenance in provenances],
        "code_fingerprints": sorted(
            {provenance["code_fingerprint"] for provenance in provenances}
        ),
        "environment_fingerprints": sorted(
            {provenance["environment_fingerprint"] for provenance in provenances}
        ),
        "config_fingerprint": fingerprint_json(provenances[0]["config"]),
        "source_attempts": [
            {
                "status": provenance["status"],
                "attempt_id": provenance["attempt_id"],
                "attempt_key": provenance["attempt_key"],
                "config_fingerprint": fingerprint_json(provenance["config"]),
                "config_index": provenance["config_index"],
                "trial_index": provenance["trial_index"],
                "code_fingerprint": provenance["code_fingerprint"],
                "environment_fingerprint": provenance["environment_fingerprint"],
                "controller_environment_fingerprint": provenance[
                    "controller_environment_fingerprint"
                ],
                "execution_policy_fingerprint": provenance[
                    "execution_policy_fingerprint"
                ],
            }
            for provenance in provenances
        ],
        "metrics": validate_public_summary(metrics),
    }


def _validate_manifest(
    manifest: object, *, manifest_path: Path
) -> dict[str, dict]:
    if not isinstance(manifest, dict):
        raise ValueError("document fidelity manifest must be an object")
    if manifest.get("schema_version") != 1 or manifest.get("task") != "ocr":
        raise ValueError("document fidelity manifest identity is invalid")
    if manifest.get("workload_class") != "generated_quality_control":
        raise ValueError("document fidelity scorer accepts generated controls only")
    items = manifest.get("items")
    references = manifest.get("references")
    if not isinstance(items, list) or not isinstance(references, dict):
        raise ValueError("document fidelity manifest requires items and references")
    item_ids = []
    for item in items:
        if not isinstance(item, dict) or type(item.get("id")) is not str:
            raise ValueError("document fidelity manifest item is invalid")
        item_ids.append(item["id"])
    expected_ids = [
        "page_007_bilingual_code",
        "frame_012_420s_formula_board",
        "page_008_table_columns",
    ]
    if item_ids != expected_ids:
        raise ValueError("document fidelity v1 fixture IDs are invalid")
    if set(item_ids) != set(references):
        raise ValueError("document fidelity item and reference IDs differ")
    generator = manifest.get("generator")
    if (
        not isinstance(generator, dict)
        or generator.get("protocol") != PROTOCOL_VERSION
        or generator.get("canvas_width") != 1920
        or generator.get("canvas_height") != 1080
        or type(generator.get("pillow_version")) is not str
    ):
        raise ValueError("document fidelity generator identity is invalid")
    expected_markers = {
        "page_007_bilingual_code": "<!-- meta:page number=7 -->",
        "frame_012_420s_formula_board": (
            "<!-- meta:frame id=frame_012_420s -->"
        ),
        "page_008_table_columns": "<!-- meta:page number=8 -->",
    }
    for item in items:
        sample_id = item["id"]
        reference = references[sample_id]
        if (
            item.get("path") != f"{sample_id}.png"
            or item.get("expected_text") is not True
            or item.get("output_marker") != expected_markers[sample_id]
            or reference.get("marker") != item.get("output_marker")
        ):
            raise ValueError("document fidelity item contract is invalid")
        image_path = manifest_path.parent / item["path"]
        if (
            not image_path.is_file()
            or reference.get("image_sha256") != _sha256(image_path)
        ):
            raise ValueError("document fidelity image identity is invalid")
        _validate_reference(reference)
    reference_characters = sum(
        len(reference["expected_markdown"]) for reference in references.values()
    )
    if reference_characters > _MAX_REFERENCE_CHARACTERS:
        raise ValueError("document fidelity reference budget exceeded")
    return {sample_id: references[sample_id] for sample_id in item_ids}


def _validate_reference(reference: object) -> None:
    if not isinstance(reference, dict):
        raise ValueError("document fidelity reference must be an object")
    required_lists = (
        "headings",
        "formulas",
        "code_blocks",
        "tables",
        "ordered_anchors",
        "protected_spans",
        "forbidden_spans",
    )
    marker = reference.get("marker")
    expected = reference.get("expected_markdown")
    if type(marker) is not str or _MARKER.fullmatch(marker) is None:
        raise ValueError("document fidelity reference marker is invalid")
    if type(expected) is not str:
        raise ValueError("document fidelity expected_markdown is invalid")
    if _canonical_markdown(expected).split("\n", 1)[0] != marker:
        raise ValueError("document fidelity reference marker must be first")
    for key in required_lists:
        if not isinstance(reference.get(key), list):
            raise ValueError(f"document fidelity reference is missing {key}")
    for formula in reference["formulas"]:
        if (
            not isinstance(formula, dict)
            or not isinstance(formula.get("accepted"), list)
            or not formula["accepted"]
            or not all(type(value) is str and value for value in formula["accepted"])
        ):
            raise ValueError("document fidelity formula alternatives are invalid")
    for code in reference["code_blocks"]:
        if (
            not isinstance(code, dict)
            or type(code.get("language")) is not str
            or type(code.get("body")) is not str
        ):
            raise ValueError("document fidelity code reference is invalid")
    for table in reference["tables"]:
        if (
            not isinstance(table, dict)
            or not isinstance(table.get("rows"), list)
            or not table["rows"]
            or not all(isinstance(row, list) and row for row in table["rows"])
        ):
            raise ValueError("document fidelity table reference is invalid")
    for key in ("ordered_anchors", "protected_spans", "forbidden_spans"):
        if not all(type(value) is str and value for value in reference[key]):
            raise ValueError(f"document fidelity {key} entries are invalid")


def _read_trial_records(
    path: Path,
    *,
    prediction_budget: list[int],
) -> dict[str, dict]:
    _require_bounded_file(
        path,
        maximum_bytes=_MAX_RECORD_FILE_BYTES,
        label="record",
    )
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"invalid document record at line {line_number}")
            sample_id = record.get("sample_id")
            if type(sample_id) is not str or sample_id in records:
                raise ValueError(
                    f"invalid or duplicate document sample ID at line {line_number}"
                )
            if len(records) >= _MAX_RECORDS_PER_TRIAL:
                raise ValueError("document fidelity record count limit exceeded")
            success = record.get("success")
            token_cap_hit = record.get("token_cap_hit", False)
            prediction = record.get("prediction")
            if type(success) is not bool:
                raise ValueError("document fidelity record success must be boolean")
            if type(token_cap_hit) is not bool:
                raise ValueError("document fidelity token cap flag must be boolean")
            if success is True and type(prediction) is not str:
                raise ValueError("document fidelity records require raw prediction strings")
            if (
                success is False
                and prediction is not None
                and type(prediction) is not str
            ):
                raise ValueError("document fidelity failed prediction must be a string")
            prediction_characters = len(prediction) if type(prediction) is str else 0
            if prediction_characters > _MAX_PREDICTION_CHARACTERS:
                raise ValueError("document fidelity prediction length limit exceeded")
            if prediction_characters > prediction_budget[0]:
                raise ValueError("document fidelity prediction budget exceeded")
            prediction_budget[0] -= prediction_characters
            records[sample_id] = record
    return records


def _read_records_provenance(
    records_path: Path,
    *,
    candidate_id: str,
    workload_fingerprint: str,
) -> dict:
    path = records_path.with_name("records-provenance.json")
    if not path.is_file():
        raise ValueError("document fidelity records provenance is missing")
    _require_bounded_file(
        path,
        maximum_bytes=_MAX_PROVENANCE_BYTES,
        label="provenance",
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("document fidelity records provenance is invalid")
    if (
        value.get("protocol") != "sustained-process-v1"
        or value.get("status") not in _SOURCE_STATUSES
        or value.get("candidate_id") != candidate_id
        or value.get("task") != "ocr"
        or value.get("phase") != "quality"
        or value.get("workload_class") != "generated_quality_control"
        or value.get("workload_fingerprint") != workload_fingerprint
        or value.get("records_sha256") != _sha256(records_path)
    ):
        raise ValueError("document fidelity records provenance does not match")
    config = value.get("config")
    if not isinstance(config, dict) or config.get("mode") != "source_faithful":
        raise ValueError("document fidelity records require source-faithful config")
    for key in ("config_index", "trial_index"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise ValueError(f"document fidelity provenance {key} is invalid")
    try:
        uuid.UUID(value.get("attempt_id", ""))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError("document fidelity provenance attempt ID is invalid") from error
    for key in (
        "attempt_key",
        "code_fingerprint",
        "environment_fingerprint",
        "controller_environment_fingerprint",
        "execution_policy_fingerprint",
    ):
        if (
            type(value.get(key)) is not str
            or re.fullmatch(r"[0-9a-f]{16}", value[key]) is None
        ):
            raise ValueError(f"document fidelity provenance {key} is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", value["workload_fingerprint"]) is None:
        raise ValueError("document fidelity workload fingerprint is invalid")
    return value


def _validate_trial_provenance(provenances: list[dict]) -> None:
    if len({value["attempt_id"] for value in provenances}) != len(provenances):
        raise ValueError("document fidelity trials require distinct attempt IDs")
    if len({value["attempt_key"] for value in provenances}) != len(provenances):
        raise ValueError("document fidelity trials require distinct attempt keys")
    if len({value["trial_index"] for value in provenances}) != len(provenances):
        raise ValueError("document fidelity trials require distinct trial indices")
    serialized_configs = {
        json.dumps(value["config"], sort_keys=True, separators=(",", ":"))
        for value in provenances
    }
    if len(serialized_configs) != 1:
        raise ValueError("document fidelity trials must use one configuration")
    for key, label in (
        ("config_index", "config index"),
        ("code_fingerprint", "code fingerprint"),
        ("environment_fingerprint", "environment fingerprint"),
        (
            "controller_environment_fingerprint",
            "controller environment fingerprint",
        ),
        ("execution_policy_fingerprint", "execution policy fingerprint"),
    ):
        if len({value[key] for value in provenances}) != 1:
            raise ValueError(f"document fidelity trials must use one {label}")


def _validate_source_status(
    source_status: str,
    *,
    expected_ids: set[str],
    records: dict[str, dict],
) -> None:
    successful_sample_count = sum(
        records[sample_id]["success"] is True for sample_id in expected_ids
    )
    expected_status = (
        "succeeded"
        if successful_sample_count == len(expected_ids)
        else "all_failed" if successful_sample_count == 0 else "partial_failure"
    )
    if source_status != expected_status:
        raise ValueError("document fidelity source status does not match records")


def _scorer_fingerprint() -> str:
    module_path = Path(__file__).resolve()
    return fingerprint_files(
        [
            module_path,
            module_path.with_name("fingerprint.py"),
            module_path.with_name("load_sustained_workload.py"),
            module_path.with_name("score_ocr_quality.py"),
            module_path.with_name("validate_public_summary.py"),
        ]
    )


def _require_bounded_file(path: Path, *, maximum_bytes: int, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"document fidelity {label} file is missing")
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"document fidelity {label} file budget exceeded")


def _canonical_markdown(value: str) -> str:
    """Normalize line endings and at most one terminal newline, nothing else."""

    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    return normalized[:-1] if normalized.endswith("\n") else normalized


def _score_sample(
    reference: dict,
    prediction: str,
    record: dict,
    *,
    edit_budget: list[int],
) -> dict:
    expected = _canonical_markdown(reference["expected_markdown"])
    markers = _MARKER.findall(prediction)
    first_line = prediction.split("\n", 1)[0]
    masked = _mask_fences(prediction)
    heading_matches = list(_HEADING.finditer(masked))
    headings = [match.group(0) for match in heading_matches]
    predicted_formulas = _extract_formulas(prediction)
    formula_matches, formula_expected = _match_formulas(
        reference["formulas"], predicted_formulas
    )
    predicted_code = _extract_fenced_code_blocks(prediction)
    code = _score_code(reference["code_blocks"], predicted_code)
    predicted_tables = _extract_gfm_tables(masked)
    table = _score_tables(reference["tables"], predicted_tables)
    anchor_hits, anchor_pairs = _score_ordered_anchors(
        prediction, reference["ordered_anchors"]
    )
    expected_tokens = Counter(_lexical_tokens(expected))
    predicted_tokens = Counter(_lexical_tokens(prediction))
    lexical_hits = sum((expected_tokens & predicted_tokens).values())
    return {
        "failed": record.get("success") is not True,
        "token_cap_hit": record.get("token_cap_hit") is True,
        "exact": prediction == expected,
        "marker_exact": first_line == reference["marker"] and len(markers) == 1,
        "heading_exact": headings == reference["headings"],
        "h1_violations": sum(len(match.group(1)) == 1 for match in heading_matches),
        "outer_fence_violations": int(_has_outer_markdown_fence(prediction)),
        "formula_matches": formula_matches,
        "formula_expected": formula_expected,
        "formula_predicted": len(predicted_formulas),
        "unicode_math_substitutions": sum(
            _mask_inline_code(_mask_fences(prediction)).count(character)
            for character in _FORMULA_UNICODE
        ),
        "table_shape_hits": table["shape_hits"],
        "table_expected": table["expected"],
        "table_cell_hits": table["cell_hits"],
        "table_cell_total": table["cell_total"],
        "html_table_count": len(re.findall(r"<\s*table\b", prediction, re.IGNORECASE)),
        "fenced_table_count": sum(
            bool(_extract_gfm_tables(block["body"])) for block in predicted_code
        ),
        "code_fence_hits": code["fence_hits"],
        "code_expected": code["expected"],
        "code_line_hits": code["line_hits"],
        "code_line_total": code["line_total"],
        "code_indent_hits": code["indent_hits"],
        "code_indent_total": code["indent_total"],
        "python_parse_hits": code["python_parse_hits"],
        "python_expected": code["python_expected"],
        "anchor_hits": anchor_hits,
        "anchor_pairs": anchor_pairs,
        "protected_hits": sum(
            _span_present(prediction, span) for span in reference["protected_spans"]
        ),
        "protected_total": len(reference["protected_spans"]),
        "forbidden_hits": sum(
            _span_present(prediction, span, case_sensitive=False)
            for span in reference["forbidden_spans"]
        ),
        "lexical_hits": lexical_hits,
        "lexical_expected": sum(expected_tokens.values()),
        "lexical_predicted": sum(predicted_tokens.values()),
        "reference_characters": len(expected),
        "edit_distance": _bounded_levenshtein(
            expected,
            prediction,
            edit_budget,
        ),
    }


def _bounded_levenshtein(
    left: str,
    right: str,
    edit_budget: list[int],
) -> int:
    if left == right or not left or not right:
        return _levenshtein(left, right)
    cells = len(left) * len(right)
    if cells > edit_budget[0]:
        raise ValueError("document fidelity edit-distance budget exceeded")
    edit_budget[0] -= cells
    return _levenshtein(left, right)


def _extract_formulas(value: str) -> list[str]:
    masked = _mask_inline_code(_mask_fences(value))
    formulas: list[tuple[int, str]] = []
    display_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"\$\$(.*?)\$\$", masked, re.DOTALL):
        formulas.append((match.start(), match.group(1)))
        display_spans.append(match.span())
    chars = list(masked)
    for start, end in display_spans:
        for index in range(start, end):
            if chars[index] != "\n":
                chars[index] = " "
    inline_source = "".join(chars)
    for match in re.finditer(
        r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", inline_source
    ):
        formulas.append((match.start(), match.group(1)))
    return [formula for _, formula in sorted(formulas)]


def _extract_fenced_code_blocks(value: str) -> list[dict[str, str]]:
    pattern = re.compile(
        rf"(?ms)^{re.escape(_FENCE)}([^\x60\n]*)\n(.*?)\n"
        rf"{re.escape(_FENCE)}[ \t]*$"
    )
    return [
        {"language": match.group(1).strip(), "body": match.group(2)}
        for match in pattern.finditer(value)
    ]


def _mask_fences(value: str) -> str:
    chars = list(value)
    pattern = re.compile(
        rf"(?ms)^{re.escape(_FENCE)}[^\x60\n]*\n.*?\n"
        rf"{re.escape(_FENCE)}[ \t]*$"
    )
    for match in pattern.finditer(value):
        for index in range(match.start(), match.end()):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def _mask_inline_code(value: str) -> str:
    chars = list(value)
    tick = re.escape(chr(96))
    pattern = re.compile(
        rf"(?<!{tick}){tick}(?!{tick})[^\n]*?(?<!{tick}){tick}(?!{tick})"
    )
    for match in pattern.finditer(value):
        for index in range(match.start(), match.end()):
            chars[index] = " "
    return "".join(chars)


def _extract_gfm_tables(value: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    lines = value.splitlines()
    index = 0
    while index < len(lines):
        if not _is_pipe_row(lines[index]):
            index += 1
            continue
        block = []
        while index < len(lines) and _is_pipe_row(lines[index]):
            block.append(_parse_pipe_row(lines[index]))
            index += 1
        if (
            len(block) >= 2
            and len({len(row) for row in block}) == 1
            and all(re.fullmatch(r":?-{3,}:?", cell) for cell in block[1])
        ):
            tables.append(block)
    return tables


def _is_pipe_row(line: str) -> bool:
    if line.startswith("\t"):
        return False
    leading_spaces = len(line) - len(line.lstrip(" "))
    if leading_spaces > 3:
        return False
    stripped = line[leading_spaces:].rstrip()
    return (
        stripped.startswith("|")
        and stripped.endswith("|")
        and stripped.count("|") >= 2
    )


def _parse_pipe_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip()[1:-1].split("|")]


def _score_tables(expected: list[dict], predicted: list[list[list[str]]]) -> dict:
    shape_hits = 0
    cell_hits = 0
    cell_total = 0
    for index in range(max(len(expected), len(predicted))):
        expected_rows = expected[index]["rows"] if index < len(expected) else []
        predicted_rows = predicted[index] if index < len(predicted) else []
        expected_shape = [len(row) for row in expected_rows]
        predicted_shape = [len(row) for row in predicted_rows]
        shape_hits += bool(expected_rows and expected_shape == predicted_shape)
        cell_total += max(
            sum(len(row) for row in expected_rows),
            sum(len(row) for row in predicted_rows),
        )
        for row_index, row in enumerate(expected_rows):
            for column_index, cell in enumerate(row):
                if (
                    row_index < len(predicted_rows)
                    and column_index < len(predicted_rows[row_index])
                    and predicted_rows[row_index][column_index] == cell
                ):
                    cell_hits += 1
    return {
        "shape_hits": shape_hits,
        "expected": max(len(expected), len(predicted)),
        "cell_hits": cell_hits,
        "cell_total": cell_total,
    }


def _score_code(expected: list[dict], predicted: list[dict]) -> dict:
    has_code = bool(expected or predicted)
    fence_hits = int(
        has_code
        and len(expected) == len(predicted)
        and all(
            left["language"] == right["language"] and left["body"] == right["body"]
            for left, right in zip(expected, predicted)
        )
    )
    line_hits = 0
    line_total = 0
    indent_hits = 0
    indent_total = 0
    python_parse_hits = 0
    python_expected = sum(block["language"] == "python" for block in expected)
    for index in range(max(len(expected), len(predicted))):
        expected_block = expected[index] if index < len(expected) else {"body": ""}
        predicted_block = predicted[index] if index < len(predicted) else {"body": ""}
        expected_lines = expected_block["body"].split("\n")
        predicted_lines = predicted_block["body"].split("\n")
        line_total += max(len(expected_lines), len(predicted_lines))
        indent_total += max(len(expected_lines), len(predicted_lines))
        for line_index in range(min(len(expected_lines), len(predicted_lines))):
            expected_line = expected_lines[line_index]
            predicted_line = predicted_lines[line_index]
            line_hits += expected_line == predicted_line
            indent_hits += _leading_whitespace(expected_line) == _leading_whitespace(
                predicted_line
            )
        if index < len(expected) and expected[index]["language"] == "python":
            if index < len(predicted):
                try:
                    ast.parse(predicted[index]["body"])
                except SyntaxError:
                    pass
                else:
                    python_parse_hits += 1
    return {
        "fence_hits": fence_hits,
        "expected": int(has_code),
        "line_hits": line_hits,
        "line_total": line_total,
        "indent_hits": indent_hits,
        "indent_total": indent_total,
        "python_parse_hits": python_parse_hits,
        "python_expected": python_expected,
    }


def _leading_whitespace(value: str) -> str:
    return value[: len(value) - len(value.lstrip(" \t"))]


def _match_formulas(expected: list[dict], predicted: list[str]) -> tuple[int, int]:
    unmatched = [
        {_normalize_formula(value) for value in group["accepted"]} for group in expected
    ]
    matches = 0
    for formula in predicted:
        normalized = _normalize_formula(formula)
        for index, accepted in enumerate(unmatched):
            if accepted and normalized in accepted:
                matches += 1
                unmatched[index] = set()
                break
    return matches, len(expected)


def _normalize_formula(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _score_ordered_anchors(value: str, anchors: list[str]) -> tuple[int, int]:
    positions = [value.find(anchor) for anchor in anchors]
    pairs = 0
    hits = 0
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            pairs += 1
            hits += (
                positions[left] >= 0
                and positions[right] >= 0
                and positions[left] < positions[right]
            )
    return hits, pairs


def _span_present(
    value: str, span: str, *, case_sensitive: bool = True
) -> bool:
    left = r"(?<!\w)" if span[0].isalnum() else ""
    right = r"(?!\w)" if span[-1].isalnum() else ""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.search(left + re.escape(span) + right, value, flags) is not None


def _lexical_tokens(value: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", value, re.UNICODE)


def _has_outer_markdown_fence(value: str) -> bool:
    first_line = value.split("\n", 1)[0].strip().casefold()
    if first_line not in {_FENCE + "markdown", _FENCE + "md"}:
        return False
    return value.rstrip().endswith(_FENCE)


def _aggregate_scores(
    *,
    scores: list[dict],
    references: dict[str, dict],
    trials: list[dict[str, dict]],
    predictions_by_sample: dict[str, list[str]],
) -> dict:
    sample_count = len(references)
    trial_count = len(trials)
    total = len(scores)

    def ratio(numerator: int, denominator: int, *, empty: float = 1.0) -> float:
        return numerator / denominator if denominator else empty

    formula_matches = sum(score["formula_matches"] for score in scores)
    formula_expected = sum(score["formula_expected"] for score in scores)
    formula_predicted = sum(score["formula_predicted"] for score in scores)
    table_shape_hits = sum(score["table_shape_hits"] for score in scores)
    table_expected = sum(score["table_expected"] for score in scores)
    table_cell_hits = sum(score["table_cell_hits"] for score in scores)
    table_cell_total = sum(score["table_cell_total"] for score in scores)
    code_fence_hits = sum(score["code_fence_hits"] for score in scores)
    code_expected = sum(score["code_expected"] for score in scores)
    code_line_hits = sum(score["code_line_hits"] for score in scores)
    code_line_total = sum(score["code_line_total"] for score in scores)
    code_indent_hits = sum(score["code_indent_hits"] for score in scores)
    code_indent_total = sum(score["code_indent_total"] for score in scores)
    python_parse_hits = sum(score["python_parse_hits"] for score in scores)
    python_expected = sum(score["python_expected"] for score in scores)
    anchor_hits = sum(score["anchor_hits"] for score in scores)
    anchor_pairs = sum(score["anchor_pairs"] for score in scores)
    protected_hits = sum(score["protected_hits"] for score in scores)
    protected_total = sum(score["protected_total"] for score in scores)
    lexical_hits = sum(score["lexical_hits"] for score in scores)
    lexical_expected = sum(score["lexical_expected"] for score in scores)
    lexical_predicted = sum(score["lexical_predicted"] for score in scores)
    reference_characters = sum(score["reference_characters"] for score in scores)
    edit_distance = sum(score["edit_distance"] for score in scores)
    repeat_hits = sum(
        len(predictions) >= 2 and len(set(predictions)) == 1
        for predictions in predictions_by_sample.values()
    )
    metrics = {
        "sample_count": sample_count,
        "trial_count": trial_count,
        "failure_count": sum(score["failed"] for score in scores),
        "token_cap_hit_count": sum(score["token_cap_hit"] for score in scores),
        "exact_document_fraction": ratio(
            sum(score["exact"] for score in scores), total, empty=0.0
        ),
        "repeat_exact_fraction": ratio(repeat_hits, sample_count, empty=0.0),
        "marker_exact_fraction": ratio(
            sum(score["marker_exact"] for score in scores), total, empty=0.0
        ),
        "page_sequence_pass": all(
            _trial_page_sequence_pass(references, records) for records in trials
        ),
        "heading_exact_fraction": ratio(
            sum(score["heading_exact"] for score in scores), total, empty=0.0
        ),
        "h1_violation_count": sum(score["h1_violations"] for score in scores),
        "outer_fence_violation_count": sum(
            score["outer_fence_violations"] for score in scores
        ),
        "formula_exact_precision": ratio(formula_matches, formula_predicted),
        "formula_exact_recall": ratio(formula_matches, formula_expected),
        "unicode_math_substitution_count": sum(
            score["unicode_math_substitutions"] for score in scores
        ),
        "table_shape_pass_fraction": ratio(table_shape_hits, table_expected),
        "table_cell_exact_fraction": ratio(table_cell_hits, table_cell_total),
        "html_table_count": sum(score["html_table_count"] for score in scores),
        "fenced_table_count": sum(score["fenced_table_count"] for score in scores),
        "code_fence_pass_fraction": ratio(code_fence_hits, code_expected),
        "code_line_exact_fraction": ratio(code_line_hits, code_line_total),
        "code_indentation_exact_fraction": ratio(code_indent_hits, code_indent_total),
        "python_parse_fraction": ratio(python_parse_hits, python_expected),
        "reading_order_pair_accuracy": ratio(anchor_hits, anchor_pairs),
        "protected_span_recall": ratio(protected_hits, protected_total),
        "forbidden_span_hit_count": sum(score["forbidden_hits"] for score in scores),
        "lexical_precision": ratio(lexical_hits, lexical_predicted),
        "lexical_recall": ratio(lexical_hits, lexical_expected),
        "markdown_character_error_rate": ratio(
            edit_distance, reference_characters, empty=0.0
        ),
    }
    metrics["semantic_gate_pass"] = _semantic_gate_pass(metrics)
    metrics["profile_gate_pass"] = (
        metrics["semantic_gate_pass"]
        and metrics["exact_document_fraction"] == 1.0
    )
    return metrics


def _trial_page_sequence_pass(
    references: dict[str, dict], records: dict[str, dict]
) -> bool:
    expected_pages: list[int] = []
    predicted_pages: list[int] = []
    for sample_id, reference in references.items():
        if "page_number" not in reference:
            continue
        expected_pages.append(reference["page_number"])
        raw_prediction = records[sample_id].get("prediction")
        if type(raw_prediction) is not str:
            return False
        prediction = _canonical_markdown(raw_prediction)
        first_line = prediction.split("\n", 1)[0]
        match = _PAGE_MARKER.fullmatch(first_line)
        if match is None:
            return False
        predicted_pages.append(int(match.group(1)))
    return (
        predicted_pages == expected_pages
        and predicted_pages == sorted(predicted_pages)
    )


def _semantic_gate_pass(metrics: dict) -> bool:
    exact_one = (
        "repeat_exact_fraction",
        "marker_exact_fraction",
        "heading_exact_fraction",
        "formula_exact_precision",
        "formula_exact_recall",
        "table_shape_pass_fraction",
        "table_cell_exact_fraction",
        "code_fence_pass_fraction",
        "code_line_exact_fraction",
        "code_indentation_exact_fraction",
        "python_parse_fraction",
        "reading_order_pair_accuracy",
        "protected_span_recall",
    )
    zero_counts = (
        "failure_count",
        "token_cap_hit_count",
        "h1_violation_count",
        "outer_fence_violation_count",
        "unicode_math_substitution_count",
        "html_table_count",
        "fenced_table_count",
        "forbidden_span_hit_count",
    )
    return (
        metrics["sample_count"] == 3
        and metrics["trial_count"] >= 2
        and metrics["page_sequence_pass"]
        and all(metrics[key] == 1.0 for key in exact_one)
        and all(metrics[key] == 0 for key in zero_counts)
        and metrics["lexical_precision"] >= 0.99
        and metrics["lexical_recall"] >= 0.99
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
