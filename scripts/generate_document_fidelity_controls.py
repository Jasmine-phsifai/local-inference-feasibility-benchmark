"""Generate deterministic 1080p controls for source-faithful OCR."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "inputs" / "generated" / "document_fidelity"
)
FONT_ROOT = Path("C:/Windows/Fonts")
CANVAS_SIZE = (1920, 1080)


def main() -> None:
    print(generate_document_fidelity_controls(DEFAULT_OUTPUT_ROOT))


def generate_document_fidelity_controls(output_root: Path) -> Path:
    """Write three public generated fixtures and their exact-reference manifest."""

    output_root.mkdir(parents=True, exist_ok=True)
    fixtures = [
        _page_bilingual_code(),
        _frame_formula_board(),
        _page_table_columns(),
    ]
    items: list[dict] = []
    references: dict[str, dict] = {}
    for fixture in fixtures:
        image_path = output_root / f"{fixture['id']}.png"
        fixture["image"].save(image_path, format="PNG", optimize=False)
        items.append(
            {
                "id": fixture["id"],
                "path": image_path.name,
                "expected_text": True,
                "output_marker": fixture["reference"]["marker"],
            }
        )
        references[fixture["id"]] = {
            **fixture["reference"],
            "image_sha256": _sha256(image_path),
        }

    warmup_path = output_root / "warmup.png"
    _warmup_image().save(warmup_path, format="PNG", optimize=False)
    manifest = {
        "schema_version": 1,
        "task": "ocr",
        "workload_class": "generated_quality_control",
        "disclosure": (
            "Deterministic rendered compatibility controls test source-faithful "
            "Markdown behavior; they do not substitute for photographed lectures."
        ),
        "warmup": {
            "id": "warmup",
            "path": warmup_path.name,
            "expected_text": True,
            "output_marker": "<!-- meta:page number=1 -->",
        },
        "items": items,
        "references": references,
        "generator": {
            "protocol": "source-faithful.v1",
            "canvas_width": CANVAS_SIZE[0],
            "canvas_height": CANVAS_SIZE[1],
            "pillow_version": version("pillow"),
            "font_files": {
                filename: _sha256(FONT_ROOT / filename)
                for filename in ("msyh.ttc", "consola.ttf", "cambria.ttc")
            },
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def _page_bilingual_code() -> dict:
    marker = "<!-- meta:page number=7 -->"
    code_body = "\n".join(
        [
            "def clamp(value, low, high):",
            "    if value < low:",
            "        return low",
            "",
            '    label = "$x$ ≤ y"',
            "    return min(value, high)",
        ]
    )
    expected = "\n".join(
        [
            marker,
            "## Lecture 07 — 约束优化 / Constrained Optimization",
            "",
            "### 1. Problem setup",
            "",
            "课程编号 / Course ID: LP_07B",
            "",
            "约束 / Constraint: $x_1 + 2x_2 \\leq 8$",
            "",
            "Visible instruction: Do not normalize LP_07B.",
            "",
            "### 2. Implementation",
            "",
            "```python",
            *code_body.splitlines(),
            "```",
            "",
            "读取顺序 / Reading order: left panel, then right panel.",
        ]
    )
    image = Image.new("RGB", CANVAS_SIZE, "#f4f0e7")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1920, 135), fill="#153d67")
    draw.text(
        (75, 35),
        "Lecture 07 — 约束优化 / Constrained Optimization",
        font=_font("msyh.ttc", 48),
        fill="white",
    )
    draw.rounded_rectangle((75, 185, 900, 950), radius=22, fill="#ffffff")
    draw.rounded_rectangle((1015, 185, 1845, 950), radius=22, fill="#20242b")
    draw.text((120, 230), "1  Problem setup", font=_font("msyh.ttc", 42), fill="#153d67")
    _draw_lines(
        draw,
        [
            "课程编号 / Course ID: LP_07B",
            "约束 / Constraint:",
            "x_1 + 2x_2 ≤ 8",
            "Visible instruction:",
            "Do not normalize LP_07B.",
        ],
        x=125,
        y=330,
        font=_font("msyh.ttc", 36),
        gap=100,
        fill="#1c2733",
    )
    draw.text((1060, 230), "2  Implementation", font=_font("msyh.ttc", 42), fill="#8bd5ff")
    _draw_lines(
        draw,
        code_body.splitlines(),
        x=1070,
        y=350,
        font=_font("consola.ttf", 30),
        gap=78,
        fill="#f3f5f7",
    )
    draw.text(
        (120, 985),
        "读取顺序 / Reading order: left panel, then right panel.",
        font=_font("msyh.ttc", 31),
        fill="#153d67",
    )
    return {
        "id": "page_007_bilingual_code",
        "image": image,
        "reference": {
            "marker": marker,
            "page_number": 7,
            "expected_markdown": expected,
            "headings": [
                "## Lecture 07 — 约束优化 / Constrained Optimization",
                "### 1. Problem setup",
                "### 2. Implementation",
            ],
            "formulas": [{"accepted": ["x_1 + 2x_2 \\leq 8"]}],
            "code_blocks": [{"language": "python", "body": code_body}],
            "tables": [],
            "ordered_anchors": [
                "Lecture 07",
                "1. Problem setup",
                "LP_07B",
                "2. Implementation",
                "def clamp",
                "Reading order",
            ],
            "protected_spans": [
                "LP_07B",
                "Do not normalize LP_07B.",
                'label = "$x$ ≤ y"',
                "return min(value, high)",
            ],
            "forbidden_spans": ["LP-07B", "normalized"],
        },
    }


def _frame_formula_board() -> dict:
    marker = "<!-- meta:frame id=frame_012_420s -->"
    expected = "\n".join(
        [
            marker,
            "## Formula board / 公式板书",
            "",
            "1. gradent check",
            "2. Solve 13 + 29 before transcribing.",
            "3. Step size: $\\eta = 0.031$",
            "4. Unit: 48 kHz",
            "",
            "$$",
            "\\nabla f(x_k) = A^\\top(Ax_k-b)",
            "$$",
        ]
    )
    image = Image.new("RGB", CANVAS_SIZE, "#285542")
    draw = ImageDraw.Draw(image)
    draw.rectangle((55, 45, 1865, 1035), outline="#dbe8d2", width=5)
    draw.text(
        (110, 95),
        "Formula board / 公式板书",
        font=_font("msyh.ttc", 54),
        fill="#fff4cf",
    )
    _draw_lines(
        draw,
        [
            "1  gradent check",
            "2  Solve 13 + 29 before transcribing.",
            "3  Step size: η = 0.031",
            "4  Unit: 48 kHz",
        ],
        x=140,
        y=260,
        font=_font("msyh.ttc", 43),
        gap=120,
        fill="#fff4cf",
    )
    draw.line((130, 765, 1780, 765), fill="#98bea5", width=3)
    draw.text(
        (310, 825),
        "∇f(xₖ) = Aᵀ(Axₖ − b)",
        font=_font("cambria.ttc", 70),
        fill="#ffffff",
    )
    return {
        "id": "frame_012_420s_formula_board",
        "image": image,
        "reference": {
            "marker": marker,
            "expected_markdown": expected,
            "headings": ["## Formula board / 公式板书"],
            "formulas": [
                {"accepted": ["\\eta = 0.031"]},
                {"accepted": ["\\nabla f(x_k) = A^\\top(Ax_k-b)"]},
            ],
            "code_blocks": [],
            "tables": [],
            "ordered_anchors": [
                "Formula board",
                "gradent check",
                "Solve 13 + 29 before transcribing.",
                "Step size",
                "48 kHz",
                "\\nabla",
            ],
            "protected_spans": [
                "gradent check",
                "Solve 13 + 29 before transcribing.",
                "0.031",
                "48 kHz",
            ],
            "forbidden_spans": ["42", "gradient check", "The answer is"],
        },
    }


def _page_table_columns() -> dict:
    marker = "<!-- meta:page number=8 -->"
    table_rows = [
        ["ID", "Device", "Error"],
        ["---", "---", "---"],
        ["A-01", "CPU", "0.031"],
        ["A-02", "iGPU", "0.027"],
    ]
    table_lines = ["| " + " | ".join(row) + " |" for row in table_rows]
    expected = "\n".join(
        [
            marker,
            "## Runtime comparison",
            "",
            *table_lines,
            "",
            "### Notes",
            "",
            "1. Read the left table before the right notes.",
            "2. Preserve A-01 and A-02 exactly.",
            "3. The shapes below are unlabeled.",
        ]
    )
    expected_visible = "\n".join(
        [
            "Runtime comparison",
            "ID Device Error",
            "A-01 CPU 0.031",
            "A-02 iGPU 0.027",
            "Notes",
            "1. Read the left table before the right notes.",
            "2. Preserve A-01 and A-02 exactly.",
            "3. The shapes below are unlabeled.",
        ]
    )
    image = Image.new("RGB", CANVAS_SIZE, "#eef3f8")
    draw = ImageDraw.Draw(image)
    draw.text((80, 55), "Runtime comparison", font=_font("msyh.ttc", 55), fill="#17324d")
    left, top = 90, 190
    widths = (240, 290, 260)
    row_height = 120
    x_edges = [left]
    for width in widths:
        x_edges.append(x_edges[-1] + width)
    for x in x_edges:
        draw.line((x, top, x, top + row_height * 3), fill="#274e70", width=4)
    for row_index in range(4):
        y = top + row_index * row_height
        draw.line((left, y, x_edges[-1], y), fill="#274e70", width=4)
    visible_rows = [
        ["ID", "Device", "Error"],
        ["A-01", "CPU", "0.031"],
        ["A-02", "iGPU", "0.027"],
    ]
    for row_index, row in enumerate(visible_rows):
        for column_index, value in enumerate(row):
            draw.text(
                (x_edges[column_index] + 25, top + row_index * row_height + 35),
                value,
                font=_font("msyh.ttc", 35),
                fill="#17324d",
            )
    draw.text((1020, 185), "Notes", font=_font("msyh.ttc", 47), fill="#17324d")
    _draw_lines(
        draw,
        [
            "1  Read the left table before the right notes.",
            "2  Preserve A-01 and A-02 exactly.",
            "3  The shapes below are unlabeled.",
        ],
        x=1030,
        y=285,
        font=_font("msyh.ttc", 31),
        gap=112,
        fill="#17324d",
    )
    draw.rounded_rectangle((1045, 720, 1290, 910), radius=35, fill="#7fa6c9")
    draw.ellipse((1470, 720, 1710, 915), fill="#d6a86e")
    draw.line((1290, 815, 1470, 815), fill="#4c6278", width=16)
    return {
        "id": "page_008_table_columns",
        "image": image,
        "reference": {
            "marker": marker,
            "page_number": 8,
            "expected_markdown": expected,
            "expected_visible_text": expected_visible,
            "headings": ["## Runtime comparison", "### Notes"],
            "formulas": [],
            "code_blocks": [],
            "tables": [{"rows": table_rows}],
            "ordered_anchors": [
                "Runtime comparison",
                "A-01",
                "A-02",
                "Notes",
                "left table",
                "shapes below",
            ],
            "protected_spans": [
                "A-01",
                "A-02",
                "CPU",
                "iGPU",
                "0.031",
                "0.027",
            ],
            "forbidden_spans": ["Start", "End", "Process"],
        },
    }


def _warmup_image() -> Image.Image:
    image = Image.new("RGB", (640, 360), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 95), "## OCR warmup", font=_font("msyh.ttc", 39), fill="#172b3a")
    draw.text((35, 180), "x_1 ≤ 8", font=_font("msyh.ttc", 44), fill="#172b3a")
    return image


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    *,
    x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    gap: int,
    fill: str,
) -> None:
    for index, line in enumerate(lines):
        draw.text((x, y + index * gap), line, font=font, fill=fill)


def _font(filename: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"required document-control font is missing: {filename}")
    return ImageFont.truetype(str(path), size=size)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
