"""Generate deterministic 1080p OCR quality controls with exact references."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "inputs" / "generated" / "ocr_quality"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
FONT_ROOT = Path("C:/Windows/Fonts")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    samples = [
        _projected_mixed_slide(),
        _code_and_formula_slide(),
        _handwriting_board(),
        _blurred_projection(),
        _occluded_slide(),
        _dense_table_slide(),
        _negative_diagram(),
    ]
    items = []
    references = {}
    for sample in samples:
        destination = OUTPUT_ROOT / f"{sample['id']}.png"
        sample["image"].save(destination, format="PNG", optimize=False)
        items.append(
            {
                "id": sample["id"],
                "path": destination.name,
                "expected_text": bool(sample["lines"]),
            }
        )
        references[sample["id"]] = {
            "category": sample["category"],
            "lines": sample["lines"],
            "required_tokens": sample["required_tokens"],
            "image_sha256": _sha256(destination),
        }

    warmup_path = OUTPUT_ROOT / "warmup.png"
    _warmup_image().save(warmup_path, format="PNG", optimize=False)
    manifest = {
        "schema_version": 1,
        "task": "ocr",
        "workload_class": "generated_quality_control",
        "disclosure": (
            "Deterministic rendered controls are exact-reference tests, not a "
            "substitute for camera-captured lecture images."
        ),
        "warmup": {
            "id": "warmup",
            "path": warmup_path.name,
            "expected_text": True,
        },
        "items": items,
        "references": references,
        "generator": {
            "protocol": "rendered-ocr-controls.v1",
            "canvas_width": 1920,
            "canvas_height": 1080,
            "pillow_version": version("pillow"),
            "font_files": {
                name: _sha256(FONT_ROOT / name)
                for name in ("msyh.ttc", "consola.ttf", "Inkfree.ttf")
            },
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(MANIFEST_PATH)


def _projected_mixed_slide() -> dict:
    lines = [
        "Lecture 07 · Control Systems",
        "状态空间模型  State-space model",
        "x(k+1) = A x(k) + B u(k)",
        "Sampling rate: 48 kHz | latency ≤ 12.5 ms",
        "结论：稳定性取决于 eigenvalues.",
    ]
    image = Image.new("RGB", (1920, 1080), "#17233d")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((115, 75, 1805, 1005), radius=18, fill="#f6f2e8")
    draw.rectangle((115, 75, 1805, 190), fill="#2457a7")
    draw.text((175, 105), lines[0], font=_font("msyh.ttc", 55), fill="white")
    _draw_lines(draw, lines[1:], x=190, y=260, font=_font("msyh.ttc", 56), gap=125)
    return _sample("projected_mixed", "projected_mixed", image, lines)


def _code_and_formula_slide() -> dict:
    lines = [
        "def kalman_update(x, P, z):",
        "    K = P @ H.T @ inv(H @ P @ H.T + R)",
        "    return x + K @ (z - H @ x)",
        "P_k = (I - K_k H) P_(k-1)",
        "复杂度: O(n³)   test_id = 0x2A7F",
    ]
    image = Image.new("RGB", (1920, 1080), "#1e1f24")
    draw = ImageDraw.Draw(image)
    draw.rectangle((70, 55, 1850, 1025), outline="#63676f", width=4)
    _draw_lines(draw, lines[:3], x=130, y=135, font=_font("consola.ttf", 43), gap=100, fill="#e8e8e8")
    draw.line((115, 500, 1805, 500), fill="#63676f", width=3)
    _draw_lines(draw, lines[3:], x=130, y=585, font=_font("msyh.ttc", 54), gap=135, fill="#f0d060")
    return _sample("code_formula", "code_formula", image, lines)


def _handwriting_board() -> dict:
    lines = [
        "Gradient descent -> minimize loss",
        "η = 0.01,  batch = 64",
        "手写提示：先检查单位",
        "∂L/∂w = 2(w - w*)",
    ]
    image = Image.new("RGB", (1920, 1080), "#315b49")
    draw = ImageDraw.Draw(image)
    for y in range(150, 1000, 155):
        draw.line((90, y, 1830, y), fill="#416b59", width=2)
    _draw_lines(draw, lines[:2], x=150, y=150, font=_font("Inkfree.ttf", 69), gap=175, fill="#f5f0d8")
    _draw_lines(draw, lines[2:], x=155, y=530, font=_font("msyh.ttc", 61), gap=175, fill="#f5f0d8")
    return _sample("handwriting_board", "handwriting", image, lines)


def _blurred_projection() -> dict:
    lines = [
        "Signal Processing / 信号处理",
        "SNR = 20 log10(Psignal / Pnoise)",
        "截止频率 fc = 2.40 GHz",
        "Dataset split: 70% / 15% / 15%",
    ]
    image = Image.new("RGB", (1920, 1080), "#d9d4c6")
    draw = ImageDraw.Draw(image)
    draw.polygon(((95, 100), (1815, 55), (1750, 1010), (145, 970)), fill="#f7f4e9")
    _draw_lines(draw, lines, x=210, y=210, font=_font("msyh.ttc", 58), gap=165, fill="#414141")
    image = image.filter(ImageFilter.GaussianBlur(radius=1.8))
    image = ImageEnhance.Contrast(image).enhance(0.72)
    return _sample("blurred_projection", "blur", image, lines)


def _occluded_slide() -> dict:
    lines = [
        "Robotics Seminar 2026",
        "Pose: [x, y, θ] = [1.25, -0.80, 90°]",
        "传感器融合: camera + IMU + LiDAR",
        "Next checkpoint: 14:35:20",
    ]
    image = Image.new("RGB", (1920, 1080), "#eceff4")
    draw = ImageDraw.Draw(image)
    _draw_lines(draw, lines, x=145, y=145, font=_font("msyh.ttc", 57), gap=170, fill="#18243a")
    draw.ellipse((1060, 610, 1360, 1050), fill="#30343a")
    draw.ellipse((1160, 490, 1260, 590), fill="#30343a")
    return _sample("occluded_slide", "occlusion", image, lines)


def _dense_table_slide() -> dict:
    lines = [
        "Model Precision Recall Throughput",
        "Tiny 92.4% 88.1% 28,743 img/h",
        "Small 95.2% 91.7% 13,952 img/h",
        "备注：CPU-only, batch = 1",
    ]
    image = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(image)
    x_positions = (100, 650, 1030, 1370, 1820)
    y_positions = (155, 330, 505, 680)
    for x in x_positions:
        draw.line((x, 120, x, 690), fill="#274060", width=4)
    for y in (120, *y_positions):
        draw.line((100, y, 1820, y), fill="#274060", width=4)
    font = _font("msyh.ttc", 42)
    rows = (
        ("Model", "Precision", "Recall", "Throughput"),
        ("Tiny", "92.4%", "88.1%", "28,743 img/h"),
        ("Small", "95.2%", "91.7%", "13,952 img/h"),
    )
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            draw.text(
                (x_positions[column_index] + 25, 185 + row_index * 175),
                cell,
                font=font,
                fill="#132238",
            )
    draw.text((120, 800), lines[3], font=_font("msyh.ttc", 52), fill="#132238")
    return _sample("dense_table", "dense_table", image, lines)


def _negative_diagram() -> dict:
    image = Image.new("RGB", (1920, 1080), "#e9edf2")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((180, 170, 650, 500), radius=40, fill="#7a9cc6")
    draw.rounded_rectangle((1260, 580, 1740, 900), radius=40, fill="#90b68d")
    draw.ellipse((760, 350, 1160, 750), fill="#d1a16b")
    draw.line((650, 335, 805, 465), fill="#4f5f70", width=18)
    draw.line((1110, 650, 1260, 720), fill="#4f5f70", width=18)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.1))
    return _sample("negative_diagram", "negative", image, [])


def _sample(sample_id: str, category: str, image: Image.Image, lines: list[str]) -> dict:
    return {
        "id": sample_id,
        "category": category,
        "image": image,
        "lines": lines,
        "required_tokens": _required_tokens(lines),
    }


def _required_tokens(lines: list[str]) -> list[str]:
    return [line for line in lines if line]


def _draw_lines(draw, lines, *, x, y, font, gap, fill="#202020") -> None:
    for index, line in enumerate(lines):
        draw.text((x, y + index * gap), line, font=font, fill=fill)


def _font(filename: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"required quality-control font is missing: {filename}")
    return ImageFont.truetype(str(path), size=size)


def _warmup_image() -> Image.Image:
    image = Image.new("RGB", (640, 360), "white")
    ImageDraw.Draw(image).text((40, 130), "OCR warmup 2026", font=_font("msyh.ttc", 40), fill="black")
    return image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
