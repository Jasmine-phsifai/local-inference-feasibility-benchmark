import json
import math
import struct
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/arial.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _write_tone_audio(path: Path, seconds: float = 12.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        frames = bytearray()
        for index in range(int(seconds * rate)):
            value = int(1200 * math.sin(2 * math.pi * 220 * index / rate))
            frames.extend(struct.pack("<h", value))
        output.writeframes(frames)


def generate_inputs(root: Path, public_root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    texts = [
        "Local inference benchmark 2026\nChapter 1: vectors and motion\n速度 velocity = distance / time",
        "课程板书 OCR 测试\n线性代数：Ax = b\nEnglish + 中文 mixed lecture slide",
        "Table\nModel     Images/hour\nTiny      12000\nMedium     5000",
    ]
    images = []
    for index, text in enumerate(texts, start=1):
        path = root / f"synthetic_slide_{index}.png"
        image = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(image)
        draw.multiline_text((70, 70), text, fill="black", font=_font(42), spacing=24)
        image.save(path)
        images.append({"path": str(path), "kind": "synthetic_rendered_text", "representative": False})
    public_audio = public_root / "jfk.wav"
    if public_audio.exists():
        audio_path = public_audio
        audio_kind = "public_domain_jfk_speech_from_whisper_cpp"
        audio_duration = 11.0
    else:
        audio_path = root / "synthetic_tone_12s.wav"
        _write_tone_audio(audio_path)
        audio_kind = "synthetic_tone_no_speech"
        audio_duration = 12.0
    manifest = {
        "disclosure": "Fallback synthetic inputs; not representative lecture data and not usable for quality scoring.",
        "images": images,
        "audio": [{"path": str(audio_path), "duration_seconds": audio_duration, "kind": audio_kind, "representative": False,
                   "source": "https://github.com/ggml-org/whisper.cpp/blob/b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav" if public_audio.exists() else None}],
    }
    return manifest


def write_manifest(root: Path, manifest_path: Path) -> None:
    manifest = generate_inputs(root, root.parent / "public")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
