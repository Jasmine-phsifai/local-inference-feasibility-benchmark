"""Extract ignored generic audio and frame samples from one local lecture."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


AUDIO_DURATIONS_SECONDS = (900, 900, 900, 900, 1800)
FRAME_COUNT = 36
EDGE_MARGIN_SECONDS = 300


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workload-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--ffprobe", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.source, args.ffmpeg, args.ffprobe):
        if not path.is_file():
            raise FileNotFoundError("required private sampling input is missing")

    duration = _media_duration(args.ffprobe, args.source)
    audio_offsets = _nonoverlapping_offsets(duration, AUDIO_DURATIONS_SECONDS)
    frame_offsets = _even_offsets(duration, FRAME_COUNT)
    audio_dir = args.output_dir / "audio"
    frame_dir = args.output_dir / "frames"
    audio_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    args.workload_dir.mkdir(parents=True, exist_ok=True)

    audio_items = []
    for index, (offset, segment_seconds) in enumerate(
        zip(audio_offsets, AUDIO_DURATIONS_SECONDS, strict=True),
        start=1,
    ):
        sample_id = f"audio_{index:03d}"
        destination = audio_dir / f"{sample_id}.wav"
        _extract_audio(args.ffmpeg, args.source, offset, segment_seconds, destination)
        audio_items.append(
            {
                "id": sample_id,
                "path": str(destination.resolve()),
                "duration_seconds": _media_duration(args.ffprobe, destination),
                "expected_speech": True,
            }
        )

    image_items = []
    for index, offset in enumerate(frame_offsets, start=1):
        sample_id = f"image_{index:03d}"
        destination = frame_dir / f"{sample_id}.png"
        _extract_frame(args.ffmpeg, args.source, offset, destination)
        image_items.append(
            {
                "id": sample_id,
                "path": str(destination.resolve()),
                "expected_text": False,
            }
        )

    _write_json(
        args.workload_dir / "local_private_asr.json",
        _workload("asr", audio_items),
    )
    _write_json(
        args.workload_dir / "local_private_ocr.json",
        _workload("ocr", image_items),
    )
    _write_json(
        args.output_dir / "private-source-offsets.json",
        {
            "schema_version": 1,
            "source_duration_seconds": duration,
            "audio": [
                {
                    "id": item["id"],
                    "start_seconds": offset,
                    "requested_duration_seconds": segment_seconds,
                }
                for item, offset, segment_seconds in zip(
                    audio_items,
                    audio_offsets,
                    AUDIO_DURATIONS_SECONDS,
                    strict=True,
                )
            ],
            "frames": [
                {"id": item["id"], "start_seconds": offset}
                for item, offset in zip(image_items, frame_offsets, strict=True)
            ],
        },
    )
    print(
        json.dumps(
            {
                "source_duration_seconds": duration,
                "audio_sample_count": len(audio_items),
                "audio_duration_seconds": sum(
                    item["duration_seconds"] for item in audio_items
                ),
                "frame_sample_count": len(image_items),
            },
            sort_keys=True,
        )
    )


def _nonoverlapping_offsets(
    source_duration: float,
    segment_durations: tuple[int, ...],
) -> list[float]:
    usable = source_duration - 2 * EDGE_MARGIN_SECONDS
    required = sum(segment_durations)
    if usable <= required:
        raise ValueError("lecture is too short for the requested private samples")
    gap = (usable - required) / (len(segment_durations) + 1)
    offsets = []
    cursor = float(EDGE_MARGIN_SECONDS)
    for segment_duration in segment_durations:
        cursor += gap
        offsets.append(round(cursor, 3))
        cursor += segment_duration
    return offsets


def _even_offsets(source_duration: float, count: int) -> list[float]:
    usable = source_duration - 2 * EDGE_MARGIN_SECONDS
    if usable <= 0:
        raise ValueError("lecture is too short for frame sampling")
    return [
        round(EDGE_MARGIN_SECONDS + usable * index / (count + 1), 3)
        for index in range(1, count + 1)
    ]


def _extract_audio(
    ffmpeg: Path,
    source: Path,
    offset: float,
    duration: int,
    destination: Path,
) -> None:
    if destination.is_file():
        return
    partial = destination.with_suffix(".part.wav")
    _run_media_tool(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(offset),
            "-i",
            str(source),
            "-t",
            str(duration),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(partial),
        ],
        "audio extraction",
    )
    os.replace(partial, destination)


def _extract_frame(
    ffmpeg: Path,
    source: Path,
    offset: float,
    destination: Path,
) -> None:
    if destination.is_file():
        return
    partial = destination.with_suffix(".part.png")
    _run_media_tool(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(offset),
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(partial),
        ],
        "frame extraction",
    )
    os.replace(partial, destination)


def _media_duration(ffprobe: Path, media: Path) -> float:
    completed = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(media),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    if completed.returncode != 0:
        raise RuntimeError("duration probe failed")
    try:
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("duration probe returned invalid data") from error
    if duration <= 0:
        raise RuntimeError("duration probe returned a nonpositive duration")
    return duration


def _run_media_tool(command: list[str], operation: str) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=3600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{operation} failed")


def _workload(task: str, items: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "task": task,
        "workload_class": "private_course",
        "warmup_item_id": items[0]["id"],
        "items": items,
    }


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
