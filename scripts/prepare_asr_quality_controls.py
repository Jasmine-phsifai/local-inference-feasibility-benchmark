"""Build ten minutes of exact-reference mixed-language ASR controls."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "inputs" / "generated" / "asr_quality"
SOURCE_ROOT = OUTPUT_ROOT / "sources"
GENERATED_ROOT = OUTPUT_ROOT / "generated"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
SAMPLE_RATE = 16_000
CONTROL_SECONDS = 60
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_REPOSITORY = "google/fleurs"

FLEURS_ARCHIVES = {
    "cmn_hans_cn": {
        "filename": "fleurs-cmn-hans-cn-dev.tar.gz",
        "repository_path": "data/cmn_hans_cn/audio/dev.tar.gz",
        "bytes": 217_347_747,
        "sha256": "3bc33212d5974eef7feb04bc4792458d6cd7e14ff10a1a24772f3c45ea87a822",
    },
    "en_us": {
        "filename": "fleurs-en-us-dev.tar.gz",
        "repository_path": "data/en_us/audio/dev.tar.gz",
        "bytes": 171_250_900,
        "sha256": "2658fda72f199e12676ecac9415094667a4e14e149b146e568ea00b2a2f0954c",
    },
}

PUBLIC_SOURCES = {
    "zh_wifi": {
        "config": "cmn_hans_cn",
        "row": 11,
        "id": 1554,
        "filename": "10519077850963840950.wav",
        "num_samples": 89_280,
        "transcript": "他称，他制作了一个 WiFi 门铃。",
        "required_terms": [
            {"aliases": ["WiFi", "Wi-Fi"]},
            {"aliases": ["门铃"]},
        ],
    },
    "zh_numbers": {
        "config": "cmn_hans_cn",
        "row": 32,
        "id": 1520,
        "filename": "11788433184081135999.wav",
        "num_samples": 244_800,
        "transcript": (
            "在 2010 年联邦大选之前曾对 1,400 名受访者做过调查，"
            "受访者中反对澳大利亚成为共和国的人数自 2008 年以来"
            "增长了 8%。"
        ),
        "required_terms": [
            {"aliases": ["2010"]},
            {"aliases": ["1,400", "1400"]},
            {"aliases": ["2008"]},
            {"aliases": ["8%"]},
        ],
    },
    "zh_ai": {
        "config": "cmn_hans_cn",
        "row": 68,
        "id": 1517,
        "filename": "12905939626654587005.wav",
        "num_samples": 131_520,
        "transcript": (
            "人工智能 (AI) 的研究涉及制造机器让需要智能行为的任务自动化。"
        ),
        "required_terms": [
            {"aliases": ["人工智能"]},
            {"aliases": ["AI", "A I"]},
            {"aliases": ["自动化"]},
        ],
    },
    "en_ai": {
        "config": "en_us",
        "row": 29,
        "id": 1517,
        "filename": "10885549230041454053.wav",
        "num_samples": 112_320,
        "transcript": (
            "Research in AI involves making machines to automate tasks that "
            "require intelligent behavior."
        ),
        "required_terms": [
            {"aliases": ["AI", "A I"]},
            {"aliases": ["automate"]},
            {"aliases": ["intelligent behavior"]},
        ],
    },
    "en_ms": {
        "config": "en_us",
        "row": 95,
        "id": 1580,
        "filename": "13999045526176313011.wav",
        "num_samples": 256_000,
        "transcript": (
            "Across the United States of America, there are approximately "
            "400,000 known cases of Multiple Sclerosis (MS), leaving it as "
            "the leading neurological disease in younger and middle aged adults."
        ),
        "required_terms": [
            {"aliases": ["400,000", "400000"]},
            {"aliases": ["Multiple Sclerosis"]},
            {"aliases": ["MS", "M S"]},
            {"aliases": ["neurological disease"]},
        ],
    },
}

SAPI_SOURCES = {
    "technical": {
        "voice": "Microsoft David Desktop",
        "text": (
            "During lecture seven, the Kalman filter updates the state vector "
            "every twenty milliseconds. The C P U uses A V X two, and the "
            "sample rate is forty eight kilohertz. Latency must remain below "
            "twelve point five milliseconds."
        ),
        "required_terms": [
            {"aliases": ["Kalman filter"]},
            {"aliases": ["CPU", "C P U"]},
            {"aliases": ["AVX2", "A V X two"]},
            {"aliases": ["48 kilohertz", "forty eight kilohertz"]},
            {"aliases": ["12.5 milliseconds", "twelve point five milliseconds"]},
        ],
    },
    "formula": {
        "voice": "Microsoft Zira Desktop",
        "text": (
            "For the state equation, x at k plus one equals A x at k plus B u "
            "at k. The learning rate eta is zero point zero one, batch size is "
            "sixty four, and the eigenvalue is negative one point two five."
        ),
        "required_terms": [
            {"aliases": ["state equation"]},
            {"aliases": ["eta"]},
            {"aliases": ["0.01", "zero point zero one"]},
            {"aliases": ["64", "sixty four"]},
            {"aliases": ["negative 1.25", "negative one point two five"]},
        ],
    },
    "abbreviations": {
        "voice": "Microsoft David Desktop",
        "text": (
            "Call function Kalman update with matrix H transpose. The test "
            "identifier is zero x two A seven F. Read I M U, lidar, R A M, and "
            "A S R separately. The checkpoint time is fourteen thirty five "
            "and twenty seconds."
        ),
        "required_terms": [
            {"aliases": ["Kalman update"]},
            {"aliases": ["H transpose"]},
            {"aliases": ["0x2A7F", "zero x two A seven F"]},
            {"aliases": ["IMU", "I M U"]},
            {"aliases": ["LiDAR", "lidar"]},
            {"aliases": ["RAM", "R A M"]},
            {"aliases": ["ASR", "A S R"]},
            {"aliases": ["14:35:20", "fourteen thirty five and twenty seconds"]},
        ],
    },
}


def main() -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)
    ffmpeg = _find_ffmpeg()
    sources = _prepare_public_sources(ffmpeg)
    sources.update(_prepare_sapi_sources(ffmpeg))
    controls = _build_controls(sources)

    items = []
    references = {}
    for control in controls:
        path = OUTPUT_ROOT / f"{control['id']}.wav"
        _write_pcm(path, control["frames"])
        expected_speech = bool(control["transcript"])
        items.append(
            {
                "id": control["id"],
                "path": path.name,
                "duration_seconds": CONTROL_SECONDS,
                "expected_speech": expected_speech,
            }
        )
        references[control["id"]] = {
            "category": control["category"],
            "transcript": control["transcript"],
            "required_terms": control["required_terms"],
            "expected_speech": expected_speech,
            "speech_intervals": control["speech_intervals"],
            "audio_sha256": _sha256(path),
        }

    warmup = _assemble_control(
        "warmup",
        "warmup",
        ["en_ai"],
        sources,
        target_seconds=10,
        repeat=False,
    )
    warmup_path = OUTPUT_ROOT / "warmup.wav"
    _write_pcm(warmup_path, warmup["frames"])
    manifest = {
        "schema_version": 1,
        "task": "asr",
        "workload_class": "generated_quality_control",
        "disclosure": (
            "Exact-reference controls combine five revision-pinned CC-BY-4.0 "
            "FLEURS validation clips, Windows SAPI speech, and digital silence. "
            "They do not substitute for independently transcribed natural lectures."
        ),
        "warmup": {
            "id": "warmup",
            "path": warmup_path.name,
            "duration_seconds": 10,
            "expected_speech": True,
        },
        "items": items,
        "references": references,
        "generator": {
            "protocol": "mixed-asr-controls.v1",
            "sample_rate_hz": SAMPLE_RATE,
            "sample_width_bytes": 2,
            "channels": 1,
            "control_seconds": CONTROL_SECONDS,
            "public_source_repository": FLEURS_REPOSITORY,
            "public_source_revision": FLEURS_REVISION,
            "public_source_license": "CC-BY-4.0",
            "public_archive_sha256": {
                config: metadata["sha256"]
                for config, metadata in FLEURS_ARCHIVES.items()
            },
            "public_source_audio_sha256": {
                source_id: sources[source_id]["source_sha256"]
                for source_id in PUBLIC_SOURCES
            },
            "sapi_voices": sorted(
                {source["voice"] for source in SAPI_SOURCES.values()}
            ),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(MANIFEST_PATH)


def _prepare_public_sources(ffmpeg: Path) -> dict:
    archives = {}
    for config, metadata in FLEURS_ARCHIVES.items():
        archive_path = SOURCE_ROOT / metadata["filename"]
        url = (
            f"https://huggingface.co/datasets/{FLEURS_REPOSITORY}/resolve/"
            f"{FLEURS_REVISION}/{metadata['repository_path']}?download=true"
        )
        _download_archive_verified(
            url,
            archive_path,
            expected_bytes=metadata["bytes"],
            expected_sha256=metadata["sha256"],
        )
        archives[config] = archive_path

    selected_by_config = {}
    for source_id, metadata in PUBLIC_SOURCES.items():
        selected_by_config.setdefault(metadata["config"], {})[
            metadata["filename"]
        ] = source_id
    for config, selected in selected_by_config.items():
        with tarfile.open(archives[config], "r:gz") as archive:
            matching = {
                Path(member.name).name: member
                for member in archive.getmembers()
                if member.isfile() and Path(member.name).name in selected
            }
            if set(matching) != set(selected):
                raise ValueError(f"FLEURS archive members are incomplete: {config}")
            for filename, source_id in selected.items():
                source_path = SOURCE_ROOT / filename
                extracted = archive.extractfile(matching[filename])
                if extracted is None:
                    raise ValueError(f"failed to read FLEURS member: {filename}")
                with source_path.open("wb") as destination:
                    shutil.copyfileobj(extracted, destination)

    prepared = {}
    for source_id, metadata in PUBLIC_SOURCES.items():
        source_path = SOURCE_ROOT / metadata["filename"]
        normalized_path = GENERATED_ROOT / f"{source_id}.wav"
        _normalize_wav(ffmpeg, source_path, normalized_path)
        frames = _read_pcm(normalized_path)
        if len(frames) // 2 != metadata["num_samples"]:
            raise ValueError(f"FLEURS sample count mismatch: {source_id}")
        prepared[source_id] = {
            "frames": frames,
            "transcript": metadata["transcript"],
            "required_terms": metadata["required_terms"],
            "source_sha256": _sha256(source_path),
        }
    return prepared


def _prepare_sapi_sources(ffmpeg: Path) -> dict:
    helper = PROJECT_ROOT / "scripts" / "synthesize_sapi_control.ps1"
    powershell = Path(
        "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    prepared = {}
    for source_id, metadata in SAPI_SOURCES.items():
        raw_path = GENERATED_ROOT / f"{source_id}-sapi.wav"
        normalized_path = GENERATED_ROOT / f"{source_id}.wav"
        subprocess.run(
            [
                str(powershell),
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(helper),
                "-OutputPath",
                str(raw_path),
                "-Voice",
                metadata["voice"],
                "-Text",
                metadata["text"],
            ],
            check=True,
        )
        _normalize_wav(ffmpeg, raw_path, normalized_path)
        prepared[source_id] = {
            "frames": _read_pcm(normalized_path),
            "transcript": metadata["text"],
            "required_terms": metadata["required_terms"],
        }
    return prepared


def _build_controls(sources: dict) -> list[dict]:
    return [
        _assemble_control(
            "mandarin_repeated",
            "mandarin_public",
            ["zh_wifi", 1.0, "zh_ai", 1.0, "zh_numbers", 1.0],
            sources,
        ),
        _assemble_control(
            "english_repeated",
            "english_public",
            ["en_ai", 1.5, "en_ms", 1.5],
            sources,
        ),
        _assemble_control(
            "technical_repeated",
            "technical_terms",
            ["technical", 2.0],
            sources,
        ),
        _assemble_control(
            "numbers_formula", "numbers_formula", ["formula", 2.0], sources
        ),
        _assemble_control(
            "code_abbreviations",
            "abbreviations",
            ["abbreviations", 2.0],
            sources,
        ),
        _assemble_control(
            "mixed_bilingual",
            "mixed_language",
            ["zh_ai", 1.0, "technical", 1.0, "en_ai", 1.0, "formula", 1.0],
            sources,
        ),
        _assemble_control(
            "mixed_long_silence",
            "long_silence",
            [8.0, "zh_numbers", 8.0, "technical", 8.0, "en_ai"],
            sources,
            repeat=False,
        ),
        _assemble_control(
            "sparse_speech",
            "sparse_speech",
            [20.0, "en_ms", 15.0, "zh_wifi"],
            sources,
            repeat=False,
        ),
        _assemble_control(
            "lecture_queue_mix",
            "mixed_language",
            [
                "zh_wifi",
                0.5,
                "en_ai",
                0.5,
                "zh_numbers",
                0.5,
                "technical",
                0.5,
                "formula",
                0.5,
                "abbreviations",
                0.5,
            ],
            sources,
        ),
        _assemble_control(
            "digital_silence",
            "silence",
            [60.0],
            sources,
            repeat=False,
        ),
    ]


def _assemble_control(
    sample_id: str,
    category: str,
    pattern: list[str | float],
    sources: dict,
    *,
    target_seconds: int = CONTROL_SECONDS,
    repeat: bool = True,
) -> dict:
    target_frames = target_seconds * SAMPLE_RATE
    audio = bytearray()
    transcripts = []
    speech_intervals = []
    required_terms = []
    seen_terms = set()
    pattern_index = 0
    while len(audio) // 2 < target_frames:
        if pattern_index >= len(pattern):
            if not repeat:
                break
            pattern_index = 0
        component = pattern[pattern_index]
        pattern_index += 1
        current_frame = len(audio) // 2
        remaining_frames = target_frames - current_frame
        if isinstance(component, str):
            source = sources[component]
            source_frames = len(source["frames"]) // 2
            if source_frames > remaining_frames:
                break
            audio.extend(source["frames"])
            speech_intervals.append(
                [
                    current_frame / SAMPLE_RATE,
                    (current_frame + source_frames) / SAMPLE_RATE,
                ]
            )
            transcripts.append(source["transcript"])
            for term in source["required_terms"]:
                key = tuple(term["aliases"])
                if key not in seen_terms:
                    required_terms.append(term)
                    seen_terms.add(key)
        else:
            silence_frames = min(
                remaining_frames,
                round(float(component) * SAMPLE_RATE),
            )
            audio.extend(b"\0\0" * silence_frames)
    missing_frames = target_frames - len(audio) // 2
    audio.extend(b"\0\0" * missing_frames)
    return {
        "id": sample_id,
        "category": category,
        "frames": bytes(audio),
        "transcript": " ".join(transcripts),
        "required_terms": required_terms,
        "speech_intervals": speech_intervals,
    }


def _download_archive_verified(
    url: str,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    if destination.is_file():
        if (
            destination.stat().st_size == expected_bytes
            and _sha256(destination) == expected_sha256
        ):
            return
        raise ValueError(f"existing archive identity mismatch: {destination.name}")
    partial = destination.with_suffix(destination.suffix + ".part")
    curl = Path("C:/Windows/System32/curl.exe")
    subprocess.run(
        [
            str(curl),
            "--fail",
            "--location",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    actual_sha256 = _sha256(partial)
    if partial.stat().st_size != expected_bytes or actual_sha256 != expected_sha256:
        raise ValueError(
            f"archive identity mismatch for {destination.name}: "
            f"{partial.stat().st_size} bytes, {actual_sha256}"
        )
    partial.replace(destination)


def _find_ffmpeg() -> Path:
    environment_binary = Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe"
    if environment_binary.is_file():
        return environment_binary
    resolved = shutil.which("ffmpeg")
    if resolved:
        return Path(resolved)
    raise FileNotFoundError("ffmpeg is required to normalize ASR controls")


def _normalize_wav(ffmpeg: Path, source: Path, destination: Path) -> None:
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
    )


def _read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != SAMPLE_RATE
        ):
            raise ValueError(f"unexpected normalized WAV format: {path.name}")
        return handle.readframes(handle.getnframes())


def _write_pcm(path: Path, frames: bytes) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(frames)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
