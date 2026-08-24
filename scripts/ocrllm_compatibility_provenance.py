"""Bind an installed OCRLLM package to one clean local source snapshot."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import sysconfig
from importlib.metadata import distribution, distributions
from pathlib import Path
from typing import get_args
from urllib.parse import unquote, urlparse


EXPECTED_REVISION = "2827c98b802932d6bbc0b71bd8d8d4188fa6a0b0"
EXPECTED_VERSION = "0.1.0"
EXPECTED_PYTHON_VERSION = "3.11.15"
REVIEWED_BASELINE = "47c12efe91640659a711c8bd3429dae6a4fe44f5"
SNAPSHOT_RELATIVE_PATH = Path("data/vendor/ocrllm-master-2827c98")
EXPECTED_RUNTIME_VERSIONS = {
    "numpy": "2.4.6",
    "onnxruntime": "1.23.2",
    "opencv-python": "5.0.0.93",
    "pillow": "12.3.0",
    "rapidocr": "3.9.2",
    "requests": "2.32.5",
}
EXPECTED_RAPIDOCR_MODEL_HASHES = {
    "rapidocr/models/PP-OCRv6_det_small.onnx": (
        "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"
    ),
    "rapidocr/models/PP-OCRv6_rec_small.onnx": (
        "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"
    ),
    "rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx": (
        "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"
    ),
}


def verify_ocrllm_installation(snapshot_root: Path) -> dict:
    """Return privacy-safe evidence after checking source and installed bytes."""

    import ocrllm

    snapshot_root = snapshot_root.resolve()
    revision = _git(snapshot_root, "rev-parse", "HEAD")
    if revision != EXPECTED_REVISION:
        raise RuntimeError(
            "OCRLLM snapshot revision mismatch: "
            f"expected {EXPECTED_REVISION}, got {revision}"
        )
    if _git(snapshot_root, "status", "--porcelain=v1"):
        raise RuntimeError("OCRLLM snapshot has local changes")
    _git(
        snapshot_root,
        "merge-base",
        "--is-ancestor",
        REVIEWED_BASELINE,
        EXPECTED_REVISION,
    )

    installed_distribution = distribution("ocrllm")
    if installed_distribution.version != EXPECTED_VERSION:
        raise RuntimeError(
            "OCRLLM version mismatch: "
            f"expected {EXPECTED_VERSION}, got {installed_distribution.version}"
        )
    direct_url_text = installed_distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("OCRLLM remained editable")
    installed_from = _file_url_to_path(direct_url.get("url"))
    if installed_from != snapshot_root:
        raise RuntimeError("OCRLLM was not installed from the pinned snapshot")

    module_root = Path(ocrllm.__file__).resolve().parent
    if "site-packages" not in {part.casefold() for part in module_root.parts}:
        raise RuntimeError("OCRLLM did not install into site-packages")
    source_root = snapshot_root / "src" / "ocrllm"
    source_hashes = _python_file_hashes(source_root)
    installed_hashes = _python_file_hashes(module_root)
    if not source_hashes or installed_hashes != source_hashes:
        raise RuntimeError("installed OCRLLM Python sources differ from the snapshot")
    runtime_versions = {
        name: distribution(name).version for name in EXPECTED_RUNTIME_VERSIONS
    }
    if runtime_versions != EXPECTED_RUNTIME_VERSIONS:
        raise RuntimeError("OCRLLM compatibility runtime version mismatch")
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise RuntimeError("OCRLLM compatibility Python version mismatch")
    runtime_components = _installed_runtime_components()
    rapidocr_model_hashes = _rapidocr_model_hashes()
    _verify_rapidocr_model_hashes(rapidocr_model_hashes)
    authority_boundary = _verify_authority_boundary(ocrllm)

    return {
        "revision": revision,
        "version": installed_distribution.version,
        "reviewed_baseline_ancestor": True,
        "snapshot_clean": True,
        "installed_noneditable": True,
        "installed_from_pinned_snapshot": True,
        "installed_source_matches_snapshot": True,
        "python_file_count": len(source_hashes),
        "python_source_fingerprint": _hash_mapping(source_hashes),
        "runtime_component_count": len(runtime_components),
        "runtime_environment_fingerprint": _hash_mapping(runtime_components),
        "rapidocr_model_file_count": len(rapidocr_model_hashes),
        "rapidocr_model_fingerprint": _hash_mapping(rapidocr_model_hashes),
        "source_tree_fingerprint": _git(snapshot_root, "rev-parse", "HEAD^{tree}"),
        **authority_boundary,
    }


def _git(snapshot_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(snapshot_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError("OCRLLM snapshot Git verification failed")
    return completed.stdout.strip()


def _file_url_to_path(value: object) -> Path:
    if type(value) is not str:
        raise RuntimeError("OCRLLM installation has no source URL")
    parsed = urlparse(value)
    if parsed.scheme.casefold() != "file" or parsed.query or parsed.fragment:
        raise RuntimeError("OCRLLM installation source URL is not a local directory")
    decoded = unquote(parsed.path)
    if parsed.netloc:
        decoded = f"//{parsed.netloc}{decoded}"
    elif re.match(r"^/[A-Za-z]:/", decoded):
        decoded = decoded[1:]
    return Path(decoded).resolve()


def _python_file_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        raise RuntimeError("OCRLLM Python source directory is missing")
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    }


def _installed_runtime_components() -> dict[str, str]:
    components = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "sys_platform": sys.platform,
    }
    site_package_roots = sorted(
        {
            str(Path(sysconfig.get_paths()[key]).resolve())
            for key in ("purelib", "platlib")
        }
    )
    for installed in distributions(path=site_package_roots):
        name = installed.metadata.get("Name")
        if type(name) is not str or not name:
            raise RuntimeError("installed distribution has no canonical name")
        key = f"distribution:{_canonical_distribution_name(name)}"
        version = installed.version
        previous = components.setdefault(key, version)
        if previous != version:
            raise RuntimeError("conflicting installed distribution versions")
    return components


def _rapidocr_model_hashes() -> dict[str, str]:
    installed = distribution("rapidocr")
    model_root = Path(installed.locate_file("rapidocr/models"))
    if not model_root.is_dir():
        raise RuntimeError("RapidOCR installed model directory is missing")
    return {
        (Path("rapidocr/models") / path.relative_to(model_root)).as_posix(): _sha256(
            path
        )
        for path in sorted(model_root.rglob("*.onnx"))
        if path.is_file()
    }


def _verify_rapidocr_model_hashes(actual: dict[str, str]) -> None:
    if actual != EXPECTED_RAPIDOCR_MODEL_HASHES:
        missing = sorted(set(EXPECTED_RAPIDOCR_MODEL_HASHES) - set(actual))
        unexpected = sorted(set(actual) - set(EXPECTED_RAPIDOCR_MODEL_HASHES))
        mismatched = sorted(
            path
            for path in set(actual) & set(EXPECTED_RAPIDOCR_MODEL_HASHES)
            if actual[path] != EXPECTED_RAPIDOCR_MODEL_HASHES[path]
        )
        raise RuntimeError(
            "RapidOCR bundled model inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}, mismatched={mismatched}"
        )


def _canonical_distribution_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _verify_authority_boundary(ocrllm_module: object) -> dict:
    from ocrllm.audio.probe_short_mp3 import MAX_SHORT_MP3_DURATION_SECONDS
    from ocrllm.contracts.worker_command import WorkerCommand
    from ocrllm.validate_short_audio_options import validate_short_audio_options

    public_names = set(dir(ocrllm_module))
    direct_short_symbols = {
        "AudioModelSettings",
        "GoogleGenAISettings",
        "recognize",
    }
    local_asr_symbols = {
        "LocalASRSettings",
        "recognize_local_audio",
        "transcribe_local",
    }
    filetrans_symbols = {"FileTransSettings", "recognize_filetrans"}
    if not direct_short_symbols.issubset(public_names):
        raise RuntimeError("OCRLLM direct short-MP3 public facade is missing")
    if local_asr_symbols & public_names:
        raise RuntimeError("OCRLLM unexpectedly exposes a local ASR facade")
    if filetrans_symbols & public_names:
        raise RuntimeError("OCRLLM unexpectedly exposes a FileTrans facade")
    if MAX_SHORT_MP3_DURATION_SECONDS != 300.0:
        raise RuntimeError("OCRLLM direct audio duration boundary changed")

    audio_capabilities = [
        report
        for report in ocrllm_module.get_capabilities()
        if "audio" in report.name.casefold() or "filetrans" in report.name.casefold()
    ]
    registered_audio_available_count = sum(
        report.status == "available" for report in audio_capabilities
    )
    if not audio_capabilities or registered_audio_available_count:
        raise RuntimeError("OCRLLM registered audio capability boundary changed")

    worker_command_types = get_args(WorkerCommand)
    audio_worker_command_count = sum(
        "audio" in command_type.__name__.casefold()
        for command_type in worker_command_types
    )
    if audio_worker_command_count:
        raise RuntimeError("OCRLLM unexpectedly exposes an audio worker command")

    direct_config = ocrllm_module.Config(
        provider=ocrllm_module.GoogleGenAISettings(api_key="compatibility-probe"),
        audio_model=ocrllm_module.AudioModelSettings(name="compatibility-probe"),
    )
    validate_short_audio_options((Path("probe.mp3"),), config=direct_config)

    nonmemory_option_rejections = 0
    for options in (
        {"output_dir": "compatibility-probe"},
        {"output_dir": "compatibility-probe", "resume": True},
        {"output_dir": "compatibility-probe", "overwrite": True},
    ):
        config = ocrllm_module.Config(**options)
        try:
            validate_short_audio_options((Path("probe.mp3"),), config=config)
        except ocrllm_module.ConfigError as error:
            if error.code == "CONFIG_INVALID":
                nonmemory_option_rejections += 1
                continue
            raise RuntimeError("OCRLLM audio option rejection changed") from error
        raise RuntimeError("OCRLLM accepted an unsupported audio output option")

    return {
        "experimental_direct_short_mp3_public_facade": True,
        "local_asr_public_symbol_count": 0,
        "local_asr_facade_available": False,
        "filetrans_public_symbol_count": 0,
        "filetrans_facade_available": False,
        "configured_direct_audio_limit_seconds": MAX_SHORT_MP3_DURATION_SECONDS,
        "long_audio_facade_available": False,
        "direct_short_inmemory_options_accepted": True,
        "registered_audio_available_capability_count": 0,
        "audio_nonmemory_option_rejection_count": nonmemory_option_rejections,
        "audio_persistence_available": False,
        "audio_resume_available": False,
        "audio_worker_command_count": audio_worker_command_count,
        "audio_worker_support_available": False,
        "benchmark_owned_asr_adapters_required": True,
    }


def _hash_mapping(values: dict[str, str]) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
