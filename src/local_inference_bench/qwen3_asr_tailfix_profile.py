"""Validate the pinned OpenVINO GenAI runtime that contains the Qwen3-ASR tail fix."""

from __future__ import annotations

import json
from pathlib import Path


TAILFIX_PROFILE_RELATIVE_PATH = Path(
    "environments/qwen3_asr_openvino_genai_tailfix_20260821/"
    "runtime-provenance.json"
)
TAILFIX_PROFILE_ID = "openvino_genai_tailfix_20260821"
TAILFIX_ENVIRONMENT_NAME = (
    "local-bench-qwen3-asr-openvino-genai-tailfix-20260821"
)
TAILFIX_PRODUCT_VERSION = "2026.4.0.0-3387-98ae8c32197"
TAILFIX_SOURCE_REVISION = "98ae8c32197d1afe88ebaff89968283493c25786"
TAILFIX_REQUIRED_FIX_REVISION = "0d35ded5bac2d39bf45d52cbc7156c087f50c80d"
TAILFIX_STABLE_SOURCE_REVISION = "bd8d6542e3ca1ac30042d5d8d4202ce00b5f4af0"
TAILFIX_SOURCE_REPOSITORY = (
    "https://github.com/openvinotoolkit/openvino.genai.git"
)
TAILFIX_PACKAGE_VERSIONS = {
    "numpy": "2.4.6",
    "openvino": "2026.4.0.dev20260821",
    "openvino-genai": "2026.4.0.0.dev20260821",
    "openvino-telemetry": "2025.2.0",
    "openvino-tokenizers": "2026.4.0.0.dev20260821",
    "packaging": "26.3",
    "pip": "26.1.2",
    "setuptools": "83.0.0",
    "wheel": "0.47.0",
}
TAILFIX_WHEEL_ARTIFACTS = {
    "openvino": {
        "filename": (
            "openvino-2026.4.0.dev20260821-22849-"
            "cp311-cp311-win_amd64.whl"
        ),
        "url": (
            "https://storage.openvinotoolkit.org/wheels/nightly/openvino/"
            "openvino-2026.4.0.dev20260821-22849-cp311-cp311-win_amd64.whl"
        ),
        "size_bytes": 83_366_994,
        "sha256": (
            "ffc445f117dd210d46e26704066a8f140f4bc54caad9ff65455038a4849697d2"
        ),
    },
    "openvino-genai": {
        "filename": (
            "openvino_genai-2026.4.0.0.dev20260821-2617-"
            "cp311-cp311-win_amd64.whl"
        ),
        "url": (
            "https://storage.openvinotoolkit.org/wheels/nightly/openvino-genai/"
            "openvino_genai-2026.4.0.0.dev20260821-2617-"
            "cp311-cp311-win_amd64.whl"
        ),
        "size_bytes": 3_761_955,
        "sha256": (
            "abd3f5aad8f290995ea53b94612aa0462dae331921975657463eeaa7e1925cd4"
        ),
    },
    "openvino-tokenizers": {
        "filename": (
            "openvino_tokenizers-2026.4.0.0.dev20260821-"
            "py3-none-win_amd64.whl"
        ),
        "url": (
            "https://storage.openvinotoolkit.org/wheels/nightly/"
            "openvino-tokenizers/openvino_tokenizers-"
            "2026.4.0.0.dev20260821-py3-none-win_amd64.whl"
        ),
        "size_bytes": 1_544_778,
        "sha256": (
            "54d3881cb869ddeb5c0730f035ddf330360154f1b7f88c0c3ec1ef8078a35f2b"
        ),
    },
}
EXPECTED_TAILFIX_PROFILE = {
    "schema_version": 1,
    "profile_id": TAILFIX_PROFILE_ID,
    "environment_name": TAILFIX_ENVIRONMENT_NAME,
    "python_version": "3.11.15",
    "package_versions": TAILFIX_PACKAGE_VERSIONS,
    "openvino_genai_product_version": TAILFIX_PRODUCT_VERSION,
    "source_repository": TAILFIX_SOURCE_REPOSITORY,
    "stable_source_revision": TAILFIX_STABLE_SOURCE_REVISION,
    "associated_source_revision": TAILFIX_SOURCE_REVISION,
    "required_fix_revision": TAILFIX_REQUIRED_FIX_REVISION,
    "source_checkout": "data/vendor/openvino.genai-tail-fix-source",
    "wheel_artifacts": TAILFIX_WHEEL_ARTIFACTS,
}


def load_qwen3_asr_tailfix_profile(path: Path) -> dict:
    """Load the tracked profile only when every pinned field is unchanged."""

    if not path.is_file() or path.stat().st_size > 65_536:
        raise RuntimeError("tail-fix runtime profile is unavailable or too large")
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("tail-fix runtime profile is unreadable") from error
    try:
        canonical_profile = json.dumps(
            profile,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("tail-fix runtime profile changed") from error
    canonical_expected = json.dumps(
        EXPECTED_TAILFIX_PROFILE,
        ensure_ascii=True,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    if canonical_profile != canonical_expected:
        raise RuntimeError("tail-fix runtime profile changed")
    return profile
