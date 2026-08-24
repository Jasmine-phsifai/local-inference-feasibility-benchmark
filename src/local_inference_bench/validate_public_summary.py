"""Reject free-form, identifying, secret, or path-bearing public values."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence


_BANNED_KEY_PARTS = {
    "path",
    "text",
    "transcript",
    "preview",
    "stdout",
    "stderr",
    "reference",
    "prediction",
    "course",
    "teacher",
    "student",
}
_ALLOWED_STRING_KEYS = {
    "backend",
    "candidate_id",
    "compute_type",
    "device",
    "device_name",
    "execution_devices",
    "failure_kind",
    "load_semantics",
    "mode",
    "model_revision",
    "phase",
    "prompt_version",
    "protocol",
    "runtime_name",
    "runtime_version",
    "stability_status",
    "status",
    "task",
    "unit",
    "vad_revision",
    "workload_class",
}
_MACHINE_LABEL_KEYS = _ALLOWED_STRING_KEYS - {"device_name"}
_MACHINE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+,()-]{0,159}$")
_DEVICE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:+,#()-]{0,159}$")
_EMAIL_ADDRESS = re.compile(
    r"(?i)(?<![a-z0-9._%+-])"
    r"[a-z0-9.!#$%&'*+=?^_`{|}~-]+@"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+"
    r"(?![a-z0-9_-])"
)
_PHONE_NUMBER = re.compile(
    r"(?x)(?:"
    r"(?<!\d)1[3-9]\d{9}(?!\d)"
    r"|(?<![A-Za-z0-9])\+\d{1,3}(?:[ .-]?\d){7,14}(?!\d)"
    r"|(?<![A-Za-z0-9])(?:\(?\d{2,4}\)?[ .-]){2,4}\d{3,4}(?!\d)"
    r")"
)
_CREDENTIAL_VALUE = re.compile(
    r"(?ix)(?:"
    r"-----BEGIN[ ](?:[A-Z0-9]+[ ])*PRIVATE[ ]KEY-----"
    r"|\b(?:api[_ -]?key|access[_ -]?token|auth(?:orization)?|password|passwd|secret)"
    r"\s*[:=]\s*\S{4,}"
    r"|\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"
    r"|(?<![A-Za-z0-9])(?:"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{12,}"
    r"|github_pat_[A-Za-z0-9_]{12,}"
    r"|glpat-[A-Za-z0-9_-]{12,}"
    r"|hf_[A-Za-z0-9]{12,}"
    r"|xox[baprs]-[A-Za-z0-9-]{12,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|AIza[0-9A-Za-z_-]{20,}"
    r")"
    r"|\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    r")"
)
_SENSITIVE_RELATIVE_FILENAME = re.compile(
    r"(?i)^(?:"
    r"\.env|\.netrc|\.npmrc|\.pypirc|\.git-credentials|credentials\.json"
    r"|[^/\\]+\.(?:"
    r"exe|dll|so(?:\.\d+)*|dylib|bin|gguf|onnx|safetensors|pt|pth|"
    r"wav|mp3|mp4|png|jpe?g|pdf|jsonl?|ya?ml|toml|ini|cfg|txt|md|"
    r"py|ps1|bat|cmd"
    r")"
    r")$"
)
_OPAQUE_TOKEN = re.compile(r"^[A-Za-z0-9]{32,160}={0,2}$")
_OPAQUE_TOKEN_EXEMPT_KEY_SUFFIXES = ("sha256", "revision", "fingerprint")
_PUBLIC_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MAX_CONTAINER_ENTRIES = 256
_MAX_NESTING_DEPTH = 12
_MAX_TOTAL_VALUES = 5_000


def validate_public_summary(summary: object) -> dict:
    """Return a plain validated mapping safe for the tracked event journal."""

    if not isinstance(summary, Mapping):
        raise ValueError("public_summary must be a mapping")
    normalized = _validate_value(summary, key=None, depth=0, budget=[_MAX_TOTAL_VALUES])
    assert isinstance(normalized, dict)
    return normalized


def validate_sustained_public_summary(
    summary: object,
    *,
    candidate_id: str,
    task: str,
    workload_class: str,
    target_wall_seconds: float,
    phase: str,
    config: Mapping | None = None,
) -> dict:
    """Validate benchmark evidence and bind it to the runner-owned request."""

    normalized = validate_public_summary(summary)
    expected_identity = {
        "candidate_id": candidate_id,
        "task": task,
        "workload_class": workload_class,
    }
    for key, expected in expected_identity.items():
        if normalized.get(key) != expected:
            raise ValueError(f"sustained public_summary identity mismatch: {key}")
    for key in ("runtime_name", "runtime_version", "load_semantics"):
        if not isinstance(normalized.get(key), str) or not normalized[key]:
            raise ValueError(f"sustained public_summary is missing {key}")

    counts = normalized.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("sustained public_summary is missing counts")
    completed = _nonnegative_int(counts, "completed")
    failed = _nonnegative_int(counts, "failed")
    attempted = _nonnegative_int(counts, "attempted")
    if attempted == 0 or completed + failed != attempted:
        raise ValueError("sustained public_summary count invariant failed")
    expected_status = (
        "all_failed"
        if completed == 0
        else "partial_failure" if failed else "complete"
    )
    if normalized.get("status") is None:
        normalized["status"] = expected_status
    elif normalized.get("status") != expected_status:
        raise ValueError("sustained public_summary status does not match counts")

    throughput = normalized.get("throughput")
    if not isinstance(throughput, Mapping):
        raise ValueError("sustained public_summary is missing throughput")
    value = throughput.get("value")
    if type(value) not in {int, float} or value < 0:
        raise ValueError("sustained public_summary throughput is invalid")
    expected_unit = {
        "asr": "audio_hours_per_wall_hour",
        "ocr": "images_per_hour",
    }[task]
    if throughput.get("unit") != expected_unit:
        raise ValueError("sustained public_summary throughput unit mismatch")

    timing = normalized.get("timing")
    if not isinstance(timing, Mapping):
        raise ValueError("sustained public_summary is missing timing")
    steady_wall_seconds = timing.get("steady_wall_seconds")
    if type(steady_wall_seconds) not in {int, float} or steady_wall_seconds <= 0:
        raise ValueError("sustained public_summary steady timing is invalid")
    if completed == 0:
        if float(value) != 0.0:
            raise ValueError("all-failed sustained public_summary has throughput")
    elif float(value) <= 0.0:
        raise ValueError("completed sustained public_summary has no throughput")
    if task == "ocr":
        expected_throughput = completed / float(steady_wall_seconds) * 3600.0
        if not math.isclose(
            float(value),
            expected_throughput,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("OCR sustained public_summary throughput mismatch")
    reported_target = timing.get("target_wall_seconds")
    if (
        type(reported_target) not in {int, float}
        or not math.isclose(
            float(reported_target),
            float(target_wall_seconds),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("sustained public_summary target timing mismatch")
    if (
        phase in {"screen", "sustained"}
        and normalized.get("status") == "complete"
        and float(steady_wall_seconds) + 1e-6 < float(target_wall_seconds)
    ):
        raise ValueError("complete sustained public_summary ended before its target")
    if candidate_id == "faster_whisper_cpu":
        _validate_faster_whisper_concurrency(normalized, config, phase)
    return normalized


def _nonnegative_int(mapping: Mapping, key: str) -> int:
    value = mapping.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"sustained public_summary count is invalid: {key}")
    return value


def _validate_faster_whisper_concurrency(
    summary: Mapping,
    config: Mapping | None,
    phase: str,
) -> None:
    if not isinstance(config, Mapping):
        raise ValueError("faster-whisper concurrency requires runner config")
    concurrency = summary.get("concurrency")
    if not isinstance(concurrency, Mapping):
        raise ValueError("faster-whisper concurrency evidence is missing")
    processes = _nonnegative_int(config, "processes")
    workers = _nonnegative_int(config, "model_workers")
    exact = {
        "configured_processes": processes,
        "configured_model_workers_per_process": workers,
        "configured_total_model_workers": processes * workers,
        "instrumented_process_count": processes,
        "runtime_model_workers_min": workers,
        "runtime_model_workers_max": workers,
    }
    for key, expected in exact.items():
        if _nonnegative_int(concurrency, key) != expected:
            raise ValueError(f"faster-whisper concurrency mismatch: {key}")
    python_peak = _nonnegative_int(
        concurrency,
        "python_calls_in_flight_peak_per_process",
    )
    python_peak_min = _nonnegative_int(
        concurrency,
        "python_calls_in_flight_peak_min_per_process",
    )
    processing_peak = _nonnegative_int(
        concurrency,
        "ctranslate2_processing_batches_peak_per_process",
    )
    processing_peak_min = _nonnegative_int(
        concurrency,
        "ctranslate2_processing_batches_peak_min_per_process",
    )
    if not (
        python_peak_min <= python_peak <= workers
        and processing_peak_min <= processing_peak <= workers
    ):
        raise ValueError("faster-whisper concurrency peak is invalid")
    active_peak = _nonnegative_int(
        concurrency,
        "ctranslate2_active_batches_peak_per_process",
    )
    queued_peak = _nonnegative_int(
        concurrency,
        "ctranslate2_queued_batches_peak_per_process",
    )
    if (
        queued_peak > active_peak
        or processing_peak > active_peak
        or active_peak > queued_peak + workers
    ):
        raise ValueError("faster-whisper queued batches exceed active batches")
    sample_count = _nonnegative_int(
        concurrency,
        "ctranslate2_sampler_sample_count",
    )
    busy_count = _nonnegative_int(
        concurrency,
        "ctranslate2_busy_sample_count",
    )
    fully_busy_count = _nonnegative_int(
        concurrency,
        "ctranslate2_fully_busy_sample_count",
    )
    failures = _nonnegative_int(
        concurrency,
        "ctranslate2_sampler_failure_count",
    )
    _nonnegative_int(
        concurrency,
        "ctranslate2_discarded_sample_count",
    )
    sample_min = _nonnegative_int(
        concurrency,
        "ctranslate2_sampler_sample_count_min_per_process",
    )
    busy_min = _nonnegative_int(
        concurrency,
        "ctranslate2_busy_sample_count_min_per_process",
    )
    if (
        sample_count == 0
        or sample_min == 0
        or not fully_busy_count <= busy_count <= sample_count
        or busy_min > sample_min
        or sample_count < processes * sample_min
        or busy_count < processes * busy_min
        or failures != 0
    ):
        raise ValueError("faster-whisper concurrency samples are invalid")
    completed = _nonnegative_int(summary["counts"], "completed")
    if (
        completed
        and phase in {"screen", "sustained"}
        and min(busy_min, python_peak_min, processing_peak_min) == 0
    ):
        raise ValueError("faster-whisper active concurrency was not observed")
    fraction = concurrency.get("ctranslate2_fully_busy_fraction_when_busy")
    expected_fraction = fully_busy_count / busy_count if busy_count else 0.0
    if (
        type(fraction) not in {int, float}
        or not 0.0 <= float(fraction) <= 1.0
        or not math.isclose(
            float(fraction),
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("faster-whisper concurrency fraction is invalid")


def _validate_value(
    value: object,
    *,
    key: str | None,
    depth: int,
    budget: list[int],
) -> object:
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("public_summary value count exceeds the limit")
    if depth > _MAX_NESTING_DEPTH:
        raise ValueError("public_summary nesting exceeds the limit")
    if isinstance(value, Mapping):
        if len(value) > _MAX_CONTAINER_ENTRIES:
            raise ValueError("public_summary mapping exceeds the entry limit")
        result = {}
        for child_key, child in value.items():
            if (
                type(child_key) is not str
                or _PUBLIC_KEY.fullmatch(child_key) is None
            ):
                raise ValueError("public_summary keys must be public identifiers")
            folded = child_key.casefold()
            if any(part in folded for part in _BANNED_KEY_PARTS):
                raise ValueError(f"private field is forbidden in public_summary: {child_key}")
            result[child_key] = _validate_value(
                child,
                key=child_key,
                depth=depth + 1,
                budget=budget,
            )
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_CONTAINER_ENTRIES:
            raise ValueError("public_summary sequence exceeds the entry limit")
        return [
            _validate_value(child, key=key, depth=depth + 1, budget=budget)
            for child in value
        ]
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public_summary numbers must be finite")
        return value
    if isinstance(value, str):
        if key not in _ALLOWED_STRING_KEYS:
            raise ValueError(f"free-form string field is forbidden: {key}")
        if len(value) > 160 or "\n" in value or "\r" in value:
            raise ValueError(f"public_summary string is not bounded: {key}")
        _validate_public_string(value, key=key)
        return value
    raise ValueError(f"unsupported public_summary value for {key}: {type(value).__name__}")


def _validate_public_string(value: str, *, key: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"public_summary string is not a bounded label: {key}")
    validate_public_string_privacy(value, key=key)
    pattern = _MACHINE_LABEL if key in _MACHINE_LABEL_KEYS else _DEVICE_LABEL
    if pattern.fullmatch(value) is None:
        raise ValueError(f"public_summary string is not a bounded label: {key}")


def validate_public_string_privacy(value: str, *, key: str) -> None:
    """Reject a bounded string that resembles private identity or secret material."""

    if _EMAIL_ADDRESS.search(value) or _PHONE_NUMBER.search(value):
        raise ValueError(f"identifying value is forbidden in public_summary: {key}")
    if _CREDENTIAL_VALUE.search(value):
        raise ValueError(f"credential-like value is forbidden in public_summary: {key}")
    if _looks_like_opaque_token(value, key=key):
        raise ValueError(f"opaque token-like value is forbidden in public_summary: {key}")
    if _looks_like_local_path(value):
        raise ValueError(f"local path is forbidden in public_summary: {key}")


def _looks_like_local_path(value: str) -> bool:
    folded = value.casefold()
    return (
        "/" in value
        or "\\" in value
        or value in {".", "..", "~"}
        or folded.startswith("file:")
        or (len(value) >= 2 and value[0].isalpha() and value[1] == ":")
        or _SENSITIVE_RELATIVE_FILENAME.fullmatch(value) is not None
    )


def _looks_like_opaque_token(value: str, *, key: str) -> bool:
    if key.casefold().endswith(_OPAQUE_TOKEN_EXEMPT_KEY_SUFFIXES):
        return False
    if _OPAQUE_TOKEN.fullmatch(value) is None:
        return False
    character_classes = sum(
        (
            any(character.islower() for character in value),
            any(character.isupper() for character in value),
            any(character.isdigit() for character in value),
        )
    )
    is_plain_hex = re.fullmatch(r"(?i)[0-9a-f]+", value) is not None
    return character_classes >= 3 or (len(value) >= 48 and not is_plain_hex)
