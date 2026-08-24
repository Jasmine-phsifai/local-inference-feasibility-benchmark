"""Compare the Qwen3-ASR encoder-tail fix without publishing transcripts.

The parent process creates one deterministic public-audio control, then launches
the pinned stable and nightly OpenVINO GenAI environments separately. Child
responses contain only transcript hashes and counts. A local, full-history
official source checkout supplies executable Git ancestry evidence; matching or
differing hashes measure the narrower transcript-level effect on these controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import distribution, version
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.qwen3_asr_openvino_genai_export_manifest import verify_export


SOURCE_REPOSITORY = "https://github.com/openvinotoolkit/openvino.genai.git"
FIX_REVISION = "0d35ded5bac2d39bf45d52cbc7156c087f50c80d"
MODEL_SOURCE_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
MODEL_EXPORTER_REVISION = "f48d93fddff8c91e198389c47a6d5974789b67f4"
PUBLIC_CONTROL_SHA256 = (
    "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
)
TAIL_CONTROL_SHA256 = (
    "53477ea64bf492703e06cca13171296d3032b07dd1c96732908aa41ce62421f2"
)
EXPORT_MARKER_SHA256 = (
    "77ab740f4828f8663469738e70bd7d247db51d9292057e6155a41ff30673b1c9"
)
EXPORT_PROVENANCE_SHA256 = (
    "82585af7052b471b0db914b4b4c8faca7cc3db2ff8fcae0b5c5f2c997811333a"
)
MAX_CHILD_SECONDS = 180
TAIL_SILENCE_SAMPLES = 1_280
ENCODER_CHUNK_FRAMES = 100
TOKENS_PER_FULL_CHUNK = 13
EVENT_PROTOCOL = "openvino-genai-qwen3-asr-tail-fix-v3"


@dataclass(frozen=True)
class RuntimeSpec:
    label: str
    python: Path
    source_revision: str
    packages: dict[str, tuple[str, str]]


STABLE_SPEC = RuntimeSpec(
    label="stable-2026.3.0.0",
    python=Path(
        "D:/Anaconda/envs/local-bench-qwen3-asr-openvino-genai-official/python.exe"
    ),
    source_revision="bd8d6542e3ca1ac30042d5d8d4202ce00b5f4af0",
    packages={
        "openvino": (
            "2026.3.0",
            "7b688e38cc129cc268253c17d29a70d57e6e3ef8fb477a211294d1506440da15",
        ),
        "openvino-genai": (
            "2026.3.0.0",
            "ff5c1306387c4bf8fe57857b240442799ed5ed0440928e6bdfb27316b3c35827",
        ),
        "openvino-tokenizers": (
            "2026.3.0.0",
            "7c6fb2f1e9b6c3c4b2bdb992e69ef97869787d8486b193d155d24aaa9cd5fed0",
        ),
    },
)

NIGHTLY_SPEC = RuntimeSpec(
    label="nightly-2026.4.0.0.dev20260821",
    python=Path(
        "D:/Anaconda/envs/"
        "local-bench-qwen3-asr-openvino-genai-tailfix-20260821/python.exe"
    ),
    source_revision="98ae8c32197d1afe88ebaff89968283493c25786",
    packages={
        "openvino": (
            "2026.4.0.dev20260821",
            "ffc445f117dd210d46e26704066a8f140f4bc54caad9ff65455038a4849697d2",
        ),
        "openvino-genai": (
            "2026.4.0.0.dev20260821",
            "abd3f5aad8f290995ea53b94612aa0462dae331921975657463eeaa7e1925cd4",
        ),
        "openvino-tokenizers": (
            "2026.4.0.0.dev20260821",
            "54d3881cb869ddeb5c0730f035ddf330360154f1b7f88c0c3ec1ef8078a35f2b",
        ),
    },
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stable-python", type=Path, default=STABLE_SPEC.python
    )
    parser.add_argument(
        "--nightly-python", type=Path, default=NIGHTLY_SPEC.python
    )
    parser.add_argument(
        "--source-model",
        type=Path,
        default=PROJECT_ROOT / "data/models/qwen3-asr-0.6b-original",
    )
    parser.add_argument(
        "--export-model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data/models/qwen3-asr-0.6b-openvino-genai-official-f48d93f"
        ),
    )
    parser.add_argument(
        "--public-control",
        type=Path,
        default=PROJECT_ROOT / "data/inputs/public/jfk.wav",
    )
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=PROJECT_ROOT / "data/vendor/openvino.genai-tail-fix-source",
    )
    parser.add_argument("--timeout-seconds", type=int, default=MAX_CHILD_SECONDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--append-journal", type=Path)
    parser.add_argument("--child-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-response", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_request is not None or args.child_response is not None:
        if args.child_request is None or args.child_response is None:
            raise ValueError("both child request and response paths are required")
        _run_child(args.child_request, args.child_response)
        return

    timeout_seconds = _validate_timeout(args.timeout_seconds)
    stable_spec = _with_python(STABLE_SPEC, args.stable_python)
    nightly_spec = _with_python(NIGHTLY_SPEC, args.nightly_python)
    event = run_comparison(
        stable_spec=stable_spec,
        nightly_spec=nightly_spec,
        source_model=args.source_model,
        export_model=args.export_model,
        public_control=args.public_control,
        source_checkout=args.source_checkout,
        timeout_seconds=timeout_seconds,
    )
    serialized = json.dumps(event, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    if args.append_journal is not None:
        from local_inference_bench.event_journal import append_event

        append_event(args.append_journal, event)
    print(serialized, end="")


def run_comparison(
    *,
    stable_spec: RuntimeSpec,
    nightly_spec: RuntimeSpec,
    source_model: Path,
    export_model: Path,
    public_control: Path,
    source_checkout: Path,
    timeout_seconds: int,
) -> dict:
    timeout_seconds = _validate_timeout(timeout_seconds)
    _verify_model_identity(source_model, export_model)
    _verify_public_control(public_control)
    source_ancestry = _verify_source_ancestry(source_checkout)
    for spec in (stable_spec, nightly_spec):
        if not spec.python.is_file():
            raise FileNotFoundError(f"pinned runtime is missing: {spec.label}")

    with tempfile.TemporaryDirectory(prefix="qwen3-asr-tail-fix-") as temp_name:
        temp_dir = Path(temp_name)
        tail_control = temp_dir / "jfk-plus-80ms-silence.wav"
        _write_tail_control(public_control, tail_control)
        controls = [
            _control_descriptor(
                "public-jfk-11s", public_control, mel_frames=1_100
            ),
            _control_descriptor(
                "generated-jfk-plus-80ms-silence",
                tail_control,
                mel_frames=1_108,
            ),
        ]
        stable = _invoke_runtime(
            stable_spec,
            controls,
            export_model,
            temp_dir / "stable",
            timeout_seconds,
        )
        nightly = _invoke_runtime(
            nightly_spec,
            controls,
            export_model,
            temp_dir / "nightly",
            timeout_seconds,
        )
    return _build_event(stable, nightly, controls, source_ancestry)


def _run_child(request_path: Path, response_path: Path) -> None:
    stage = "request_validation"
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        spec = _spec_from_request(request["runtime"])
        stage = "runtime_provenance"
        provenance = _installed_runtime_provenance(spec)

        stage = "runtime_import"
        import openvino_genai
        import numpy

        stage = "pipeline_load"
        pipeline = openvino_genai.ASRPipeline(
            str(Path(request["export_model"])),
            "CPU",
            CACHE_DIR=str(Path(request["cache_dir"])),
            PERFORMANCE_HINT="LATENCY",
            INFERENCE_NUM_THREADS=24,
        )
        generation_config = pipeline.get_generation_config()
        generation_config.max_new_tokens = 512
        generation_config.return_timestamps = False
        outputs = []
        for control in request["controls"]:
            stage = "control_validation"
            path = Path(control["path"])
            if _sha256(path) != control["sha256"]:
                raise RuntimeError("control audio hash changed before inference")
            audio = _read_pcm16_float32(path, numpy)
            stage = "inference"
            result = pipeline.generate(audio, generation_config)
            transcript = result.texts[0].strip() if result.texts else ""
            outputs.append(_summarize_transcript(control["id"], transcript, result))
        stage = "response_validation"
        response = {
            "schema": "qwen3-asr-openvino-genai-tail-fix-child-v1",
            "runtime": provenance,
            "outputs": outputs,
        }
        _validate_child_response(
            response,
            expected_control_count=len(request["controls"]),
        )
    except Exception as error:
        response_path.write_text(
            json.dumps(
                {
                    "schema": "qwen3-asr-openvino-genai-tail-fix-child-failure-v1",
                    "failure": {
                        "stage": stage,
                        "kind": type(error).__name__,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return
    response_path.write_text(
        json.dumps(response, sort_keys=True) + "\n", encoding="utf-8"
    )


def _invoke_runtime(
    spec: RuntimeSpec,
    controls: list[dict],
    export_model: Path,
    run_dir: Path,
    timeout_seconds: int,
) -> dict:
    run_dir.mkdir(parents=True)
    request_path = run_dir / "request.json"
    response_path = run_dir / "response.json"
    request = {
        "runtime": _spec_to_request(spec),
        "export_model": str(export_model.resolve(strict=True)),
        "cache_dir": str(run_dir / "compile-cache"),
        "controls": [
            {"id": item["id"], "path": item["path"], "sha256": item["sha256"]}
            for item in controls
        ],
    }
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    command = [
        str(spec.python),
        str(Path(__file__).resolve()),
        "--child-request",
        str(request_path),
        "--child-response",
        str(response_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"{spec.label} exceeded the {timeout_seconds}-second run limit"
        ) from error
    if completed.returncode != 0 or not response_path.is_file():
        # Captured streams are deliberately not surfaced: a dependency must not
        # be able to leak a decoded transcript through diagnostics.
        raise RuntimeError(f"{spec.label} failed with exit code {completed.returncode}")
    response = json.loads(response_path.read_text(encoding="utf-8"))
    if response.get("schema") == (
        "qwen3-asr-openvino-genai-tail-fix-child-failure-v1"
    ):
        _validate_child_failure(response)
        failure = response["failure"]
        raise RuntimeError(
            f"{spec.label} failed during {failure['stage']} "
            f"with {failure['kind']}"
        )
    _validate_child_response(response, expected_control_count=len(controls))
    _validate_runtime_response(response["runtime"], spec)
    return response


def _validate_child_failure(response: dict) -> None:
    allowed_stages = {
        "request_validation",
        "runtime_provenance",
        "runtime_import",
        "pipeline_load",
        "control_validation",
        "inference",
        "response_validation",
    }
    if set(response) != {"schema", "failure"}:
        raise RuntimeError("child failure contains an unapproved field")
    failure = response["failure"]
    if (
        not isinstance(failure, dict)
        or set(failure) != {"stage", "kind"}
        or failure.get("stage") not in allowed_stages
        or not isinstance(failure.get("kind"), str)
        or not failure["kind"].isidentifier()
        or len(failure["kind"]) > 64
    ):
        raise RuntimeError("child failure is malformed")


def _build_event(
    stable: dict,
    nightly: dict,
    controls: list[dict],
    source_ancestry: dict,
) -> dict:
    stable_by_id = {item["control_id"]: item for item in stable["outputs"]}
    nightly_by_id = {item["control_id"]: item for item in nightly["outputs"]}
    comparisons = []
    for control in controls:
        control_id = control["id"]
        stable_output = stable_by_id[control_id]
        nightly_output = nightly_by_id[control_id]
        transcript_equal = (
            stable_output["transcript_sha256"]
            == nightly_output["transcript_sha256"]
        )
        counts_equal = all(
            stable_output[key] == nightly_output[key]
            for key in (
                "unicode_character_count",
                "utf8_byte_count",
                "generated_token_count",
            )
        )
        comparisons.append(
            {
                "control_id": control_id,
                "input": {
                    key: control[key]
                    for key in (
                        "sha256",
                        "sample_count",
                        "sample_rate_hz",
                        "mel_frame_count",
                        "encoder_remainder_frames",
                        "legacy_tail_output_token_count",
                        "fixed_tail_output_token_count",
                    )
                },
                "stable_output": stable_output,
                "nightly_output": nightly_output,
                "transcript_exactly_equal": transcript_equal,
                "output_counts_equal": counts_equal,
            }
        )
    all_transcripts_equal = all(
        item["transcript_exactly_equal"] for item in comparisons
    )
    all_counts_equal = all(item["output_counts_equal"] for item in comparisons)
    return {
        "event": "openvino_genai_qwen3_asr_tail_fix_compared",
        "protocol": EVENT_PROTOCOL,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": _producer_sha256(),
        "privacy": {
            "input_scope": "public_and_deterministically_generated_only",
            "raw_transcripts_published": False,
            "published_output_fields": "sha256_and_counts_only",
        },
        "model": {
            "source_revision": MODEL_SOURCE_REVISION,
            "exporter_revision": MODEL_EXPORTER_REVISION,
            "export_marker_sha256": EXPORT_MARKER_SHA256,
            "export_provenance_sha256": EXPORT_PROVENANCE_SHA256,
        },
        "static_implementation_containment": source_ancestry,
        "source_binding_interpretation": {
            "git_ancestry_executed": True,
            "runtime_wheel_build_attested_to_source_revision": False,
            "runtime_to_source_relationship": (
                "pinned_release_association_not_cryptographic_build_attestation"
            ),
        },
        "runtimes": {
            "stable": stable["runtime"],
            "nightly": nightly["runtime"],
        },
        "transcript_level_effect": {
            "control_count": len(comparisons),
            "all_transcripts_exactly_equal": all_transcripts_equal,
            "all_output_counts_equal": all_counts_equal,
            "effect_observed_on_bounded_controls": not all_transcripts_equal,
            "scope_note": (
                "output equality does not negate static fix containment; it only "
                "bounds transcript-level impact to these controls"
            ),
            "controls": comparisons,
        },
    }


def _control_descriptor(control_id: str, path: Path, *, mel_frames: int) -> dict:
    with wave.open(str(path), "rb") as stream:
        if (
            stream.getnchannels() != 1
            or stream.getsampwidth() != 2
            or stream.getframerate() != 16_000
            or stream.getcomptype() != "NONE"
        ):
            raise RuntimeError("tail-fix controls must be mono PCM16 16 kHz WAV")
        sample_count = stream.getnframes()
    remainder = mel_frames % ENCODER_CHUNK_FRAMES
    legacy, fixed = _tail_output_token_counts(remainder)
    return {
        "id": control_id,
        "path": str(path.resolve(strict=True)),
        "sha256": _sha256(path),
        "sample_count": sample_count,
        "sample_rate_hz": 16_000,
        "mel_frame_count": mel_frames,
        "encoder_remainder_frames": remainder,
        "legacy_tail_output_token_count": legacy,
        "fixed_tail_output_token_count": fixed,
    }


def _write_tail_control(source: Path, target: Path) -> None:
    _append_pcm16_silence(
        source,
        target,
        silence_samples=TAIL_SILENCE_SAMPLES,
    )
    if _sha256(target) != TAIL_CONTROL_SHA256:
        raise RuntimeError("generated tail control identity changed")


def _read_pcm16_float32(path: Path, numpy_module):
    with wave.open(str(path), "rb") as reader:
        if (
            reader.getnchannels() != 1
            or reader.getsampwidth() != 2
            or reader.getframerate() != 16_000
            or reader.getcomptype() != "NONE"
        ):
            raise RuntimeError("tail-fix controls must be mono PCM16 16 kHz audio")
        frames = reader.readframes(reader.getnframes())
    return numpy_module.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0


def _append_pcm16_silence(
    source: Path,
    target: Path,
    *,
    silence_samples: int,
) -> None:
    with wave.open(str(source), "rb") as reader:
        parameters = reader.getparams()
        frames = reader.readframes(reader.getnframes())
    if (
        parameters.nchannels != 1
        or parameters.sampwidth != 2
        or parameters.framerate != 16_000
        or parameters.comptype != "NONE"
    ):
        raise RuntimeError("public control identity changed")
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(parameters.nchannels)
        writer.setsampwidth(parameters.sampwidth)
        writer.setframerate(parameters.framerate)
        writer.setcomptype(parameters.comptype, "not compressed")
        writer.writeframes(frames + bytes(silence_samples * 2))


def _tail_output_token_counts(remainder_frames: int) -> tuple[int, int]:
    if not 0 <= remainder_frames < ENCODER_CHUNK_FRAMES:
        raise ValueError("remainder frames must be in [0, 100)")
    if remainder_frames == 0:
        return 0, 0
    legacy = (
        remainder_frames * TOKENS_PER_FULL_CHUNK + ENCODER_CHUNK_FRAMES - 1
    ) // ENCODER_CHUNK_FRAMES
    fixed = remainder_frames
    for _ in range(3):
        fixed = (fixed + 1) // 2
    return legacy, fixed


def _summarize_transcript(control_id: str, transcript: str, result: object) -> dict:
    encoded = transcript.encode("utf-8")
    perf_metrics = getattr(result, "perf_metrics")
    return {
        "control_id": control_id,
        "transcript_sha256": hashlib.sha256(encoded).hexdigest(),
        "unicode_character_count": len(transcript),
        "utf8_byte_count": len(encoded),
        "generated_token_count": int(perf_metrics.get_num_generated_tokens()),
    }


def _installed_runtime_provenance(spec: RuntimeSpec) -> dict:
    if sys.version_info[:3] != (3, 11, 15):
        raise RuntimeError("tail-fix probe requires pinned Python 3.11.15")
    packages = {}
    for package, (expected_version, expected_sha256) in spec.packages.items():
        actual_version = version(package)
        direct_url_text = distribution(package).read_text("direct_url.json")
        direct_url = json.loads(direct_url_text) if direct_url_text else {}
        hashes = direct_url.get("archive_info", {}).get("hashes", {})
        actual_sha256 = hashes.get("sha256")
        if actual_version != expected_version or actual_sha256 != expected_sha256:
            raise RuntimeError(f"pinned package provenance changed: {package}")
        packages[package] = {
            "version": actual_version,
            "wheel_sha256": actual_sha256,
        }
    return {
        "label": spec.label,
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "openvino_genai_source_repository": SOURCE_REPOSITORY,
        "associated_openvino_genai_source_revision": spec.source_revision,
        "packages": packages,
    }


def _validate_child_response(response: dict, *, expected_control_count: int) -> None:
    if set(response) != {"schema", "runtime", "outputs"}:
        raise RuntimeError("child response contains an unapproved field")
    if response["schema"] != "qwen3-asr-openvino-genai-tail-fix-child-v1":
        raise RuntimeError("child response schema changed")
    runtime = response["runtime"]
    allowed_runtime = {
        "label",
        "python_version",
        "openvino_genai_source_repository",
        "associated_openvino_genai_source_revision",
        "packages",
    }
    if not isinstance(runtime, dict) or set(runtime) != allowed_runtime:
        raise RuntimeError("child runtime contains an unapproved field")
    packages = runtime["packages"]
    if not isinstance(packages, dict) or set(packages) != {
        "openvino",
        "openvino-genai",
        "openvino-tokenizers",
    }:
        raise RuntimeError("child package inventory changed")
    for package in packages.values():
        if not isinstance(package, dict) or set(package) != {
            "version",
            "wheel_sha256",
        }:
            raise RuntimeError("child package contains an unapproved field")
    outputs = response["outputs"]
    if not isinstance(outputs, list) or len(outputs) != expected_control_count:
        raise RuntimeError("child response output count changed")
    allowed = {
        "control_id",
        "transcript_sha256",
        "unicode_character_count",
        "utf8_byte_count",
        "generated_token_count",
    }
    for output in outputs:
        if set(output) != allowed:
            raise RuntimeError("child output contains an unapproved field")
        if len(output["transcript_sha256"]) != 64:
            raise RuntimeError("child output hash is malformed")


def _validate_runtime_response(runtime: dict, spec: RuntimeSpec) -> None:
    expected_packages = {
        name: {"version": values[0], "wheel_sha256": values[1]}
        for name, values in spec.packages.items()
    }
    if (
        runtime["label"] != spec.label
        or runtime["python_version"] != "3.11.15"
        or runtime["openvino_genai_source_repository"] != SOURCE_REPOSITORY
        or runtime["associated_openvino_genai_source_revision"]
        != spec.source_revision
        or runtime["packages"] != expected_packages
    ):
        raise RuntimeError("child runtime provenance does not match its pin")


def _verify_source_ancestry(source_checkout: Path) -> dict:
    if not source_checkout.is_dir():
        raise FileNotFoundError("OpenVINO GenAI source checkout is missing")
    remote = _run_git(source_checkout, "remote", "get-url", "origin")
    if remote != SOURCE_REPOSITORY:
        raise RuntimeError("OpenVINO GenAI source remote does not match its pin")
    if _run_git(source_checkout, "rev-parse", "--is-shallow-repository") != "false":
        raise RuntimeError("OpenVINO GenAI ancestry requires a full-history clone")
    for revision in (
        FIX_REVISION,
        STABLE_SPEC.source_revision,
        NIGHTLY_SPEC.source_revision,
    ):
        resolved = _run_git(
            source_checkout,
            "rev-parse",
            f"{revision}^{{commit}}",
        )
        if resolved != revision:
            raise RuntimeError("OpenVINO GenAI source revision is missing")
    stable_contains_fix = _git_is_ancestor(
        source_checkout,
        FIX_REVISION,
        STABLE_SPEC.source_revision,
    )
    nightly_contains_fix = _git_is_ancestor(
        source_checkout,
        FIX_REVISION,
        NIGHTLY_SPEC.source_revision,
    )
    if stable_contains_fix or not nightly_contains_fix:
        raise RuntimeError("OpenVINO GenAI source ancestry changed")
    return {
        "repository": SOURCE_REPOSITORY,
        "fix_revision": FIX_REVISION,
        "stable_source_revision": STABLE_SPEC.source_revision,
        "nightly_source_revision": NIGHTLY_SPEC.source_revision,
        "stable_contains_fix": stable_contains_fix,
        "nightly_contains_fix": nightly_contains_fix,
        "evidence_method": "executed_git_merge_base_is_ancestor_full_clone",
        "verified_conclusion": (
            "source ancestry establishes that only the associated nightly "
            "revision contains the fix"
        ),
    }


def _run_git(source_checkout: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_checkout), *arguments],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("OpenVINO GenAI source verification failed")
    return completed.stdout.strip()


def _git_is_ancestor(
    source_checkout: Path,
    ancestor: str,
    descendant: str,
) -> bool:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(source_checkout),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError("OpenVINO GenAI source ancestry check failed")
    return completed.returncode == 0


def _verify_model_identity(source_model: Path, export_model: Path) -> None:
    summary = verify_export(source_model, export_model)
    if (
        summary["source_revision"] != MODEL_SOURCE_REVISION
        or summary["exporter_revision"] != MODEL_EXPORTER_REVISION
        or _sha256(export_model / "export-complete.json")
        != EXPORT_MARKER_SHA256
        or _sha256(export_model / "official-export-provenance.json")
        != EXPORT_PROVENANCE_SHA256
    ):
        raise RuntimeError("pinned Qwen3-ASR export identity changed")


def _verify_public_control(path: Path) -> None:
    if not path.is_file() or _sha256(path) != PUBLIC_CONTROL_SHA256:
        raise RuntimeError("pinned public JFK control identity changed")
    descriptor = _control_descriptor("public-jfk-11s", path, mel_frames=1_100)
    if descriptor["sample_count"] != 176_000:
        raise RuntimeError("pinned public JFK sample count changed")


def _validate_timeout(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_CHILD_SECONDS:
        raise ValueError("timeout must be an integer from 1 through 180 seconds")
    return value


def _with_python(spec: RuntimeSpec, python: Path) -> RuntimeSpec:
    return RuntimeSpec(
        label=spec.label,
        python=python,
        source_revision=spec.source_revision,
        packages=spec.packages,
    )


def _spec_to_request(spec: RuntimeSpec) -> dict:
    return {
        "label": spec.label,
        "source_revision": spec.source_revision,
        "packages": {
            name: {"version": values[0], "wheel_sha256": values[1]}
            for name, values in spec.packages.items()
        },
    }


def _spec_from_request(value: dict) -> RuntimeSpec:
    packages = {
        name: (details["version"], details["wheel_sha256"])
        for name, details in value["packages"].items()
    }
    matching = [
        spec
        for spec in (STABLE_SPEC, NIGHTLY_SPEC)
        if (
            value["label"] == spec.label
            and value["source_revision"] == spec.source_revision
            and packages == spec.packages
        )
    ]
    if len(matching) != 1:
        raise RuntimeError("child runtime specification is not pinned")
    return matching[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _producer_sha256() -> dict[str, str]:
    """Bind the event to every repository-owned direct producer it executes."""

    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "workers/qwen3_asr_openvino_genai_export_manifest.py",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256(path)
        for path in paths
    }


if __name__ == "__main__":
    main()
