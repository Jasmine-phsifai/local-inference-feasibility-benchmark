"""Run the fixed one-page OvisOCR2 b10598 CPU quality gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.bounded_vlm_assets import (
    load_and_verify_candidate_assets,
)
from local_inference_bench.terminate_process_tree import terminate_process_tree


CANDIDATE_ID = "ovisocr2_q8_cpu"
REQUEST_PROTOCOL = "bounded-vlm-b10598-run-v2"
RUNTIME_REVISION = "b10598-56db501e73cfb10c8fcce61be708f5c3ee749271"
SAMPLE_ID = "page_008_table_columns"
RESPONSE_MARKER = "<MTMD_TEST_RESPONSE_MARKER_OVISOCR2_B10598_V3>"
IDENTITY_TEMPLATE = '{{ messages[0]["content"] }}'
OFFICIAL_ASSISTANT_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
OFFICIAL_PROMPT = (
    "\nExtract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/'
    "bbox_{left}_{top}_{right}_{bottom}.jpg\" />, where left, top, right, bottom "
    "are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. "
    "Format tables as HTML: <table>...</table>. Transcribe all other text as "
    "standard Markdown. Preserve the original text without translation or "
    "paraphrasing."
)
EXPECTED_CONFIG = {
    "processes": 1,
    "threads_per_process": 24,
    "max_new_tokens": 4096,
    "mode": "source_faithful",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    run_quality_gate(request)


def run_quality_gate(request: dict) -> None:
    project_root = PROJECT_ROOT
    _validate_request(request, project_root=project_root)
    assets = load_and_verify_candidate_assets(
        project_root=project_root,
        candidate_id=CANDIDATE_ID,
    )
    output_root = Path(request["response_path"]).resolve().parent
    output_paths = {
        "response": Path(request["response_path"]).resolve(),
        "records": Path(request["private_records_path"]).resolve(),
        "stdout": output_root / "ovis.stdout.txt",
        "stderr": output_root / "ovis.stderr.txt",
        "llama_log": output_root / "ovis.llama.log",
        "command": output_root / "ovis.command.json",
    }
    collisions = [path.name for path in output_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"Ovis quality worker refuses overwrite: {collisions}")

    image_path = assets["fixtures"]["images"][SAMPLE_ID]["path"]
    executable = assets["runtime"]["entrypoints"]["llama_mtmd_cli"]["path"]
    model = assets["artifacts"]["model"]["path"]
    projector = assets["artifacts"]["projector"]["path"]
    prompt = build_rendered_prompt()
    command = build_command(
        executable=executable,
        model=model,
        projector=projector,
        image=image_path,
        prompt=prompt,
        log_path=output_paths["llama_log"],
        threads=EXPECTED_CONFIG["threads_per_process"],
        max_new_tokens=EXPECTED_CONFIG["max_new_tokens"],
    )
    output_paths["command"].write_text(
        json.dumps(
            {
                "argv": command,
                "cwd": str(project_root),
                "rendered_prompt_sha256": _sha256_text(prompt),
                "response_marker": RESPONSE_MARKER,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    timeout_seconds = float(request["timeout_seconds"])
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    timed_out = False
    environment = os.environ.copy()
    environment["MTMD_TEST_RESPONSE_MARKER"] = RESPONSE_MARKER
    with (
        output_paths["stdout"].open("xb") as stdout,
        output_paths["stderr"].open("xb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            ),
        )
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process_tree(process.pid)
            return_code = process.wait(timeout=15.0)
    wall_seconds = time.perf_counter() - started

    stdout_text = output_paths["stdout"].read_text(
        encoding="utf-8", errors="replace"
    )
    stderr_text = output_paths["stderr"].read_text(
        encoding="utf-8", errors="replace"
    )
    llama_text = (
        output_paths["llama_log"].read_text(encoding="utf-8", errors="replace")
        if output_paths["llama_log"].is_file()
        else ""
    )
    prediction = extract_prediction(stdout_text)
    perf_text = stderr_text + "\n" + llama_text
    completion_tokens = extract_completion_tokens(perf_text)
    token_cap_hit = (
        completion_tokens is not None
        and completion_tokens >= EXPECTED_CONFIG["max_new_tokens"] - 1
    )
    success = (
        return_code == 0
        and not timed_out
        and bool(prediction)
        and completion_tokens is not None
    )
    stop_finish = success and not token_cap_hit
    record = {
        "sample_id": SAMPLE_ID,
        "success": success,
        "failure_kind": None if success else _failure_kind(
            return_code=return_code,
            timed_out=timed_out,
            prediction=prediction,
            completion_tokens=completion_tokens,
        ),
        "prediction": prediction,
        "lines": [
            {"text": line} for line in prediction.splitlines() if line.strip()
        ],
        "latency_seconds": wall_seconds,
        "units": 1.0 if success else 0.0,
        "completion_tokens": completion_tokens or 0,
        "token_cap_hit": token_cap_hit,
        "length_finish": token_cap_hit,
        "stop_finish": stop_finish,
        "complete_html_table": bool(
            re.search(r"<table\b.*?</table>", prediction, flags=re.I | re.S)
        ),
    }
    output_paths["records"].write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "candidate_id": CANDIDATE_ID,
        "task": "ocr",
        "runtime_name": "llama_cpp_mtmd_cli",
        "runtime_version": RUNTIME_REVISION,
        "workload_class": "generated_quality_control",
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "attempted": 1,
            "completed": int(success),
            "failed": int(not success),
        },
        "timing": {
            "steady_wall_seconds": wall_seconds,
            "load_seconds": _milliseconds_to_seconds(
                _extract_milliseconds(perf_text, "load time")
            ),
            "prompt_eval_seconds": _milliseconds_to_seconds(
                _extract_milliseconds(perf_text, "prompt eval time")
            ),
            "generation_seconds": _milliseconds_to_seconds(
                _extract_milliseconds(perf_text, "eval time")
            ),
            "total_seconds": _milliseconds_to_seconds(
                _extract_milliseconds(perf_text, "total time")
            ),
            "image_encode_seconds": _milliseconds_to_seconds(
                _extract_integer_milliseconds(perf_text, "mtmd batch encoding done in")
            ),
            "image_decode_seconds": _milliseconds_to_seconds(
                _extract_integer_milliseconds(perf_text, "image decoded (batch 1/1) in")
            ),
        },
        "generation": {
            "max_new_tokens": EXPECTED_CONFIG["max_new_tokens"],
            "completion_tokens_total": completion_tokens or 0,
            "token_cap_hit_count": int(token_cap_hit),
            "stop_finish_count": int(stop_finish),
        },
        "model": {
            "backend": "cpu",
            "compute_type": "q8_0_text_bf16_projector",
            "threads": EXPECTED_CONFIG["threads_per_process"],
        },
    }
    output_paths["response"].write_text(
        json.dumps({"public_summary": summary}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if not success:
        raise RuntimeError(f"Ovis quality inference failed: {record['failure_kind']}")


def build_rendered_prompt() -> str:
    """Build the exact one-image raw prompt used by the upstream Ovis card."""

    prompt = (
        "<|im_start|>user\n<__media__>"
        + OFFICIAL_PROMPT
        + "<|im_end|>\n"
        + OFFICIAL_ASSISTANT_SUFFIX
    )
    if (
        prompt.count("<__media__>") != 1
        or not prompt.startswith(
            "<|im_start|>user\n<__media__>\nExtract all readable content"
        )
        or not prompt.endswith(OFFICIAL_ASSISTANT_SUFFIX)
    ):
        raise RuntimeError("Ovis rendered prompt identity is invalid")
    return prompt


def build_command(
    *,
    executable: Path,
    model: Path,
    projector: Path,
    image: Path,
    prompt: str,
    log_path: Path,
    threads: int,
    max_new_tokens: int,
) -> list[str]:
    """Return the fixed CPU-only b10598 mtmd command."""

    return [
        str(executable),
        "-m",
        str(model),
        "-mm",
        str(projector),
        "--image",
        str(image),
        "-p",
        prompt,
        "--jinja",
        "--chat-template",
        IDENTITY_TEMPLATE,
        "--ctx-size",
        "32768",
        "--predict",
        str(max_new_tokens),
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads),
        "--temperature",
        "0",
        "--seed",
        "1",
        "--device",
        "none",
        "-ngl",
        "0",
        "--no-mmproj-offload",
        "--mmproj-device",
        "none",
        "--no-warmup",
        "--perf",
        "--verbose",
        "--log-timestamps",
        "--log-file",
        str(log_path),
    ]


def extract_prediction(stdout_text: str) -> str:
    if stdout_text.count(RESPONSE_MARKER) != 1:
        return ""
    return stdout_text.split(RESPONSE_MARKER, 1)[1].strip()


def extract_completion_tokens(log_text: str) -> int | None:
    matches = re.findall(
        r"llama_perf_context_print:\s+eval time\s*=\s*[0-9.]+\s*ms\s*/\s*(\d+)\s*runs",
        log_text,
    )
    return int(matches[-1]) if matches else None


def _validate_request(request: object, *, project_root: Path) -> None:
    if not isinstance(request, dict):
        raise ValueError("Ovis quality request must be an object")
    if (
        request.get("protocol") != REQUEST_PROTOCOL
        or request.get("candidate_id") != CANDIDATE_ID
        or request.get("task") != "ocr"
        or request.get("phase") != "quality"
        or request.get("capture_predictions") is not True
        or request.get("config") != EXPECTED_CONFIG
    ):
        raise ValueError("Ovis quality request identity changed")
    workload = request.get("workload")
    if (
        not isinstance(workload, dict)
        or workload.get("workload_class") != "generated_quality_control"
        or [item.get("id") for item in workload.get("items", [])] != [SAMPLE_ID]
    ):
        raise ValueError("Ovis quality workload changed")
    timeout = request.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("Ovis quality timeout is invalid")
    if not 60 <= float(timeout) <= 1800:
        raise ValueError("Ovis quality timeout must remain bounded")
    output_paths = [request.get("response_path"), request.get("private_records_path")]
    if any(type(value) is not str or not value for value in output_paths):
        raise ValueError("Ovis quality output paths are invalid")
    resolved = [Path(value).resolve() for value in output_paths]
    if resolved[0].parent != resolved[1].parent:
        raise ValueError("Ovis quality outputs must share one run directory")
    if not resolved[0].is_relative_to(project_root.resolve() / "results" / "artifacts"):
        raise ValueError("Ovis quality outputs must stay under ignored artifacts")


def _failure_kind(
    *,
    return_code: int,
    timed_out: bool,
    prediction: str,
    completion_tokens: int | None,
) -> str:
    if timed_out:
        return "timeout"
    if return_code != 0:
        return "runtime_exit"
    if not prediction:
        return "missing_response_marker_or_empty_output"
    if completion_tokens is None:
        return "missing_generation_timing"
    return "invalid_output"


def _extract_milliseconds(log_text: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)}\s*=\s*([0-9.]+)\s*ms", log_text)
    return float(matches[-1]) if matches else None


def _extract_integer_milliseconds(log_text: str, label: str) -> float | None:
    matches = re.findall(rf"{re.escape(label)}\s+(\d+)\s+ms", log_text)
    return float(matches[-1]) if matches else None


def _milliseconds_to_seconds(value: float | None) -> float | None:
    return None if value is None else value / 1000.0


def _sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
