"""Measure HunyuanOCR 1.5 through one resident CPU-only llama-server."""

from __future__ import annotations

import argparse
import base64
import json
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


MODEL_REVISION = "449e7d471a8a1ef5bd5d652e4881183d7252cbc7"
RUNTIME_REVISION = "b10588-70adb1b4cea5ee39f867792c78dc59320921eda7"
PROMPTS = {
    "doc_parse": (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    ),
    "structured_parse": "提取图中的文字。",
}
_LATEX_MARKER = re.compile(
    r"(?:\\\(|\\\[|\$\$|\\begin\{|\\frac|\\sum|\\int)"
)
_HUNYUAN_BEGIN = "<\uff5chy_begin\u2581of\u2581sentence\uff5c>"
_HUNYUAN_USER = "<\uff5chy_User\uff5c>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    if int(config["processes"]) != 1:
        raise ValueError("HunyuanOCR quality uses one resident server")
    if request["phase"] not in {"screen", "quality", "compatibility"}:
        raise ValueError("HunyuanOCR worker is bounded to screen or quality phases")
    threads = int(config["threads_per_process"])
    max_new_tokens = int(config["max_new_tokens"])
    mode = str(config["mode"])
    if mode not in PROMPTS:
        raise ValueError("unknown HunyuanOCR task mode")

    project_root = Path(__file__).resolve().parents[1]
    tool_root = project_root / "data" / "tools" / "llama-b10588-cpu"
    model_root = project_root / "data" / "models" / "hunyuanocr-1.5-449e7d47"
    server_executable = tool_root / "llama-server.exe"
    model_path = model_root / "hyocr-f16.gguf"
    projector_path = model_root / "mmproj-hyocr-f16.gguf"
    for required in (server_executable, model_path, projector_path):
        if not required.is_file():
            raise FileNotFoundError("verified HunyuanOCR runtime assets are incomplete")

    port = _reserve_local_port()
    server_command = _server_command(
        server_executable=server_executable,
        model_path=model_path,
        projector_path=projector_path,
        port=port,
        threads=threads,
    )
    artifact_root = Path(request["response_path"]).parent
    server_stdout_path = artifact_root / "hunyuan-server.stdout.txt"
    server_stderr_path = artifact_root / "hunyuan-server.stderr.txt"
    loaded_at = time.perf_counter()
    with (
        server_stdout_path.open("w", encoding="utf-8") as server_stdout,
        server_stderr_path.open("w", encoding="utf-8") as server_stderr,
    ):
        server = subprocess.Popen(
            server_command,
            cwd=project_root,
            stdout=server_stdout,
            stderr=server_stderr,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            _wait_until_ready(server, port, timeout_seconds=900.0)
            load_seconds = time.perf_counter() - loaded_at
            media_marker = str(_get_json(port, "/props")["media_marker"])
            warmup = _recognize(
                port=port,
                item=request["workload"]["warmup_item"],
                prompt=PROMPTS["structured_parse"],
                max_new_tokens=min(128, max_new_tokens),
                capture_prediction=False,
                media_marker=media_marker,
            )
            if not warmup["success"]:
                raise RuntimeError("HunyuanOCR warmup failed")

            started = time.perf_counter()
            records = []
            for item in request["workload"]["items"]:
                record = _recognize(
                    port=port,
                    item=item,
                    prompt=PROMPTS[mode],
                    max_new_tokens=max_new_tokens,
                    capture_prediction=bool(request["capture_predictions"]),
                    media_marker=media_marker,
                )
                record["completed_offset_seconds"] = time.perf_counter() - started
                records.append(record)
            steady_wall_seconds = time.perf_counter() - started
        finally:
            _stop_server(server)

    write_private_records(Path(request["private_records_path"]), records)
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="ocr",
        runtime_name="llama_cpp_hunyuanocr_1_5",
        runtime_version=RUNTIME_REVISION,
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[load_seconds],
        warmup_seconds=[warmup["latency_seconds"]],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="resident_server",
    )
    public_summary["generation"] = _generation_summary(records, max_new_tokens)
    public_summary["model"] = {
        "model_revision": MODEL_REVISION,
        "compute_type": "f16",
        "backend": "cpu",
        "mode": mode,
        "threads": threads,
        "parallel_slots": 1,
        "sequence_limit_tokens": 10240,
    }
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _server_command(
    *,
    server_executable: Path,
    model_path: Path,
    projector_path: Path,
    port: int,
    threads: int,
) -> list[str]:
    return [
        str(server_executable),
        "--offline",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--alias",
        "HYOCR15",
        "--model",
        str(model_path),
        "--mmproj",
        str(projector_path),
        "--device",
        "none",
        "--n-gpu-layers",
        "0",
        "--fit",
        "off",
        "--no-kv-offload",
        "--no-op-offload",
        "--mmproj-device",
        "none",
        "--no-mmproj-offload",
        "--no-warmup",
        "--threads",
        str(threads),
        "--threads-batch",
        str(threads),
        "--ctx-size",
        "10240",
        "--n-predict",
        "4096",
        "--parallel",
        "1",
        "--chat-template",
        "hunyuan-vl",
        "--timeout",
        "1200",
    ]


def _recognize(
    *,
    port: int,
    item: dict,
    prompt: str,
    max_new_tokens: int,
    capture_prediction: bool,
    media_marker: str,
) -> dict:
    started = time.perf_counter()
    try:
        response = _post_json(
            port,
            "/completion",
            _raw_completion_payload(
                image_path=Path(item["path"]),
                prompt=_raw_hunyuan_prompt(media_marker, prompt),
                max_new_tokens=max_new_tokens,
            ),
            timeout_seconds=1200.0,
        )
        text = str(response["content"]).strip()
        stop_type = str(response.get("stop_type", ""))
        timings = response.get("timings", {})
        completion_tokens = int(timings.get("predicted_n", 0))
    except Exception as error:
        return {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
    latency_seconds = time.perf_counter() - started
    output_is_valid = bool(text) or not item.get("expected_text", True)
    format_metrics = _format_metrics(text)
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": None if output_is_valid else "empty_output",
        "latency_seconds": latency_seconds,
        "units": 1.0 if output_is_valid else 0.0,
        "output_character_count": len(text),
        "completion_tokens": completion_tokens,
        "token_cap_hit": (
            completion_tokens == max_new_tokens and stop_type == "limit"
        ),
        "length_finish": stop_type == "limit",
        "stop_finish": stop_type in {"eos", "word"},
        "prompt_seconds": _milliseconds_to_seconds(timings.get("prompt_ms")),
        "generation_seconds": _milliseconds_to_seconds(
            timings.get("predicted_ms")
        ),
        "generated_tokens_per_second": _safe_number(
            timings.get("predicted_per_second")
        ),
        **format_metrics,
    }
    if capture_prediction:
        record["prediction"] = text
        record["lines"] = [
            {"text": line}
            for line in text.splitlines()
            if line.strip()
        ]
        record["raw_response"] = response
    return record


def _raw_hunyuan_prompt(media_marker: str, prompt: str) -> str:
    # Equivalent to llama.cpp's pinned LLM_CHAT_TEMPLATE_HUNYUAN_VL branch.
    return f"{_HUNYUAN_BEGIN}{media_marker}{prompt}{_HUNYUAN_USER}"


def _raw_completion_payload(
    *,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
) -> dict:
    return {
        "prompt": {
            "prompt_string": prompt,
            "multimodal_data": [
                base64.b64encode(image_path.read_bytes()).decode("ascii")
            ],
        },
        "n_predict": max_new_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0,
        "repeat_penalty": 1.08,
        "seed": 1,
        "stream": False,
        "cache_prompt": False,
    }


def _generation_summary(records: list[dict], max_new_tokens: int) -> dict:
    rates = [
        record["generated_tokens_per_second"]
        for record in records
        if isinstance(record.get("generated_tokens_per_second"), (int, float))
    ]
    completion_tokens = [
        int(record.get("completion_tokens", 0)) for record in records
    ]
    return {
        "max_new_tokens": max_new_tokens,
        "completion_tokens_total": sum(completion_tokens),
        "completion_tokens_max": max(completion_tokens, default=0),
        "token_cap_hit_count": sum(
            bool(record.get("token_cap_hit")) for record in records
        ),
        "length_finish_count": sum(
            bool(record.get("length_finish")) for record in records
        ),
        "stop_finish_count": sum(
            bool(record.get("stop_finish")) for record in records
        ),
        "latex_marker_count": sum(
            bool(record.get("latex_marker")) for record in records
        ),
        "complete_html_table_count": sum(
            bool(record.get("complete_html_table")) for record in records
        ),
        "mean_generated_tokens_per_second": (
            sum(rates) / len(rates) if rates else 0.0
        ),
    }


def _format_metrics(text: str) -> dict:
    folded = text.casefold()
    return {
        "latex_marker": bool(_LATEX_MARKER.search(text)),
        "complete_html_table": "<table" in folded and "</table>" in folded,
    }


def _reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _wait_until_ready(
    process: subprocess.Popen,
    port: int,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(f"llama-server exited during load: {exit_code}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health",
                timeout=2.0,
            ) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)
    raise TimeoutError("llama-server did not become ready")


def _post_json(
    port: int,
    route: str,
    payload: dict,
    *,
    timeout_seconds: float,
) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(port: int, route: str) -> dict:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{route}",
        timeout=30.0,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _milliseconds_to_seconds(value: object) -> float | None:
    number = _safe_number(value)
    return None if number is None else number / 1000.0


def _safe_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15.0)


if __name__ == "__main__":
    main()
