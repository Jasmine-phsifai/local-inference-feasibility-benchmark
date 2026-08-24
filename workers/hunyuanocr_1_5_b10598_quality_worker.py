"""Select llama.cpp b10598 for the fixed HunyuanOCR 1.5 quality gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from local_inference_bench.bounded_vlm_assets import (  # noqa: E402
    load_and_verify_candidate_assets,
)
from workers import hunyuanocr_1_5_server_worker as worker  # noqa: E402


CANDIDATE_ID = "hunyuanocr_1_5_gguf_cpu"
REQUEST_PROTOCOL = "bounded-vlm-b10598-run-v2"
RUNTIME_REVISION = "b10598-56db501e73cfb10c8fcce61be708f5c3ee749271"
EXPECTED_CONFIG = {
    "processes": 1,
    "threads_per_process": 24,
    "max_new_tokens": 4096,
    "mode": "doc_parse",
}
EXPECTED_ITEM_IDS = ["code_formula", "dense_table", "negative_diagram"]
_ORIGINAL_SERVER_COMMAND = worker._server_command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    assets = load_and_verify_candidate_assets(
        project_root=PROJECT_ROOT,
        candidate_id=CANDIDATE_ID,
    )
    _validate_request(request, assets=assets)
    server_path = assets["runtime"]["entrypoints"]["llama_server"]["path"]

    def b10598_server_command(
        *,
        server_executable: Path,
        model_path: Path,
        projector_path: Path,
        port: int,
        threads: int,
    ) -> list[str]:
        del server_executable
        return build_b10598_server_command(
            b10598_server=server_path,
            model_path=model_path,
            projector_path=projector_path,
            port=port,
            threads=threads,
        )

    worker.RUNTIME_REVISION = RUNTIME_REVISION
    worker._server_command = b10598_server_command
    sys.argv = [sys.argv[0], "--request", str(args.request)]
    worker.main()


def build_b10598_server_command(
    *,
    b10598_server: Path,
    model_path: Path,
    projector_path: Path,
    port: int,
    threads: int,
) -> list[str]:
    """Preserve the proven Hunyuan command while changing only the runtime."""

    if not b10598_server.is_file():
        raise FileNotFoundError("verified llama.cpp b10598 server is missing")
    return _ORIGINAL_SERVER_COMMAND(
        server_executable=b10598_server,
        model_path=model_path,
        projector_path=projector_path,
        port=port,
        threads=threads,
    )


def _validate_request(request: object, *, assets: dict) -> None:
    if not isinstance(request, dict):
        raise ValueError("Hunyuan quality request must be an object")
    if (
        request.get("protocol") != REQUEST_PROTOCOL
        or request.get("candidate_id") != CANDIDATE_ID
        or request.get("task") != "ocr"
        or request.get("phase") != "quality"
        or request.get("capture_predictions") is not True
        or request.get("config") != EXPECTED_CONFIG
    ):
        raise ValueError("Hunyuan quality request identity changed")
    workload = request.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("Hunyuan quality workload is invalid")
    if (
        workload.get("workload_class") != "generated_quality_control"
        or [item.get("id") for item in workload.get("items", [])]
        != EXPECTED_ITEM_IDS
        or workload.get("warmup_item", {}).get("id") != "warmup"
    ):
        raise ValueError("Hunyuan quality workload changed")
    expected_images = assets["fixtures"]["images"]
    requested_items = [workload["warmup_item"], *workload["items"]]
    for item in requested_items:
        expected = expected_images.get(item["id"])
        if (
            expected is None
            or Path(item.get("path", "")).resolve() != expected["path"]
        ):
            raise ValueError("Hunyuan quality fixture identity changed")
    output_paths = [request.get("response_path"), request.get("private_records_path")]
    if any(type(value) is not str or not value for value in output_paths):
        raise ValueError("Hunyuan quality output paths are invalid")
    resolved = [Path(value).resolve() for value in output_paths]
    if resolved[0].parent != resolved[1].parent:
        raise ValueError("Hunyuan quality outputs must share one run directory")
    if not resolved[0].is_relative_to(PROJECT_ROOT / "results" / "artifacts"):
        raise ValueError("Hunyuan quality outputs must stay under ignored artifacts")


if __name__ == "__main__":
    main()
