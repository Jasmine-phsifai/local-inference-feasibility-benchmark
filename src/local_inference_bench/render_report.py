import json
from collections import defaultdict
from datetime import datetime, timezone

from .event_journal import read_events
from .load_registry import load_json
from .project_paths import EVENTS_PATH, HARDWARE_PATH, REGISTRY_PATH, REPORT_PATH


def _throughput_value(event: dict) -> float:
    throughput = event.get("result", {}).get("throughput", {})
    return throughput.get("audio_hours_per_wall_hour", throughput.get("images_per_hour", 0.0))


def _format_hours(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.1f} min"
    if hours < 1000:
        return f"{hours:.2f} h"
    return f"{hours:,.0f} h ({hours / 24:.1f} days)"


def _projection_cells(result: dict, task: str) -> list[str]:
    projections = result.get("projections", {}).get("projected_wall_hours", {})
    if task.startswith("asr"):
        return [_format_hours(projections.get(key, 0)) for key in ("2.5", "37.5", "375.0", "56000.0")]
    return [
        f"{_format_hours(projections.get(key, [0, 0])[0])}–{_format_hours(projections.get(key, [0, 0])[1])}"
        for key in ("50-80", "750-1200", "7500-12000", "1120000-1790000")
    ]


def render_report() -> None:
    events = read_events(EVENTS_PATH)
    registry = load_json(REGISTRY_PATH)
    hardware = load_json(HARDWARE_PATH)
    successes: dict[str, list[dict]] = defaultdict(list)
    failures: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        if event.get("event") == "attempt_succeeded":
            successes[event["candidate_id"]].append(event)
        elif event.get("event") in {"attempt_failed", "attempt_interrupted"}:
            failures[event["candidate_id"]].append(event)

    best = {}
    for candidate_id, candidate_events in successes.items():
        latest_fingerprint = candidate_events[-1].get("code_fingerprint")
        current = [event for event in candidate_events if event.get("code_fingerprint") == latest_fingerprint]
        best[candidate_id] = max(current, key=_throughput_value)

    gpu_names = ", ".join(device["name"] for device in hardware["gpu"])
    lines = [
        "# Local inference feasibility report", "", f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "> This stage measures installation, execution, and speed feasibility. It does not compare final recognition quality.", "",
        "## Host and test boundary", "",
        f"- CPU: {hardware['cpu']['name'].strip()}, {hardware['cpu']['physical_cores']} physical / {hardware['cpu']['logical_processors']} logical processors.",
        f"- RAM: {hardware['memory']['total_bytes'] / 2**30:.1f} GiB usable; {hardware['memory']['available_bytes'] / 2**30:.1f} GiB available at capture.",
        f"- Display adapters: {gpu_names}. The Intel iGPU uses shared memory; the reported adapter aperture is not a hard allocation ceiling.",
        "- NPU: no visible ComputeAccelerator/NPU device. NVIDIA/CUDA: no device or runtime present.",
        "- The official SenseVoice Vulkan package enumerated Intel Graphics but then rejected Vulkan graph execution, so iGPU acceleration is detected but not runnable in that package on this host.",
        "- Audio: public-domain JFK speech sample distributed by whisper.cpp (~11 s). Images: three synthetic 1280×720 bilingual lecture-like slides.",
        "- Inputs are fallback samples, not representative lecture recordings/screenshots; projections are performance estimates only.", "",
        "## Candidate scope", "",
        "Candidates were selected from current official implementations and model repositories; the registry preserves exact roles, environment manifests, configurations, and source URLs.", "",
    ]
    for candidate in registry["candidates"]:
        lines.append(f"- [{candidate['id']}]({candidate['official_url']}): {candidate['role']} (`{candidate['status']}`).")
    environment_sizes_path = EVENTS_PATH.parent / "environment-sizes.json"
    if environment_sizes_path.exists():
        environment_sizes = load_json(environment_sizes_path)
        lines.extend(["", "## Installation footprint", "", environment_sizes["note"], "", "| Environment | Apparent size |", "|---|---:|"])
        for item in environment_sizes["environments"]:
            lines.append(f"| {item['environment']} | {item['apparent_bytes'] / 2**30:.2f} GiB |")
    lines.extend(["",
        "## Feasibility summary", "",
        "| Candidate | Role | Classification | Best tested configuration | Warm throughput | Load | Peak host CPU | Peak process RAM | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ])
    for candidate in registry["candidates"]:
        candidate_id = candidate["id"]
        if candidate_id in best:
            event = best[candidate_id]
            result = event["result"]
            throughput = result["throughput"]
            if "audio_hours_per_wall_hour" in throughput:
                rate = f"{throughput['audio_hours_per_wall_hour']:.2f} audio h/wall h (RTF {throughput['real_time_factor']:.4f})"
            else:
                seconds_per_image = throughput.get("seconds_per_image", result["inference_seconds"] / result["input_count"])
                rate = f"{throughput['images_per_hour']:,.0f} images/h ({seconds_per_image:.3f} s/image)"
            if result.get("below_generation_cutoff"):
                classification = "runnable but below generation cutoff"
            elif throughput.get("real_time_factor", 0) >= 1:
                classification = "runnable but slower than real time"
            else:
                classification = "feasible on CPU"
            model_mib = (result.get("model_size_bytes") or 0) / 2**20
            note = f"model files {model_mib:.1f} MiB; valid nonempty output"
            runtime_version = candidate.get("runtime_pin") or result.get("runtime", {}).get("version") or result.get("runtime", {}).get("release")
            precision = result.get("model", {}).get("quantization") or result.get("model", {}).get("dtype")
            if runtime_version:
                note += f"; runtime {runtime_version}"
            if precision:
                note += f"; {precision}"
            if result.get("below_generation_cutoff"):
                token_rate = result.get("tokens_per_second", throughput.get("estimated_output_tokens_per_second"))
                note += f"; {token_rate:.3f} generated tok/s"
            load = f"{result['load_seconds']:.2f} s" if result.get("load_seconds") is not None else "included in CLI timing"
            lines.append(f"| {candidate_id} | {candidate['role']} | {classification} | `{event['config']}` | {rate} | {load} | {event['resources']['peak_cpu_percent_of_host']:.1f}% | {event['resources']['peak_rss_bytes'] / 2**30:.2f} GiB | {note} |")
        elif failures.get(candidate_id):
            lines.append(f"| {candidate_id} | {candidate['role']} | installation or runtime failure | — | — | — | — | — | {len(failures[candidate_id])} persisted failed/interrupted attempt(s) |")
        else:
            note = "optional; not measured" if candidate["status"] == "optional" else "setup/benchmark pending"
            lines.append(f"| {candidate_id} | {candidate['role']} | experimental | — | — | — | — | — | {note} |")

    lines.extend(["", "## Workload projections", ""])
    for candidate_id, event in best.items():
        candidate = next(item for item in registry["candidates"] if item["id"] == candidate_id)
        headers = "lecture 2.5 h; course 37.5 h; ten courses 375 h; long-term 56,000 h" if candidate["task"].startswith("asr") else "lecture 50–80; course 750–1,200; ten courses 7,500–12,000; long-term 1.12–1.79M images"
        cells = _projection_cells(event["result"], candidate["task"])
        lines.extend([f"### {candidate_id}", "", f"Best configuration: `{event['config']}`. Scales: {headers}.", "", "| Lecture | Course | Ten courses | Long-term |", "|---:|---:|---:|---:|", f"| {' | '.join(cells)} |", ""])

    setup_failure_path = EVENTS_PATH.parent / "setup-failures.jsonl"
    setup_failures = []
    if setup_failure_path.exists():
        setup_failures = [json.loads(line) for line in setup_failure_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines.extend(["## Persisted failures and limitations", ""])
    for failure in setup_failures:
        lines.append(f"- `{failure['candidate_id']}` {failure['phase']}: {failure['summary']} Remediation: {failure['remediation']}")
    for candidate_id, candidate_failures in failures.items():
        lines.append(f"- `{candidate_id}`: {len(candidate_failures)} failed/interrupted benchmark attempt(s); bounded stderr tails are in `results/events.jsonl` and ignored artifacts.")
    lines.extend([
        "- Windows PowerShell policy blocked direct `.ps1` execution. Reproduction uses process-local `-ExecutionPolicy Bypass`; machine policy was not changed.",
        "- Peak CPU comes from process-tree CPU-time deltas normalized against 24 logical processors. Intel iGPU memory/utilization counters were unavailable in the controller.",
        "- Model download and load time are separate from warmed inference. One-time downloads are excluded from workload projections.",
        "- The fallback samples prove execution and plausible output only; use representative inputs for the later quality/error-rate stage.",
        "- PP-OCRv6 tiny was also tested with a three-image input batch; it did not beat the best sequential 8-thread run on this small sample.", "",
        "## Current recommendation", "",
        "Use PP-OCRv6 tiny or multi-worker RapidOCR for bulk slides and SenseVoice GGUF for bulk audio. Reserve larger OCR/VLM models for difficult pages only after representative quality testing. Maximum threads were often slower, and the packaged Intel Vulkan path is not currently usable.", "",
        "No paid APIs, crawler, RAG, note generation, privacy filtering, fine-tuning, or production integration were used.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
