from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "registries" / "candidates.json"
PLAN_PATH = PROJECT_ROOT / "registries" / "benchmark_plan.json"
HARDWARE_PATH = PROJECT_ROOT / "results" / "hardware.json"
EVENTS_PATH = PROJECT_ROOT / "results" / "events.jsonl"
ARTIFACTS_PATH = PROJECT_ROOT / "results" / "artifacts"
INPUT_MANIFEST_PATH = PROJECT_ROOT / "data" / "inputs" / "manifest.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "local-inference-feasibility.md"
SUSTAINED_REGISTRY_PATH = PROJECT_ROOT / "registries" / "sustained_candidates.json"
SUSTAINED_EVENTS_PATH = PROJECT_ROOT / "results" / "sustained-events.jsonl"
SUSTAINED_ARTIFACTS_PATH = PROJECT_ROOT / "results" / "artifacts" / "sustained"
QUALITY_EVENTS_PATH = PROJECT_ROOT / "results" / "quality-events.jsonl"
