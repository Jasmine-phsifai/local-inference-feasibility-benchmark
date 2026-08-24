"""Verify the independently installed pinned OCRLLM compatibility snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_locked_environment import verify_locked_environment

if __package__:
    from scripts.ocrllm_compatibility_provenance import (
        SNAPSHOT_RELATIVE_PATH,
        verify_ocrllm_installation,
    )
else:
    from ocrllm_compatibility_provenance import (  # type: ignore[no-redef]
        SNAPSHOT_RELATIVE_PATH,
        verify_ocrllm_installation,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(__file__).resolve().parents[1] / SNAPSHOT_RELATIVE_PATH,
    )
    args = parser.parse_args()
    environment = verify_locked_environment(
        PROJECT_ROOT
        / "environments"
        / "ocrllm_compatibility"
        / "requirements.lock.txt",
        expected_python=(3, 11, 15),
        allowed_extra_packages={"ocrllm": "0.1.0"},
    )
    evidence = verify_ocrllm_installation(args.snapshot)
    evidence["environment"] = environment
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
