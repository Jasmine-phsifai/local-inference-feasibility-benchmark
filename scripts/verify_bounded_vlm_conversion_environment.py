"""Verify the exact CPU-only environment used for Hunyuan GGUF conversion."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_locked_environment import verify_locked_environment


def main() -> None:
    environment = verify_locked_environment(
        PROJECT_ROOT
        / "environments"
        / "bounded_vlm_conversion"
        / "requirements.lock.txt",
        expected_python=(3, 12, 13),
    )

    import numpy
    import sentencepiece
    import torch
    import transformers

    if torch.cuda.is_available():
        raise RuntimeError("bounded VLM conversion environment must remain CPU-only")
    if not all((numpy, sentencepiece, transformers)):
        raise RuntimeError("bounded VLM conversion import surface is incomplete")
    print(
        json.dumps(
            {
                "status": "verified",
                "environment": environment,
                "device": "CPU",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
