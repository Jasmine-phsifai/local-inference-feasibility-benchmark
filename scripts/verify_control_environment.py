"""Verify the exact environment used by the sustained-run controller."""

from __future__ import annotations

import json
import os
import re
import sys
from importlib.metadata import distributions
from pathlib import Path
from site import getsitepackages
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_inference_bench.verify_locked_environment import verify_locked_environment


def main() -> None:
    environment = verify_locked_environment(
        PROJECT_ROOT / "environments" / "control" / "requirements.lock.txt",
        expected_python=(3, 12, 14),
        allowed_extra_packages={"local-inference-bench": "0.1.0"},
    )
    _verify_editable_project_origin()
    print(
        json.dumps(
            {
                "status": "verified",
                "environment": environment,
                "editable_project_origin": "verified",
            },
            sort_keys=True,
        )
    )


def _verify_editable_project_origin() -> None:
    installed = next(
        (
            value
            for value in distributions(path=getsitepackages())
            if value.metadata.get("Name", "").casefold()
            == "local-inference-bench"
        ),
        None,
    )
    if installed is None:
        raise RuntimeError("control project distribution is missing")
    direct_url_text = installed.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError("control project installation has no direct origin")
    payload = json.loads(direct_url_text)
    parsed = urlparse(payload.get("url", ""))
    raw_path = unquote(parsed.path)
    if os.name == "nt" and re.match(r"^/[A-Za-z]:/", raw_path):
        raw_path = raw_path[1:]
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or payload.get("dir_info", {}).get("editable") is not True
        or Path(raw_path).resolve() != PROJECT_ROOT.resolve()
    ):
        raise RuntimeError("control project editable origin is not this checkout")


if __name__ == "__main__":
    main()
