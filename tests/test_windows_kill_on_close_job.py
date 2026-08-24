import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from local_inference_bench.windows_kill_on_close_job import (
    CREATE_SUSPENDED,
    WindowsKillOnCloseJob,
)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_closing_job_kills_child_after_root_crash(tmp_path: Path) -> None:
    for iteration in range(5):
        child_pid_path = tmp_path / f"child-pid-{iteration}.json"
        root_code = (
            "import json,subprocess,sys;"
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            f"open({str(child_pid_path)!r},'w').write(json.dumps(child.pid));"
            "raise SystemExit(7)"
        )
        job = WindowsKillOnCloseJob()
        root = subprocess.Popen(
            [sys.executable, "-c", root_code],
            creationflags=CREATE_SUSPENDED,
        )
        job.assign(root)
        job.resume(root)
        assert root.wait(timeout=10) == 7
        child_pid = json.loads(child_pid_path.read_text(encoding="utf-8"))
        assert psutil.pid_exists(child_pid)

        job.close()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and psutil.pid_exists(child_pid):
            time.sleep(0.05)

        assert not psutil.pid_exists(child_pid)
