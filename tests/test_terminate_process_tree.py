import subprocess
import sys
import time

import psutil

from local_inference_bench.terminate_process_tree import terminate_process_tree


def test_terminates_owned_root_and_child():
    child_code = "import time; time.sleep(60)"
    root_code = (
        "import subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        "time.sleep(60)"
    )
    root = subprocess.Popen([sys.executable, "-c", root_code])
    root_process = psutil.Process(root.pid)
    deadline = time.monotonic() + 10
    children = []
    while time.monotonic() < deadline and not children:
        children = root_process.children(recursive=True)
        time.sleep(0.05)
    assert children
    child_pid = children[0].pid

    result = terminate_process_tree(root.pid, grace_seconds=2)

    assert result["found"] >= 2
    assert not psutil.pid_exists(root.pid)
    assert not psutil.pid_exists(child_pid)
