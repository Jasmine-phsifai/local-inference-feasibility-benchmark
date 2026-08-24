import subprocess
import sys
import time
import importlib

import psutil

from local_inference_bench.terminate_process_tree import terminate_process_tree


terminate_process_tree_module = importlib.import_module(
    "local_inference_bench.terminate_process_tree"
)


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


def test_process_lookup_access_denied_is_reported_as_unverified(monkeypatch):
    def deny_process(_pid):
        raise psutil.AccessDenied(pid=1234)

    monkeypatch.setattr(terminate_process_tree_module.psutil, "Process", deny_process)

    result = terminate_process_tree(1234, grace_seconds=0)

    assert result["surviving"] == 1
    assert result["error_count"] == 1


def test_running_process_probe_access_denied_remains_a_survivor():
    class UnverifiableProcess:
        @staticmethod
        def is_running():
            raise psutil.AccessDenied(pid=1234)

    process = UnverifiableProcess()

    assert terminate_process_tree_module._running_processes([process]) == [process]
