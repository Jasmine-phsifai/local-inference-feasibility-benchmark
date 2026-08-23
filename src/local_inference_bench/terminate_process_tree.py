"""Terminate only one benchmark-owned process tree."""

from __future__ import annotations

import psutil


def terminate_process_tree(pid: int, grace_seconds: float = 5.0) -> dict:
    """Terminate descendants and root, then kill only survivors."""

    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {"found": 0, "terminated": 0, "killed": 0}
    descendants = root.children(recursive=True)
    owned = [*reversed(descendants), root]
    for process in owned:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(owned, timeout=grace_seconds)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=grace_seconds)
    return {
        "found": len(owned),
        "terminated": len(owned) - len(alive),
        "killed": len(alive),
    }
