"""Terminate only one benchmark-owned process tree."""

from __future__ import annotations

import psutil


def terminate_process_tree(pid: int, grace_seconds: float = 5.0) -> dict:
    """Terminate descendants and root, then report any verified survivors."""

    error_count = 0
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {
            "found": 0,
            "terminated": 0,
            "killed": 0,
            "surviving": 0,
            "error_count": 0,
        }
    except (psutil.AccessDenied, OSError):
        return {
            "found": 0,
            "terminated": 0,
            "killed": 0,
            "surviving": 1,
            "error_count": 1,
        }
    try:
        descendants = root.children(recursive=True)
    except (psutil.AccessDenied, OSError):
        descendants = []
        error_count += 1
    owned = [*reversed(descendants), root]
    for process in owned:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError):
            error_count += 1
    try:
        _, alive = psutil.wait_procs(owned, timeout=grace_seconds)
    except (psutil.Error, OSError):
        alive = _running_processes(owned)
        error_count += 1
    graceful_count = len(owned) - len(alive)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
        except (psutil.AccessDenied, OSError):
            error_count += 1
    survivors = alive
    if survivors:
        try:
            _, survivors = psutil.wait_procs(survivors, timeout=grace_seconds)
        except (psutil.Error, OSError):
            survivors = _running_processes(survivors)
            error_count += 1
    return {
        "found": len(owned),
        "terminated": graceful_count,
        "killed": len(alive) - len(survivors),
        "surviving": len(survivors),
        "error_count": error_count,
    }


def _running_processes(processes: list[psutil.Process]) -> list[psutil.Process]:
    running = []
    for process in processes:
        try:
            if process.is_running():
                running.append(process)
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, OSError):
            running.append(process)
    return running
