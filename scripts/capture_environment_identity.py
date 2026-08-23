"""Print a path-free identity for the active Python environment."""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import distributions


def main() -> None:
    packages = []
    for installed in distributions():
        name = installed.metadata.get("Name")
        if not name:
            continue
        packages.append(
            {
                "name": name.casefold(),
                "version": installed.version,
                "direct_install": _direct_install_identity(
                    installed.read_text("direct_url.json")
                ),
            }
        )
    packages.sort(
        key=lambda package: (
            package["name"],
            package["version"],
            json.dumps(package["direct_install"], sort_keys=True),
        )
    )
    identity = {
        "python": list(sys.version_info[:3]),
        "implementation": platform.python_implementation(),
        "packages": packages,
    }
    print(json.dumps(identity, sort_keys=True, separators=(",", ":")))


def _direct_install_identity(direct_url_text: str | None) -> dict | None:
    if not direct_url_text:
        return None
    payload = json.loads(direct_url_text)
    identity = {}
    vcs_info = payload.get("vcs_info")
    if isinstance(vcs_info, dict):
        identity["vcs"] = vcs_info.get("vcs")
        identity["commit_id"] = vcs_info.get("commit_id")
        identity["requested_revision"] = vcs_info.get("requested_revision")
    dir_info = payload.get("dir_info")
    if isinstance(dir_info, dict):
        identity["editable"] = bool(dir_info.get("editable"))
    archive_info = payload.get("archive_info")
    if isinstance(archive_info, dict):
        identity["hash"] = archive_info.get("hash")
        identity["hashes"] = archive_info.get("hashes")
    return identity or None


if __name__ == "__main__":
    main()
