import json

from scripts.capture_environment_identity import _direct_install_identity


def test_direct_install_identity_keeps_commit_but_drops_local_url() -> None:
    identity = _direct_install_identity(
        json.dumps(
            {
                "url": "file:///D:/private/source",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "abc123",
                    "requested_revision": "main",
                },
                "dir_info": {"editable": False},
            }
        )
    )

    assert identity == {
        "vcs": "git",
        "commit_id": "abc123",
        "requested_revision": "main",
        "editable": False,
    }
    assert "private" not in json.dumps(identity)
