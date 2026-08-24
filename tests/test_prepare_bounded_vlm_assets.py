import hashlib
import io
import json
import stat
import urllib.error
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import scripts.prepare_bounded_vlm_b10598_assets as preparer
from local_inference_bench.acquire_bounded_vlm_assets import (
    _archive_member_relative,
    acquire_verified_file,
    extract_verified_zip_tree,
    hugging_face_resolve_url,
)
from local_inference_bench.bounded_vlm_assets import fingerprint_directory


class _Response(io.BytesIO):
    def __init__(
        self,
        content: bytes,
        *,
        status: int = 200,
        headers: dict | None = None,
        final_url: str = "https://downloads.example.test/asset",
    ):
        super().__init__(content)
        self.status = status
        self.headers = headers or {}
        self._final_url = final_url

    def getcode(self):
        return self.status

    def geturl(self):
        return self._final_url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _record(relative_path: str, content: bytes) -> dict:
    return {
        "path": relative_path,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_hugging_face_url_requires_and_preserves_exact_revision():
    revision = "8a22290a0c42dbcf84739d6d4c2763f877494ae0"
    identity = {
        "url": f"https://huggingface.co/Abiray/OvisOCR2-GGUF/tree/{revision}",
        "revision": revision,
    }

    url = hugging_face_resolve_url(identity, "folder/model Q8.gguf")

    assert url == (
        "https://huggingface.co/Abiray/OvisOCR2-GGUF/resolve/"
        f"{revision}/folder/model%20Q8.gguf"
    )
    identity["url"] = "https://huggingface.co/Abiray/OvisOCR2-GGUF/tree/main"
    with pytest.raises(ValueError, match="exact revision"):
        hugging_face_resolve_url(identity, "model.gguf")


def test_resumable_download_appends_only_matching_206(tmp_path):
    content = b"abcdef"
    record = _record("assets/model.gguf", content)
    partial = tmp_path / "assets" / "model.gguf.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")
    requests = []

    def open_url(request, *, timeout):
        requests.append((dict(request.header_items()), timeout))
        return _Response(
            b"def",
            status=206,
            headers={"Content-Range": "bytes 3-5/6"},
        )

    target = acquire_verified_file(
        project_root=tmp_path,
        url="https://example.test/model.gguf",
        record=record,
        open_url=open_url,
        sleep=lambda _seconds: None,
    )

    assert target.read_bytes() == content
    assert requests == [
        (
            {
                "User-agent": "local-inference-feasibility-benchmark/1",
                "Range": "bytes=3-",
            },
            120.0,
        )
    ]
    assert not partial.exists()


def test_ignored_range_restarts_partial_instead_of_appending(tmp_path):
    content = b"abcdef"
    record = _record("assets/model.gguf", content)
    partial = tmp_path / "assets" / "model.gguf.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")

    target = acquire_verified_file(
        project_root=tmp_path,
        url="https://example.test/model.gguf",
        record=record,
        open_url=lambda *_args, **_kwargs: _Response(content, status=200),
        sleep=lambda _seconds: None,
    )

    assert target.read_bytes() == content


def test_truncated_transfer_retries_from_preserved_partial(tmp_path):
    content = b"abcdef"
    record = _record("assets/model.gguf", content)
    calls = []

    def open_url(request, *, timeout):
        calls.append(dict(request.header_items()))
        if len(calls) == 1:
            return _Response(b"abc")
        return _Response(
            b"def",
            status=206,
            headers={"Content-Range": "bytes 3-5/6"},
        )

    target = acquire_verified_file(
        project_root=tmp_path,
        url="https://example.test/model.gguf",
        record=record,
        open_url=open_url,
        max_attempts=2,
        sleep=lambda _seconds: None,
    )

    assert target.read_bytes() == content
    assert "Range" not in calls[0]
    assert calls[1]["Range"] == "bytes=3-"


@pytest.mark.parametrize("status", (408, 429, 500, 503, 599))
def test_transient_http_status_recovers_with_bounded_retry(tmp_path, status):
    content = b"expected"
    record = _record("assets/model.gguf", content)
    calls = []
    sleeps = []

    def open_url(*_args, **_kwargs):
        calls.append(status)
        if len(calls) == 1:
            raise urllib.error.HTTPError(
                "https://example.test/model.gguf",
                status,
                "transient",
                hdrs=None,
                fp=None,
            )
        return _Response(content)

    target = acquire_verified_file(
        project_root=tmp_path,
        url="https://example.test/model.gguf",
        record=record,
        open_url=open_url,
        max_attempts=2,
        sleep=sleeps.append,
    )

    assert target.read_bytes() == content
    assert calls == [status, status]
    assert sleeps == [1.0]


def test_transient_http_exhaustion_does_not_promote(tmp_path):
    record = _record("assets/model.gguf", b"expected")
    calls = []
    sleeps = []

    def open_url(*_args, **_kwargs):
        calls.append(503)
        raise urllib.error.HTTPError(
            "https://example.test/model.gguf",
            503,
            "unavailable",
            hdrs=None,
            fp=None,
        )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        acquire_verified_file(
            project_root=tmp_path,
            url="https://example.test/model.gguf",
            record=record,
            open_url=open_url,
            max_attempts=3,
            sleep=sleeps.append,
        )

    assert calls == [503, 503, 503]
    assert sleeps == [1.0, 2.0]
    assert not (tmp_path / record["path"]).exists()


def test_permanent_http_4xx_fails_without_retry(tmp_path):
    record = _record("assets/model.gguf", b"expected")
    calls = []
    sleeps = []

    def open_url(*_args, **_kwargs):
        calls.append(404)
        raise urllib.error.HTTPError(
            "https://example.test/model.gguf",
            404,
            "not found",
            hdrs=None,
            fp=None,
        )

    with pytest.raises(urllib.error.HTTPError) as error:
        acquire_verified_file(
            project_root=tmp_path,
            url="https://example.test/model.gguf",
            record=record,
            open_url=open_url,
            max_attempts=64,
            sleep=sleeps.append,
        )

    assert error.value.code == 404
    assert calls == [404]
    assert sleeps == []


def test_truncated_transfer_exhaustion_keeps_partial_without_promotion(tmp_path):
    record = _record("assets/model.gguf", b"abcdef")

    with pytest.raises(RuntimeError, match="after 1 attempts"):
        acquire_verified_file(
            project_root=tmp_path,
            url="https://example.test/model.gguf",
            record=record,
            open_url=lambda *_args, **_kwargs: _Response(b"abc"),
            max_attempts=1,
            sleep=lambda _seconds: None,
        )

    assert not (tmp_path / record["path"]).exists()
    assert (tmp_path / "assets" / "model.gguf.part").read_bytes() == b"abc"


def test_resume_rejects_content_range_with_wrong_total(tmp_path):
    record = _record("assets/model.gguf", b"abcdef")
    partial = tmp_path / "assets" / "model.gguf.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"abc")

    with pytest.raises(RuntimeError, match="wrong byte range"):
        acquire_verified_file(
            project_root=tmp_path,
            url="https://example.test/model.gguf",
            record=record,
            open_url=lambda *_args, **_kwargs: _Response(
                b"def",
                status=206,
                headers={"Content-Range": "bytes 3-5/999"},
            ),
            max_attempts=1,
        )

    assert partial.read_bytes() == b"abc"


@pytest.mark.parametrize("existing_name", ("model.gguf", "model.gguf.part"))
def test_existing_wrong_sha_fails_without_overwrite(tmp_path, existing_name):
    record = _record("assets/model.gguf", b"expected")
    existing = tmp_path / "assets" / existing_name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"mismatch")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        acquire_verified_file(
            project_root=tmp_path,
            url="https://example.test/model.gguf",
            record=record,
            open_url=lambda *_args, **_kwargs: pytest.fail("network must not run"),
        )

    assert existing.read_bytes() == b"mismatch"


def test_download_rejects_https_redirect_downgrade(tmp_path):
    record = _record("assets/model.gguf", b"expected")

    with pytest.raises(RuntimeError, match="outside HTTPS"):
        acquire_verified_file(
            project_root=tmp_path,
            url="https://example.test/model.gguf",
            record=record,
            open_url=lambda *_args, **_kwargs: _Response(
                b"expected",
                final_url="http://downloads.example.test/model.gguf",
            ),
            max_attempts=1,
        )

    assert not (tmp_path / record["path"]).exists()


def test_exact_zip_tree_is_promoted_and_existing_mismatch_is_preserved(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "a.txt").write_bytes(b"a")
    (reference / "nested").mkdir()
    (reference / "nested" / "b.txt").write_bytes(b"b")
    tree_record = {
        "path": "runtime",
        **fingerprint_directory(reference),
    }
    archive = tmp_path / "runtime.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("a.txt", b"a")
        bundle.writestr("nested/b.txt", b"b")

    target = extract_verified_zip_tree(
        project_root=tmp_path,
        archive=archive,
        tree_record=tree_record,
    )
    assert fingerprint_directory(target) == fingerprint_directory(reference)

    (target / "a.txt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="tree identity mismatch"):
        extract_verified_zip_tree(
            project_root=tmp_path,
            archive=archive,
            tree_record=tree_record,
        )
    assert (target / "a.txt").read_bytes() == b"changed"


@pytest.mark.parametrize(
    "member",
    ("../escape.txt", "wrong-root/file.txt"),
)
def test_zip_extraction_rejects_untrusted_member_paths(tmp_path, member):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"payload")
    tree_record = {
        "path": "source",
        "file_count": 1,
        "total_bytes": 7,
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError):
        extract_verified_zip_tree(
            project_root=tmp_path,
            archive=archive,
            tree_record=tree_record,
            archive_root="declared",
        )
    assert not (tmp_path / "source").exists()


def test_zip_member_parser_rejects_backslash():
    member = zipfile.ZipInfo("safe.txt")
    member.filename = "declared\\backslash.txt"

    with pytest.raises(ValueError, match="path is invalid"):
        _archive_member_relative(member, "declared")


def test_zip_extraction_rejects_symlink(tmp_path):
    archive = tmp_path / "source.zip"
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "target")

    with pytest.raises(ValueError, match="symlinks"):
        extract_verified_zip_tree(
            project_root=tmp_path,
            archive=archive,
            tree_record={
                "path": "source",
                "file_count": 1,
                "total_bytes": 6,
                "sha256": "0" * 64,
            },
        )


def test_hunyuan_acquisition_includes_generation_config(monkeypatch):
    registry = json.loads(
        Path("registries/bounded_vlm_b10598_assets.json").read_text(encoding="utf-8")
    )
    candidate = registry["candidates"][preparer.HUNYUAN]
    filenames = []
    monkeypatch.setattr(
        preparer,
        "acquire_verified_file",
        lambda **kwargs: filenames.append(Path(kwargs["record"]["path"]).name),
    )

    preparer._acquire_hugging_face_records(
        identity=candidate["upstream"],
        records=candidate["lineage_files"],
    )

    assert "generation_config.json" in filenames
    assert set(filenames) == {
        Path(record["path"]).name for record in candidate["lineage_files"].values()
    }


@pytest.mark.parametrize(
    ("artifact_name", "arguments"),
    (
        ("model", ["--outtype", "f16"]),
        ("projector", ["--outtype", "f16", "--mmproj"]),
    ),
)
def test_conversion_is_offline_and_promotes_only_exact_output(
    tmp_path,
    monkeypatch,
    artifact_name,
    arguments,
):
    monkeypatch.setattr(preparer, "PROJECT_ROOT", tmp_path)
    converter = tmp_path / "source" / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.write_text("pass\n", encoding="utf-8")
    model_root = tmp_path / "models" / "source"
    model_root.mkdir(parents=True)
    expected = f"exact-{artifact_name}".encode()
    artifact_record = _record(f"models/output/{artifact_name}.gguf", expected)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        output = Path(command[command.index("--outfile") + 1])
        output.write_bytes(expected)

    monkeypatch.setattr(preparer.subprocess, "run", run)
    monkeypatch.setenv("PYTHONPATH", "untrusted")

    preparer._convert_hunyuan_artifact(
        conversion_python=Path("conversion-python.exe"),
        converter=converter,
        source_tree=converter.parent,
        model_root=model_root,
        artifact_name=artifact_name,
        artifact_record=artifact_record,
        arguments=arguments,
    )

    command, options = calls[0]
    assert command[:3] == [
        "conversion-python.exe",
        str(converter),
        str(model_root),
    ]
    assert command[3 : 3 + len(arguments)] == arguments
    assert options["timeout"] == 1800.0
    assert options["env"]["PYTHONNOUSERSITE"] == "1"
    assert options["env"]["HF_HUB_OFFLINE"] == "1"
    assert "PYTHONPATH" not in options["env"]
    assert (tmp_path / artifact_record["path"]).read_bytes() == expected


def test_conversion_hash_mismatch_is_not_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(preparer, "PROJECT_ROOT", tmp_path)
    converter = tmp_path / "source" / "convert_hf_to_gguf.py"
    converter.parent.mkdir(parents=True)
    converter.write_text("pass\n", encoding="utf-8")
    model_root = tmp_path / "models" / "source"
    model_root.mkdir(parents=True)
    artifact_record = _record("models/output/model.gguf", b"exact")

    def run(command, **_kwargs):
        Path(command[command.index("--outfile") + 1]).write_bytes(b"wrong")

    monkeypatch.setattr(preparer.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="(size|SHA-256) mismatch"):
        preparer._convert_hunyuan_artifact(
            conversion_python=Path("conversion-python.exe"),
            converter=converter,
            source_tree=converter.parent,
            model_root=model_root,
            artifact_name="model",
            artifact_record=artifact_record,
            arguments=["--outtype", "f16"],
        )
    assert not (tmp_path / artifact_record["path"]).exists()


def test_verify_only_main_does_not_enter_preparation(monkeypatch, capsys):
    monkeypatch.setattr(
        preparer,
        "parse_args",
        lambda: Namespace(
            candidate=preparer.OVIS,
            conversion_python=None,
            verify_only=True,
        ),
    )
    monkeypatch.setattr(preparer, "load_registry", lambda: {"unused": True})
    monkeypatch.setattr(
        preparer,
        "prepare_runtime",
        lambda _registry: pytest.fail("verify-only must not prepare or download"),
    )
    verified = []
    monkeypatch.setattr(preparer, "verify_candidate", verified.append)

    preparer.main()

    assert verified == [preparer.OVIS]
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
