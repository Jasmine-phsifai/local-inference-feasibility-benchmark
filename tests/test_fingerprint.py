from pathlib import Path

from local_inference_bench.fingerprint import fingerprint_files


def test_file_fingerprint_binds_relative_path_when_names_collide(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left" / "requirements.txt"
    right = tmp_path / "right" / "requirements.txt"
    left.parent.mkdir()
    right.parent.mkdir()
    left.write_text("first", encoding="utf-8")
    right.write_text("second", encoding="utf-8")
    initial = fingerprint_files([left, right])

    left.write_text("second", encoding="utf-8")
    right.write_text("first", encoding="utf-8")

    assert fingerprint_files([left, right]) != initial


def test_file_fingerprint_is_independent_of_input_order(tmp_path: Path) -> None:
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("left", encoding="utf-8")
    right.write_text("right", encoding="utf-8")

    assert fingerprint_files([left, right]) == fingerprint_files([right, left])


def test_file_fingerprint_frames_content_between_paths(tmp_path: Path) -> None:
    left = tmp_path / "a.py"
    right = tmp_path / "b.py"
    left.write_bytes(b"X")
    right.write_bytes(b"b.py\0Y")
    initial = fingerprint_files([left, right])

    # Without explicit entry framing, both states encode as:
    # a.py\0Xb.py\0b.py\0Y
    left.write_bytes(b"Xb.py\0")
    right.write_bytes(b"Y")

    assert fingerprint_files([left, right]) != initial


def test_file_fingerprint_rejects_empty_or_duplicate_inputs(tmp_path: Path) -> None:
    path = tmp_path / "only.txt"
    path.write_text("content", encoding="utf-8")

    for invalid in ([], [path, path]):
        try:
            fingerprint_files(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover - explicit assertion keeps the test dependency-free
            raise AssertionError("invalid fingerprint inputs were accepted")
