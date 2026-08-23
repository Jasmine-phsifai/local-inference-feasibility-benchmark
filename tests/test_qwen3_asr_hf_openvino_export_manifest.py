import hashlib
import json

import pytest

from workers.qwen3_asr_hf_openvino_export_manifest import (
    build_ir_manifest,
    decoder_interface,
    verify_export,
    write_export_marker,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path):
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    weight = b"weight"
    (source / "model.safetensors").write_bytes(weight)
    (export / "openvino_encoder_model.xml").write_bytes(b"encoder-xml")
    (export / "openvino_encoder_model.bin").write_bytes(b"encoder-bin")
    (export / "openvino_decoder_model.xml").write_text(
        """<net><layers>
<layer id="0" type="Parameter"><output><port names="encoder_hidden_states" /></output></layer>
<layer id="1" type="Parameter"><output><port names="input_ids" /></output></layer>
<layer id="2" type="Parameter"><output><port names="beam_idx" /></output></layer>
<layer id="3" type="Result" output_names="logits" />
</layers></net>""",
        encoding="utf-8",
    )
    (export / "openvino_decoder_model.bin").write_bytes(b"decoder-bin")
    return source, export, _sha256(weight)


def test_export_marker_binds_every_ir_digest(tmp_path) -> None:
    source, export, expected_weight = _fixture(tmp_path)
    marker = write_export_marker(
        source,
        export,
        expected_weight_sha256=expected_weight,
    )

    assert marker["ir_manifest"] == build_ir_manifest(export)
    assert marker["decoder_interface"] == decoder_interface(export)
    assert verify_export(
        source,
        export,
        expected_weight_sha256=expected_weight,
    )["source_weight_sha256"] == expected_weight


def test_export_verification_rejects_mutated_ir(tmp_path) -> None:
    source, export, expected_weight = _fixture(tmp_path)
    write_export_marker(source, export, expected_weight_sha256=expected_weight)
    (export / "openvino_decoder_model.bin").write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="manifest mismatch"):
        verify_export(source, export, expected_weight_sha256=expected_weight)


def test_export_verification_rejects_self_attested_revision(tmp_path) -> None:
    source, export, expected_weight = _fixture(tmp_path)
    write_export_marker(source, export, expected_weight_sha256=expected_weight)
    marker_path = export / "export-complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["optimum_intel_revision"] = "wrong"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        verify_export(source, export, expected_weight_sha256=expected_weight)


def test_export_verification_rejects_non_stateful_decoder(tmp_path) -> None:
    source, export, expected_weight = _fixture(tmp_path)
    decoder_xml = export / "openvino_decoder_model.xml"
    decoder_xml.write_text(
        decoder_xml.read_text(encoding="utf-8").replace("beam_idx", "attention_mask"),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="cache interface mismatch"):
        write_export_marker(source, export, expected_weight_sha256=expected_weight)
