"""Verify the isolated native HunyuanOCR CPU environment."""

import sys
from importlib.metadata import version
from pathlib import Path


EXPECTED_PACKAGE_VERSIONS = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.28.0",
    "numpy": "2.2.6",
    "pillow": "12.3.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0+cpu",
    "torchvision": "0.26.0+cpu",
    "transformers": "5.13.0",
}


def main() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Native Hunyuan environment requires Python 3.12")
    actual_versions = {
        package: version(package) for package in EXPECTED_PACKAGE_VERSIONS
    }
    if actual_versions != EXPECTED_PACKAGE_VERSIONS:
        raise RuntimeError(
            f"Native Hunyuan environment version mismatch: {actual_versions}"
        )

    import torch
    import torchvision
    from PIL import Image
    from transformers import (
        AutoProcessor,
        GenerationConfig,
        HunYuanVLForConditionalGeneration,
    )

    if torch.cuda.is_available():
        raise RuntimeError("Native Hunyuan CPU environment unexpectedly exposes CUDA")
    if not torchvision.extension._has_ops():
        raise RuntimeError("Native Hunyuan torchvision binary operators are unavailable")
    model_root = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "models"
        / "hunyuanocr-1.5-449e7d47"
    )
    sys.path.insert(0, str(model_root.parents[2]))
    from workers.hunyuanocr_1_5_transformers_cpu_worker import (  # noqa: PLC0415
        _verify_model_bundle,
    )

    bundle_fingerprint = _verify_model_bundle(model_root)
    processor = AutoProcessor.from_pretrained(
        str(model_root),
        backend="pil",
        use_fast=False,
        local_files_only=True,
    )
    if type(processor).__name__ != "HunYuanVLProcessor":
        raise RuntimeError("Native Hunyuan processor class mismatch")
    image_processor = processor.image_processor
    if type(image_processor).__name__ != "HunYuanVLImageProcessorPil":
        raise RuntimeError("Native Hunyuan PIL image processor was not selected")
    if (
        int(image_processor.patch_size) != 16
        or int(image_processor.merge_size) != 2
    ):
        raise RuntimeError("Native Hunyuan image patch metadata mismatch")
    size = image_processor.size
    shortest_edge = (
        size.shortest_edge if hasattr(size, "shortest_edge") else size["shortest_edge"]
    )
    if int(shortest_edge) != 512 * 512:
        raise RuntimeError("Native Hunyuan minimum pixel metadata mismatch")
    generation_config = GenerationConfig.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    if (
        int(generation_config.bos_token_id) != 120000
        or tuple(generation_config.eos_token_id) != (120007, 120020)
        or int(generation_config.pad_token_id) != 120002
    ):
        raise RuntimeError("Native Hunyuan generation-token metadata mismatch")
    if not all((Image, AutoProcessor, HunYuanVLForConditionalGeneration)):
        raise RuntimeError("Native Hunyuan import surface is incomplete")
    print(
        {
            "versions": actual_versions,
            "device": "CPU",
            "bundle_fingerprint": bundle_fingerprint,
        }
    )


if __name__ == "__main__":
    main()
