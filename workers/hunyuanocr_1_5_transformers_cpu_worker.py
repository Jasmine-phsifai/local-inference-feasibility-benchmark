"""Measure bounded native Transformers inference for HunyuanOCR 1.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from importlib.metadata import version
from pathlib import Path

try:
    from build_source_faithful_ocr_prompt import (
        SOURCE_FAITHFUL_PROMPT_VERSION,
        build_source_faithful_ocr_prompt,
    )
    from sustained_worker_metrics import build_public_summary, write_private_records
except ModuleNotFoundError:
    from workers.build_source_faithful_ocr_prompt import (
        SOURCE_FAITHFUL_PROMPT_VERSION,
        build_source_faithful_ocr_prompt,
    )
    from workers.sustained_worker_metrics import (
        build_public_summary,
        write_private_records,
    )


MODEL_REVISION = "449e7d471a8a1ef5bd5d652e4881183d7252cbc7"
EXPECTED_WEIGHT_SHA256 = (
    "632a1e082c4dd5a3284cf1ffcdba2fdaa06f435762c58c2f34aff0f3bd6c0249"
)
EXPECTED_AUXILIARY_SHA256 = {
    "chat_template.jinja": "be3371395b9e67a8f981d86543eb5a93d132a1dc3f54058a2d75b4ed1efc73fe",
    "config.json": "cc34ab90d0b873a1832c06e0f3fe127b47d7f390e8fda19445e4144068ed2af9",
    "generation_config.json": "e9f4d443b97de6cb40767d12b5fc045a5ca3fb6d2f911124fce307bfbe1ad585",
    "preprocessor_config.json": "e17baf5f25f542380a3a8231fefa08f359d86db1ac088feb12ecb8b06ddb01c3",
    "special_tokens_map.json": "71442c8c43669f4cedd669f1700f89a741773c49aef55783fbd533f72f050c92",
    "tokenizer.json": "3e2ab46bcc5ed8bce013b245c6daecf19fa1d2a18f48a9c88a1f571dcbf7dfd3",
    "tokenizer_config.json": "804e8a7fb5a129afb19f6ad88c51c5d3c1aa643b6abb9536d10bcdcf633b4d74",
}
EXPECTED_BOS_TOKEN_ID = 120000
EXPECTED_EOS_TOKEN_IDS = (120007, 120020)
EXPECTED_PAD_TOKEN_ID = 120002
EXPECTED_IMAGE_PATCH_SIZE = 16
EXPECTED_MIN_PIXELS = 512 * 512
EXPECTED_RESIZE_FACTOR = 32
MAX_CPU_THREADS = 24
MAX_NEW_TOKENS = 512
MAX_SIDE_LEN = 1920
MAX_PIXELS = 1920 * 1088
MAX_VISION_PATCHES = 8160
MAX_ITEMS = 3
MAX_SOURCE_PIXELS = 4096 * 4096
PROMPTS = {
    "doc_parse": (
        "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
        "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
    ),
    "structured_parse": "提取图中的文字。",
}
_LATEX_MARKER = re.compile(
    r"(?:\\\(|\\\[|\$\$|\\begin\{|\\frac|\\sum|\\int)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    config = request["config"]
    if int(config["processes"]) != 1:
        raise ValueError("native HunyuanOCR uses one resident CPU process")
    if request["phase"] not in {"screen", "quality", "compatibility"}:
        raise ValueError("native HunyuanOCR is bounded to declared probe phases")
    if request["phase"] not in config.get("phases", []):
        raise ValueError("native HunyuanOCR config does not support this phase")
    threads = int(config["threads_per_process"])
    max_new_tokens = int(config["max_new_tokens"])
    max_side_len = int(config["max_side_len"])
    max_pixels = int(config["max_pixels"])
    max_vision_patches = int(config["max_vision_patches"])
    max_items = int(config.get("max_items", len(request["workload"]["items"])))
    mode = str(config["mode"])
    if mode not in {*PROMPTS, "source_faithful"}:
        raise ValueError("unknown native HunyuanOCR task mode")
    _validate_native_cpu_bounds(
        threads=threads,
        max_new_tokens=max_new_tokens,
        max_side_len=max_side_len,
        max_pixels=max_pixels,
        max_vision_patches=max_vision_patches,
        max_items=max_items,
    )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    import torch
    from PIL import Image
    from transformers import (
        AutoProcessor,
        GenerationConfig,
        HunYuanVLForConditionalGeneration,
    )

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    project_root = Path(__file__).resolve().parents[1]
    model_root = project_root / "data" / "models" / "hunyuanocr-1.5-449e7d47"
    weight_path = model_root / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError("pinned native HunyuanOCR checkpoint is missing")
    artifact_verification_started = time.perf_counter()
    bundle_fingerprint = _verify_model_bundle(model_root)
    artifact_verification_seconds = (
        time.perf_counter() - artifact_verification_started
    )

    loaded_at = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        str(model_root),
        backend="pil",
        use_fast=False,
        local_files_only=True,
    )
    generation_config = GenerationConfig.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    eos_token_ids, pad_token_id = _validated_generation_tokens(generation_config)
    model = HunYuanVLForConditionalGeneration.from_pretrained(
        str(model_root),
        dtype=torch.float32,
        attn_implementation="eager",
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    if getattr(model.config, "architectures", None) != [
        "HunYuanVLForConditionalGeneration"
    ]:
        raise RuntimeError("native HunyuanOCR architecture metadata mismatch")
    model.generation_config = generation_config
    load_seconds = time.perf_counter() - loaded_at
    warmup = _recognize(
        item=request["workload"]["warmup_item"],
        prompt=_prompt_for_item(mode, request["workload"]["warmup_item"]),
        processor=processor,
        model=model,
        torch=torch,
        Image=Image,
        max_new_tokens=min(64, max_new_tokens),
        max_side_len=max_side_len,
        max_pixels=max_pixels,
        max_vision_patches=max_vision_patches,
        eos_token_ids=eos_token_ids,
        pad_token_id=pad_token_id,
        capture_prediction=False,
        preserve_raw_prediction=mode == "source_faithful",
        raise_on_error=True,
    )
    if not warmup["success"]:
        raise RuntimeError("native HunyuanOCR warmup produced empty output")
    started = time.perf_counter()
    records = []
    for item in request["workload"]["items"][:max_items]:
        record = _recognize(
            item=item,
            prompt=_prompt_for_item(mode, item),
            processor=processor,
            model=model,
            torch=torch,
            Image=Image,
            max_new_tokens=max_new_tokens,
            max_side_len=max_side_len,
            max_pixels=max_pixels,
            max_vision_patches=max_vision_patches,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
            capture_prediction=bool(request["capture_predictions"]),
            preserve_raw_prediction=mode == "source_faithful",
        )
        record["completed_offset_seconds"] = time.perf_counter() - started
        records.append(record)
    steady_wall_seconds = time.perf_counter() - started

    write_private_records(Path(request["private_records_path"]), records)
    if config.get("require_all_success") and not _all_records_pass_safety(records):
        raise RuntimeError("native HunyuanOCR safety probe failed")
    public_summary = build_public_summary(
        candidate_id=request["candidate_id"],
        task="ocr",
        runtime_name="transformers_hunyuanocr_1_5",
        runtime_version=(
            f"transformers-{version('transformers')}+torch-{version('torch')}"
        ),
        workload_class=request["workload"]["workload_class"],
        records=records,
        load_seconds=[load_seconds],
        warmup_seconds=[warmup["latency_seconds"]],
        steady_wall_seconds=steady_wall_seconds,
        target_wall_seconds=float(request["target_wall_seconds"]),
        load_semantics="resident_model_cpu",
    )
    public_summary["generation"] = _generation_summary(records, max_new_tokens)
    public_summary["generation"].update(
        {
            "bos_token_id": EXPECTED_BOS_TOKEN_ID,
            "eos_token_ids": eos_token_ids,
            "pad_token_id": pad_token_id,
            **_decode_semantics(mode),
            "semantic_decode_cleanup_applied": False,
            "tail_repetition_cleanup_applied": False,
            "doc_parse_normalization_applied": False,
        }
    )
    public_summary["model"] = {
        "model_revision": f"{MODEL_REVISION}:bundle:{bundle_fingerprint}",
        "model_size_bytes": weight_path.stat().st_size,
        "artifact_verification_seconds": artifact_verification_seconds,
        "compute_type": "float32",
        "backend": "transformers",
        "device": "CPU",
        "mode": mode,
        "prompt_version": (
            SOURCE_FAITHFUL_PROMPT_VERSION
            if mode == "source_faithful"
            else f"vendor-{mode}"
        ),
        "threads": threads,
        "interop_threads": 1,
        "max_side_len": max_side_len,
        "max_pixels": max_pixels,
        "min_pixels": EXPECTED_MIN_PIXELS,
        "max_vision_patches": max_vision_patches,
        "max_items": max_items,
        "max_source_pixels": MAX_SOURCE_PIXELS,
        "native_pil_processor": True,
        "eager_attention": True,
        "low_cpu_mem_usage": True,
    }
    Path(request["response_path"]).write_text(
        json.dumps({"public_summary": public_summary}, indent=2),
        encoding="utf-8",
    )


def _recognize(
    *,
    item: dict,
    prompt: str,
    processor,
    model,
    torch,
    Image,
    max_new_tokens: int,
    max_side_len: int,
    max_pixels: int,
    max_vision_patches: int,
    eos_token_ids: list[int],
    pad_token_id: int,
    capture_prediction: bool,
    preserve_raw_prediction: bool,
    raise_on_error: bool = False,
) -> dict:
    started = time.perf_counter()
    try:
        with Image.open(item["path"]) as raw:
            if raw.width * raw.height > MAX_SOURCE_PIXELS:
                raise MemoryError("native HunyuanOCR source raster bound exceeded")
            image = raw.convert("RGB")
        original_width, original_height = image.size
        output_width, output_height = _bounded_aligned_dimensions(
            original_width,
            original_height,
            max_side_len,
            max_pixels,
        )
        if (output_width, output_height) != image.size:
            image = image.resize(
                (output_width, output_height),
                resample=Image.Resampling.BICUBIC,
            )
        messages = _chat_messages(str(item["path"]), prompt)
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(
            text=[text_prompt],
            images=image,
            images_kwargs={"do_resize": False},
            padding=True,
            return_tensors="pt",
        ).to("cpu")
        image_grid = inputs.get("image_grid_thw")
        if image_grid is None or len(image_grid) != 1:
            raise RuntimeError("native HunyuanOCR image grid metadata is missing")
        vision_patch_count = int(image_grid[0].prod().item())
        processed_height = int(image_grid[0, 1].item()) * EXPECTED_IMAGE_PATCH_SIZE
        processed_width = int(image_grid[0, 2].item()) * EXPECTED_IMAGE_PATCH_SIZE
        if (processed_width, processed_height) != (output_width, output_height):
            raise RuntimeError("native HunyuanOCR processor changed bounded dimensions")
        if max(processed_width, processed_height) > max_side_len:
            raise MemoryError("native HunyuanOCR processed side bound exceeded")
        if processed_width * processed_height > max_pixels:
            raise MemoryError("native HunyuanOCR processed pixel bound exceeded")
        if vision_patch_count > max_vision_patches:
            raise MemoryError("native HunyuanOCR vision patch safety bound exceeded")
        input_ids = inputs["input_ids"]
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.08,
            "use_cache": True,
        }
        generation_kwargs["eos_token_id"] = eos_token_ids
        generation_kwargs["pad_token_id"] = pad_token_id
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generation_kwargs)
        trimmed_ids = generated_ids[:, input_ids.shape[1] :]
        decoded = processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = _decoded_output(decoded, preserve_raw=preserve_raw_prediction)
        completion_tokens = int(trimmed_ids.shape[-1])
        eos_ids = set(eos_token_ids)
        last_token = (
            int(trimmed_ids[0, -1].item()) if completion_tokens else None
        )
        eos_finish = last_token in eos_ids if last_token is not None else False
    except Exception as error:
        if raise_on_error:
            raise
        failure_record = {
            "sample_id": item["id"],
            "success": False,
            "failure_kind": type(error).__name__,
            "latency_seconds": time.perf_counter() - started,
            "units": 0.0,
        }
        if capture_prediction:
            failure_record["prediction"] = ""
        return failure_record
    latency_seconds = time.perf_counter() - started
    output_is_valid = bool(text.strip()) or not item.get("expected_text", True)
    record = {
        "sample_id": item["id"],
        "success": output_is_valid,
        "failure_kind": None if output_is_valid else "empty_output",
        "latency_seconds": latency_seconds,
        "units": 1.0 if output_is_valid else 0.0,
        "output_character_count": len(text),
        "completion_tokens": completion_tokens,
        "token_cap_hit": completion_tokens >= max_new_tokens and not eos_finish,
        "eos_finish": eos_finish,
        "end_to_end_completion_tokens_per_second": (
            completion_tokens / latency_seconds if latency_seconds else 0.0
        ),
        "input_width": original_width,
        "input_height": original_height,
        "preprocessor_input_width": output_width,
        "preprocessor_input_height": output_height,
        "processed_width": processed_width,
        "processed_height": processed_height,
        "vision_patch_count": vision_patch_count,
        **_format_metrics(text),
    }
    if capture_prediction:
        record["prediction"] = text
        record["lines"] = [
            {"text": line} for line in text.splitlines() if line.strip()
        ]
    return record


def _decoded_output(decoded: list[str], *, preserve_raw: bool) -> str:
    raw = decoded[0] if decoded else ""
    return raw if preserve_raw else raw.strip()


def _decode_semantics(mode: str) -> dict[str, bool]:
    raw = mode == "source_faithful"
    return {
        "raw_decode_scored": raw,
        "outer_whitespace_trimmed": not raw,
    }


def _all_records_pass_safety(records: list[dict]) -> bool:
    return bool(records) and all(
        record.get("success") is True and record.get("token_cap_hit") is not True
        for record in records
    )


def _chat_messages(image_reference: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_reference},
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _prompt_for_item(mode: str, item: dict) -> str:
    if mode == "source_faithful":
        marker = item.get("output_marker")
        if not isinstance(marker, str):
            raise ValueError("source-faithful HunyuanOCR item requires output_marker")
        return build_source_faithful_ocr_prompt(marker)
    return PROMPTS[mode]


def _validate_native_cpu_bounds(
    *,
    threads: int,
    max_new_tokens: int,
    max_side_len: int,
    max_pixels: int,
    max_vision_patches: int,
    max_items: int,
) -> None:
    values = {
        "threads": (threads, MAX_CPU_THREADS),
        "max_new_tokens": (max_new_tokens, MAX_NEW_TOKENS),
        "max_side_len": (max_side_len, MAX_SIDE_LEN),
        "max_pixels": (max_pixels, MAX_PIXELS),
        "max_vision_patches": (max_vision_patches, MAX_VISION_PATCHES),
        "max_items": (max_items, MAX_ITEMS),
    }
    for name, (value, ceiling) in values.items():
        if value < 1 or value > ceiling:
            raise ValueError(f"native HunyuanOCR bound is invalid: {name}")
    minimum_side = math.isqrt(EXPECTED_MIN_PIXELS)
    minimum_patches = EXPECTED_MIN_PIXELS // (EXPECTED_IMAGE_PATCH_SIZE**2)
    if max_side_len < minimum_side:
        raise ValueError("native HunyuanOCR max_side_len is below the processor minimum")
    if max_pixels < EXPECTED_MIN_PIXELS:
        raise ValueError("native HunyuanOCR max_pixels is below the processor minimum")
    if max_vision_patches < minimum_patches:
        raise ValueError(
            "native HunyuanOCR max_vision_patches is below the processor minimum"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_model_bundle(
    model_root: Path,
    *,
    expected_auxiliary_sha256: dict[str, str] = EXPECTED_AUXILIARY_SHA256,
    expected_weight_sha256: str = EXPECTED_WEIGHT_SHA256,
) -> str:
    weight_path = model_root / "model.safetensors"
    if not weight_path.is_file() or _sha256(weight_path) != expected_weight_sha256:
        raise RuntimeError("native HunyuanOCR model bundle mismatch: model.safetensors")
    identities = {"model.safetensors": expected_weight_sha256}
    for name, expected_hash in expected_auxiliary_sha256.items():
        path = model_root / name
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"native HunyuanOCR model bundle mismatch: {name}")
        identities[name] = expected_hash
    canonical = json.dumps(identities, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _bounded_aligned_dimensions(
    width: int,
    height: int,
    max_side_len: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Apply pinned smart-resize semantics plus the native CPU side ceiling."""

    if width < 1 or height < 1:
        raise ValueError("native HunyuanOCR image dimensions must be positive")
    aligned_side = max_side_len // EXPECTED_RESIZE_FACTOR * EXPECTED_RESIZE_FACTOR
    if aligned_side < EXPECTED_RESIZE_FACTOR or max_pixels < EXPECTED_MIN_PIXELS:
        raise ValueError("native HunyuanOCR image bounds are internally inconsistent")
    output_height = round(height / EXPECTED_RESIZE_FACTOR) * EXPECTED_RESIZE_FACTOR
    output_width = round(width / EXPECTED_RESIZE_FACTOR) * EXPECTED_RESIZE_FACTOR
    output_height = max(EXPECTED_RESIZE_FACTOR, output_height)
    output_width = max(EXPECTED_RESIZE_FACTOR, output_width)
    output_area = output_width * output_height
    source_area = width * height
    if output_area > max_pixels:
        scale = math.sqrt(source_area / max_pixels)
        output_height = max(
            EXPECTED_RESIZE_FACTOR,
            math.floor(height / scale / EXPECTED_RESIZE_FACTOR)
            * EXPECTED_RESIZE_FACTOR,
        )
        output_width = max(
            EXPECTED_RESIZE_FACTOR,
            math.floor(width / scale / EXPECTED_RESIZE_FACTOR)
            * EXPECTED_RESIZE_FACTOR,
        )
    elif output_area < EXPECTED_MIN_PIXELS:
        scale = math.sqrt(EXPECTED_MIN_PIXELS / source_area)
        output_height = (
            math.ceil(height * scale / EXPECTED_RESIZE_FACTOR)
            * EXPECTED_RESIZE_FACTOR
        )
        output_width = (
            math.ceil(width * scale / EXPECTED_RESIZE_FACTOR)
            * EXPECTED_RESIZE_FACTOR
        )

    if max(output_width, output_height) > aligned_side:
        scale = max(output_width, output_height) / aligned_side
        output_height = max(
            EXPECTED_RESIZE_FACTOR,
            math.floor(output_height / scale / EXPECTED_RESIZE_FACTOR)
            * EXPECTED_RESIZE_FACTOR,
        )
        output_width = max(
            EXPECTED_RESIZE_FACTOR,
            math.floor(output_width / scale / EXPECTED_RESIZE_FACTOR)
            * EXPECTED_RESIZE_FACTOR,
        )
    output_area = output_width * output_height
    if not EXPECTED_MIN_PIXELS <= output_area <= max_pixels:
        raise ValueError("native HunyuanOCR aspect ratio conflicts with image bounds")
    return output_width, output_height


def _token_id_set(token_ids: object) -> set[int]:
    if token_ids is None:
        return set()
    if isinstance(token_ids, (list, tuple, set)):
        return {int(token_id) for token_id in token_ids}
    return {int(token_ids)}


def _validated_generation_tokens(generation_config) -> tuple[list[int], int]:
    eos_token_ids = tuple(
        int(token_id)
        for token_id in (
            generation_config.eos_token_id
            if isinstance(generation_config.eos_token_id, (list, tuple))
            else [generation_config.eos_token_id]
        )
    )
    if (
        int(generation_config.bos_token_id) != EXPECTED_BOS_TOKEN_ID
        or eos_token_ids != EXPECTED_EOS_TOKEN_IDS
        or int(generation_config.pad_token_id) != EXPECTED_PAD_TOKEN_ID
    ):
        raise RuntimeError("native HunyuanOCR generation-token metadata mismatch")
    return list(eos_token_ids), EXPECTED_PAD_TOKEN_ID


def _format_metrics(text: str) -> dict:
    folded = text.casefold()
    return {
        "latex_marker": bool(_LATEX_MARKER.search(text)),
        "complete_html_table": "<table" in folded and "</table>" in folded,
    }


def _generation_summary(records: list[dict], max_new_tokens: int) -> dict:
    rates = [
        float(record["end_to_end_completion_tokens_per_second"])
        for record in records
        if isinstance(
            record.get("end_to_end_completion_tokens_per_second"),
            (int, float),
        )
    ]
    completion_tokens = [
        int(record.get("completion_tokens", 0)) for record in records
    ]
    return {
        "max_new_tokens": max_new_tokens,
        "completion_tokens_total": sum(completion_tokens),
        "completion_tokens_max": max(completion_tokens, default=0),
        "token_cap_hit_count": sum(
            bool(record.get("token_cap_hit")) for record in records
        ),
        "eos_finish_count": sum(bool(record.get("eos_finish")) for record in records),
        "latex_marker_count": sum(bool(record.get("latex_marker")) for record in records),
        "complete_html_table_count": sum(
            bool(record.get("complete_html_table")) for record in records
        ),
        "mean_end_to_end_completion_tokens_per_second": (
            sum(rates) / len(rates) if rates else 0.0
        ),
    }


if __name__ == "__main__":
    main()
