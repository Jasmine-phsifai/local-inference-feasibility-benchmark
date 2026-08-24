# Local inference feasibility report

> Historical installation-stage snapshot. Its short synthetic throughput and
> pending-candidate statements are preserved as measured history, but they are
> superseded for current recommendations by
> [sustained-quality-and-compatibility.md](sustained-quality-and-compatibility.md).

Generated: 2026-08-22T20:40:28.709537+00:00

> This stage measures installation, execution, and speed feasibility. It does not compare final recognition quality.

## Host and test boundary

- CPU: Intel(R) Core(TM) Ultra 9 285K, 24 physical / 24 logical processors.
- RAM: 63.4 GiB usable; 39.9 GiB available at capture.
- Display adapters: GameViewer Virtual Display Adapter, Intel(R) Graphics. The Intel iGPU uses shared memory; the reported adapter aperture is not a hard allocation ceiling.
- NPU: no visible ComputeAccelerator/NPU device. NVIDIA/CUDA: no device or runtime present.
- The official SenseVoice Vulkan package enumerated Intel Graphics but then rejected Vulkan graph execution, so iGPU acceleration is detected but not runnable in that package on this host.
- Audio: public-domain JFK speech sample distributed by whisper.cpp (~11 s). Images: three synthetic 1280×720 bilingual lecture-like slides.
- Inputs are fallback samples, not representative lecture recordings/screenshots; projections are performance estimates only.

## Candidate scope

Candidates were selected from current official implementations and model repositories; the registry preserves exact roles, environment manifests, configurations, and source URLs.

- [rapidocr_cpu](https://github.com/RapidAI/RapidOCR): fast conventional Chinese and English OCR (`enabled`).
- [faster_whisper_cpu](https://github.com/SYSTRAN/faster-whisper): optimized Whisper implementation (`enabled`).
- [sensevoice_small_gguf_cpu](https://github.com/QwenAudio/SenseVoice): efficient Chinese and multilingual ASR (`enabled`).
- [qwen3_asr_0_6b_cpu](https://github.com/QwenLM/Qwen3-ASR): new compact generative multilingual ASR (`enabled`).
- [ppocrv6_tiny_cpu](https://github.com/PaddlePaddle/PaddleOCR): fast current conventional Chinese and English OCR (`enabled`).
- [ppocrv6_medium_cpu](https://github.com/PaddlePaddle/PaddleOCR): strong current conventional Chinese and English OCR (`enabled`).
- [paddleocr_vl_1_6_cpu](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6): compact OCR-oriented vision-language model (`enabled`).
- [hunyuanocr_1_5_gguf_cpu](https://github.com/Tencent-Hunyuan/HunyuanOCR): OCR-specialized VLM with official PC llama.cpp path (`experimental`).
- [qwen2_5_vl_3b_gguf_intel](https://github.com/ggml-org/llama.cpp): general VLM Intel CPU/iGPU comparator (`optional`).

## Installation footprint

Apparent file sizes; Conda hard links may reduce physical incremental disk use.

| Environment | Apparent size |
|---|---:|
| local-bench-control | 0.18 GiB |
| local-bench-faster-whisper | 0.41 GiB |
| local-bench-paddleocr | 1.16 GiB |
| local-bench-qwen3-asr | 1.57 GiB |
| OCRLLM | 0.90 GiB |

## Feasibility summary

| Candidate | Role | Classification | Best tested configuration | Warm throughput | Load | Peak host CPU | Peak process RAM | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| rapidocr_cpu | fast conventional Chinese and English OCR | feasible on CPU | `{'threads': 2, 'workers': 4}` | 30,488 images/h (0.118 s/image) | 0.67 s | 64.6% | 0.80 GiB | model files 30.3 MiB; valid nonempty output; runtime rapidocr 3.9.2; onnxruntime 1.23.2 |
| faster_whisper_cpu | optimized Whisper implementation | feasible on CPU | `{'compute_type': 'int8', 'threads': 16, 'workers': 1}` | 6.92 audio h/wall h (RTF 0.1445) | 2.12 s | 71.4% | 0.66 GiB | model files 463.7 MiB; valid nonempty output; runtime faster-whisper 1.2.1; CTranslate2 4.8.1 |
| sensevoice_small_gguf_cpu | efficient Chinese and multilingual ASR | feasible on CPU | `{'backend': 'cpu', 'threads': 24}` | 36.47 audio h/wall h (RTF 0.0274) | included in CLI timing | 20.2% | 0.27 GiB | model files 242.4 MiB; valid nonempty output; runtime FunASR llama.cpp runtime-llamacpp-v0.1.9; q8 |
| qwen3_asr_0_6b_cpu | new compact generative multilingual ASR | feasible on CPU | `{'threads': 24}` | 7.56 audio h/wall h (RTF 0.1322) | 1.02 s | 98.3% | 4.24 GiB | model files 1503.3 MiB; valid nonempty output; runtime transformers 5.13.0; torch 2.13.0+cpu; float32 |
| ppocrv6_tiny_cpu | fast current conventional Chinese and English OCR | feasible on CPU | `{'threads': 8}` | 36,406 images/h (0.099 s/image) | 0.73 s | 35.2% | 0.53 GiB | model files 6.3 MiB; valid nonempty output; runtime paddleocr 3.7.0; paddlepaddle 3.2.2 |
| ppocrv6_medium_cpu | strong current conventional Chinese and English OCR | feasible on CPU | `{'threads': 24}` | 3,206 images/h (1.123 s/image) | 1.06 s | 99.0% | 1.00 GiB | model files 132.7 MiB; valid nonempty output; runtime paddleocr 3.7.0; paddlepaddle 3.2.2 |
| paddleocr_vl_1_6_cpu | compact OCR-oriented vision-language model | runnable but below generation cutoff | `{'threads': 16}` | 35 images/h (101.899 s/image) | 32.24 s | 69.7% | 8.31 GiB | model files 1841.0 MiB; valid nonempty output; runtime paddleocr 3.7.0; paddlepaddle 3.2.2 native backend; 0.221 generated tok/s |
| hunyuanocr_1_5_gguf_cpu | OCR-specialized VLM with official PC llama.cpp path | experimental | — | — | — | — | — | setup/benchmark pending |
| qwen2_5_vl_3b_gguf_intel | general VLM Intel CPU/iGPU comparator | experimental | — | — | — | — | — | optional; not measured |

## Workload projections

### rapidocr_cpu

Best configuration: `{'threads': 2, 'workers': 4}`. Scales: lecture 50–80; course 750–1,200; ten courses 7,500–12,000; long-term 1.12–1.79M images.

| Lecture | Course | Ten courses | Long-term |
|---:|---:|---:|---:|
| 0.1 min–0.2 min | 1.5 min–2.4 min | 14.8 min–23.6 min | 36.74 h–58.71 h |

### faster_whisper_cpu

Best configuration: `{'compute_type': 'int8', 'threads': 16, 'workers': 1}`. Scales: lecture 2.5 h; course 37.5 h; ten courses 375 h; long-term 56,000 h.

| Lecture | Course | Ten courses | Long-term |
|---:|---:|---:|---:|
| 21.7 min | 5.42 h | 54.18 h | 8,091 h (337.1 days) |

### ppocrv6_tiny_cpu

Best configuration: `{'threads': 8}`. Scales: lecture 50–80; course 750–1,200; ten courses 7,500–12,000; long-term 1.12–1.79M images.

| Lecture | Course | Ten courses | Long-term |
|---:|---:|---:|---:|
| 0.1 min–0.1 min | 1.2 min–2.0 min | 12.4 min–19.8 min | 30.76 h–49.17 h |

### ppocrv6_medium_cpu

Best configuration: `{'threads': 24}`. Scales: lecture 50–80; course 750–1,200; ten courses 7,500–12,000; long-term 1.12–1.79M images.

| Lecture | Course | Ten courses | Long-term |
|---:|---:|---:|---:|
| 0.9 min–1.5 min | 14.0 min–22.5 min | 2.34 h–3.74 h | 349.32 h–558.28 h |

### sensevoice_small_gguf_cpu

Best configuration: `{'backend': 'cpu', 'threads': 24}`. Scales: lecture 2.5 h; course 37.5 h; ten courses 375 h; long-term 56,000 h.

| Lecture | Course | Ten courses | Long-term |
|---:|---:|---:|---:|
| 4.1 min | 1.03 h | 10.28 h | 1,536 h (64.0 days) |

### paddleocr_vl_1_6_cpu

Best configuration: `{'threads': 16}`. Scales: lecture 50–80; course 750–1,200; ten courses 7,500–12,000; long-term 1.12–1.79M images.

| Lecture | Course | Ten courses | Long-term |
|---:|---:|---:|---:|
| 1.42 h–2.26 h | 21.23 h–33.97 h | 212.29 h–339.66 h | 31,702 h (1320.9 days)–50,667 h (2111.1 days) |

### qwen3_asr_0_6b_cpu

Best configuration: `{'threads': 24}`. Scales: lecture 2.5 h; course 37.5 h; ten courses 375 h; long-term 56,000 h.

| Lecture | Course | Ten courses | Long-term |
|---:|---:|---:|---:|
| 19.8 min | 4.96 h | 49.58 h | 7,404 h (308.5 days) |

## Persisted failures and limitations

- `faster_whisper_cpu` install: PyPI connections reset or timed out while pip backtracked CTranslate2 4.0.0 through 4.8.1; resolver then reported no available NumPy distribution. Remediation: Pinned NumPy 2.3.5 and CTranslate2 4.8.1, increased timeout and retries, then retried in the existing isolated environment.
- `paddleocr_vl_1_6_cpu` model_acquisition: The official snapshot downloader fetched 19 of 20 files but made no byte progress on the 1.917 GB weight through the local proxy. Remediation: Stopped only the benchmark-owned process and used a pinned, resumable direct official revision URL with exact size verification.
- `sensevoice_small_gguf_cpu` accelerator_runtime: The official Windows Vulkan binary enumerated Intel Graphics and its capabilities, then reported that no Vulkan GPU graph backend was available in the build. Remediation: Preserved the stderr and retained the measured CPU backend; no driver or security setting was changed.
- `qwen3_asr_0_6b_cpu` environment_setup: The environment creator passed an empty optional package value to conda, which reported too few arguments before the pip phase continued. Remediation: Filtered null optional packages and added explicit native-command exit checks; the isolated environment then installed and imported successfully.
- `qwen3_asr_0_6b_cpu` model_acquisition_optimization: The official aria2 Windows client could not negotiate TLS through the local proxy, with and without an explicit proxy argument. Remediation: Returned to the slower proven curl transfer with byte-range resume; no proxy or security setting was changed.
- `qwen3_asr_0_6b_cpu` runtime_selection: The qwen-asr 0.0.6 package expects the legacy Qwen3-ASR checkpoint schema, but the downloaded -hf checkpoint declares native Transformers 5.13 support and failed during model construction under the legacy backend. Remediation: Matched the existing pinned -hf checkpoint to its documented native Transformers 5.13.0 API and reran both CPU configurations without modifying third-party model code.
- `faster_whisper_cpu`: 3 failed/interrupted benchmark attempt(s); bounded stderr tails are in `results/events.jsonl` and ignored artifacts.
- `sensevoice_small_gguf_cpu`: 1 failed/interrupted benchmark attempt(s); bounded stderr tails are in `results/events.jsonl` and ignored artifacts.
- `paddleocr_vl_1_6_cpu`: 2 failed/interrupted benchmark attempt(s); bounded stderr tails are in `results/events.jsonl` and ignored artifacts.
- `qwen3_asr_0_6b_cpu`: 2 failed/interrupted benchmark attempt(s); bounded stderr tails are in `results/events.jsonl` and ignored artifacts.
- Windows PowerShell policy blocked direct `.ps1` execution. Reproduction uses process-local `-ExecutionPolicy Bypass`; machine policy was not changed.
- Peak CPU comes from process-tree CPU-time deltas normalized against 24 logical processors. Intel iGPU memory/utilization counters were unavailable in the controller.
- Model download and load time are separate from warmed inference. One-time downloads are excluded from workload projections.
- The fallback samples prove execution and plausible output only; use representative inputs for the later quality/error-rate stage.
- PP-OCRv6 tiny was also tested with a three-image input batch; it did not beat the best sequential 8-thread run on this small sample.

## Historical recommendation (superseded)

Use PP-OCRv6 tiny or multi-worker RapidOCR for bulk slides and SenseVoice GGUF for bulk audio. Reserve larger OCR/VLM models for difficult pages only after representative quality testing. Maximum threads were often slower, and the packaged Intel Vulkan path is not currently usable.

No paid APIs, crawler, RAG, note generation, privacy filtering, fine-tuning, or production integration were used.
