# Sustained OCR, source-faithful document OCR, and ASR on Core Ultra 9 285K

Status: final measured recommendation

Date: 2026-08-23

## Decision summary

| Decision | Recommendation | Measured basis |
|---|---|---|
| Bulk default OCR | RapidOCR 3.9.2 / PP-OCRv6 Small, ONNX Runtime, full 2000-pixel mode, classifier enabled, 8 processes x 2 threads | 14,646.44 images/hour over 10 minutes; generated-control NCER 0.001757; full mode won both separate private-course blind comparisons 9-to-2 |
| Throughput-only OCR option | RapidOCR OpenVINO, same model/settings, 8 x 2 | Matched ABBA mean 16,451.33 images/hour, 6.62% faster than ORT, with identical generated quality; mean peak RSS was 9.00 GiB versus 2.98 GiB, so it is not the practical default |
| Maximum-throughput OCR | PP-OCRv6 Tiny, 6 x 4 | 29,591.46 images/hour over 10 minutes, CV 0.0051; lower generated recall and a separate 9-to-2 blind loss to full RapidOCR |
| Difficult-image escalation | PaddleOCR-VL 1.6 at a 512-token cap, only as a manual-review lane | It emitted structured layout/formula/table blocks and no negative-control text, but only 46.67 images/hour and poor control quality. No tested CPU VLM passed the unattended source-faithful gate |
| Bulk default ASR | SenseVoice Small GGUF Q8, 3 processes x 8 threads | 116.86394 audio-hours/hour over 30 minutes, 195/195 successes, CV 0.0364, peak RSS 2.35 GiB, and best required-term recall 0.8718 |
| Timestamped ASR | faster-whisper Small int8, 6 model workers x 4 threads | 33.79100 audio-hours/hour over 10 minutes; timestamp precision/recall 0.9329/0.7342. Transcript error and silence hallucination were worse than SenseVoice |
| Higher-quality ASR escalation | Qwen3-ASR 0.6B HF-native OpenVINO CPU as a transcript-only second opinion, not an automatic replacement | NCER was slightly lower than SenseVoice, 0.1748 versus 0.1766, but term recall was lower, 0.7308 versus 0.8718, with no timestamps and about eight times lower throughput |
| Practical CPU concurrency | Use runtime-specific modest workers, not one nominal 24-thread configuration | SenseVoice peaked at 3 x 8; RapidOCR at 8 x 2; PP Tiny selected 6 x 4; faster-whisper 24 x 1 was slower, less stable, and more than twice the memory of 6 x 4 |
| Likely GPU crossover | None on this Intel iGPU; expect a discrete-GPU crossover first for autoregressive document VLMs, but it is unmeasured | Qwen CPU was 13.54% faster than `GPU.0` on identical outputs. No discrete GPU or CUDA device is present, so no numeric discrete-GPU crossover is claimed |
| Future OCRLLM compatibility | Reuse the active image facade only for text-line compatibility; preserve structured/raw OCR outside it and keep ASR behind benchmark-owned adapters | The independently installed package retained 5/5 lines and 4/4 formula-like lines, but its public facade exposed neither geometry/confidence nor ASR |

The difficult-image recommendation is deliberately a manual-review fallback,
not a claim that PaddleOCR-VL is the most accurate OCR model. Conventional OCR
remains the only tested unattended bulk path. Source-faithful VLM output must be
validated before it can replace it.

## Host, safety, cost, and evidence boundary

- Intel Core Ultra 9 285K; 24 physical and 24 logical processors visible.
- 63.4 GiB usable RAM.
- Intel integrated graphics using shared memory; no NVIDIA/CUDA device and no
  visible NPU.
- No firmware, BIOS, affinity, cooling, power-limit, security, or temperature
  control was changed. Heavy inference jobs were serialized.
- Accepted monitored runs recorded no performance-limit or thermal-throttle
  flags. The accepted faster-whisper sustained event did not have host
  thermal/power telemetry, so this statement is not generalized to it.
- CPU package temperature is unavailable. The ACPI-zone reading is not treated
  as package temperature.
- No paid API was used. Electricity, network transfer, and unknown free-service
  quota costs were not priced; this report does not claim zero cost.
- Raw course video, frames, audio, references, predictions, and private paths
  remain ignored. Tracked journals contain bounded aggregate metadata only.
- The crawler was not used and was never on the critical path.

Read-only inspection did not recover the specifically requested lecture files
at their recorded locations. They are not claimed as benchmark inputs. An
unrelated validated complete lecture, other existing private course samples,
public audio, and deterministic generated controls supplied the representative
workloads. No course identity, title, teacher/student name, transcript, frame,
audio, credential, private path, or private hash is published.

## What this stage establishes

The original approximately 11-second public audio sample and three generated
images established installation only. This stage adds:

- 10- and 30-minute sustained runs with process/worker sweeps across all 24
  visible processors;
- 10-minute mixed-language and silence-heavy ASR controls;
- realistic 1080p OCR controls for projection, handwriting, formulas, code,
  tables, blur, occlusion, and negative images;
- two randomized blind comparisons on ignored course frames;
- matched RapidOCR ORT/OpenVINO trials;
- Qwen HF-native OpenVINO CPU/iGPU quality and throughput;
- native Hunyuan FP32 compatibility plus two exact-repeat source-fidelity trials;
- bounded community/runtime falsification gates;
- active-library OCRLLM image-facade compatibility.

Model download, environment creation, bundle verification, cold load/compile,
warm-up, steady inference, and quality scoring are recorded separately.

## Sustained throughput and concurrency

| Candidate | Exact configuration | Duration | Completed | Throughput | CV | Peak RSS |
|---|---|---:|---:|---:|---:|---:|
| SenseVoice Small Q8 | 3 processes x 8 threads | 30 min | 195/195 | 116.863942 audio h/h | 0.036375 | 2.3547 GiB |
| faster-whisper Small int8 | 1 process, 6 model workers x 4 threads | 10 min | 36/36 | 33.791004 audio h/h | 0.082769 | 2.4633 GiB |
| RapidOCR full ORT | 8 x 2 | 10 min | 2,447/2,447 | 14,646.443 images/h | 0.010234 | 3.6669 GiB |
| RapidOCR 1280/classifier off | 8 x 2 | 10 min | 4,155/4,155 | 24,882.066 images/h | 0.012604 | 2.2213 GiB |
| PP-OCRv6 Tiny | 6 x 4 | 10 min | 4,939/4,939 | 29,591.461 images/h | 0.005120 | 4.9006 GiB |

The 30-minute SenseVoice attempt
`f5baeb93-929f-4ee7-9c5e-f53eb7af3be2` ran 1,803.487 seconds:

- steady state 1,802.096 seconds;
- p50/p95/max item latency 27.632/30.668/33.109 seconds;
- mean process-tree CPU 89.42% of host and mean host CPU 96.25%;
- mid-run median RSS ratio 1.000123 across 1,016 samples;
- mean/p95 RAPL package power 231.52/252.42 W;
- zero performance-limit and throttle flags.

Its separate 10-minute validation reached 103.953679 audio h/h. The two runs
have different code/workload fingerprints and are independent validations, not
a matched duration-only A/B, so they are not averaged.

### Concurrency screens

| Candidate | Selected | Challenger | Outcome |
|---|---:|---:|---|
| SenseVoice | 3 x 8: 104.318 audio h/h screen | 4 x 8: 92.177; 8 x 8: 73.511 | Additional processes lost to contention |
| faster-whisper | 6 x 4: 33.791 sustained, 2.463 GiB | 24 x 1: 32.273 screen, CV 0.2421, 5.448 GiB | 24 x 1 failed the promotion floor; no sustained follow-up |
| RapidOCR full | 8 x 2: 13,951.733 screen | same | Selected |
| PP-OCRv6 Tiny | 6 x 4: 28,079.423 screen | 8 x 3: 28,743.445 | 6 x 4 was only 2.37% slower and about 2.02 GiB smaller |

The newest `transcribe.cpp` v0.2.1 true-batch path was also measured
against SenseVoice. At 24 threads, batch 8 reached 51.9199 audio h/h, 2.72 times
its own batch-1 rate, with zero native failures and 1.157 GiB steady peak RSS.
It achieved only 44.4% of the established 116.864 audio h/h worker-pool result,
so no sustained promotion was justified.

### RapidOCR ORT versus OpenVINO

Both backends used RapidOCR 3.9.2, the same verified PP-OCRv6 Small ONNX files,
classifier on, full 2000-pixel mode, and 8 x 2. Two 60-second trials used
alternating A/B order.

| Backend | Trial rates | Mean | Mean peak RSS | Generated quality |
|---|---:|---:|---:|---|
| ONNX Runtime 1.29 | 15,425.902; 15,433.733 images/h | 15,429.818 | 2.977 GiB | NCER 0.001757469 |
| OpenVINO 2026.3 | 16,093.946; 16,808.706 images/h | 16,451.326 | 9.002 GiB | NCER 0.001757469 |

OpenVINO gained 6.62%, below the 10% promotion threshold, while using about
three times the process-tree memory. It remains a throughput-only opt-in.
RapidOCR creates one synchronous request per stage; sharing one engine across
threads is not treated as safe concurrency. One engine per process is retained.

## Recognition quality

### ASR generated controls

All rows share the 10-sample `asr-quality-v3` set with public or
independently generated references.

| Candidate | NCER | Mixed-token error | Term recall | Silence FP chars/min | Timestamp precision/recall |
|---|---:|---:|---:|---:|---:|
| SenseVoice Q8 | 0.176576 | 0.237113 | 0.871795 | 0 | unavailable |
| Qwen3-ASR HF OpenVINO CPU | 0.174783 | 0.246392 | 0.730769 | 0 | unavailable |
| Qwen3-ASR HF OpenVINO iGPU | 0.174783 | 0.246392 | 0.730769 | 0 | unavailable |
| faster-whisper Small int8 | 0.351957 | 0.605155 | 0.653846 | 6.0 | 0.932902 / 0.734159 |

Qwen CPU processed the controls at 14.875039 audio h/h with 5.275 GiB peak
host RSS. `GPU.0` produced exactly the same scored output at 13.100663
audio h/h with 5.006 GiB host RSS and 2,085,217,046 bytes of observed current
GPU allocation. CPU was 13.54% faster end to end. The GPU statistic is a
current-allocation snapshot from a separate OpenVINO core, not an attributed
historical peak.

The released legacy Qwen OpenVINO route remains an implementation-specific
blocker: both CPU and iGPU hit the token cap with degenerate repeated output.
The HF-native `automatic-speech-recognition-with-past` export fixes that
failure. It uses exact EOS/pad checks, per-component execution-device proof,
and raw token health metrics.

### Conventional OCR generated controls

| Candidate | Samples | NCER | Required-token recall | False-positive chars | Line-count MAE |
|---|---:|---:|---:|---:|---:|
| RapidOCR full | 7 | 0.001757 | 0.961538 | 0 | 1.285714 |
| RapidOCR 1280/off | 7 | 0.003515 | 0.961538 | 0 | 1.428571 |
| PP-OCRv6 Medium | 7 | 0.003515 | 0.923077 | 0 | 1.285714 |
| PP-OCRv6 Tiny | 7 | 0.005272 | 0.923077 | 0 | 1.714286 |

Two separate randomized, anonymized private-course comparisons used 12 samples,
three judges, and 36 votes each. Full RapidOCR beat PP Tiny by 9 consensus wins
to 2, with one tie. It beat RapidOCR 1280/off by the same 9-to-2 consensus
count. These comparisons have different fingerprints and are not represented
as one paired dataset.

## Source-faithful document OCR

The benchmark-owned `source-faithful.v1` contract follows the active and
legacy OCRLLM behavior without importing the legacy package:

- exactly one page/frame marker as the first line;
- Markdown headings, paragraphs, lists, and reading order;
- LaTeX delimiters and commands rather than Unicode lookalikes;
- GitHub-Flavored Markdown pipe tables;
- code fences only for visible code, with exact language, indentation, blank
  lines, spelling, identifiers, signs, numbers, and units;
- visible instructions treated as source content;
- no solving, summarizing, translating, normalizing, autocorrecting, or
  inventing.

Three deterministic 1920 x 1080 controls exercise bilingual projected code,
formula-board writing with a deliberate misspelling and an instruction-looking
sentence, and a table/two-column reading-order page. The scorer uses raw output,
normalizes only line endings and one terminal newline, and checks formula,
table, code, indentation, Python parseability, protected spans, targeted
inventions, reading order, Markdown CER, and exact repeatability. It requires
two distinct records, attempts, keys, and trial indexes with identical
configuration, code, environment, workload, and record hashes.

### Native HunyuanOCR 1.5

The native Transformers 5.13.0 / Torch 2.11 CPU path is compatible:

- exact 2,239,932,512-byte checkpoint and auxiliary bundle verified;
- FP32, eager attention, PIL/slow image processor, both EOS IDs
  `[120007, 120020]`, pad `120002`;
- one bounded 960-side compatibility page finished in 8.126 seconds,
  443.04 images/hour, EOS 1, cap 0, peak RSS 5.218 GiB;
- a prior 64-token result was explicitly invalidated after the compatibility
  gate incorrectly accepted a token cap.

The upstream slow-processor and dual-EOS requirements matter: Tencent and
Transformers documented that stale EOS/input handling and the earlier fast
processor could create repetition or garbled OCR
([Tencent discussion](https://huggingface.co/tencent/HunyuanOCR/discussions/34),
[Transformers fix](https://github.com/huggingface/transformers/pull/47499)).
The native result therefore falsifies the old implementation blocker without
reusing pre-fix output as model-quality evidence.

Two full-resolution, three-page source-faithful trials then produced:

| Metric | Result |
|---|---:|
| Mean throughput | 56.7236 images/hour |
| Mean peak RSS | 13.515 GiB |
| Runtime failures / token caps | 0 / 0 |
| EOS finishes | 6/6 |
| Exact repeatability | 1.0 |
| Markdown CER | 0.363739 |
| Lexical precision / recall | 0.965909 / 0.611511 |
| Protected-span recall | 0.928571 |
| Exact marker / page sequence | 0 / fail |
| Formula precision / recall | 0.333333 / 0.333333 |
| Table shape / cells | 1.0 / 1.0 |
| Code-fence / exact code lines / Python parse | 0 / 0 / 0 |
| Reading-order pair accuracy | 0.355556 |
| Forbidden inventions / Unicode-math substitutions | 2 / 4 |
| Semantic gate / exact-profile gate | fail / fail |

The raw outputs explain the aggregate: the model deterministically omitted
markers and lower-page content, autocorrected the deliberate misspelling,
replaced LaTeX operators with Unicode, altered formula notation, and emitted
visible code without a fence or the required blank line. It did reconstruct the
simple GFM table exactly. Native Hunyuan is therefore runnable and useful as a
manual comparator, but not an unattended source-faithful escalation.

### Other structured/VLM gates

| Candidate | Bounded result | Decision |
|---|---|---|
| HunyuanOCR 1.5 F16 GGUF | 229.33 images/h on three smaller controls; NCER 1.2195, recall 0.4444, 18 negative-control false characters | Not a quality winner |
| PaddleOCR-VL 1.6, 512 tokens | 46.67 images/h; NCER 1.0221, recall 0.4615; structured blocks; zero negative false text | Manual-review lane |
| PaddleOCR-VL 1.6, 4096 tokens | 46.90 images/h with identical scored output | Larger cap adds no value |
| Granite-Docling 258M FP32 | Official one-page path timed out fail-closed at 300.328 s with no DocTags or Markdown; peak RSS 2.438 GiB | Base CPU compatibility remains unverified; no retry or lane |
| OvisOCR2 Q8/BF16 projector | Immutable pair could not finish downloading in fast mode; independent resume measured 0.183 MiB/s and about 84 minutes remaining | Resumable future falsification gate, no current inference claim |

Granite used the pinned
[`982fe3b` model](https://huggingface.co/ibm-granite/granite-docling-258M/tree/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe),
FP32 SDPA, and the official prompt. Its weight was independently verified as
515,093,104 bytes with SHA-256
`1cdad234deb1cde18ee6a586f849057f19851daf1fedce2e40aff791dbe46f61`.
An independent falsification found that 1920 x 1080, 640 x 360, and 128 x 72
inputs all become the same 13-tile 512 x 512 tensor under the official
processor, so downscaling would not make a defensible smaller CPU gate.

OvisOCR2 was the only new community discovery likely to change the
difficult-page conclusion. The pinned
[official model](https://huggingface.co/ATH-MaaS/OvisOCR2/tree/65c619d374b55d4152e85150fc1b003700bc1f0c)
targets ordered Markdown/LaTeX but trains tables as HTML. Its raw custom-prompt
gate remains unrun. HTML-to-GFM conversion is not assumed semantics-safe and
would require a separately provenance-bound adapted-output evaluation.

## OCRLLM compatibility

The active package was installed independently and non-editably at version
0.1.0 from captured master snapshot
`379726281e3c374bda65c1bd4a6bdf5c32cde0b3`. The existing local image
facade:

- recognized the generated control in 2.904 seconds;
- retained 5/5 raw text lines and 4/4 formula-like lines;
- stayed memory-only and made zero network attempts;
- did not expose RapidOCR polygons or line confidences;
- exposed no public ASR symbols.

Text-only OCR can enter the facade without line loss, but the facade cannot
preserve layout or formula semantics by itself after geometry/confidence is
dropped. Future OCRLLM integration should retain a raw/structured sidecar and
make any Markdown adapter explicit. Image checkpoint/resume and the worker
remain image features. ASR, long audio, and FileTrans stay behind
benchmark-owned thin adapters until the active library exposes a public API.
`legacy_app.OCRLLM` is not a new dependency, and no OCRLLM repository
file was edited.

## Latest upstream and community audit

Community material was used as a lead for local tests, never as ground truth.
The search was frozen after candidates could no longer change a recommendation
within a bounded run.

- [`transcribe.cpp` v0.2.1](https://github.com/handy-computer/transcribe.cpp/releases/tag/v0.2.1)
  introduced genuine SenseVoice tensor batching. The local batch-1/2/4/8 screen
  measured a real 2.72-times gain, but its best result was still less than half
  the established multiprocess throughput.
- Current
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.2.1)
  and [CTranslate2 4.8.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.8.1)
  support the measured physical-core worker design. The new 24 x 1 challenger
  failed locally. Batched-pipeline timestamp/context behavior is not assumed
  equivalent to direct `WhisperModel` output.
- [RapidOCR 3.9.2](https://github.com/RapidAI/RapidOCR/releases/tag/v3.9.2)
  exposed the OpenVINO thread controls used in the matched test. Its synchronous
  request path supports process-level isolation, not a shared-engine thread
  shortcut.
- [OpenVINO GenAI 2026.3](https://github.com/openvinotoolkit/openvino.genai/releases/tag/2026.3.0.0)
  now lists Qwen3-ASR CPU/GPU as early release. It was not silently mixed into
  the completed Optimum experiment: the pinned `openvino-genai` wheel is
  absent, and neither existing IR is proven compatible with `ASRPipeline`.
  Fast mode froze this as a future one-wheel constructor/parity gate, not a
  technical infeasibility claim.
- OvisOCR2 was retained only as a resumable future screen. GLM-OCR requires a
  separate layout pipeline for full document structure; MinerU2.5 Pro marks
  pure VLM CPU unsupported; NaviDC-OCR has only GPU/vLLM evidence; and
  LightOnOCR's current GGUF path has an unresolved repeated-token regression.
  None overturned a measured default.

No Reddit or community result supplied a controlled Windows/x86 measurement
that overrode local evidence. Anecdotal quality reports are not substituted for
the generated controls, blinded comparisons, or append-only events.

## Linear capacity projections

These use only quality-qualified sustained winners. Downloads, cold load,
queueing, I/O, human review, and workload shift are excluded.

### ASR

| Workload | SenseVoice 116.8639 x | faster-whisper 33.7910 x |
|---:|---:|---:|
| 2.5 audio h | 1.28 min | 4.44 min |
| 37.5 audio h | 19.25 min | 1.11 h |
| 375 audio h | 3.21 h | 11.10 h |
| 56,000 audio h | 479.19 h (19.97 d) | 1,657.25 h (69.05 d) |

### OCR

| Workload | Rapid full 14,646.44/h | Rapid 1280/off 24,882.07/h | PP Tiny 29,591.46/h |
|---:|---:|---:|---:|
| 50-80 images | 0.20-0.33 min | 0.12-0.19 min | 0.10-0.16 min |
| 750-1,200 | 3.07-4.92 min | 1.81-2.89 min | 1.52-2.43 min |
| 7,500-12,000 | 0.51-0.82 h | 0.30-0.48 h | 0.25-0.41 h |
| 1.12-1.79 million | 76.47-122.21 h | 45.01-71.94 h | 37.85-60.49 h |

The OpenVINO RapidOCR and native Hunyuan rates are not used for scale
projections: the former has only matched 60-second evidence, and the latter
failed the source-faithful quality gate.

## Reproduction and provenance

Primary entry point:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' `
  -m local_inference_bench.cli sustained `
  --candidate <candidate-id> `
  --workload <manifest.json> `
  --phase <phase> `
  --target-wall-seconds <seconds> `
  --config-index <index>
```

Generate the deterministic document controls:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' `
  scripts\generate_document_fidelity_controls.py
```

Score two source-faithful trials:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' `
  -m local_inference_bench.score_document_fidelity `
  --manifest data\inputs\generated\document_fidelity\manifest.json `
  --records <trial-0-private-records.jsonl> `
  --records <trial-1-private-records.jsonl> `
  --candidate hunyuanocr_1_5_native_cpu `
  --mode raw `
  --append-journal results\quality-events.jsonl
```

Append-only sources of truth:

- `results/events.jsonl` for installation and short feasibility;
- `results/sustained-events.jsonl` for starts, successes, failures, and
  explicit invalidations;
- `results/quality-events.jsonl` for privacy-safe aggregate quality;
- `results/bounded-events.jsonl` for aggregate community screens and
  independently verified blockers;
- ignored per-attempt artifacts for raw outputs and resource samples.

The sustained runner verifies isolated environments before reuse and includes
worker, monitor, loader, validator, setup verifier, environment manifest,
candidate-specific prompt/export files, model/export artifact manifests,
workload, hardware, phase, duration, and trial index in the attempt identity.
Successful private records receive an ignored sidecar that binds their SHA-256
to candidate, task, phase, workload, config, trial, code, and environment.

This is a trusted-local provenance chain, not a cryptographic transparency log:
sidecars are not signed or independently corroborated against the journal, and
same-host hardware is implicit in the attempt identity rather than repeated in
the sidecar. The exact-profile control also represents the chosen GFM contract;
it is stricter than semantic readability and is not a complete measure of
photographed-lecture quality.
