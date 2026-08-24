# Sustained OCR and ASR on Core Ultra 9 285K

Status: current measured recommendation

Evidence cutoff: 2026-08-24

The numeric winner rows below are serialized source-frozen measurements under
the tightened workload, containment, provenance, and steady-window contracts.

## Decision summary

| Decision | Recommendation | Measured basis |
|---|---|---|
| Bulk default OCR | RapidOCR 3.9.2 / PP-OCRv6 Small, ONNX Runtime, full 2000-pixel mode, classifier enabled, 8 processes x 2 threads, OpenCV = 1 | 12,517.86 representative 1080p frames/h for 10 minutes; strongest generated accuracy; 30/30 blind preference votes and 10/10 consensus wins |
| Throughput OCR | PP-OCRv6 Tiny, 6 x 4, OpenCV = 1 | 22,780.07 representative frames/h for 10 minutes; 3,801/3,801; approximately 1.82x Rapid throughput, but weaker quality |
| Difficult-image escalation | Full-resolution RapidOCR plus human review | Ovis passes one generated HTML-table compatibility gate, but neither its coverage nor any other tested CPU VLM qualifies for unattended escalation |
| Bulk default ASR | SenseVoice Small GGUF Q8, pinned v0.2 source runtime, 8 x 3 | 124.3449 audio h/h for 10 minutes; 629/629; generated-control NCER 0.176576 and zero silence false positives |
| Timestamp ASR | faster-whisper Small int8, 10 resident workers x 2 threads | 28.5292 audio h/h; 150/150 and full timestamps, but the sustained run is explicitly variable |
| Higher-quality ASR | No automatic lane qualifies; official OpenVINO GenAI Qwen3-ASR CPU is a manual second opinion | Better generated NCER than SenseVoice, but worse overall required-term recall and failed/capped representative 120-second chunks |
| Practical CPU concurrency | SenseVoice 8 x 3; faster-whisper 10 x 2; RapidOCR 8 x 2; PP Tiny 6 x 4 | Several modest workers materially outperform one 24-thread process; library pools create more OS threads than the configured native-thread budget |
| Likely GPU crossover | No qualified sustained/general Intel-iGPU crossover; a discrete GPU is most likely to cross first for generative OCR/VLM work, but was not present to measure | A public 11-second after-load smoke favored the iGPU, while CPU was 9.527% faster in the tracked matched 32-second Qwen comparison; other durations remain unqualified |
| OCRLLM compatibility | Image facade for plain text, structured sidecar for geometry/formulas, benchmark-owned ASR adapters | Exact active master `2827c98` works locally; no public local-ASR, FileTrans, long-audio, persistence/resume, or audio-worker facade exists |

These choices optimize for this 24-processor, 64 GB, no-discrete-GPU host. They
are not generic model rankings.

## Evidence and privacy boundary

A validated complete local lecture was located through read-only inspection of
existing downloader state. It supplied ignored, de-identified samples only:

- one 20-minute mono PCM16 audio item;
- ten independent 2-minute chunks for process/worker concurrency;
- two bounded 32-second RMS-selected near-silence/speech controls;
- ten masked 1920 x 1080 frames.

Tracked lecture-derived evidence contains aggregate workload descriptions and
measurements only. It contains no course identity, title, people, transcript,
frame, audio, private path, source hash, original media metadata, source
timestamp, or credential. Five tracked PNG fixtures and two manifests contain
deterministic, invented benchmark content only. The crawler was not used and is
not on the benchmark critical path.

The masked frames cover projected slides, formulas, plots, tables, small mixed
Chinese/English text, low contrast, UI chrome, and occlusion. They do not cover
handwriting or code; deterministic public/generated controls cover those two
cases. No trusted private transcript was available. Private ASR agreement is
therefore a consistency/hallucination diagnostic, not WER or accuracy. The
RMS-selected low-energy clip is not asserted to be absolute silence.

The journals are append-only. Explicit invalidation/redaction rows supersede
stale evidence; consumers must not count both an invalidated row and its
replacement. The committed pre-existing prefixes were preserved exactly, while
content-derived private attempt keys and misplaced aggregates were removed from
the new suffix. Preserved legacy rows still contain opaque 16-hex attempt keys
and 64-hex blind-judgment fingerprints whose inputs included private benchmark
data. They are content-derived identifiers, not source-file hashes; no raw
private content or path is published. Raw outputs and the pre-projection
quarantine remain ignored.

## Sustained concurrency results

The four recommended configurations each ran for at least 600 steady seconds
and recorded zero inference failures.

| Candidate | Configuration | Completed | Steady throughput | CV / last:first | Peak RSS | Peak OS threads | Mean host CPU | Package power mean / p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SenseVoice v0.2 | 8 x 3 | 629/629 | 124.3449 audio h/h | 0.01504 / 1.01587 | 2.413 GiB | 76 | 98.42% | 244.02 / 260.50 W |
| faster-whisper | 10 x 2 | 150/150 | 28.5292 audio h/h | 0.24663 / 0.84211 | 1.875 GiB | 194 | 84.68% | 157.14 / 189.18 W |
| RapidOCR full | 8 x 2, OpenCV = 1 | 2,092/2,092 | 12,517.86 images/h | 0.03029 / 1.00980 | 2.938 GiB | 100 | 94.21% | 180.15 / 190.62 W |
| PP-OCRv6 Tiny | 6 x 4, OpenCV = 1 | 3,801/3,801 | 22,780.07 images/h | 0.01397 / 0.97927 | 5.001 GiB | 130 | 99.33% | 191.37 / 203.64 W |

Canonical events are
[SenseVoice line 383](../results/sustained-events.jsonl#L383),
[faster-whisper line 385](../results/sustained-events.jsonl#L385),
[RapidOCR line 389](../results/sustained-events.jsonl#L389), and
[PP Tiny line 387](../results/sustained-events.jsonl#L387).

Cold/load and warm-up remain separate from steady throughput:

| Candidate | Load mean | Warm-up mean | Interpretation |
|---|---:|---:|---|
| SenseVoice | 0.105 s | 1.082 s | Per-file CLI startup estimate after integrity hashing |
| faster-whisper | 0.872 s | 6.631 s | Resident model |
| RapidOCR | 0.249 s | 3.147 s | Eight resident pipelines |
| PP Tiny | 1.503 s | 1.431 s | Six resident pipelines |

The load semantics differ, so these values are not a model-loading leaderboard.
Downloads, environment creation, and model acquisition are excluded.
Process-tree RSS, available RAM, page input, CPU, and RAPL package power were
sampled. The host exposed no trustworthy DRAM-bandwidth counter, so the report
does not claim measured memory-bandwidth saturation or headroom.

### What changed the scale conclusion

- SenseVoice v0.2 at 8 x 3 improved the matched v0.1.9 pool by 7.991% and
  reached 124.34x real time in the source-frozen repeat. A separate 20-minute,
  one-process compatibility run completed in 23.527 seconds (51.0052x); it
  establishes long-file execution, while the 124.34x result requires independent
  chunk concurrency.
- faster-whisper 10 x 2 reached 28.53x versus only 12.02x for one 24-thread
  worker. It is the practical timestamp setting, but its CV 0.24663 and
  last:first 0.84211 require a variable-throughput warning. The 24 x 1 screen
  created 419 OS threads and unstable tail latency and was rejected.
- PP Tiny's matched OpenCV cap increased throughput only 1.36%, but reduced
  peak thread count from about 300 to 130 and peak RSS by 2.67%. That reduction
  in oversubscription is enough to make `opencv_threads=1` the default.
- RapidOCR's balanced comparison improved mean throughput by 1.10%, reduced
  peak thread count by 69.3%, and preserved quality. Its representative
  sustained rate is 3.37% above the earlier uncapped baseline.
- `KMP_BLOCKTIME=0` changed PP throughput by only about 0.3% and package power
  by about 0.9%; it was not promoted.

This host exposes 24 processors, but `processes x configured threads` is not an
OS-thread ceiling. OpenCV, ONNX Runtime, oneDNN/OpenMP, CTranslate2, Python, and
monitoring add their own threads. The measured process-tree thread counts are
therefore part of the recommendation.

## OCR quality

### Deterministic generated controls

| Candidate | Samples | NCER | Required-token recall | Negative false characters |
|---|---:|---:|---:|---:|
| RapidOCR full | 7/7 | 0.001757 | 0.961538 | 0 |
| PP-OCRv6 Tiny | 7/7 | 0.005272 | 0.923077 | 0 |

The current provenance-bound events are
[RapidOCR line 29](../results/quality-events.jsonl#L29) and
[PP Tiny line 30](../results/quality-events.jsonl#L30). Generated controls test
code, formulas, dense tables, bilingual text, and negative images. They are
absolute references, but they are not a substitute for photographed-course
evaluation.

### Randomized representative-frame comparison

Three accepted judge runs—two Luna max and one Terra high—scored the
precommitted, salted, randomized mapping of ten masked lecture frames. The
packet commitment is at [quality line 45](../results/quality-events.jsonl#L45)
and the v10 aggregate is at
[quality line 46](../results/quality-events.jsonl#L46):

| Candidate | Win votes | Consensus wins | Mean error severity | Usable vote fraction |
|---|---:|---:|---:|---:|
| RapidOCR full | 30 | 10 | 0.9667 | 1.0000 |
| PP-OCRv6 Tiny | 0 | 0 | 2.2000 | 0.7333 |

There were no ties, 30 total votes, pairwise winner agreement 1.0, and
unanimity on every sample. Mapping consistency and the semantic-duplicate guard
passed. This is procedural blinding and a strong evaluator preference, not
ground truth: judge identity and semantic independence were not verified,
pre-judgment chronology is not machine-verified, and three judges do not turn
ten frames into thirty independent source samples.

The result explains the two-tier recommendation: RapidOCR is the quality-biased
bulk default; PP Tiny is the throughput option.

## ASR quality and long-audio behavior

The public/generated set contains ten Chinese, English, mixed-language,
technical-term, abbreviation, number/formula narration, sparse speech,
long-silence, and silence controls.

| Candidate | NCER | Mixed-token error | Required-term recall | Silence false chars/min | Timestamp evidence |
|---|---:|---:|---:|---:|---|
| SenseVoice v0.2 | 0.176576 | 0.237113 | 0.871795 | 0 | unavailable |
| Official Qwen3-ASR OpenVINO GenAI CPU | 0.128772 | 0.214433 | 0.743590 | 0 | unavailable |
| faster-whisper auto language | 0.401852 | 0.646392 | 0.615385 | 6 | precision 0.90737; recall 0.70329 |
| faster-whisper forced Chinese | 0.723036 | 0.950515 | 0.461538 | 108 | worse; rejected |

The canonical events are
[faster-whisper lines 26-27](../results/quality-events.jsonl#L26-L27),
[SenseVoice line 31](../results/quality-events.jsonl#L31), and
[Qwen line 32](../results/quality-events.jsonl#L32).

SenseVoice is the only candidate that passes the bulk gate of high sustained
throughput, generated-control NCER at or below 0.20, term recall at or above
0.80, and zero silence false positives. It does not provide timestamps in this
runtime. faster-whisper remains valuable specifically for timestamps; forcing
Chinese is not a safe default for mixed lecture audio.

Qwen's lower generated-control NCER makes it a useful manual second opinion,
not an automatic escalation. Its term recall is 12.82 percentage points below
SenseVoice. On the current ten-item representative 120-second cohort it
completed 9/10, hit the 512-token cap once, and produced too few steady windows
for a sustained-throughput claim at
[sustained line 403](../results/sustained-events.jsonl#L403).

The three-source private ASR agreement event at
[quality line 47](../results/quality-events.jsonl#L47) covers ten samples from
SenseVoice, faster-whisper, and Qwen. Qwen has nine comparable outputs and one
unavailable item, so its successful-output exact aggregates are deliberately
withheld. The event is text-only agreement: it has no trusted gold transcript,
no timestamp comparison, and cannot select a quality winner.

## Official Qwen3-ASR OpenVINO GenAI and Intel iGPU

The official stateful `ASRPipeline` export is independently pinned and verified;
it is separate from the older Transformers/Optimum experiments.

Four matched short CPU/iGPU pairs established timing. The earlier device-only
agreement event at [quality line 33](../results/quality-events.jsonl#L33) binds
one CPU source attempt and one iGPU source attempt and reports exact-equal text
for that evidenced pair; it is separate from the ten-item cross-model cohort:

| Device | Mean throughput | Result |
|---|---:|---|
| CPU | 15.7637 audio h/h | 9.527% faster than iGPU |
| Intel iGPU | 14.3925 audio h/h | More variable; no speed crossover |

The OpenVINO GenAI encoder-tail comparison at
[bounded line 8](../results/bounded-events.jsonl#L8) establishes source ancestry:
stable 2026.3 lacks the fix and the associated nightly source contains it. Both
produced identical transcript hashes on only two bounded controls. The event
explicitly does not claim cryptographic wheel-to-source attestation, and it does
not resolve the 120-second failures. In the decisive same-item tail-sensitive
120.08-second check, both stable and tail-fixed runtimes failed 0/1 at the same
512-token cap ([stable line 391](../results/sustained-events.jsonl#L391),
[tail-fixed line 393](../results/sustained-events.jsonl#L393)). The source fix
corrects encoder tail-token retention; it does not remedy the separate decoder
runaway observed here. The tested association is
`2026.4.0.0.dev20260821`; no earlier universal nightly floor is claimed. Any
future release must be checked by actual tag/build ancestry rather than inferred
from version number.

The practical conclusion is limited to the tracked duration and backend, not
"iGPU always slower." No discrete GPU was present. A discrete GPU is most likely
to change the economics first for generative VLM/OCR and long generative ASR,
not for the already-fast conventional OCR and SenseVoice lanes, but that
crossover remains unmeasured on this machine.

## Difficult-image and OCR VLM gates

Completing an inference path is not the same as passing the source-fidelity
gate.

| Candidate | Bounded result | Fidelity result | Decision |
|---|---|---|---|
| OvisOCR2 Q8/BF16 projector, llama.cpp b10598 | 1/1, no cap, 324.87 images/h single-page estimate, 1.89 GB peak RSS | raw P/R 0.3312/0.5532; visible P/R 1.0/0.9412; 9/9 semantic HTML cells exact | narrow `quality_gate_passed`; coverage insufficient for escalation |
| HunyuanOCR 1.5 F16 GGUF, llama.cpp b10598 | 3/3, no caps, 216.46 images/h, 4.09 GB peak RSS | visible NCER 0.2683, recall 0.7778, and 18 false characters on the negative control | `quality_blocker` |
| Native HunyuanOCR 1.5 | Runnable on full-resolution controls | deterministic omissions, altered formulas, bad reading order, semantic/profile gates failed | manual comparator only |
| PaddleOCR-VL 1.6 | Runnable at about 46.7 images/h in the earlier gate | NCER above 1 and low recall | manual structured comparator only |
| Granite-Docling 258M | Official path timed out fail-closed at 300.328 s | no DocTags or Markdown emitted | no CPU lane |

The effective, fully hash-bound b10598 events are
[Ovis line 13](../results/bounded-events.jsonl#L13) and
[Hunyuan line 14](../results/bounded-events.jsonl#L14). The adversarially rejected
scorer rows are retained and explicitly invalidated at
[lines 15-16](../results/bounded-events.jsonl#L15-L16). Both candidates falsify
their old implementation/acquisition blockers. Hunyuan's remaining blocker is
recognition fidelity and negative-control hallucination.

Ovis uses the generic llama.cpp Qwen3.5 text plus Qwen3-VL projector route; the
upstream llama.cpp tree has no Ovis-native converter or graph. The tested build
is bound to the b10598 archive and source revision containing the Qwen-VL resize
correction. That makes the local run reproducible but does not turn community
GGUF compatibility into upstream Ovis support or establish equivalence to the
official BF16 checkpoint. Its one deterministic HTML-table fixture preserves
all nine semantic cells and passes the visible-text gate. It does not test
formulas or photographed lecture frames, so HTML-to-GFM conversion is still not
assumed semantics-safe.

No tested CPU VLM is approved for unattended difficult-image escalation. Ovis's
one-fixture pass is compatibility evidence, not representative qualification.
The operational route is full-resolution RapidOCR plus human review; VLM output
may be shown as a second opinion with raw structured output preserved.

## OCRLLM compatibility

The active `master` package was installed independently, non-editably, and
hash-compared against clean revision
`2827c98b802932d6bbc0b71bd8d8d4188fa6a0b0`, which descends from the reviewed
baseline `47c12efe91640659a711c8bd3429dae6a4fe44f5`. No OCRLLM repository file
was edited.

The current canonical check at
[quality line 48](../results/quality-events.jsonl#L48) reports:

- 195 installed Python files byte-equal to the pinned snapshot, three bundled
  RapidOCR model files, and 40 runtime components;
- 5 detected and 5 retained text lines, 147 output characters;
- 2/5 exact required-token hits and 5/5 whitespace-insensitive hits;
- mean internal RapidOCR confidence 0.978482;
- zero facade/provider calls recorded by the bounded instrumentation; this is
  not a universal network-zero claim;
- memory-only output;
- no facade-exposed polygons or line confidences and no detected LaTeX marker.

Plain text can enter the active image facade. Geometry, confidence, and
structured Markdown/HTML/LaTeX must remain in a sidecar; the benchmark does not
claim that HTML-to-GFM conversion preserves layout or formula semantics.

Current master also exposes experimental provider-backed, short, in-memory MP3
options with a configured 300-second validation limit. This is not a tested
maximum, and no provider call was needed here. It rejects the three tested
non-memory audio options and does not expose a public local-ASR facade,
FileTrans, long-audio path, audio persistence/resume, or audio worker. Local ASR
candidates therefore remain behind benchmark-owned thin adapters.
`legacy_app.OCRLLM` is not a new dependency.

Older OCRLLM and unbound blind rows remain in append-only history; their
supersession/corrections are recorded at
[quality lines 39-44](../results/quality-events.jsonl#L39-L44), and the current
effective checks are lines 45-48.

## Linear warmed-capacity projections

These projections use only the representative sustained winners. They exclude
downloads, cold load, queueing, media decode/frame selection, I/O, human review,
and workload shift.

### ASR

The scale assumptions are 2.5 audio hours per lecture, 37.5 per course, 375 for
ten courses, and 56,000 long-term audio hours.

| Scale | SenseVoice 124.3449x | faster-whisper 28.5292x |
|---|---:|---:|
| One lecture | 1.21 min | 5.26 min |
| One course | 18.09 min | 1.31 h |
| Ten courses | 3.02 h | 13.14 h |
| Long term | 18.77 d | 81.79 d |

### OCR

Frame assumptions are 50-80 per lecture, 750-1,200 per course, 7,500-12,000
for ten courses, and 1.12-1.79 million long term.

| Scale | RapidOCR 12,517.86/h | PP Tiny 22,780.07/h |
|---|---:|---:|
| One lecture | 0.24-0.38 min | 0.13-0.21 min |
| One course | 3.59-5.75 min | 1.98-3.16 min |
| Ten courses | 0.60-0.96 h | 0.33-0.53 h |
| Long term | 89.47-143.00 h | 49.17-78.58 h |

For the validated approximately 2.79-hour lecture used in this investigation,
the stable SenseVoice rate projects to 80.82 seconds of warmed ASR wall time.
Scaling the public 50-80-frames-per-2.5-hour assumption gives about 56-89 OCR
frames and 16.06-25.69 seconds for RapidOCR. Sequential OCR plus ASR is therefore
96.88-106.51 seconds. This is a capacity estimate, not an end-to-end delivery
time or a claim that the two heavy lanes were run concurrently.

Public/generated-control rates remain separately labeled in the journal and are
not averaged with representative 1080p rates.

## Upstream and community findings that changed experiments

Community sources were treated as leads, not ground truth. Primary source code,
official documentation, and pinned assets determined what was tested locally.

- [ONNX Runtime threading guidance](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
  supports one session set per process and explicit intra/inter-op control; it
  also motivated measuring spin/oversubscription rather than assuming thread
  counts.
- [OpenCV `setNumThreads`](https://docs.opencv.org/4.10.0/db/de0/group__core__utils.html)
  led directly to the successful RapidOCR and PP-OCR process-pool caps.
- [Paddle CPU configuration](https://www.paddlepaddle.org.cn/inference/v3.0/api_reference/python_api_doc/Config/CPUConfig.html)
  and predictor-pool guidance supported process isolation and the 24 configured
  native-thread budget.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
  [CTranslate2](https://github.com/OpenNMT/CTranslate2) source behavior led to
  instrumenting actual queued/processing batches; Python calls in flight alone
  are not proof of native concurrency.
- The official
  [OpenVINO GenAI tail fix](https://github.com/openvinotoolkit/openvino.genai/commit/0d35ded5bac2d39bf45d52cbc7156c087f50c80d)
  narrowed the Qwen investigation to a specific partial-mel-chunk correction.
- The pinned [llama.cpp b10598 release](https://github.com/ggml-org/llama.cpp/releases/tag/b10598),
  [official OvisOCR2 checkpoint](https://huggingface.co/ATH-MaaS/OvisOCR2/tree/65c619d374b55d4152e85150fc1b003700bc1f0c),
  and separately identified community GGUF enabled the bounded Ovis
  falsification run without claiming upstream-native support.

No Reddit or community anecdote supplied a controlled Windows/x86 result that
overrode the local measurements. Community reports saved search time, but every
recommendation above is tied to local append-only evidence.

## Reproduction and provenance

Primary sustained entry point:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' `
  -m local_inference_bench.cli sustained `
  --candidate <candidate-id> `
  --workload <manifest.json> `
  --phase <phase> `
  --target-wall-seconds <seconds> `
  --config-index <index>
```

SenseVoice source runtimes are rebuilt through:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\prepare_sensevoice_sustained_assets.ps1
```

The tracked patches preserve the upstream eight-thread default and add an
explicit thread option. Current rebuilds verify exact source/tag, llama.cpp,
patch, compiler, CMake, Ninja, archive, and executable hashes. Retained earlier
historical measurements predate the last build-tool hash hardening and are not
retroactively claimed to have used current enforcement.

One immutable VLM run and its independent public-event build use:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\run_bounded_vlm_b10598_quality.py `
  --candidate <candidate-id> `
  --output-dir results\artifacts\<new-run-directory>

& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\build_bounded_vlm_v3_event.py `
  --candidate <candidate-id> `
  --run-dir results\artifacts\<new-run-directory> `
  --output results\artifacts\<new-event.json>
```

The bounded VLM registry verifies full SHA-256 and size for the release archive,
runtime tree, entrypoints, models, projector/conversion lineage, manifests, and
tracked generated images. The public builder independently rehashes request,
response, records, logs, telemetry, producer code, and assets before emitting an
aggregate-only event.

The sustained runner verifies required assets before candidate startup,
fingerprints configuration-specific artifacts, rejects duplicate config
indices, and records complete, partial, and all-failed outcomes separately.
Private successful records receive an ignored provenance sidecar binding their
hash to candidate, task, phase, workload, configuration, trial, code, and
environment. The corrected runner does not publish new private content-derived
attempt keys; the preserved legacy-key limitation is disclosed above.

The current regression result is 871 passed and 5 intentionally skipped. This
is a trusted-local provenance chain, not a signed transparency log. Hardware
identity, local monitors, and ignored sidecars can still be altered by someone
with write access to the machine; the report claims reproducibility and bounded
validation, not tamper-proof attestation.

## Remaining uncertainty

- faster-whisper needs another stable 10-minute repeat before its timestamp lane
  can carry a stable-throughput claim; the current source-frozen run is variable.
- Qwen needs a bounded decoder-runaway remedy, 10/10 representative 120-second
  success, and an independently checked at-least-10-minute reference subset
  before it can become an automatic ASR escalation. Encoder tail-fix ancestry
  alone is insufficient.
- A discrete-GPU crossover requires actual matched hardware. It cannot be
  inferred from Intel-iGPU behavior.
- More representative handwriting/code frames and an independently checked
  private speech subset would reduce the current quality uncertainty.
- Ovis needs photographed lecture frames, formulas, and official-checkpoint or
  conversion-lineage parity evidence before its one-table pass can support an
  unattended escalation decision.
