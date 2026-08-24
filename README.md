# Local inference feasibility benchmark

This repository measures sustained local OCR, source-faithful document
transcription, and ASR on the hardware that is actually visible: an Intel Core
Ultra 9 285K with 24 physical/logical processors, 64 GB RAM, Intel integrated
graphics, and no discrete GPU.

## Current decisions

| Use case | Recommendation |
|---|---|
| Bulk OCR | RapidOCR 3.9.2 / PP-OCRv6 Small, ONNX Runtime, full 2000-pixel mode, classifier enabled, 8 processes x 2 ORT threads, OpenCV threads = 1 |
| Maximum OCR throughput | PP-OCRv6 Tiny, 6 processes x 4 Paddle threads, OpenCV threads = 1 |
| Difficult-image escalation | Full-resolution RapidOCR plus human review; no tested CPU VLM qualifies for unattended escalation |
| Bulk ASR | SenseVoice Small GGUF Q8 on the pinned v0.2 source runtime, 8 processes x 3 threads |
| Timestamped ASR | faster-whisper Small int8, 10 resident model workers x 2 threads; accept mild sustained variability |
| Higher-quality ASR | No automatic lane qualifies; official OpenVINO GenAI Qwen3-ASR CPU is a manual second opinion only |
| Intel iGPU | No qualified sustained/general crossover: a public 11-second after-load smoke favored the iGPU, while CPU was 9.527% faster in the tracked matched 32-second comparison; other durations remain unqualified |
| OCRLLM | Use the active image facade for plain text, preserve geometry/structured output in a sidecar, and keep local ASR behind benchmark-owned adapters |

The measured basis, limitations, scale projections, and compatibility findings
are in
[reports/sustained-quality-and-compatibility.md](reports/sustained-quality-and-compatibility.md).
The earlier installation-only report is retained as a historical snapshot and
is explicitly superseded for recommendations.

## Representative sustained evidence

| Candidate | Configuration | Steady result | Stability and resources |
|---|---|---:|---|
| SenseVoice v0.2 | 8 x 3 | 123.4018 audio h/h; 622/622 in 604.9 s | CV 0.02460; 2.460 GiB; 76 threads; 98.75% host CPU |
| faster-whisper | 10 x 2 | 27.9238 audio h/h; 140/140 in 601.6 s | CV 0.05127; 1.863 GiB; 193 threads; 85.76% host CPU |
| RapidOCR full | 8 x 2, OpenCV = 1 | 12,710.79 images/h; 2,125/2,125 in 601.9 s | CV 0.02325; 2.937 GiB; 100 threads; 91.13% host CPU |
| PP-OCRv6 Tiny | 6 x 4, OpenCV = 1 | 23,858.80 images/h; 3,985/3,985 in 601.3 s | CV 0.01669; 4.809 GiB; 130 threads; 99.57% host CPU |

All four runs recorded zero inference failures and zero available
performance-limit or thermal-throttle flags. Package temperature is not
available from this host, so no package-temperature claim is made. Normal OS,
firmware, cooling, and power controls remained enabled.

These rows preserve the latest pre-source-freeze measurements. The worker,
workload-fingerprint, containment, and steady-window contracts have since been
tightened, so the final evidence checkpoint must replace them with serialized
runs from the frozen source before treating the exact rates as current.

The representative private workload came from a validated complete local
lecture found through read-only inspection. Only ignored, de-identified samples
were used: one 20-minute audio item, ten independent 2-minute chunks, bounded
32-second RMS-selected near-silence/speech controls, and ten masked 1080p
frames. Tracked lecture-derived evidence contains no identity, title, people,
transcript, frame, audio, path, source hash, original media metadata, or source
timestamps. Five tracked PNG fixtures and two manifests contain deterministic,
invented benchmark content only. The crawler was not used and is not a
dependency.

## Evidence boundary

The project separates acquisition, environment creation, cold load/compile,
warm-up, steady inference, quality scoring, and failure records. The original
short synthetic pass remains useful installation evidence, but it is not
treated as lecture-quality or sustained-throughput evidence.

- `results/events.jsonl` records installation and short feasibility attempts.
- `results/sustained-events.jsonl` is the append-only process and sustained-run history.
- `results/quality-events.jsonl` contains fixed-key, privacy-safe quality aggregates and explicit invalidations.
- `results/bounded-events.jsonl` records aggregate community screens, compatibility gates, and verified blockers.
- Candidate stdout/stderr, raw predictions, private samples, resource traces,
  provenance sidecars, and resumable downloads remain under ignored paths.
- Public/generated quality controls are the only absolute recognition references.
  Private ASR comparisons without a trusted transcript are agreement diagnostics,
  not WER or accuracy.

## Reproduce

PowerShell script execution is disabled by policy on the measured host. Invoke
scripts with a process-local bypass; this does not change system policy.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\capture_hardware.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\create_control_environment.ps1
```

Prepare and verify the deterministic public/generated controls used for
absolute quality scoring:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\prepare_generated_quality_controls.py `
  --suite ocr --suite document-fidelity --suite asr
& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\verify_generated_quality_controls.py --suite ocr
& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\verify_generated_quality_controls.py --suite document-fidelity
& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\verify_generated_quality_controls.py --suite asr
```

Create only the isolated candidate environments needed for a run. Each wrapper
installs the pinned environment, acquires resumable pinned assets where needed,
and runs its verifier:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\prepare_sensevoice_sustained_assets.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\create_faster_whisper_environment.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\create_paddleocr_environment.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\create_rapidocr_openvino_environment.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\prepare_qwen3_asr_openvino_genai_official.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\create_ocrllm_compatibility_environment.ps1
```

Run a registry candidate:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' -m local_inference_bench.cli sustained `
  --candidate <candidate-id> `
  --workload <manifest.json> `
  --phase <screen|quality|compatibility|sustained> `
  --target-wall-seconds <seconds> `
  --config-index <index>
```

Run one immutable b10598 VLM gate, then independently build its public event:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\prepare_bounded_vlm_b10598_assets.ps1 `
  -CandidateId <ovisocr2_q8_cpu|hunyuanocr_1_5_gguf_cpu>

& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\run_bounded_vlm_b10598_quality.py `
  --candidate <ovisocr2_q8_cpu|hunyuanocr_1_5_gguf_cpu> `
  --output-dir results\artifacts\<new-run-directory>

& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\build_bounded_vlm_v3_event.py `
  --candidate <candidate-id> `
  --run-dir results\artifacts\<new-run-directory> `
  --output results\artifacts\<new-public-event.json>
```

Exercise the independently installed active OCRLLM image facade on the tracked
generated code/formula control without appending an event:

```powershell
& 'D:\Anaconda\envs\local-bench-ocrllm-master-f234f39\python.exe' scripts\check_ocrllm_image_facade.py `
  --image data\inputs\generated\ocr_quality\code_formula.png `
  --manifest data\inputs\generated\ocr_quality\manifest.json
```

The runner verifies isolated environments and required assets before reuse,
fingerprints candidate-specific code and model/export manifests, distinguishes
complete, partial, and all-failed outcomes, and keeps new content-derived
private attempt identity in ignored provenance only. Preserved legacy
append-only rows still contain opaque 16-hex attempt keys and 64-hex blind-
judgment fingerprints whose inputs included private benchmark data. They are
content-derived identifiers, not source-file hashes; no raw private content or
path is published. Quality scorers require status-consistent, hash-bound records
and deterministic generated references; partial and all-failed attempts remain
explicit in their availability and quality denominators.

Run the regression suite with:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' -m pytest -q
```

## Safety and interpretation

No firmware, BIOS, affinity, cooling, power-limit, security, or temperature
control was changed. Heavy inference runs were serialized. Resource monitors
record process-tree CPU, RSS, host memory, RAPL package power, available
performance-limit evidence, and failures. The ACPI-zone value is not reported
as CPU package temperature.

Throughput projections are linear warmed-capacity estimates, not delivery
promises. They exclude downloads, cold load, queueing, media decode/frame
selection, storage I/O, human review, and workload shift. No paid API was used,
and the repository does not claim zero electricity, network, or unknown
free-service quota cost.
