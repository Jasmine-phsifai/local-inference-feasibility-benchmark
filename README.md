# Local inference feasibility benchmark

This repository measures sustained local OCR, source-faithful document
transcription, and ASR on the hardware that is actually visible: an Intel Core
Ultra 9 285K with 24 physical/logical processors, 64 GB RAM, Intel integrated
graphics, and no discrete GPU.

The current measured choices are:

| Use case | Recommendation |
|---|---|
| Bulk OCR | RapidOCR 3.9.2 / PP-OCRv6 Small, ONNX Runtime, full 2000-pixel mode, classifier on, 8 processes x 2 threads |
| Maximum OCR throughput | PP-OCRv6 Tiny, 6 x 4 |
| Difficult-page OCR | PaddleOCR-VL 1.6 at 512 tokens as a manual-review lane only; no tested CPU VLM is approved for unattended source-faithful conversion |
| Bulk ASR | SenseVoice Small GGUF Q8, 3 x 8 |
| Timestamped ASR | faster-whisper Small int8, 6 workers x 4 threads |
| Transcript second opinion | Qwen3-ASR 0.6B HF-native OpenVINO on CPU; it is not an automatic quality winner |
| Intel iGPU | No measured crossover: Qwen CPU was 13.5% faster than the iGPU |

The full evidence, limitations, scale projections, and OCRLLM compatibility
findings are in
[reports/sustained-quality-and-compatibility.md](reports/sustained-quality-and-compatibility.md).

## Evidence boundary

The project separates acquisition, environment creation, cold load/compile,
warm-up, steady inference, quality scoring, and failure records. The first
short synthetic pass is retained, but it is not treated as lecture-quality or
sustained-throughput evidence.

- `results/events.jsonl` records installation and short feasibility attempts.
- `results/sustained-events.jsonl` is the append-only sustained/process history.
- `results/quality-events.jsonl` contains fixed-key, privacy-safe quality aggregates.
- `results/bounded-events.jsonl` records aggregate community screens and verified blockers.
- Candidate stdout, stderr, raw predictions, private samples, and resource traces
  remain under ignored `results/artifacts/` and `data/` paths.
- Large model files and resumable partial downloads are ignored.

Requested course files were sought only through read-only local state. The
specific requested lectures were not available at their recorded locations, so
they are not claimed as test inputs. An unrelated validated complete lecture,
other existing private course samples, public audio, and deterministic generated
controls were used where appropriate. No course identity, title, name,
transcript, frame, audio, credential, private path, or private hash is committed.
The crawler is not a dependency.

## Reproduce

PowerShell script execution is disabled by policy on the measured host. Invoke
scripts with a process-local bypass; this does not change system policy.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\capture_hardware.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\create_control_environment.ps1
```

Create only the candidate environment needed for a run. Dedicated setup and
verification scripts pin the newer comparison paths:

- `scripts/create_rapidocr_openvino_environment.ps1`
- `scripts/create_qwen3_asr_hf_openvino_environment.ps1`
- `scripts/export_qwen3_asr_hf_openvino.ps1`
- `scripts/create_hunyuanocr_native_environment.ps1`
- `scripts/create_ocrllm_compatibility_environment.ps1`

Generate the deterministic source-fidelity controls:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' scripts\generate_document_fidelity_controls.py
```

Run a bounded or sustained candidate:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' -m local_inference_bench.cli sustained `
  --candidate <candidate-id> `
  --workload <manifest.json> `
  --phase <screen|quality|compatibility|sustained> `
  --target-wall-seconds <seconds> `
  --config-index <index>
```

The runner verifies the isolated environment before reuse, fingerprints the
worker, validation path, environment, model/export manifests, workload, and
hardware, alternates A/B order across trials, and writes ignored provenance
sidecars for raw records. The source-fidelity scorer accepts raw output only and
requires two distinct, provenance-bound trials:

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

Run the regression suite with:

```powershell
& 'D:\Anaconda\envs\local-bench-control\python.exe' -m pytest -q
```

## Safety and interpretation

No firmware, BIOS, affinity, cooling, power-limit, security, or temperature
control is changed. Resource monitors record CPU use, process-tree RSS, host
memory, available RAPL/frequency/performance-limit evidence, and failures. CPU
package temperature is unavailable on this host; the ACPI-zone value is not
reported as package temperature. Heavy inference runs are serialized.

Throughput projections are linear capacity estimates, not delivery promises.
They exclude downloads, cold load, queueing, storage I/O, human review, and
workload shift. No paid API was used, and the repository does not claim zero
cost for electricity, network transfer, or unknown free-service quota usage.
