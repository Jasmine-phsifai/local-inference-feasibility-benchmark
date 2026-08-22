# Local inference feasibility benchmark

This repository measures installation and throughput feasibility for local ASR,
conventional OCR, and OCR-oriented vision-language models on the detected host.
It intentionally does **not** evaluate final recognition quality.

## Reproduce

PowerShell script execution is disabled by policy on the measured host. Invoke
repository scripts with `powershell.exe -NoProfile -ExecutionPolicy Bypass -File`
when necessary; this process-local override does not change system policy.

1. Run `scripts/capture_hardware.ps1`.
2. Run `scripts/create_control_environment.ps1`.
3. Download the small public-domain speech sample with
   `scripts/download_public_samples.ps1`, then generate the input manifest with
   `D:\Anaconda\envs\local-bench-control\python.exe -m local_inference_bench.cli make-inputs`.
4. Create a candidate environment with
   `scripts/create_candidate_environment.ps1 -CandidateId <id>`.
   The Paddle environment also installs Microsoft's `vcomp14` runtime through
   Conda because PaddlePaddle 3.2.2 requires `VCOMP140.dll` on this host.
5. Run one candidate with
   `D:\Anaconda\envs\local-bench-control\python.exe -m local_inference_bench.cli run --candidate <id>`.
6. Rebuild the report with
   `D:\Anaconda\envs\local-bench-control\python.exe -m local_inference_bench.cli report`.
   `scripts/capture_environment_sizes.ps1` records isolated-environment disk
   overhead (apparent size; Conda hard links can reduce physical disk use).

`results/events.jsonl` is the append-only source of truth. Successful attempts
with the same candidate, configuration, input fingerprint, and hardware
fingerprint are skipped on rerun. Candidate stdout, stderr, raw responses, and
resource samples live under `results/artifacts/`.

The checked-in report records which inputs are synthetic fallback samples and
must not be interpreted as lecture recognition quality evidence.

The audio fallback is the JFK sample distributed by the official `whisper.cpp`
repository. Generated images contain rendered bilingual lecture-like text.

Large model files are intentionally ignored by Git. The pinned, resumable
download scripts are `scripts/download_sensevoice_assets.ps1` and
`scripts/download_paddleocr_vl_model.ps1`; Qwen3-ASR uses
`scripts/download_qwen3_asr_model.ps1`. Hugging Face's Xet client stalled
behind the measured host's local proxy, so these scripts use direct official
revision URLs and `curl -C -` rather than changing proxy or security settings.
