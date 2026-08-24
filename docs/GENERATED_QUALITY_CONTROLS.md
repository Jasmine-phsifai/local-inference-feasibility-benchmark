# Generated quality controls

These controls are invented or revision-pinned public inputs. They are separate
from private course samples and do not depend on the crawler.

## Prepare one bounded suite

Create and verify the exact control environment first:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\create_control_environment.ps1
$benchmarkPython = 'D:\Anaconda\envs\local-bench-control\python.exe'
```

Preparation requires an explicit suite. Repeat `--suite` only when multiple
suites are intentionally requested:

```powershell
& $benchmarkPython .\scripts\prepare_generated_quality_controls.py --suite ocr
& $benchmarkPython .\scripts\prepare_generated_quality_controls.py --suite document-fidelity
& $benchmarkPython .\scripts\prepare_generated_quality_controls.py --suite asr
```

The OCR and document-fidelity producers render into a temporary directory,
verify the result, and only then copy it into `data/inputs/generated`. Before
promotion, any output pinned as a tracked bounded-VLM fixture must exactly match
the byte count and SHA-256 in `registries/bounded_vlm_b10598_assets.json`. This
prevents a different font or renderer from silently replacing committed public
fixtures. The ASR producer writes directly to its ignored output directory so
its two verified FLEURS archive downloads remain resumable.

Preparation never reads `data/inputs/local`, private workloads, downloader
output, or crawler state.

## Verify without regenerating

The verifier checks the manifest schema, bounded item count, safe relative
paths, exact reference hashes, RGB PNG decoding and dimensions, or PCM WAV
format and duration. It prints aggregate identity only, never reference text.

```powershell
& $benchmarkPython .\scripts\verify_generated_quality_controls.py --suite ocr
& $benchmarkPython .\scripts\verify_generated_quality_controls.py --suite document-fidelity
& $benchmarkPython .\scripts\verify_generated_quality_controls.py --suite asr
& $benchmarkPython .\scripts\verify_generated_quality_controls.py --suite hunyuan-vlm-fixture
& $benchmarkPython .\scripts\verify_generated_quality_controls.py --suite ovis-vlm-fixture
```

For a regeneration host, add `--verify-regeneration-host`. OCR manifests declare
the exact font-file hashes, and document-fidelity plus current OCR manifests
also declare the exact Pillow version. Media verification establishes that the
files match their manifest. It does not make an ignored manifest an external
trust anchor; benchmark attempt provenance must freeze the manifest and media
together. The bounded VLM registry is the external hash authority for its
tracked subset.

## Tracked and ignored state

The full generated suites remain ignored because they are reproducible local
benchmark inputs and the ASR suite includes large downloaded archives and WAVs.
Only the following small public bounded-VLM subset is intended to be tracked:

- OCR: `warmup.png`, `code_formula.png`, `dense_table.png`,
  `negative_diagram.png`, and `hunyuan_doc_quality.json`.
- Document fidelity: `page_008_table_columns.png` and
  `ovisocr2_page_quality.json`.
- ASR: no generated or downloaded media is tracked.

The full `ocr_quality/manifest.json`, `document_fidelity/manifest.json`, all
other rendered images, FLEURS archives, extracted source clips, SAPI clips, and
assembled ASR controls remain ignored. Do not force-add them. The seven intended
tracked files contain invented text only; they contain no lecture frames,
transcripts, names, source paths, or private metadata.

## Reproduction limits

- OCR rendering is byte-reproducible only when the declared Windows font files
  and the locked Pillow build match. Font filenames alone are insufficient;
  rely on the manifest hashes and the post-render media hashes.
- The ASR public clips come from `google/fleurs` at the exact revision and
  archive hashes declared in `prepare_asr_quality_controls.py`. FLEURS is used
  under CC-BY-4.0.
- Microsoft David Desktop and Microsoft Zira Desktop are Windows SAPI host
  dependencies. A voice name does not identify its installed binary, language
  data, OS servicing level, or synthesis behavior. Some Windows editions do not
  install these voices.
- The ASR producer resolves ffmpeg from its active environment or `PATH`. Its
  build is not pinned, and resampling output can differ across builds.
- Therefore SAPI and ffmpeg regeneration is verified by the resulting WAV
  hashes, not claimed byte-identical across arbitrary Windows hosts. Preserve
  the generated manifest used by an attempt and treat SAPI speech as a synthetic
  diagnostic, not unquestionable ground truth.
- Explicit LF JSON output removes Windows-versus-POSIX newline drift from newly
  generated manifests and OCR subset manifests.
