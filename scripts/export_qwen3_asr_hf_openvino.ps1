$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$pythonRoot = 'D:\Anaconda\envs\local-bench-qwen3-asr-hf-openvino'
$python = Join-Path $pythonRoot 'python.exe'
$optimumCli = Join-Path $pythonRoot 'Scripts\optimum-cli.exe'
$sourceModel = Join-Path $projectRoot 'data\models\qwen3-asr-0.6b-hf'
$outputModel = Join-Path $projectRoot 'data\models\qwen3-asr-0.6b-hf-openvino-with-past'
$partialModel = "$outputModel.partial"
$attemptLog = Join-Path $sourceModel 'openvino-hf-export-attempts.jsonl'
$marker = Join-Path $outputModel 'export-complete.json'
$environmentVerifier = Join-Path $projectRoot 'scripts\verify_qwen3_asr_hf_openvino_environment.py'
$manifestTool = Join-Path $projectRoot 'workers\qwen3_asr_hf_openvino_export_manifest.py'
$sourceRevision = '7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c'
$optimumIntelRevision = '4ca1144eafc3ef7d3d805a99c7b92953441437e5'
$env:CI = 'true'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

if (-not (Test-Path -LiteralPath $python)) {
    throw 'Pinned HF-native Qwen3-ASR Python is missing; create its environment first.'
}
& $python $environmentVerifier
if ($LASTEXITCODE -ne 0) {
    throw 'Pinned HF-native Qwen3-ASR environment verification failed.'
}
& $python $manifestTool verify-source --source-model $sourceModel
if ($LASTEXITCODE -ne 0) {
    throw 'Pinned HF-native Qwen3-ASR source verification failed.'
}

if (Test-Path -LiteralPath $marker) {
    & $python $manifestTool verify-export --source-model $sourceModel --export-model $outputModel
    if ($LASTEXITCODE -ne 0) {
        throw 'Existing HF-native OpenVINO export failed artifact verification.'
    }
    Write-Host "HF-native OpenVINO export already complete: $outputModel"
    exit 0
}
if (Test-Path -LiteralPath $outputModel) {
    throw 'HF-native OpenVINO export directory exists without its completion marker.'
}
if (Test-Path -LiteralPath $partialModel) {
    throw 'HF-native OpenVINO partial export exists; preserved for inspection.'
}
if (-not (Test-Path -LiteralPath $optimumCli)) {
    throw 'Pinned HF-native Optimum CLI is missing; create its environment first.'
}
$startedUtc = [DateTimeOffset]::UtcNow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$status = 'succeeded'
try {
    & $optimumCli export openvino `
        --model $sourceModel `
        --task automatic-speech-recognition-with-past `
        $partialModel
    if ($LASTEXITCODE -ne 0) {
        throw 'HF-native Qwen3-ASR OpenVINO export failed.'
    }
    & $python $manifestTool write-marker --source-model $sourceModel --export-model $partialModel
    if ($LASTEXITCODE -ne 0) {
        throw 'HF-native Qwen3-ASR OpenVINO export manifest creation failed.'
    }
    & $python $manifestTool verify-export --source-model $sourceModel --export-model $partialModel
    if ($LASTEXITCODE -ne 0) {
        throw 'HF-native Qwen3-ASR OpenVINO export artifact verification failed.'
    }
    Move-Item -LiteralPath $partialModel -Destination $outputModel
}
catch {
    $status = 'failed'
    throw
}
finally {
    $stopwatch.Stop()
    $attempt = [ordered]@{
        timestamp_utc = $startedUtc.ToString('o')
        status = $status
        source_revision = $sourceRevision
        optimum_intel_revision = $optimumIntelRevision
        elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
    }
    Add-Content -LiteralPath $attemptLog -Value ($attempt | ConvertTo-Json -Compress) -Encoding UTF8
}

Write-Host "Verified HF-native Qwen3-ASR OpenVINO export: $outputModel"
