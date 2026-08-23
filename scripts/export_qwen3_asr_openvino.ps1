$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$pythonRoot = 'D:\Anaconda\envs\local-bench-qwen3-asr-openvino'
$optimumCli = Join-Path $pythonRoot 'Scripts\optimum-cli.exe'
$sourceModel = Join-Path $projectRoot 'data\models\qwen3-asr-0.6b-original'
$outputModel = Join-Path $projectRoot 'data\models\qwen3-asr-0.6b-openvino-fp16'
$partialModel = "$outputModel.partial"
$attemptLog = Join-Path $sourceModel 'openvino-export-attempts.jsonl'
$marker = Join-Path $outputModel 'export-complete.json'
$revision = '5eb144179a02acc5e5ba31e748d22b0cf3e303b0'

if (Test-Path -LiteralPath $marker) {
    Write-Host "OpenVINO FP16 export already complete: $outputModel"
    exit 0
}
if (Test-Path -LiteralPath $outputModel) {
    throw 'OpenVINO export directory exists without its completion marker; preserved for inspection.'
}
if (Test-Path -LiteralPath $partialModel) {
    throw 'OpenVINO partial export exists; preserved for inspection.'
}
if (-not (Test-Path -LiteralPath $optimumCli)) {
    throw 'Pinned Optimum CLI is missing; create the OpenVINO environment first.'
}

$startedUtc = [DateTimeOffset]::UtcNow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$status = 'succeeded'
try {
    & $optimumCli export openvino `
        --model $sourceModel `
        --task automatic-speech-recognition `
        --trust-remote-code `
        --weight-format fp16 `
        $partialModel
    if ($LASTEXITCODE -ne 0) {
        throw 'Qwen3-ASR OpenVINO FP16 export failed.'
    }
    $xmlFiles = @(Get-ChildItem -LiteralPath $partialModel -Recurse -File -Filter '*.xml')
    $binFiles = @(Get-ChildItem -LiteralPath $partialModel -Recurse -File -Filter '*.bin')
    if (-not $xmlFiles -or -not $binFiles) {
        throw 'OpenVINO export produced no complete XML/BIN model pair.'
    }
    $complete = [ordered]@{
        source_revision = $revision
        weight_format = 'fp16'
        completed_utc = [DateTimeOffset]::UtcNow.ToString('o')
        elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
        xml_file_count = $xmlFiles.Count
        bin_file_count = $binFiles.Count
    }
    $complete | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $partialModel 'export-complete.json') -Encoding UTF8
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
        source_revision = $revision
        elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
    }
    Add-Content -LiteralPath $attemptLog -Value ($attempt | ConvertTo-Json -Compress) -Encoding UTF8
}

Write-Host "Verified OpenVINO FP16 export: $outputModel"
