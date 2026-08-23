$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$modelRoot = Join-Path $projectRoot 'data\models\qwen3-asr-0.6b-original'
$attemptLog = Join-Path $modelRoot 'download-attempts.jsonl'
$revision = '5eb144179a02acc5e5ba31e748d22b0cf3e303b0'
$repository = 'Qwen/Qwen3-ASR-0.6B'
$expectedWeightHash = '79D6CBD4C98C7BBFFE9DB2EDAC07F56CD6637D0D5944B27F6C2B8353840323EA'
$files = [ordered]@{
    '.gitattributes' = 1519
    'README.md' = 57456
    'chat_template.json' = 1161
    'config.json' = 6193
    'generation_config.json' = 142
    'merges.txt' = 1671853
    'model.safetensors' = 1876091704
    'preprocessor_config.json' = 330
    'tokenizer_config.json' = 12487
    'vocab.json' = 2776833
}

New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
$startedUtc = [DateTimeOffset]::UtcNow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$downloadedBytes = 0L
$status = 'succeeded'
try {
    foreach ($entry in $files.GetEnumerator()) {
        $filename = $entry.Key
        $expectedBytes = [int64]$entry.Value
        $destination = Join-Path $modelRoot $filename
        if (Test-Path -LiteralPath $destination) {
            if ((Get-Item -LiteralPath $destination).Length -ne $expectedBytes) {
                throw "Existing Qwen3-ASR file has the wrong size: $filename"
            }
            continue
        }
        $partial = "$destination.part"
        $url = "https://huggingface.co/$repository/resolve/$revision/${filename}?download=true"
        & curl.exe `
            --fail `
            --location `
            --retry 8 `
            --retry-all-errors `
            --retry-delay 2 `
            --continue-at - `
            --output $partial `
            $url
        if ($LASTEXITCODE -ne 0) {
            throw "Qwen3-ASR download failed: $filename"
        }
        if ((Get-Item -LiteralPath $partial).Length -ne $expectedBytes) {
            throw "Qwen3-ASR file size mismatch: $filename"
        }
        Move-Item -LiteralPath $partial -Destination $destination
        $downloadedBytes += $expectedBytes
    }
    $weightHash = (Get-FileHash -LiteralPath (Join-Path $modelRoot 'model.safetensors') -Algorithm SHA256).Hash
    if ($weightHash -ne $expectedWeightHash) {
        throw 'Qwen3-ASR original weight hash verification failed.'
    }
    $totalBytes = ($files.Values | Measure-Object -Sum).Sum
    if ($totalBytes -ne 1880619678) {
        throw 'Pinned Qwen3-ASR inventory total is inconsistent.'
    }
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
        revision = $revision
        elapsed_seconds = $stopwatch.Elapsed.TotalSeconds
        downloaded_bytes_this_run = $downloadedBytes
    }
    Add-Content -LiteralPath $attemptLog -Value ($attempt | ConvertTo-Json -Compress) -Encoding UTF8
}

Write-Host "Verified Qwen3-ASR original checkpoint: 1880619678 bytes"
