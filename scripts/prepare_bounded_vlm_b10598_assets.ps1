param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('ovisocr2_q8_cpu', 'hunyuanocr_1_5_gguf_cpu', 'all')]
    [string]$CandidateId,
    [switch]$VerifyOnly
)
$ErrorActionPreference = 'Stop'

$controlPython = 'D:\Anaconda\envs\local-bench-control\python.exe'
$conversionPython = 'D:\Anaconda\envs\local-bench-bounded-vlm-conversion\python.exe'
$preparer = Join-Path $PSScriptRoot 'prepare_bounded_vlm_b10598_assets.py'

if (-not (Test-Path -LiteralPath $controlPython -PathType Leaf)) {
    throw 'The exact control environment is missing; run create_control_environment.ps1.'
}
if (-not $VerifyOnly -and $CandidateId -in @('hunyuanocr_1_5_gguf_cpu', 'all')) {
    & (Join-Path $PSScriptRoot 'create_bounded_vlm_conversion_environment.ps1')
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to prepare the Hunyuan GGUF conversion environment.'
    }
}

$arguments = @($preparer, '--candidate', $CandidateId)
if ($VerifyOnly) {
    $arguments += '--verify-only'
}
elseif ($CandidateId -in @('hunyuanocr_1_5_gguf_cpu', 'all')) {
    $arguments += @('--conversion-python', $conversionPython)
}
& $controlPython @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Bounded VLM asset preparation failed for $CandidateId."
}
