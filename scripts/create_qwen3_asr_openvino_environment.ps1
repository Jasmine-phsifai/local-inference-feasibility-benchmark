$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$manifest = Join-Path $projectRoot 'environments\qwen3_asr_openvino\requirements.txt'
$sourceEnvironment = 'local-bench-qwen3-asr'
$targetEnvironment = 'local-bench-qwen3-asr-openvino'
$targetPython = "D:\Anaconda\envs\$targetEnvironment\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'

if (-not (Test-Path -LiteralPath $targetPython)) {
    & $conda create --name $targetEnvironment --clone $sourceEnvironment --yes
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to clone the isolated Qwen3-ASR OpenVINO environment.'
    }
}
& $targetPython -m pip install `
    --timeout 120 `
    --retries 8 `
    --upgrade-strategy only-if-needed `
    --requirement $manifest
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the pinned Qwen3-ASR OpenVINO stack.'
}
& $targetPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'Qwen3-ASR OpenVINO environment dependency check failed.'
}
$environmentVerifier = Join-Path $projectRoot 'scripts\verify_qwen3_asr_openvino_environment.py'
& $targetPython $environmentVerifier
if ($LASTEXITCODE -ne 0) {
    throw 'Qwen3-ASR OpenVINO import verification failed.'
}
