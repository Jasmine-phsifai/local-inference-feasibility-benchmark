$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$targetPython = 'D:\Anaconda\envs\local-bench-qwen3-asr-openvino-genai-official\python.exe'

& (Join-Path $PSScriptRoot 'create_qwen3_asr_openvino_genai_official_environment.ps1')
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to prepare the official OpenVINO GenAI environment.'
}
& (Join-Path $PSScriptRoot 'download_qwen3_asr_original_model.ps1')
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to acquire the pinned original Qwen3-ASR checkpoint.'
}
& $targetPython (Join-Path $PSScriptRoot 'export_qwen3_asr_openvino_genai_official.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to create or verify the official OpenVINO GenAI export.'
}
& $targetPython (Join-Path $PSScriptRoot 'verify_qwen3_asr_openvino_genai_official_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Official OpenVINO GenAI environment verification failed.'
}
