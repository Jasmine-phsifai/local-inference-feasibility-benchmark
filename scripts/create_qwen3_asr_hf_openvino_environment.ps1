$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentName = 'local-bench-qwen3-asr-hf-openvino'
$targetPython = "D:\Anaconda\envs\$environmentName\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$manifest = Join-Path $projectRoot 'environments\qwen3_asr_hf_openvino\requirements.txt'
$gitDirectory = 'C:\Program Files\Git\cmd'
$optimumIntelRevision = '4ca1144eafc3ef7d3d805a99c7b92953441437e5'

if (-not (Test-Path -LiteralPath $targetPython)) {
    & $conda create --name $environmentName --yes python=3.11 pip
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the isolated HF-native Qwen3-ASR OpenVINO environment.'
    }
}

& $targetPython -m pip install --timeout 120 --retries 8 `
    --index-url 'https://download.pytorch.org/whl/cpu' `
    'torch==2.11.0+cpu' `
    'torchaudio==2.11.0+cpu'
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the matched CPU Torch and Torchaudio pair.'
}

& $targetPython -m pip install --timeout 120 --retries 8 --requirement $manifest
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install pinned HF-native Qwen3-ASR OpenVINO dependencies.'
}

$env:Path = "$gitDirectory;$env:Path"
$optimumIntelSource = (
    'optimum-intel @ git+https://github.com/openvino-dev-samples/' +
    "optimum-intel.git@$optimumIntelRevision"
)
& $targetPython -m pip install --timeout 120 --retries 8 --no-deps --force-reinstall $optimumIntelSource
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the immutable HF-native Optimum Intel revision.'
}

& $targetPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'HF-native Qwen3-ASR OpenVINO dependency check failed.'
}
& $targetPython (Join-Path $PSScriptRoot 'verify_qwen3_asr_hf_openvino_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw 'HF-native Qwen3-ASR OpenVINO environment verification failed.'
}
