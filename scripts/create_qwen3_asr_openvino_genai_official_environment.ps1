$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentName = 'local-bench-qwen3-asr-openvino-genai-official'
$targetPrefix = "D:\Anaconda\envs\$environmentName"
$targetPython = Join-Path $targetPrefix 'python.exe'
$conda = 'D:\Anaconda\Scripts\conda.exe'
$manifest = Join-Path $projectRoot 'environments\qwen3_asr_openvino_genai_official\requirements.lock.txt'
$verifier = Join-Path $PSScriptRoot 'verify_qwen3_asr_openvino_genai_official_environment.py'
$optimumIntelRevision = 'f48d93fddff8c91e198389c47a6d5974789b67f4'

if (Test-Path -LiteralPath $targetPython) {
    & $targetPython $verifier
    if ($LASTEXITCODE -eq 0) {
        Write-Output 'The existing official OpenVINO GenAI environment is verified.'
        return
    }
    throw 'The existing official OpenVINO GenAI environment is incompatible; refusing in-place mutation.'
}

if (-not (Test-Path -LiteralPath $conda -PathType Leaf)) {
    throw 'Conda was not found at the benchmark-pinned installation.'
}
& $conda create --prefix $targetPrefix --yes python=3.11.15 pip=26.1.2
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to create the official OpenVINO GenAI Qwen3-ASR environment.'
}

$git = Get-Command git.exe -ErrorAction SilentlyContinue
if ($null -eq $git) {
    $fallbackGit = 'C:\Program Files\Git\cmd\git.exe'
    if (-not (Test-Path -LiteralPath $fallbackGit -PathType Leaf)) {
        throw 'Git is required to install the pinned Optimum Intel exporter.'
    }
    $gitDirectory = Split-Path -Parent $fallbackGit
} else {
    $gitDirectory = Split-Path -Parent $git.Source
}

& $targetPython -m pip install --timeout 120 --retries 8 --requirement $manifest
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install official OpenVINO GenAI Qwen3-ASR dependencies.'
}

$env:Path = "$gitDirectory;$env:Path"
$optimumIntelSource = (
    'optimum-intel @ git+https://github.com/huggingface/optimum-intel.git@' +
    $optimumIntelRevision
)
& $targetPython -m pip install --timeout 120 --retries 8 `
    --no-deps --force-reinstall --no-build-isolation $optimumIntelSource
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the pinned official Optimum Intel exporter revision.'
}

& $targetPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'Official OpenVINO GenAI Qwen3-ASR dependency check failed.'
}
& $targetPython $verifier
if ($LASTEXITCODE -ne 0) {
    throw 'Official OpenVINO GenAI Qwen3-ASR environment verification failed.'
}
