$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentName = 'local-bench-paddleocr'
$python = "D:\Anaconda\envs\$environmentName\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$lock = Join-Path $projectRoot 'environments\paddleocr_cpu\requirements.lock.txt'

if (-not (Test-Path -LiteralPath $python)) {
    & $conda create --name $environmentName --yes python=3.11.15 pip=26.1.2 vcomp14=14.44.35208=h4927774_12
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the PaddleOCR environment.'
    }
}

& $conda install --name $environmentName --yes vcomp14=14.44.35208=h4927774_12
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the pinned Paddle/OpenMP native runtime.'
}

& $python -m pip install --timeout 120 --retries 8 --requirement $lock
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the locked PaddleOCR environment.'
}
& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'The PaddleOCR environment dependency check failed.'
}
& $python (Join-Path $PSScriptRoot 'download_paddleocr_assets.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to acquire the pinned PP-OCRv6 model files.'
}
& $python (Join-Path $PSScriptRoot 'verify_paddleocr_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw 'The PaddleOCR environment or model verification failed.'
}
