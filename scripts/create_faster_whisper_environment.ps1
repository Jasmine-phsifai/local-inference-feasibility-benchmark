$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentName = 'local-bench-faster-whisper'
$python = "D:\Anaconda\envs\$environmentName\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$lock = Join-Path $projectRoot 'environments\faster_whisper_cpu\requirements.lock.txt'

if (-not (Test-Path -LiteralPath $python)) {
    & $conda create --name $environmentName --yes python=3.11.15 pip=26.1.2
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the faster-whisper environment.'
    }
}

& $python -m pip install --timeout 120 --retries 8 --requirement $lock
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the locked faster-whisper environment.'
}
& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'The faster-whisper environment dependency check failed.'
}
$env:HF_HUB_DISABLE_XET = '1'
& $python (Join-Path $PSScriptRoot 'download_faster_whisper_assets.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to acquire the pinned faster-whisper model files.'
}
& $python (Join-Path $PSScriptRoot 'verify_faster_whisper_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw 'The faster-whisper environment or model verification failed.'
}
