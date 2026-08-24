$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$targetEnvironment = 'local-bench-rapidocr-openvino'
$targetPython = "D:\Anaconda\envs\$targetEnvironment\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$manifest = Join-Path $projectRoot 'environments\rapidocr_openvino\requirements.lock.txt'

if (-not (Test-Path -LiteralPath $targetPython)) {
    & $conda create --name $targetEnvironment --yes python=3.11.15 pip=26.1.2
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the isolated RapidOCR OpenVINO environment.'
    }
}

& $targetPython -m pip install --timeout 120 --retries 8 --requirement $manifest
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the pinned RapidOCR OpenVINO stack.'
}
& $targetPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'RapidOCR OpenVINO environment dependency check failed.'
}
& $targetPython (Join-Path $PSScriptRoot 'verify_rapidocr_openvino_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw 'RapidOCR OpenVINO environment verification failed.'
}
