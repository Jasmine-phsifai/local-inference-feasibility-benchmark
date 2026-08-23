$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$sourceEnvironment = 'local-bench-rapidocr'
$targetEnvironment = 'local-bench-rapidocr-openvino'
$sourcePython = "D:\Anaconda\envs\$sourceEnvironment\python.exe"
$targetPython = "D:\Anaconda\envs\$targetEnvironment\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$manifest = Join-Path $projectRoot 'environments\rapidocr_openvino\requirements.txt'

if (-not (Test-Path -LiteralPath $targetPython)) {
    if (Test-Path -LiteralPath $sourcePython) {
        & $conda create --name $targetEnvironment --clone $sourceEnvironment --yes
    }
    else {
        & $conda create --name $targetEnvironment --yes python=3.11 pip
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the isolated RapidOCR OpenVINO environment.'
    }
}

& $targetPython -m pip install --timeout 120 --retries 8 --upgrade-strategy only-if-needed --requirement $manifest
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
