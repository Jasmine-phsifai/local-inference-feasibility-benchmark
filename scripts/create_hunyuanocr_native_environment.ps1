$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$sourceEnvironment = 'local-bench-hunyuanocr'
$targetEnvironment = 'local-bench-hunyuanocr-native'
$sourcePython = "D:\Anaconda\envs\$sourceEnvironment\python.exe"
$targetPython = "D:\Anaconda\envs\$targetEnvironment\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$manifest = Join-Path $projectRoot 'environments\hunyuanocr_native\requirements.txt'

if (-not (Test-Path -LiteralPath $targetPython)) {
    if (Test-Path -LiteralPath $sourcePython) {
        & $conda create --name $targetEnvironment --clone $sourceEnvironment --yes
    }
    else {
        & $conda create --name $targetEnvironment --yes python=3.12 pip
    }
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the isolated native HunyuanOCR environment.'
    }
}

& $targetPython -m pip install --timeout 120 --retries 8 `
    --index-url 'https://download.pytorch.org/whl/cpu' `
    'torch==2.11.0+cpu' `
    'torchvision==0.26.0+cpu'
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the matched native Hunyuan Torch stack.'
}

& $targetPython -m pip install --timeout 120 --retries 8 --requirement $manifest
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the pinned native Hunyuan dependencies.'
}
& $targetPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'Native Hunyuan environment dependency check failed.'
}
& $targetPython (Join-Path $PSScriptRoot 'verify_hunyuanocr_native_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Native Hunyuan environment verification failed.'
}
