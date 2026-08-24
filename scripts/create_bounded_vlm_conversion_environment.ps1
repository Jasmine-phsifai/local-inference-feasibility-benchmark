$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentName = 'local-bench-bounded-vlm-conversion'
$environmentPython = "D:\Anaconda\envs\$environmentName\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$environmentFile = Join-Path $projectRoot 'environments\bounded_vlm_conversion\environment.yml'
$lockFile = Join-Path $projectRoot 'environments\bounded_vlm_conversion\requirements.lock.txt'

if (-not (Test-Path -LiteralPath $environmentPython -PathType Leaf)) {
    & $conda env create --file $environmentFile
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the bounded VLM conversion environment.'
    }
}

& $environmentPython -m pip install --disable-pip-version-check `
    --no-deps `
    --index-url 'https://pypi.org/simple' `
    --extra-index-url 'https://download.pytorch.org/whl/cpu' `
    --requirement $lockFile
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the exact bounded VLM conversion lock.'
}
& $environmentPython (Join-Path $PSScriptRoot 'verify_bounded_vlm_conversion_environment.py')
if ($LASTEXITCODE -ne 0) {
    throw 'Bounded VLM conversion environment verification failed.'
}
