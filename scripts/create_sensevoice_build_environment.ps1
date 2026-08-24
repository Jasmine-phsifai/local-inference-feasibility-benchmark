$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$conda = 'D:\Anaconda\Scripts\conda.exe'
$environmentFile = Join-Path $projectRoot 'environments\sensevoice_build\environment.yml'
$environmentRoot = 'D:\Anaconda\envs\local-bench-sensevoice-build'
$cmake = Join-Path $environmentRoot 'Library\bin\cmake.exe'
$ninja = Join-Path $environmentRoot 'Library\bin\ninja.exe'

if (-not (Test-Path -LiteralPath $conda -PathType Leaf)) {
  throw 'Conda is required to create the pinned SenseVoice build environment.'
}
if (-not (Test-Path -LiteralPath $cmake -PathType Leaf) -or
    -not (Test-Path -LiteralPath $ninja -PathType Leaf)) {
  & $conda env update --file $environmentFile --prune
  if ($LASTEXITCODE -ne 0) {
    throw 'Pinned SenseVoice build environment creation failed.'
  }
}

$cmakeVersion = (& $cmake --version | Select-Object -First 1)
if ($LASTEXITCODE -ne 0) { throw 'Pinned CMake version probe failed.' }
$ninjaVersion = (& $ninja --version).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Pinned Ninja version probe failed.' }
if ($cmakeVersion -ne 'cmake version 4.2.3' -or $ninjaVersion -ne '1.13.1') {
  throw 'Pinned SenseVoice build-tool versions changed.'
}
Write-Output 'Pinned SenseVoice build environment is ready.'
