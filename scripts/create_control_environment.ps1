$ErrorActionPreference = 'Stop'
$conda = 'D:\Anaconda\Scripts\conda.exe'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentFile = Join-Path $projectRoot 'environments\control\environment.yml'
$environmentPython = 'D:\Anaconda\envs\local-bench-control\python.exe'

Push-Location $projectRoot
try {
  if (Test-Path -LiteralPath $environmentPython) {
    & $conda env update --file $environmentFile
  } else {
    & $conda env create --file $environmentFile
  }
  if ($LASTEXITCODE -ne 0) { throw 'Control Conda environment creation/update failed.' }
  & $conda run --name local-bench-control python -m pip install --editable '.[test]'
  if ($LASTEXITCODE -ne 0) { throw 'Control project installation failed.' }
} finally {
  Pop-Location
}
