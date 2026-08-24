$ErrorActionPreference = 'Stop'
$conda = 'D:\Anaconda\Scripts\conda.exe'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentFile = Join-Path $projectRoot 'environments\control\environment.yml'
$requirementsFile = Join-Path $projectRoot 'environments\control\requirements.lock.txt'
$environmentPython = 'D:\Anaconda\envs\local-bench-control\python.exe'

Push-Location $projectRoot
try {
  if (Test-Path -LiteralPath $environmentPython) {
    & $conda env update --file $environmentFile
  } else {
    & $conda env create --file $environmentFile
  }
  if ($LASTEXITCODE -ne 0) { throw 'Control Conda environment creation/update failed.' }
  & $environmentPython -m pip install --requirement $requirementsFile
  if ($LASTEXITCODE -ne 0) { throw 'Control lock installation failed.' }
  & $environmentPython -m pip install --editable $projectRoot --no-deps
  if ($LASTEXITCODE -ne 0) { throw 'Control project installation failed.' }
  & $environmentPython (Join-Path $projectRoot 'scripts\verify_control_environment.py')
  if ($LASTEXITCODE -ne 0) { throw 'Control environment verification failed.' }
} finally {
  Pop-Location
}
