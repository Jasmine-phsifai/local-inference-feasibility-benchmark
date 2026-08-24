$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$conda = 'D:\Anaconda\Scripts\conda.exe'
$environmentFile = Join-Path $projectRoot 'environments\sensevoice_official_runtime\environment.yml'
$environmentRoot = 'D:\Anaconda\envs\local-bench-sensevoice-official-runtime'
$runtime = Join-Path $environmentRoot 'Library\bin\vcomp140.dll'
$condaMeta = Join-Path $environmentRoot 'conda-meta\vs2015_runtime-14.42.34433-hbfb602d_5.json'
$expectedBytes = 192112
$expectedSha256 = 'E36A5C5E329BC7AF35D4FAA610A29AEEE826A7810E06712F0F54E9B2CFE6A728'

if (-not (Test-Path -LiteralPath $conda -PathType Leaf)) {
  throw 'Conda is required to create the pinned SenseVoice official runtime environment.'
}
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf) -or
    -not (Test-Path -LiteralPath $condaMeta -PathType Leaf)) {
  & $conda env update --file $environmentFile --prune
  if ($LASTEXITCODE -ne 0) {
    throw 'Pinned SenseVoice official runtime environment creation failed.'
  }
}
if (-not (Test-Path -LiteralPath $runtime -PathType Leaf) -or
    (Get-Item -LiteralPath $runtime).Length -ne $expectedBytes -or
    (Get-FileHash -LiteralPath $runtime -Algorithm SHA256).Hash -ne $expectedSha256 -or
    -not (Test-Path -LiteralPath $condaMeta -PathType Leaf)) {
  throw 'Pinned SenseVoice official VCOMP runtime changed.'
}

Write-Output 'Pinned SenseVoice official runtime environment is ready.'
