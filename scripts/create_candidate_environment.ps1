param([Parameter(Mandatory = $true)][string]$CandidateId)
$ErrorActionPreference = 'Stop'
$conda = 'D:\Anaconda\Scripts\conda.exe'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$registry = Get-Content -LiteralPath (Join-Path $projectRoot 'registries\candidates.json') -Raw | ConvertFrom-Json
$candidate = $registry.candidates | Where-Object id -eq $CandidateId | Select-Object -First 1
if (-not $candidate) { throw "Unknown candidate: $CandidateId" }
if ($candidate.setup_script) {
  $setupScript = Join-Path $projectRoot $candidate.setup_script
  if (-not (Test-Path -LiteralPath $setupScript -PathType Leaf)) {
    throw "Candidate setup script is missing: $setupScript"
  }
  & $setupScript
  if ($LASTEXITCODE -ne 0) { throw "Candidate setup failed: $setupScript" }
  return
}
if (-not $candidate.manifest) { throw "Candidate $CandidateId has no Python environment manifest." }
$manifest = Join-Path $projectRoot $candidate.manifest
if (-not (Test-Path -LiteralPath $manifest)) { throw "No manifest for $CandidateId at $manifest" }
$envName = $candidate.environment
& $conda create --name $envName --yes python=3.11 pip
if ($LASTEXITCODE -ne 0) {
  Write-Host "Environment $envName already exists or create failed; attempting the pinned install in the existing environment."
}
foreach ($condaPackage in @($candidate.conda_packages | Where-Object { $_ })) {
  & $conda install --name $envName --yes $condaPackage
  if ($LASTEXITCODE -ne 0) { throw "Conda package install failed: $condaPackage" }
}
& $conda run --name $envName python -m pip install --timeout 60 --retries 8 --requirement $manifest
if ($LASTEXITCODE -ne 0) { throw "Pip manifest install failed: $manifest" }
