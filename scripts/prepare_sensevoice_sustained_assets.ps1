param(
  [ValidateRange(1, 8)]
  [int]$ParallelJobs = 4,
  [string]$CMakePath = '',
  [string]$NinjaPath = ''
)

$ErrorActionPreference = 'Stop'

& (Join-Path $PSScriptRoot 'create_sensevoice_official_runtime_environment.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Pinned SenseVoice official runtime setup failed.' }

& (Join-Path $PSScriptRoot 'create_sensevoice_build_environment.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Pinned SenseVoice build-tool setup failed.' }

& (Join-Path $PSScriptRoot 'download_sensevoice_assets.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Pinned SenseVoice asset download failed.' }

foreach ($runtimeVersion in @('0.1.9', '0.2.0')) {
  & (Join-Path $PSScriptRoot 'build_sensevoice_thread_runtime.ps1') `
    -RuntimeVersion $runtimeVersion `
    -ParallelJobs $ParallelJobs `
    -CMakePath $CMakePath `
    -NinjaPath $NinjaPath
  if ($LASTEXITCODE -ne 0) {
    throw "Thread-controlled SenseVoice $runtimeVersion build failed."
  }
}
