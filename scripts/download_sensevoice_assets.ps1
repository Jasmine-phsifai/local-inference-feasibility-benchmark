$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$cpuRoot = Join-Path $projectRoot 'data\models\sensevoice'
$vulkanRoot = Join-Path $projectRoot 'data\models\sensevoice-vulkan'
New-Item -ItemType Directory -Force -Path $cpuRoot, $vulkanRoot | Out-Null

$release = 'runtime-llamacpp-v0.1.9'
$revision = '90c1c61912018b70ada0fcc024ea24aca62f2e63'
$vadRevision = '6840bae4c5c92ee8c04faaf4db23dd0105098d7f'
$cpuZip = Join-Path $cpuRoot 'runtime.zip'
$vulkanZip = Join-Path $vulkanRoot 'runtime.zip'
$modelFile = Join-Path $cpuRoot 'sensevoice-small-q8.gguf'
$vadFile = Join-Path $cpuRoot 'fsmn-vad.gguf'
function Get-PinnedFile([string]$Destination, [int64]$ExpectedBytes, [string]$Url, [string]$ExpectedSha256 = '') {
  if (Test-Path -LiteralPath $Destination) {
    if ((Get-Item -LiteralPath $Destination).Length -ne $ExpectedBytes) { throw "Existing pinned asset has the wrong size: $Destination" }
    if ($ExpectedSha256 -and (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $ExpectedSha256) { throw "Existing pinned asset has the wrong hash: $Destination" }
    return
  }
  $partial = "$Destination.part"
  & curl.exe -L --fail --retry 8 --retry-all-errors -C - -o $partial $Url
  if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $partial).Length -ne $ExpectedBytes) { throw "Pinned download failed: $Destination" }
  if ($ExpectedSha256 -and (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $ExpectedSha256) { throw "Pinned download hash verification failed: $Destination" }
  Move-Item -LiteralPath $partial -Destination $Destination
}
Get-PinnedFile $cpuZip 4917274 "https://github.com/QwenAudio/SenseVoice/releases/download/$release/funasr-llamacpp-windows-x64-avx2.zip"
Get-PinnedFile $vulkanZip 21849192 "https://github.com/QwenAudio/SenseVoice/releases/download/$release/funasr-llamacpp-windows-x64-vulkan.zip"
Get-PinnedFile $modelFile 254208320 "https://huggingface.co/QwenAudio/SenseVoiceSmall-GGUF/resolve/$revision/sensevoice-small-q8.gguf?download=true" '4AE45C94422DE949B387E2E0FB10D7E14E4C42C69DB30C3444ECC7D4B844B7C5'
Get-PinnedFile $vadFile 1720512 "https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF/resolve/$vadRevision/fsmn-vad.gguf?download=true" '1270F2559C495F4E7B6E739541151027D360761A3FDA43FC147034F5719F5479'
Expand-Archive -LiteralPath $cpuZip -DestinationPath $cpuRoot -Force
Expand-Archive -LiteralPath $vulkanZip -DestinationPath $vulkanRoot -Force
