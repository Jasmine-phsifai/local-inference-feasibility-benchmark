$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$cpuRoot = Join-Path $projectRoot 'data\models\sensevoice'
$vulkanRoot = Join-Path $projectRoot 'data\models\sensevoice-vulkan'
New-Item -ItemType Directory -Force -Path $cpuRoot, $vulkanRoot | Out-Null

$release = 'runtime-llamacpp-v0.1.9'
$revision = '90c1c61912018b70ada0fcc024ea24aca62f2e63'
$cpuZip = Join-Path $cpuRoot 'runtime.zip'
$vulkanZip = Join-Path $vulkanRoot 'runtime.zip'
$modelFile = Join-Path $cpuRoot 'sensevoice-small-q8.gguf'
function Get-PinnedFile([string]$Destination, [int64]$ExpectedBytes, [string]$Url) {
  if ((Test-Path -LiteralPath $Destination) -and (Get-Item -LiteralPath $Destination).Length -eq $ExpectedBytes) { return }
  & curl.exe -L --fail --retry 8 --retry-all-errors -C - -o $Destination $Url
  if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $Destination).Length -ne $ExpectedBytes) { throw "Pinned download failed: $Destination" }
}
Get-PinnedFile $cpuZip 4917274 "https://github.com/QwenAudio/SenseVoice/releases/download/$release/funasr-llamacpp-windows-x64-avx2.zip"
Get-PinnedFile $vulkanZip 21849192 "https://github.com/QwenAudio/SenseVoice/releases/download/$release/funasr-llamacpp-windows-x64-vulkan.zip"
Get-PinnedFile $modelFile 254208320 "https://huggingface.co/QwenAudio/SenseVoiceSmall-GGUF/resolve/$revision/sensevoice-small-q8.gguf?download=true"
if ((Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash -ne '4AE45C94422DE949B387E2E0FB10D7E14E4C42C69DB30C3444ECC7D4B844B7C5') { throw 'SenseVoice model hash verification failed.' }
Expand-Archive -LiteralPath $cpuZip -DestinationPath $cpuRoot -Force
Expand-Archive -LiteralPath $vulkanZip -DestinationPath $vulkanRoot -Force
