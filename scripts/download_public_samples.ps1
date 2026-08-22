$ErrorActionPreference = 'Stop'
$destinationDirectory = Join-Path $PSScriptRoot '..\data\inputs\public'
$destination = Join-Path $destinationDirectory 'jfk.wav'
New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
curl.exe -L --fail --retry 3 --output $destination 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav'
$expectedHash = '59DFB9A4ACB36FE2A2AFFC14BACBEE2920FF435CB13CC314A08C13F66BA7860E'
if ((Get-Item -LiteralPath $destination).Length -ne 352078 -or (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $expectedHash) {
  throw 'Downloaded audio sample failed pinned size/hash verification.'
}
