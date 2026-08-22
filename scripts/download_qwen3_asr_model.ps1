$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$modelRoot = Join-Path $projectRoot 'data\models\qwen3-asr-0.6b-hf'
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
$revision = '7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c'
$files = @(
  'chat_template.jinja', 'config.json', 'generation_config.json',
  'model.safetensors', 'processor_config.json', 'tokenizer.json',
  'tokenizer_config.json'
)
$sizes = @{
  'chat_template.jinja'=1434; 'config.json'=2398; 'generation_config.json'=165;
  'model.safetensors'=1564928088; 'processor_config.json'=487;
  'tokenizer.json'=11429653; 'tokenizer_config.json'=998
}
foreach ($file in $files) {
  $destination = Join-Path $modelRoot $file
  if ((Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -eq $sizes[$file]) { continue }
  & curl.exe -L --fail --retry 8 --retry-all-errors -C - -o $destination "https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf/resolve/$revision/${file}?download=true"
  if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $destination).Length -ne $sizes[$file]) { throw "Qwen3-ASR download failed: $file" }
}
$modelHash = (Get-FileHash -LiteralPath (Join-Path $modelRoot 'model.safetensors') -Algorithm SHA256).Hash
if ($modelHash -ne 'D3F212DD20ABECD315D830BC54AE3865E56EBFC3276484E57B771288BA27FD35') { throw 'Qwen3-ASR model hash verification failed.' }
