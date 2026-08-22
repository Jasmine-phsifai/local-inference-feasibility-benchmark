$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$modelRoot = Join-Path $projectRoot 'data\models\paddleocr-vl-1.6'
New-Item -ItemType Directory -Force -Path $modelRoot | Out-Null
$revision = 'c5630abae1d940eafe0697512a0325494b02ab42'
$files = @(
  'added_tokens.json', 'chat_template.jinja', 'config.json',
  'configuration_paddleocr_vl.py', 'generation_config.json',
  'image_processing_paddleocr_vl.py', 'inference.yml', 'model.safetensors',
  'modeling_paddleocr_vl.py', 'preprocessor_config.json',
  'processing_paddleocr_vl.py', 'processor_config.json',
  'special_tokens_map.json', 'tokenizer.json', 'tokenizer.model',
  'tokenizer_config.json'
)
$sizes = @{
  'added_tokens.json'=25381; 'chat_template.jinja'=1474; 'config.json'=2059;
  'configuration_paddleocr_vl.py'=8104; 'generation_config.json'=133;
  'image_processing_paddleocr_vl.py'=25032; 'inference.yml'=43;
  'model.safetensors'=1917255968; 'modeling_paddleocr_vl.py'=103889;
  'preprocessor_config.json'=641; 'processing_paddleocr_vl.py'=12253;
  'processor_config.json'=137; 'special_tokens_map.json'=1151;
  'tokenizer.json'=11189060; 'tokenizer.model'=1614363;
  'tokenizer_config.json'=186947
}
foreach ($file in $files) {
  $destination = Join-Path $modelRoot $file
  if ((Test-Path -LiteralPath $destination) -and (Get-Item -LiteralPath $destination).Length -eq $sizes[$file]) { continue }
  & curl.exe -L --fail --retry 8 --retry-all-errors -C - -o $destination "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6/resolve/$revision/${file}?download=true"
  if ($LASTEXITCODE -ne 0 -or (Get-Item -LiteralPath $destination).Length -ne $sizes[$file]) { throw "PaddleOCR-VL download failed: $file" }
}
$modelHash = (Get-FileHash -LiteralPath (Join-Path $modelRoot 'model.safetensors') -Algorithm SHA256).Hash
if ($modelHash -ne '85A479D506A11E724E7285D395C551BE69F41DBC16B6342D3CACFB189AED71DB') { throw 'PaddleOCR-VL model hash verification failed.' }
