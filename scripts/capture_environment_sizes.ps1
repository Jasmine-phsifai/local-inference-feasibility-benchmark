$ErrorActionPreference = 'Stop'
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$registry = Get-Content -LiteralPath (Join-Path $projectRoot 'registries\candidates.json') -Raw | ConvertFrom-Json
$names = @('local-bench-control') + @($registry.candidates.environment)
$records = foreach ($name in ($names | Where-Object { $_ -and $_ -notmatch 'external' } | Sort-Object -Unique)) {
  $path = Join-Path 'D:\Anaconda\envs' $name
  if (Test-Path -LiteralPath $path) {
    $bytes = (Get-ChildItem -LiteralPath $path -File -Recurse | Measure-Object -Property Length -Sum).Sum
    [ordered]@{ environment = $name; path = $path; apparent_bytes = [int64]$bytes }
  }
}
[ordered]@{
  captured_at_utc = (Get-Date).ToUniversalTime().ToString('o')
  note = 'Apparent file sizes; Conda hard links may reduce physical incremental disk use.'
  environments = @($records)
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $projectRoot 'results\environment-sizes.json') -Encoding utf8
