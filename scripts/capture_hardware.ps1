$ErrorActionPreference = 'Stop'
$gpu = @(Get-CimInstance Win32_VideoController | ForEach-Object {
  [ordered]@{ name=$_.Name; driver_version=$_.DriverVersion; adapter_ram_bytes=[int64]$_.AdapterRAM; status=$_.Status }
})
$npu = @(Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object {
  $_.FriendlyName -match '(?i)\bNPU\b|Neural Processing|AI Boost'
} | ForEach-Object { [ordered]@{ name=$_.FriendlyName; status=$_.Status; instance_id=$_.InstanceId } })
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
$disk = Get-PSDrive -PSProvider FileSystem | ForEach-Object {
  [ordered]@{ name=$_.Name; used_bytes=[int64]$_.Used; free_bytes=[int64]$_.Free }
}
$record = [ordered]@{
  captured_at_utc=(Get-Date).ToUniversalTime().ToString('o')
  os=[ordered]@{ caption=$os.Caption; version=$os.Version; build=$os.BuildNumber }
  cpu=[ordered]@{ name=$cpu.Name; physical_cores=$cpu.NumberOfCores; logical_processors=$cpu.NumberOfLogicalProcessors }
  memory=[ordered]@{ total_bytes=[int64]$os.TotalVisibleMemorySize * 1024; available_bytes=[int64]$os.FreePhysicalMemory * 1024 }
  gpu=$gpu
  npu=$npu
  disks=@($disk)
}
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '..\results') | Out-Null
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot '..\results\hardware.json') -Encoding utf8
