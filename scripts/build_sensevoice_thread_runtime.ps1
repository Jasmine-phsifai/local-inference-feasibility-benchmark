param(
  [ValidateRange(1, 8)]
  [int]$ParallelJobs = 4,
  [ValidateSet('0.1.9', '0.2.0')]
  [string]$RuntimeVersion = '0.1.9',
  [string]$CMakePath = '',
  [string]$NinjaPath = ''
)

$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
if ($RuntimeVersion -eq '0.1.9') {
  $buildDirectory = 'sensevoice-runtime-v0.1.9-thread-build'
  $sourceRepository = 'https://github.com/QwenAudio/SenseVoice.git'
  $sourceTag = 'runtime-llamacpp-v0.1.9'
  $sourceTagObject = '8f07e2f3624a1340bbfde5f5ddd5022ea37862d2'
  $sourceRevision = '73ccdd3577db37e92dbf22a4a9fc323b038cf13b'
  $llamaRevision = '8086439a4cea94c71a5dfb8fe4ad1546aebd640f'
  $llamaDirectory = 'llama-pinned-8086439'
  $patchRelativePath = 'patches\sensevoice-runtime-v0.1.9-thread-option.patch'
  $patchSha256 = '151D1E42490B1A0BFF7103308B3A7F424699AC38CE2A6918891F1D6DF25D52F8'
  $patchedFile = 'runtime/llama.cpp/funasr-sensevoice/funasr-sensevoice.cpp'
  $provenanceRelativePath = 'results\artifacts\sensevoice-thread-build\provenance.json'
} else {
  $buildDirectory = 'sensevoice-runtime-v0.2.0-thread-build'
  $sourceRepository = 'https://github.com/modelscope/FunASR.git'
  $sourceTag = 'runtime-llamacpp-v0.2.0'
  $sourceTagObject = 'ef27da6d5332d801ede62dcba9811151d1b936ce'
  $sourceRevision = '500956bc331bb7edbaac58d8f84a84f28bd3d29f'
  $llamaRevision = '803b7fcae893e9caaee3921779628fef83ac0965'
  $llamaDirectory = 'llama-pinned-803b7fca'
  $patchRelativePath = 'patches\sensevoice-runtime-v0.2.0-thread-option.patch'
  $patchSha256 = '3A6831762C5379CD37B5CF64435396E6AB78FA91F17FA306AD64F635A0CDD194'
  $patchedFile = 'runtime/llama.cpp/sensevoice/funasr-sensevoice/funasr-sensevoice.cpp'
  $provenanceRelativePath = 'results\artifacts\sensevoice-v020-thread-build\provenance.json'
}
$buildRoot = Join-Path $projectRoot "data\models\$buildDirectory"
$sourceRoot = Join-Path $buildRoot 'source'
$llamaRoot = Join-Path $buildRoot $llamaDirectory
$outputRoot = Join-Path $buildRoot 'build-pinned'
$binaryRoot = Join-Path $outputRoot 'bin'
$toolchainRoot = Join-Path $projectRoot 'data\models\sensevoice-build-toolchain'
$toolchainArchive = Join-Path $toolchainRoot 'llvm-mingw-20260616-ucrt-x86_64.zip'
$toolchain = Join-Path $toolchainRoot 'llvm-mingw-20260616-ucrt-x86_64'
$toolchainBin = Join-Path $toolchain 'bin'
$patch = Join-Path $projectRoot $patchRelativePath
$provenance = Join-Path $projectRoot $provenanceRelativePath
$verifier = Join-Path $PSScriptRoot 'verify_sensevoice_thread_runtime.py'
$controlPython = 'D:\Anaconda\envs\local-bench-control\python.exe'
$archiveSha256 = 'B9B68A4D276E16FA25802AABA458E4638F64B3884C290AACCDC2D87083B6CA35'
$archiveBytes = 187504083
$archiveUrl = 'https://github.com/mstorsjo/llvm-mingw/releases/download/20260616/llvm-mingw-20260616-ucrt-x86_64.zip'
$cmakeSha256 = 'C6BB52C7F9DC1728F3764A877135C282E1873DB04A34B0F025110D5412043F38'
$ninjaSha256 = '704E86746F0D63DECD9412A1CEA01335F92848FC77525538CE0C97449F2E7993'
$clangDriverSha256 = 'A8B7A614EEADD9105F814BE3701A7F312CDA4CEA51751B75B408C16100C94E85'
$clangRealSha256 = '870A1393B07B9DEBF407CFDAD19F6D4680DA32A213C33D0A6EB6AE7E8C8171A1'
$bundleNames = @('llama-funasr-sensevoice.exe', 'libc++.dll', 'libomp.dll', 'libunwind.dll')

function Assert-LastExitCode([string]$Message) {
  if ($LASTEXITCODE -ne 0) { throw $Message }
}

function Get-PinnedFile(
  [string]$Destination,
  [int64]$ExpectedBytes,
  [string]$ExpectedSha256,
  [string]$Url
) {
  if (Test-Path -LiteralPath $Destination -PathType Leaf) {
    $item = Get-Item -LiteralPath $Destination
    $hash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
    if ($item.Length -ne $ExpectedBytes -or $hash -ne $ExpectedSha256) {
      throw "Existing pinned toolchain archive is invalid: $Destination"
    }
    return
  }
  $partial = "$Destination.part"
  & curl.exe -L --fail --retry 8 --retry-all-errors -C - -o $partial $Url
  Assert-LastExitCode 'Pinned LLVM-MinGW download failed.'
  $item = Get-Item -LiteralPath $partial
  $hash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
  if ($item.Length -ne $ExpectedBytes -or $hash -ne $ExpectedSha256) {
    throw 'Pinned LLVM-MinGW archive verification failed.'
  }
  Move-Item -LiteralPath $partial -Destination $Destination
}

function Resolve-BuildTool(
  [string]$Requested,
  [string]$CommandName,
  [string]$Fallback
) {
  if ($Requested) {
    if (-not (Test-Path -LiteralPath $Requested -PathType Leaf)) {
      throw "$CommandName was not found at the requested path."
    }
    return (Resolve-Path -LiteralPath $Requested).Path
  }
  if (Test-Path -LiteralPath $Fallback -PathType Leaf) { return $Fallback }
  $command = Get-Command $CommandName -ErrorAction SilentlyContinue
  if ($null -ne $command) { return $command.Source }
  throw "$CommandName is required; pass its exact executable path."
}

function Assert-PinnedExecutable(
  [string]$Path,
  [string]$ExpectedSha256,
  [string]$Name
) {
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
    throw "$Name is missing."
  }
  $actualSha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
  if ($actualSha256 -ne $ExpectedSha256) {
    throw "$Name does not match the pinned executable hash."
  }
}

$existingBundle = @($bundleNames | ForEach-Object { Test-Path -LiteralPath (Join-Path $binaryRoot $_) -PathType Leaf })
if ((Test-Path -LiteralPath $provenance -PathType Leaf) -and -not ($existingBundle -contains $false)) {
  & $controlPython $verifier --source-only --source-version "v$RuntimeVersion"
  Assert-LastExitCode 'Existing thread-controlled SenseVoice bundle failed verification; preserving it for inspection.'
  Write-Output 'The existing thread-controlled SenseVoice bundle is verified.'
  return
}

if (-not (Test-Path -LiteralPath $controlPython -PathType Leaf)) {
  throw 'The benchmark control Python environment is missing.'
}
if ((Get-FileHash -LiteralPath $patch -Algorithm SHA256).Hash -ne $patchSha256) {
  throw 'The tracked SenseVoice thread patch changed.'
}

$cmakeFallback = 'D:\Anaconda\envs\local-bench-sensevoice-build\Library\bin\cmake.exe'
$ninjaFallback = 'D:\Anaconda\envs\local-bench-sensevoice-build\Library\bin\ninja.exe'
$cmake = Resolve-BuildTool $CMakePath 'cmake.exe' $cmakeFallback
$ninja = Resolve-BuildTool $NinjaPath 'ninja.exe' $ninjaFallback
Assert-PinnedExecutable $cmake $cmakeSha256 'CMake'
Assert-PinnedExecutable $ninja $ninjaSha256 'Ninja'
$cmakeVersion = (& $cmake --version | Select-Object -First 1)
Assert-LastExitCode 'CMake version probe failed.'
$ninjaVersion = (& $ninja --version).Trim()
Assert-LastExitCode 'Ninja version probe failed.'
if ($cmakeVersion -ne 'cmake version 4.2.3' -or $ninjaVersion -ne '1.13.1') {
  throw 'SenseVoice requires exactly CMake 4.2.3 and Ninja 1.13.1.'
}

New-Item -ItemType Directory -Force -Path $buildRoot, $toolchainRoot | Out-Null
Get-PinnedFile $toolchainArchive $archiveBytes $archiveSha256 $archiveUrl
if (-not (Test-Path -LiteralPath $toolchain -PathType Container)) {
  Expand-Archive -LiteralPath $toolchainArchive -DestinationPath $toolchainRoot
}
$clang = Join-Path $toolchainBin 'clang.exe'
$clangxx = Join-Path $toolchainBin 'clang++.exe'
$clangReal = Join-Path $toolchainBin 'clang-22.exe'
Assert-PinnedExecutable $clang $clangDriverSha256 'LLVM-MinGW clang driver'
Assert-PinnedExecutable $clangxx $clangDriverSha256 'LLVM-MinGW clang++ driver'
Assert-PinnedExecutable $clangReal $clangRealSha256 'LLVM-MinGW clang implementation'
$compilerVersion = (& $clang --version | Select-Object -First 1)
Assert-LastExitCode 'LLVM-MinGW compiler probe failed.'
if ($compilerVersion -notmatch '^clang version 22\.1\.8 ') {
  throw 'Pinned LLVM-MinGW compiler version changed.'
}

$git = (Get-Command git.exe -ErrorAction Stop).Source
if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot '.git') -PathType Container)) {
  if (Test-Path -LiteralPath $sourceRoot) {
    throw 'Existing SenseVoice source directory is not a Git checkout.'
  }
  & $git clone --branch $sourceTag --depth 1 $sourceRepository $sourceRoot
  Assert-LastExitCode 'Pinned SenseVoice source clone failed.'
}
$actualSourceRevision = (& $git -C $sourceRoot rev-parse HEAD).Trim()
Assert-LastExitCode 'SenseVoice source identity probe failed.'
$tagExpression = $sourceTag + '^{tag}'
$actualTagObject = (& $git -C $sourceRoot rev-parse $tagExpression).Trim()
Assert-LastExitCode 'SenseVoice tag identity probe failed.'
if ($actualSourceRevision -ne $sourceRevision -or $actualTagObject -ne $sourceTagObject) {
  throw 'SenseVoice source revision changed.'
}
$sourceStatus = @(& $git -C $sourceRoot status --porcelain --untracked-files=all)
if ($sourceStatus.Count -eq 0) {
  & $git -C $sourceRoot apply --check $patch
  Assert-LastExitCode 'SenseVoice thread patch no longer applies cleanly.'
  & $git -C $sourceRoot apply $patch
  Assert-LastExitCode 'SenseVoice thread patch application failed.'
  $sourceStatus = @(& $git -C $sourceRoot status --porcelain --untracked-files=all)
}
$expectedModified = " M $patchedFile"
if ($sourceStatus.Count -ne 1 -or $sourceStatus[0] -ne $expectedModified) {
  throw 'SenseVoice source contains unexpected changes.'
}
& $git -C $sourceRoot apply --reverse --check $patch
Assert-LastExitCode 'SenseVoice source does not match the tracked thread patch.'

if (-not (Test-Path -LiteralPath (Join-Path $llamaRoot '.git') -PathType Container)) {
  if (Test-Path -LiteralPath $llamaRoot) {
    throw 'Existing pinned llama.cpp directory is not a Git checkout.'
  }
  New-Item -ItemType Directory -Path $llamaRoot | Out-Null
  & $git -C $llamaRoot init
  Assert-LastExitCode 'Pinned llama.cpp repository initialization failed.'
  & $git -C $llamaRoot remote add origin https://github.com/ggml-org/llama.cpp.git
  Assert-LastExitCode 'Pinned llama.cpp remote configuration failed.'
  & $git -C $llamaRoot config core.longpaths true
  Assert-LastExitCode 'Pinned llama.cpp long-path configuration failed.'
  & $git -C $llamaRoot -c http.version=HTTP/1.1 fetch --depth 1 origin $llamaRevision
  Assert-LastExitCode 'Pinned llama.cpp fetch failed.'
  & $git -C $llamaRoot checkout --detach FETCH_HEAD
  Assert-LastExitCode 'Pinned llama.cpp checkout failed.'
}
$actualLlamaRevision = (& $git -C $llamaRoot rev-parse HEAD).Trim()
Assert-LastExitCode 'Pinned llama.cpp identity probe failed.'
$llamaStatus = @(& $git -C $llamaRoot status --porcelain)
if ($actualLlamaRevision -ne $llamaRevision -or $llamaStatus.Count -ne 0) {
  throw 'Pinned llama.cpp checkout changed.'
}

$env:Path = "$toolchainBin;$env:Path"
$cmakeSource = Join-Path $sourceRoot 'runtime\llama.cpp'
& $cmake -S $cmakeSource -B $outputRoot -G Ninja `
  "-DCMAKE_MAKE_PROGRAM=$ninja" `
  "-DCMAKE_C_COMPILER=$clang" `
  "-DCMAKE_CXX_COMPILER=$clangxx" `
  "-DFETCHCONTENT_SOURCE_DIR_LLAMA=$llamaRoot" `
  '-DCMAKE_BUILD_TYPE=Release' `
  '-DGGML_NATIVE=OFF' `
  '-DGGML_AVX=ON' `
  '-DGGML_AVX2=ON' `
  '-DGGML_FMA=ON' `
  '-DGGML_F16C=ON'
Assert-LastExitCode 'Pinned SenseVoice configuration failed.'
& $cmake --build $outputRoot --target llama-funasr-sensevoice --parallel $ParallelJobs
Assert-LastExitCode 'Pinned SenseVoice build failed.'

foreach ($name in @('libc++.dll', 'libomp.dll', 'libunwind.dll')) {
  Copy-Item -LiteralPath (Join-Path $toolchainBin $name) -Destination (Join-Path $binaryRoot $name)
}
$binary = Join-Path $binaryRoot 'llama-funasr-sensevoice.exe'
$helpOutput = (& $binary --help 2>&1 | Out-String)
if ($LASTEXITCODE -ne 0 -or $helpOutput -notmatch '--threads N \(default: 8\)') {
  throw 'Thread-controlled SenseVoice non-inference help probe failed.'
}

$bundle = @($bundleNames | ForEach-Object {
  $path = Join-Path $binaryRoot $_
  $item = Get-Item -LiteralPath $path
  [ordered]@{
    name = $_
    size_bytes = $item.Length
    sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
  }
})
$document = [ordered]@{
  schema_version = 1
  recorded_at_utc = [DateTime]::UtcNow.ToString('o')
  scope = 'source-build-only; no model inference'
  source = [ordered]@{
    repository = $sourceRepository
    tag = $sourceTag
    annotated_tag_object = $sourceTagObject
    commit = $sourceRevision
    patched_file = $patchedFile
  }
  pinned_dependency = [ordered]@{
    repository = 'https://github.com/ggml-org/llama.cpp.git'
    commit = $llamaRevision
  }
  patch = [ordered]@{
    path = $patchRelativePath.Replace('\', '/')
    sha256 = $patchSha256.ToLowerInvariant()
    behavior = 'explicit encoder and VAD thread count; upstream default remains eight'
  }
  toolchain = [ordered]@{
    distribution = 'llvm-mingw'
    release = '20260616'
    archive_url = $archiveUrl
    archive_size_bytes = $archiveBytes
    archive_sha256 = $archiveSha256.ToLowerInvariant()
    compiler = 'clang 22.1.8'
    cmake = '4.2.3'
    ninja = '1.13.1'
    clang_driver_sha256 = $clangDriverSha256.ToLowerInvariant()
    clang_implementation_sha256 = $clangRealSha256.ToLowerInvariant()
    cmake_sha256 = $cmakeSha256.ToLowerInvariant()
    ninja_sha256 = $ninjaSha256.ToLowerInvariant()
  }
  configuration = [ordered]@{
    build_type = 'Release'
    ggml_native = $false
    ggml_avx = $true
    ggml_avx2 = $true
    ggml_f16c = $true
    ggml_fma = $true
    max_parallel_compile_jobs = $ParallelJobs
    target_only = 'llama-funasr-sensevoice'
  }
  bundle = $bundle
  non_inference_verification = [ordered]@{
    help_exit_code = 0
    help_advertised_default_threads = 8
    model_loaded = $false
    audio_processed = $false
  }
  safety = [ordered]@{
    firmware_or_power_settings_changed = $false
    cooling_or_temperature_controls_changed = $false
    model_inference_run = $false
  }
}
$provenanceRoot = Split-Path -Parent $provenance
New-Item -ItemType Directory -Force -Path $provenanceRoot | Out-Null
$temporaryProvenance = "$provenance.$PID.tmp"
$document | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporaryProvenance -Encoding utf8
Move-Item -LiteralPath $temporaryProvenance -Destination $provenance
& $controlPython $verifier --source-only --source-version "v$RuntimeVersion"
Assert-LastExitCode 'Built thread-controlled SenseVoice bundle failed verification.'
