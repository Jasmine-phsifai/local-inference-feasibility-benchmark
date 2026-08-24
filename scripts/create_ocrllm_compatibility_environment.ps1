$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$snapshotRoot = Join-Path $projectRoot 'data\vendor\ocrllm-master-f234f39'
$lock = Join-Path $projectRoot 'environments\ocrllm_compatibility\requirements.lock.txt'
$environmentName = 'local-bench-ocrllm-master-f234f39'
$environmentPython = "D:\Anaconda\envs\$environmentName\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$git = 'C:\Program Files\Git\cmd\git.exe'
$sourceRepository = 'https://github.com/Jasmine-phsifai/LLM-based-OQC-scanner-for-textbook-pdfs-and-courses.git'
$expectedRevision = 'f234f3958f9e55d5bb9993338792ffc0cadc01fe'
$reviewedBaseline = '47c12efe91640659a711c8bd3429dae6a4fe44f5'

if (-not (Test-Path -LiteralPath (Join-Path $snapshotRoot '.git'))) {
    if (Test-Path -LiteralPath $snapshotRoot) {
        throw 'OCRLLM snapshot path exists without Git metadata.'
    }
    & $git clone --branch master --single-branch $sourceRepository $snapshotRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to acquire the OCRLLM master history.'
    }
    & $git -C $snapshotRoot checkout --detach $expectedRevision
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to check out the pinned OCRLLM revision.'
    }
}
$actualRevision = (& $git -C $snapshotRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualRevision -ne $expectedRevision) {
    throw "OCRLLM snapshot revision mismatch: $actualRevision"
}
& $git -C $snapshotRoot merge-base --is-ancestor $reviewedBaseline $expectedRevision
if ($LASTEXITCODE -ne 0) {
    throw 'Pinned OCRLLM snapshot does not descend from the reviewed baseline.'
}
$snapshotStatus = (& $git -C $snapshotRoot status --porcelain=v1)
if ($LASTEXITCODE -ne 0 -or $snapshotStatus) {
    throw 'Pinned OCRLLM snapshot has local changes.'
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $conda create --name $environmentName --yes python=3.11.15 pip=26.1.2
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the isolated OCRLLM compatibility environment.'
    }
}

& $environmentPython -m pip install --timeout 120 --retries 8 --requirement $lock
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the locked OCRLLM compatibility environment.'
}
& $environmentPython -m pip install --no-deps --force-reinstall --no-cache-dir `
    --no-build-isolation $snapshotRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the pinned OCRLLM snapshot independently.'
}
& $environmentPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'OCRLLM compatibility environment dependency check failed.'
}
& $environmentPython (Join-Path $PSScriptRoot 'verify_ocrllm_compatibility_environment.py') `
    --snapshot $snapshotRoot
if ($LASTEXITCODE -ne 0) {
    throw 'OCRLLM independent-install verification failed.'
}
