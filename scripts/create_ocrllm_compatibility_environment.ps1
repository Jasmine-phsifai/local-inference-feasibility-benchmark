$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$snapshotRoot = Join-Path $projectRoot 'data\vendor\ocrllm-active-master'
$manifest = Join-Path $projectRoot 'environments\ocrllm_compatibility\requirements.txt'
$environmentName = 'local-bench-ocrllm'
$environmentPython = "D:\Anaconda\envs\$environmentName\python.exe"
$conda = 'D:\Anaconda\Scripts\conda.exe'
$git = 'C:\Program Files\Git\cmd\git.exe'
$expectedRevision = '379726281e3c374bda65c1bd4a6bdf5c32cde0b3'
$reviewedBaseline = '47c12efe91640659a711c8bd3429dae6a4fe44f5'

if (-not (Test-Path -LiteralPath (Join-Path $snapshotRoot '.git'))) {
    throw 'Pinned OCRLLM snapshot Git metadata is missing.'
}
$actualRevision = (& $git -C $snapshotRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $actualRevision -ne $expectedRevision) {
    throw "OCRLLM snapshot revision mismatch: $actualRevision"
}
& $git -C $snapshotRoot merge-base --is-ancestor $reviewedBaseline $expectedRevision
if ($LASTEXITCODE -ne 0) {
    throw 'Pinned OCRLLM snapshot does not descend from the reviewed baseline.'
}

if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $conda create --name $environmentName --yes python=3.11 pip
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to create the isolated OCRLLM compatibility environment.'
    }
}

& $environmentPython -m pip install --timeout 60 --retries 8 --requirement $manifest
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install pinned OCRLLM compatibility dependencies.'
}
& $environmentPython -m pip install --no-deps --force-reinstall $snapshotRoot
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to install the pinned OCRLLM snapshot independently.'
}
& $environmentPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw 'OCRLLM compatibility environment dependency check failed.'
}
& $environmentPython -c @'
import importlib.metadata
import json
from pathlib import Path

import ocrllm

distribution = importlib.metadata.distribution("ocrllm")
direct_url_text = distribution.read_text("direct_url.json")
direct_url = json.loads(direct_url_text) if direct_url_text else {}
if direct_url.get("dir_info", {}).get("editable"):
    raise SystemExit("OCRLLM remained editable")
module_path = Path(ocrllm.__file__).resolve()
if "site-packages" not in {part.casefold() for part in module_path.parts}:
    raise SystemExit(f"OCRLLM did not install into site-packages: {module_path}")
print(distribution.version)
'@
if ($LASTEXITCODE -ne 0) {
    throw 'OCRLLM independent-install verification failed.'
}
