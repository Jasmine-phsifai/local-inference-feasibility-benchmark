$ErrorActionPreference = 'Stop'

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$environmentName = 'local-bench-qwen3-asr-openvino-genai-tailfix-20260821'
$targetPrefix = "D:\Anaconda\envs\$environmentName"
$targetPython = Join-Path $targetPrefix 'python.exe'
$conda = 'D:\Anaconda\Scripts\conda.exe'
$profilePath = Join-Path $projectRoot 'environments\qwen3_asr_openvino_genai_tailfix_20260821\runtime-provenance.json'
$lockPath = Join-Path $projectRoot 'environments\qwen3_asr_openvino_genai_tailfix_20260821\requirements.lock.txt'
$expectedProfileSha256 = '33c8925cf8a9a219cadaee22d60fea5ea30a6a1c0fea0abb67202d8a592623aa'
$expectedLockSha256 = '6077ebedd0e1613b653f35568d1c4d0e93ebd27b7f11756babc9a599c4bb6275'
if (
    (Get-FileHash -LiteralPath $profilePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    $expectedProfileSha256
) {
    throw 'The tail-fix runtime profile identity changed.'
}
if (
    (Get-FileHash -LiteralPath $lockPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne
    $expectedLockSha256
) {
    throw 'The tail-fix runtime lock identity changed.'
}
$profile = Get-Content -LiteralPath $profilePath -Raw | ConvertFrom-Json
if ($profile.source_repository -ne 'https://github.com/openvinotoolkit/openvino.genai.git') {
    throw 'The tail-fix source repository identity changed.'
}

$inProgressMarkerSuffix = '.local-bench-tailfix-preparation-in-progress-v1'
$completeMarkerSuffix = '.local-bench-tailfix-preparation-complete-v1'

function Get-PreparationMarkerState {
    param(
        [Parameter(Mandatory = $true)][string]$ResourceKind,
        [Parameter(Mandatory = $true)][string]$ResourcePath
    )
    $normalizedPath = [System.IO.Path]::GetFullPath($ResourcePath).TrimEnd('\')
    $payload = @(
        "profile_sha256=$expectedProfileSha256",
        "lock_sha256=$expectedLockSha256",
        "resource_kind=$ResourceKind",
        "resource_path=$normalizedPath"
    ) -join "`n"
    [PSCustomObject]@{
        ResourcePath = $normalizedPath
        InProgressPath = "$normalizedPath$inProgressMarkerSuffix"
        CompletePath = "$normalizedPath$completeMarkerSuffix"
        Payload = "$payload`n"
        Stream = $null
    }
}

function Assert-PreparationMarkerContent {
    param(
        [Parameter(Mandatory = $true)]$State,
        [Parameter(Mandatory = $true)][string]$MarkerPath
    )
    if (
        -not (Test-Path -LiteralPath $MarkerPath -PathType Leaf) -or
        (Get-Content -LiteralPath $MarkerPath -Raw) -cne $State.Payload
    ) {
        throw 'The tail-fix preparation ownership marker is invalid.'
    }
}

function Open-OwnedPreparation {
    param(
        [Parameter(Mandatory = $true)][string]$ResourceKind,
        [Parameter(Mandatory = $true)][string]$ResourcePath
    )
    $state = Get-PreparationMarkerState $ResourceKind $ResourcePath
    $resourceExists = Test-Path -LiteralPath $state.ResourcePath
    $inProgressExists = Test-Path -LiteralPath $state.InProgressPath
    $completeExists = Test-Path -LiteralPath $state.CompletePath
    if ($inProgressExists -and $completeExists) {
        throw 'The tail-fix preparation ownership markers conflict.'
    }
    if ($completeExists) {
        Assert-PreparationMarkerContent $state $state.CompletePath
        throw 'The completed tail-fix resource is invalid; refusing repair.'
    }
    if (-not $inProgressExists -and $resourceExists) {
        throw 'The invalid tail-fix resource is not owned by this preparation.'
    }
    try {
        if ($inProgressExists) {
            Assert-PreparationMarkerContent $state $state.InProgressPath
            $state.Stream = [System.IO.File]::Open(
                $state.InProgressPath,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::Delete
            )
        } else {
            $state.Stream = [System.IO.File]::Open(
                $state.InProgressPath,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::Delete
            )
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($state.Payload)
            $state.Stream.Write($bytes, 0, $bytes.Length)
            $state.Stream.Flush($true)
            $state.Stream.Position = 0
        }
    } catch {
        throw 'Another process owns or changed the tail-fix preparation marker.'
    }
    return $state
}

function Complete-OwnedPreparation {
    param([Parameter(Mandatory = $true)]$State)
    [System.IO.File]::Move($State.InProgressPath, $State.CompletePath)
    $State.Stream.Dispose()
    $State.Stream = $null
}

function Complete-VerifiedPreparationMarkerIfPresent {
    param(
        [Parameter(Mandatory = $true)][string]$ResourceKind,
        [Parameter(Mandatory = $true)][string]$ResourcePath
    )
    $state = Get-PreparationMarkerState $ResourceKind $ResourcePath
    $inProgressExists = Test-Path -LiteralPath $state.InProgressPath
    $completeExists = Test-Path -LiteralPath $state.CompletePath
    if ($inProgressExists -and $completeExists) {
        throw 'The tail-fix preparation ownership markers conflict.'
    }
    if ($completeExists) {
        Assert-PreparationMarkerContent $state $state.CompletePath
        return
    }
    if (-not $inProgressExists) {
        return
    }
    Assert-PreparationMarkerContent $state $state.InProgressPath
    try {
        $state.Stream = [System.IO.File]::Open(
            $state.InProgressPath,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::Delete
        )
        Complete-OwnedPreparation $state
    } finally {
        if ($null -ne $state.Stream) {
            $state.Stream.Dispose()
        }
    }
}
$verifier = Join-Path $PSScriptRoot 'verify_qwen3_asr_openvino_genai_tailfix_20260821_environment.py'
$downloadDirectory = Join-Path $projectRoot 'data\downloads\openvino-genai-tailfix-20260821'
$sourceCheckout = [System.IO.Path]::GetFullPath(
    [string](Join-Path $projectRoot $profile.source_checkout)
)
$vendorRoot = [System.IO.Path]::GetFullPath(
    [string](Join-Path $projectRoot 'data\vendor')
).TrimEnd('\') + '\'
if (-not $sourceCheckout.StartsWith(
    $vendorRoot,
    [System.StringComparison]::OrdinalIgnoreCase
)) {
    throw 'The tail-fix source checkout escaped the ignored vendor directory.'
}
$git = Get-Command git.exe -ErrorAction SilentlyContinue
if ($null -eq $git) {
    $fallbackGit = 'C:\Program Files\Git\cmd\git.exe'
    if (-not (Test-Path -LiteralPath $fallbackGit -PathType Leaf)) {
        throw 'Git is required for the OpenVINO GenAI source-ancestry proof.'
    }
    $gitExecutable = $fallbackGit
} else {
    $gitExecutable = $git.Source
}

function Test-ExactSourceCheckout {
    if (-not (Test-Path -LiteralPath $sourceCheckout -PathType Container)) {
        return $false
    }
    $sourceOrigin = (
        & $gitExecutable -C $sourceCheckout remote get-url origin 2>$null |
        Out-String
    ).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceOrigin -ne $profile.source_repository) {
        return $false
    }
    $sourceIsShallow = (
        & $gitExecutable -C $sourceCheckout rev-parse --is-shallow-repository 2>$null |
        Out-String
    ).Trim()
    if ($LASTEXITCODE -ne 0 -or $sourceIsShallow -ne 'false') {
        return $false
    }
    foreach ($revision in @(
        $profile.stable_source_revision,
        $profile.required_fix_revision,
        $profile.associated_source_revision
    )) {
        $resolvedRevision = (
            & $gitExecutable -C $sourceCheckout rev-parse "$revision^{commit}" 2>$null |
            Out-String
        ).Trim()
        if ($LASTEXITCODE -ne 0 -or $resolvedRevision -ne $revision) {
            return $false
        }
    }
    & $gitExecutable -C $sourceCheckout merge-base --is-ancestor `
        $profile.required_fix_revision $profile.associated_source_revision
    return $LASTEXITCODE -eq 0
}

if (Test-ExactSourceCheckout) {
    Complete-VerifiedPreparationMarkerIfPresent 'source' $sourceCheckout
} else {
    New-Item -ItemType Directory -Path (Split-Path -Parent $sourceCheckout) -Force | Out-Null
    $sourcePreparation = Open-OwnedPreparation 'source' $sourceCheckout
    try {
        if (-not (Test-Path -LiteralPath $sourceCheckout)) {
            New-Item -ItemType Directory -Path $sourceCheckout | Out-Null
        }
        if (-not (Test-Path -LiteralPath $sourceCheckout -PathType Container)) {
            throw 'The owned OpenVINO GenAI source path is not a directory.'
        }
        & $gitExecutable -C $sourceCheckout rev-parse --git-dir 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            & $gitExecutable -C $sourceCheckout init
            if ($LASTEXITCODE -ne 0) {
                throw 'Failed to initialize the OpenVINO GenAI ancestry checkout.'
            }
        }
        $sourceOrigin = (
            & $gitExecutable -C $sourceCheckout remote get-url origin 2>$null |
            Out-String
        ).Trim()
        if ($LASTEXITCODE -ne 0) {
            & $gitExecutable -C $sourceCheckout remote add origin $profile.source_repository
            if ($LASTEXITCODE -ne 0) {
                throw 'Failed to bind the OpenVINO GenAI source origin.'
            }
        } elseif ($sourceOrigin -ne $profile.source_repository) {
            throw 'The owned OpenVINO GenAI source origin changed.'
        }
        $sourceIsShallow = (
            & $gitExecutable -C $sourceCheckout rev-parse --is-shallow-repository 2>$null |
            Out-String
        ).Trim()
        if ($sourceIsShallow -eq 'true') {
            & $gitExecutable -C $sourceCheckout fetch --filter=blob:none --unshallow origin `
                '+refs/heads/*:refs/remotes/origin/*'
        } else {
            & $gitExecutable -C $sourceCheckout fetch --filter=blob:none origin `
                '+refs/heads/*:refs/remotes/origin/*'
        }
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to fetch the OpenVINO GenAI source ancestry.'
        }
        if (-not (Test-ExactSourceCheckout)) {
            throw 'The recovered OpenVINO GenAI source ancestry is incomplete.'
        }
        Complete-OwnedPreparation $sourcePreparation
    } finally {
        if ($null -ne $sourcePreparation.Stream) {
            $sourcePreparation.Stream.Dispose()
        }
    }
}

if (Test-Path -LiteralPath $targetPython -PathType Leaf) {
    & $targetPython $verifier
    $environmentVerified = $LASTEXITCODE -eq 0
} else {
    $environmentVerified = $false
}
if ($environmentVerified) {
    Complete-VerifiedPreparationMarkerIfPresent 'environment' $targetPrefix
}
if (-not $environmentVerified -and -not (Test-Path -LiteralPath $conda -PathType Leaf)) {
    throw 'Conda was not found at the benchmark-pinned installation.'
}

if (-not $environmentVerified) {
    $environmentPreparation = Open-OwnedPreparation 'environment' $targetPrefix
    try {
        New-Item -ItemType Directory -Path $downloadDirectory -Force | Out-Null
        $wheelPaths = @()
        foreach ($property in $profile.wheel_artifacts.PSObject.Properties) {
        $artifact = $property.Value
        $target = Join-Path $downloadDirectory $artifact.filename
        $partial = "$target.part"
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
            if (Test-Path -LiteralPath $partial -PathType Leaf) {
                $partialSize = (Get-Item -LiteralPath $partial).Length
                if ($partialSize -gt [int64]$artifact.size_bytes) {
                    throw 'Partial OpenVINO nightly wheel exceeds its pinned size.'
                }
                if ($partialSize -eq [int64]$artifact.size_bytes) {
                    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $artifact.sha256) {
                        throw 'Complete partial OpenVINO nightly wheel has the wrong hash.'
                    }
                    Move-Item -LiteralPath $partial -Destination $target
                }
            }
            if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
                & curl.exe --location --fail --retry 8 --continue-at - `
                    --output $partial $artifact.url
                if ($LASTEXITCODE -ne 0) {
                    throw 'Failed to download a pinned OpenVINO nightly wheel.'
                }
                if ((Get-Item -LiteralPath $partial).Length -ne [int64]$artifact.size_bytes) {
                    throw 'Downloaded OpenVINO nightly wheel size changed.'
                }
                if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $artifact.sha256) {
                    throw 'Downloaded OpenVINO nightly wheel hash changed.'
                }
                Move-Item -LiteralPath $partial -Destination $target
            }
        }
        if (
            (Get-Item -LiteralPath $target).Length -ne [int64]$artifact.size_bytes -or
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $artifact.sha256
        ) {
            throw 'Cached OpenVINO nightly wheel identity changed.'
        }
            $wheelPaths += $target
        }

        $condaHistory = Join-Path $targetPrefix 'conda-meta\history'
        if (-not (Test-Path -LiteralPath $condaHistory -PathType Leaf)) {
            & $conda create --prefix $targetPrefix --yes `
                python=3.11.15 pip=26.1.2 setuptools=83.0.0 wheel=0.47.0
            if ($LASTEXITCODE -ne 0) {
                throw 'Failed to create the tail-fix OpenVINO GenAI environment.'
            }
        } else {
            if (-not (Test-Path -LiteralPath $targetPython -PathType Leaf)) {
                throw 'The owned tail-fix Conda base is incomplete; refusing repair.'
            }
            & $targetPython -c `
                'import os,sys; expected=os.path.normcase(os.path.realpath(sys.argv[1])); actual=os.path.normcase(os.path.realpath(sys.prefix)); raise SystemExit(0 if actual == expected and sys.version_info[:3] == (3, 11, 15) else 1)' `
                $targetPrefix
            if ($LASTEXITCODE -ne 0) {
                throw 'The owned tail-fix Conda base identity changed; refusing repair.'
            }
            & $targetPython -m pip --version | Out-Null
            if ($LASTEXITCODE -ne 0) {
                throw 'The owned tail-fix Conda base has no working pip; refusing repair.'
            }
        }
        & $targetPython -m pip install --no-deps --upgrade `
            numpy==2.4.6 openvino-telemetry==2025.2.0 packaging==26.3 `
            pip==26.1.2 setuptools==83.0.0 wheel==0.47.0
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to install the exact non-wheel tail-fix dependencies.'
        }
        & $targetPython -m pip install --no-deps --force-reinstall @wheelPaths
        if ($LASTEXITCODE -ne 0) {
            throw 'Failed to install the pinned OpenVINO nightly wheels.'
        }
        & $targetPython $verifier
        if ($LASTEXITCODE -ne 0) {
            throw 'Tail-fix OpenVINO GenAI environment verification failed.'
        }
        Complete-OwnedPreparation $environmentPreparation
    } finally {
        if ($null -ne $environmentPreparation.Stream) {
            $environmentPreparation.Stream.Dispose()
        }
    }
}

$officialPython = 'D:\Anaconda\envs\local-bench-qwen3-asr-openvino-genai-official\python.exe'
$exportScript = Join-Path $PSScriptRoot 'export_qwen3_asr_openvino_genai_official.py'
$exportDirectory = Join-Path $projectRoot 'data\models\qwen3-asr-0.6b-openvino-genai-official-f48d93f'
$exportMarker = Join-Path $exportDirectory 'export-complete.json'
$exportProvenance = Join-Path $exportDirectory 'official-export-provenance.json'
if (
    (Test-Path -LiteralPath $officialPython -PathType Leaf) -and
    (Test-Path -LiteralPath $exportMarker -PathType Leaf) -and
    (Test-Path -LiteralPath $exportProvenance -PathType Leaf)
) {
    & $officialPython $exportScript
    if ($LASTEXITCODE -ne 0) {
        throw 'The existing pinned official Qwen3-ASR export failed verification.'
    }
} else {
    & (Join-Path $PSScriptRoot 'prepare_qwen3_asr_openvino_genai_official.ps1')
    if ($LASTEXITCODE -ne 0) {
        throw 'Failed to prepare the pinned official Qwen3-ASR export.'
    }
}
& $targetPython $verifier
if ($LASTEXITCODE -ne 0) {
    throw 'Final tail-fix OpenVINO GenAI environment verification failed.'
}
