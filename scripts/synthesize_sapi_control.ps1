param(
    [Parameter(Mandatory = $true)]
    [string]$OutputPath,
    [Parameter(Mandatory = $true)]
    [string]$Voice,
    [Parameter(Mandatory = $true)]
    [string]$Text,
    [int]$Rate = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Speech
$synthesizer = [System.Speech.Synthesis.SpeechSynthesizer]::new()
try {
    $availableVoices = $synthesizer.GetInstalledVoices() |
        ForEach-Object { $_.VoiceInfo.Name }
    if ($Voice -notin $availableVoices) {
        throw "Required SAPI voice is not installed: $Voice"
    }
    $parent = Split-Path -Parent $OutputPath
    if ($parent) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    $synthesizer.SelectVoice($Voice)
    $synthesizer.Rate = $Rate
    $synthesizer.SetOutputToWaveFile($OutputPath)
    $synthesizer.Speak($Text)
}
finally {
    $synthesizer.SetOutputToNull()
    $synthesizer.Dispose()
}
