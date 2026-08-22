param([string[]]$Candidate = @('rapidocr_cpu','faster_whisper_cpu'))
$ErrorActionPreference = 'Stop'
$python = 'D:\Anaconda\envs\local-bench-control\python.exe'
foreach ($candidateId in $Candidate) {
  & $python -m local_inference_bench.cli run --candidate $candidateId
}
& $python -m local_inference_bench.cli report
