param([string]$EvalsWheel, [string]$EvalsSHA256)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 environment creation failed' }
}
.venv/Scripts/python.exe -m pip install -c requirements.lock.txt setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'Build dependency installation failed' }
if (Test-Path -LiteralPath 'src/llm_evals') {
    .venv/Scripts/python.exe -m pip install --no-build-isolation -c requirements.lock.txt -e . pytest
} else {
    if (-not $EvalsWheel -or -not $EvalsSHA256) { throw 'Supply -EvalsWheel and -EvalsSHA256 for the versioned llm-evals release wheel' }
    $dependency = Get-Content -LiteralPath 'evals-dependency.json' -Raw | ConvertFrom-Json
    if ($EvalsSHA256.ToLowerInvariant() -ne $dependency.sha256) { throw 'Pinned release checksum required; use evals-dependency.json' }
    $actual = (Get-FileHash -LiteralPath $EvalsWheel -Algorithm SHA256).Hash
    if ($actual.ToLowerInvariant() -ne $EvalsSHA256.ToLowerInvariant()) { throw 'llm-evals wheel checksum mismatch' }
    .venv/Scripts/python.exe -m pip install -c requirements.lock.txt $EvalsWheel
    if ($LASTEXITCODE -ne 0) { throw 'Shared evaluation wheel installation failed' }
    .venv/Scripts/python.exe -m pip install --no-build-isolation -c requirements.lock.txt -e . pytest
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
.venv/Scripts/python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
