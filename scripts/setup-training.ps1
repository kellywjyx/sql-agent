param([Parameter(Mandatory=$true)][string]$EvalsWheel,
      [Parameter(Mandatory=$true)][string]$EvalsSHA256)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$dependency = Get-Content -LiteralPath 'evals-dependency.json' -Raw | ConvertFrom-Json
if ($EvalsSHA256.ToLowerInvariant() -ne $dependency.sha256) { throw 'Pinned llm-evals checksum required' }
$actual = (Get-FileHash -LiteralPath $EvalsWheel -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $EvalsSHA256.ToLowerInvariant()) { throw 'llm-evals wheel checksum mismatch' }
if (-not (Test-Path -LiteralPath '.venv-train/Scripts/python.exe')) {
    $python313 = $null
    if (Test-Path -LiteralPath '.venv/Scripts/python.exe') {
        $candidate = & .venv/Scripts/python.exe -c "import sys; print(sys.base_prefix if sys.version_info[:2] == (3, 13) else '')"
        if ($LASTEXITCODE -eq 0 -and $candidate) {
            $basePython = Join-Path $candidate 'python.exe'
            if (Test-Path -LiteralPath $basePython) { $python313 = $basePython }
        }
    }
    if (-not $python313) {
        $python313 = (Get-Command py -ErrorAction SilentlyContinue).Source
        if (-not $python313) { throw 'Python 3.13 was not found. Install it or create the application .venv first.' }
        & $python313 -3.13 -m venv .venv-train
    } else {
        & $python313 -m venv .venv-train
    }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 training environment creation failed' }
}
.venv-train/Scripts/python.exe -m pip install $EvalsWheel
if ($LASTEXITCODE -ne 0) { throw 'llm-evals installation failed' }
.venv-train/Scripts/python.exe -m pip install --no-deps torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA 12.8 PyTorch installation failed' }
.venv-train/Scripts/python.exe -m pip install -r requirements-training.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed' }
.venv-train/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw 'sql-agent training package installation failed' }
.venv-train/Scripts/python.exe -c "import importlib.metadata as m, torch; expected={'transformers':'4.57.6','trl':'0.25.1','peft':'0.18.1','bitsandbytes':'0.49.2','accelerate':'1.10.1','datasets':'4.4.1','huggingface-hub':'0.36.0'}; actual={k:m.version(k) for k in expected}; assert actual == expected, (actual, expected); assert m.version('torch') == '2.8.0+cu128', m.version('torch'); assert torch.cuda.is_available(), 'CUDA PyTorch is not active'; print({'torch':m.version('torch'),**actual})"
if ($LASTEXITCODE -ne 0) { throw 'Pinned training dependency verification failed' }
