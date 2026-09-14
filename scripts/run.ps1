param([ValidateSet('api','ui')][string]$Role='ui', [string]$Artifacts)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $Artifacts) { $Artifacts = Join-Path $projectRoot 'artifacts' }
$env:PORTFOLIO_ARTIFACTS = $Artifacts
$env:PORTFOLIO_TRACES = Join-Path $Artifacts 'traces.jsonl'
$env:HF_HUB_OFFLINE = '1'
$interpreter = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $interpreter)) { throw 'Run scripts/setup.ps1 first' }
if ($Role -eq 'api') {
    & $interpreter -m uvicorn sql_agent.api:create_app --factory --host 127.0.0.1 --port 8102
} else {
    & $interpreter -m streamlit run src/sql_agent/ui.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false
}
exit $LASTEXITCODE
