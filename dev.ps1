# Start Beat The Market for local development on Windows: API on :8000, web app on :5173.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Host "Installing uv (Python package manager)..."
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  Write-Error "Node.js 20+ is required: https://nodejs.org"
}

Push-Location backend; uv sync; Pop-Location
if (-not (Test-Path frontend/node_modules)) { Push-Location frontend; npm install; Pop-Location }

$api = Start-Process -PassThru -NoNewWindow -WorkingDirectory backend uv "run uvicorn app.main:app --reload --port 8000"
$web = Start-Process -PassThru -NoNewWindow -WorkingDirectory frontend npm "run dev -- --open"
Write-Host "`n  Beat The Market is starting: http://localhost:5173`n  Press Ctrl+C to stop.`n"
try { Wait-Process -Id $api.Id, $web.Id } finally {
  Stop-Process -Id $api.Id, $web.Id -ErrorAction SilentlyContinue
}
