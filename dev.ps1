# Start Beat The Market for local development on Windows: API on :8000, web app on :5173.
# .\dev.ps1 -Phone also opens it to phones and tablets on the same Wi-Fi.
param([switch]$Phone)
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
$webArgs = if ($Phone) { "run dev -- --open --host 0.0.0.0" } else { "run dev -- --open" }
$web = Start-Process -PassThru -NoNewWindow -WorkingDirectory frontend npm $webArgs
Write-Host "`n  Beat The Market is starting: http://localhost:5173"
if ($Phone) {
  $ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.PrefixOrigin -in "Dhcp","Manual" -and $_.IPAddress -notlike "169.*" -and $_.IPAddress -ne "127.0.0.1" } | Select-Object -First 1).IPAddress
  Write-Host "  On your phone (same Wi-Fi):   http://$($ip):5173"
}
Write-Host "  Press Ctrl+C to stop.`n"
try { Wait-Process -Id $api.Id, $web.Id } finally {
  Stop-Process -Id $api.Id, $web.Id -ErrorAction SilentlyContinue
}
