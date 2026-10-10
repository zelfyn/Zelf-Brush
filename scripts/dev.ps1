<#
  Run RotoBrush in development.
    scripts\dev.ps1            # desktop window (Tauri)
    scripts\dev.ps1 -Browser   # backend + Vite only, open http://127.0.0.1:1420
#>
param([switch]$Browser)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = (Resolve-Path backend\.venv\Scripts\python.exe).Path
$env:ROTOBRUSH_PYTHON = $py
if ($Browser) {
  $be = Start-Process $py -ArgumentList "-m","rotobrush","--port","8765" -WorkingDirectory backend -PassThru -WindowStyle Hidden
  try { npx vite --port 1420 --host 127.0.0.1 } finally { Stop-Process -Id $be.Id -Force -ErrorAction SilentlyContinue }
} else {
  npm run tauri dev
}
