$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
& backend\.venv\Scripts\python.exe -m pytest backend\tests -q
npm run typecheck
