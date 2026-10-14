<# Build the Windows installer (NSIS). Needs Rust + Node + Python venv from setup.ps1. #>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "backend\.venv\Scripts\python.exe"
& $py -m pip install pyinstaller
Push-Location backend
& ..\$py -m PyInstaller --noconfirm --clean --onedir --name rotobrush-backend `
   --collect-submodules uvicorn --collect-submodules rotobrush --collect-all onnxruntime --hidden-import tkinter run_backend.py
# AI models (segmentation General/Anime/Fast + depth), verified by hash.
& (Join-Path $PSScriptRoot "download-models.ps1") -Dir "models"
Copy-Item models dist\rotobrush-backend\models -Recurse -Force

# Bundle REAL ffmpeg/ffprobe (not shims/aliases): put them in backend\bin first. Falls back to PATH only if they are real files.
New-Item -ItemType Directory -Force dist\rotobrush-backend\bin | Out-Null
foreach ($n in "ffmpeg.exe", "ffprobe.exe") {
  $src = if (Test-Path "bin\$n") { Get-Item "bin\$n" } else { (Get-Command $n -ErrorAction SilentlyContinue | Select-Object -First 1).Source | ForEach-Object { Get-Item $_ } }
  if (-not $src -or $src.Length -lt 10MB) { throw "No real $n found (shims and aliases are too small to be the real tool). Put the full $n in backend\bin." }
  Copy-Item $src.FullName dist\rotobrush-backend\bin -Force
}
Pop-Location
& $py scripts\smoke_test.py --backend backend\dist\rotobrush-backend\rotobrush-backend.exe --ffmpeg backend\dist\rotobrush-backend\bin\ffmpeg.exe --expect-engine onnx --expect-models
if ($LASTEXITCODE -ne 0) { throw "Smoke test of the packaged backend failed - not building the installer." }
npm run tauri build
Write-Host "Installer: src-tauri\target\release\bundle\nsis" -ForegroundColor Green
