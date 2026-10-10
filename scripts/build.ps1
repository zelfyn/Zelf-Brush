<# Build the Windows installer (NSIS). Needs Rust + Node + Python venv from setup.ps1. #>
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = "backend\.venv\Scripts\python.exe"
& $py -m pip install pyinstaller
Push-Location backend
& ..\$py -m PyInstaller --noconfirm --clean --onedir --name rotobrush-backend `
   --collect-submodules uvicorn --collect-submodules rotobrush --collect-all onnxruntime --hidden-import tkinter run_backend.py
# AI model for the GPU engine (U2-Net, ONNX). Downloaded once into backend\models and verified by hash.
$model = "models\u2net.onnx"
if (-not (Test-Path $model)) {
  New-Item -ItemType Directory -Force models | Out-Null
  Invoke-WebRequest "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx" -OutFile $model
}
if ((Get-FileHash $model -Algorithm SHA256).Hash.ToLower() -ne "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491") { throw "u2net.onnx hash mismatch - delete backend\models\u2net.onnx and retry." }
Copy-Item models dist\rotobrush-backend\models -Recurse -Force

# Bundle REAL ffmpeg/ffprobe (not shims/aliases): put them in backend\bin first. Falls back to PATH only if they are real files.
New-Item -ItemType Directory -Force dist\rotobrush-backend\bin | Out-Null
foreach ($n in "ffmpeg.exe", "ffprobe.exe") {
  $src = if (Test-Path "bin\$n") { Get-Item "bin\$n" } else { (Get-Command $n -ErrorAction SilentlyContinue | Select-Object -First 1).Source | ForEach-Object { Get-Item $_ } }
  if (-not $src -or $src.Length -lt 10MB) { throw "No real $n found (shims and aliases are too small to be the real tool). Put the full $n in backend\bin." }
  Copy-Item $src.FullName dist\rotobrush-backend\bin -Force
}
Pop-Location
& $py scripts\smoke_test.py --backend backend\dist\rotobrush-backend\rotobrush-backend.exe --ffmpeg backend\dist\rotobrush-backend\bin\ffmpeg.exe --expect-engine onnx
if ($LASTEXITCODE -ne 0) { throw "Smoke test of the packaged backend failed - not building the installer." }
npm run tauri build
Write-Host "Installer: src-tauri\target\release\bundle\nsis" -ForegroundColor Green
