<#
  One-time setup on Windows 10/11.
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1          # GPU-capable ONNX engine (DirectML) + model
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Gpu     # additionally PyTorch CUDA + SAM 2 (optional, ~3 GB)
#>
param([switch]$Gpu)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Need($cmd, $hint) {
  if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { Write-Host "MISSING: $cmd  ->  $hint" -ForegroundColor Red; $script:missing = $true }
}
$missing = $false
Need python "winget install Python.Python.3.12"
Need node   "winget install OpenJS.NodeJS.LTS"
Need ffmpeg "winget install Gyan.FFmpeg   (or put ffmpeg.exe + ffprobe.exe in backend\bin)"
Need cargo  "winget install Rustlang.Rustup   (only needed to build the Tauri desktop shell)"
if ($missing) { Write-Host "Install the missing tools, open a NEW terminal and run this script again." -ForegroundColor Yellow }

Write-Host "== Python backend ==" -ForegroundColor Cyan
python -m venv backend\.venv
& backend\.venv\Scripts\python.exe -m pip install --upgrade pip
& backend\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt

Write-Host "== AI model (U2-Net, ONNX; runs on the GPU through DirectML) ==" -ForegroundColor Cyan
New-Item -ItemType Directory -Force backend\models | Out-Null
$model = "backend\models\u2net.onnx"
if (-not (Test-Path $model)) { Invoke-WebRequest "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx" -OutFile $model }
if ((Get-FileHash $model -Algorithm SHA256).Hash.ToLower() -ne "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491") { Remove-Item $model; throw "u2net.onnx hash mismatch - download deleted, run setup again." }

if ($Gpu) {
  Write-Host "== Deep-learning engine (PyTorch CUDA + SAM 2) ==" -ForegroundColor Cyan
  & backend\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  & backend\.venv\Scripts\python.exe -m pip install "git+https://github.com/facebookresearch/sam2.git"
  $models = Join-Path $env:LOCALAPPDATA "RotoBrush\models"
  New-Item -ItemType Directory -Force $models | Out-Null
  $ckpt = Join-Path $models "sam2.1_hiera_small.pt"
  if (-not (Test-Path $ckpt)) {
    Invoke-WebRequest "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt" -OutFile $ckpt
  }
}

Write-Host "== Frontend ==" -ForegroundColor Cyan
npm install
Write-Host "Setup finished. Next: scripts\dev.ps1" -ForegroundColor Green
