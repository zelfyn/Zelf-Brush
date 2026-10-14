<#
  Downloads every AI model Zelf-Brush needs into backend\models and verifies each file by SHA-256.
  Single source of truth: used by setup.ps1, build.ps1 and the GitHub workflow.
#>
param([string]$Dir = "backend\models")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Dir | Out-Null
$rembg = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"
$models = @(
  @{ f = "isnet-general-use.onnx";      u = "$rembg/isnet-general-use.onnx"; h = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a" },
  @{ f = "isnet-anime.onnx";            u = "$rembg/isnet-anime.onnx";       h = "f15622d853e8260172812b657053460e20806f04b9e05147d49af7bed31a6e99" },
  @{ f = "u2net.onnx";                  u = "$rembg/u2net.onnx";             h = "8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491" },
  @{ f = "depth_anything_v2_vits.onnx"; u = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"; h = "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be" }
)
foreach ($m in $models) {
  $dst = Join-Path $Dir $m.f
  if (-not (Test-Path $dst)) {
    Write-Host "Downloading $($m.f) ..."
    Invoke-WebRequest $m.u -OutFile $dst
  }
  $got = (Get-FileHash $dst -Algorithm SHA256).Hash.ToLower()
  if ($got -ne $m.h) { Remove-Item $dst -Force; throw "$($m.f): hash mismatch ($got). The download was removed - run again." }
  Write-Host ("OK  {0}  ({1} MB)" -f $m.f, [math]::Round((Get-Item $dst).Length / 1MB))
}
