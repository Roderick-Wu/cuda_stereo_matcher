<#
PowerShell helper to configure and build the CUDA kernel project on Windows.

Usage: run from repository root in PowerShell.
  cd <repo-root>
  .\cuda\build\build_cuda.ps1

Notes:
- Requires CMake, a supported Visual Studio generator (MSVC), and the CUDA toolkit (nvcc) installed and on PATH.
- This script only configures and builds the placeholder project; adjust paths and targets as needed.
#>

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
$buildDir = Join-Path $repoRoot "cuda\build\cmake-build"

Write-Host "Configuring build directory: $buildDir"
if (-Not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

Push-Location $buildDir

cmake ..\.. -G "Visual Studio 16 2019" -A x64
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configuration failed"; Exit 1 }

cmake --build . --config Release
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; Exit 1 }

Write-Host "Build completed (if tools installed). Check $buildDir for outputs."
Pop-Location
