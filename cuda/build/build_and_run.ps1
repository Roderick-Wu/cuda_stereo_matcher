<#
Configure, build, and run the CUDA test executable on Windows.

Usage: Run from any location. The script will create a build dir under
`cuda/build/cmake-build` and attempt to build the `vector_add_test` target,
then run the produced executable.

Requires: CMake, Visual Studio / MSVC, and CUDA toolkit (nvcc).
#>

$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptDir
$buildDir = Join-Path $scriptDir "cmake-build"

Write-Host "Repo root: $repoRoot"
Write-Host "Build directory: $buildDir"

if (-Not (Test-Path $buildDir)) { New-Item -ItemType Directory -Path $buildDir | Out-Null }

Push-Location $buildDir

Write-Host "Running CMake configure..."
cmake ..\.. -G "Visual Studio 16 2019" -A x64
if ($LASTEXITCODE -ne 0) { Write-Error "CMake configuration failed"; Exit 1 }

Write-Host "Building vector_add_test (Release)..."
cmake --build . --config Release --target vector_add_test
if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; Exit 1 }

$exePath = Join-Path $buildDir "Release\vector_add_test.exe"
if (-Not (Test-Path $exePath)) {
    Write-Error "Executable not found: $exePath"
    Exit 1
}

Write-Host "Running test executable: $exePath"
& $exePath

Pop-Location
