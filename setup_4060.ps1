# GPU installation for Windows; run from any directory.
param(
    [string]$PythonVersion = '3.12',
    [string]$PipIndex = 'https://pypi.org/simple',
    [string]$TorchIndex = 'https://download.pytorch.org/whl/cu126'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    & py "-$PythonVersion" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Install 64-bit Python $PythonVersion with Python Launcher and Tcl/Tk, or set -PythonVersion." }
}
& $pythonExe -m pip install --upgrade pip --index-url $PipIndex
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed. Check the selected PipIndex and retry.' }
& $pythonExe -m pip install -r requirements-torch-cu126.txt --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed. Check TorchIndex/network access and retry.' }
& $pythonExe -m pip install -r requirements.txt --index-url $PipIndex
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }
& $pythonExe check_environment.py --require-cuda
if ($LASTEXITCODE -ne 0) { throw 'CUDA check failed. Update the NVIDIA driver and inspect the output above.' }
Write-Host 'Setup complete. See README.md for annotation, training and prediction.'
