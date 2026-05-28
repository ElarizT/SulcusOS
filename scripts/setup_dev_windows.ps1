param(
    [string]$PythonVersion = "3.11",
    [switch]$Recreate,
    [switch]$SkipMaturinDevelop
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPath = Join-Path $RepoRoot ".venv"
$PythonInVenv = Join-Path $VenvPath "Scripts\python.exe"

Set-Location $RepoRoot

if ($Recreate -and (Test-Path $VenvPath)) {
    Remove-Item -Recurse -Force $VenvPath
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Windows py launcher was not found. Install Python 3.11+ from python.org with the py launcher enabled."
}

if (-not (Test-Path $PythonInVenv)) {
    py "-$PythonVersion" -m venv .venv
}

& $PythonInVenv -m pip install -U pip
& $PythonInVenv -m pip install -r requirements-dev.txt

if (-not $SkipMaturinDevelop) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "cargo was not found. Install the Rust stable toolchain before running maturin develop."
    }
    & $PythonInVenv -m maturin develop
}

Write-Host "Agent OS dev environment is ready."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
