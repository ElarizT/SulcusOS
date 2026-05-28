param(
    [switch]$SkipCargo
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PythonInVenv = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Set-Location $RepoRoot

if (-not (Test-Path $PythonInVenv)) {
    throw "Missing .venv. Run .\scripts\setup_dev_windows.ps1 first."
}

& $PythonInVenv -m pytest

if (-not $SkipCargo) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw "cargo was not found. Install Rust or rerun with -SkipCargo."
    }
    $env:PYO3_PYTHON = $PythonInVenv
    cargo test
}
