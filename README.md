# Agent OS Core

A lightweight hybrid runtime that combines:

- A Rust core crate (`agent_os_core`)
- Python orchestration (`main.py`, `kernel/`)
- Optional WASM sandboxed execution for constrained agent code

## Project Structure

- `src/` Rust crate source
- `kernel/` Python runtime modules (processes, dashboard, toolchain, LLM integration)
- `tests/` Test suite
- `examples/` Example assets/configurations
- `docs/` Documentation
- `main.py` Python entrypoint
- `Cargo.toml` Rust crate config

## Documentation

- `docs/interactive_shell.md` covers the process shell and isolation modes.
- `docs/ipc_protocol.md` covers the structured Agent-to-Agent IPC protocol.

## Requirements

- Python 3.10+
- Rust (stable toolchain)
- Cargo

## Quick Start

For Windows test/development setup, see `docs/windows_dev_setup.md`.

### 1) Python environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt
```

Install the native Python extension into the active virtual environment:

```powershell
maturin develop
```

### 2) Build Rust core

```powershell
cargo build
```

### 3) Run runtime

```powershell
python main.py
```

## Development

Run tests:

```powershell
python -m pytest
```

Run Rust checks:

```powershell
$env:PYO3_PYTHON = ".\.venv\Scripts\python.exe"
cargo check
```

## Notes

- Runtime behavior is configurable through environment variables used in `main.py`.
- Logs are written to files such as `agent_runtime.log` and `agent_debug.log`.
