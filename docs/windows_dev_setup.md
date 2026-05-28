# Windows Dev Setup

Agent OS should be tested with a normal Windows Python installation, not the
Blender-bundled Python runtime. The recommended path is Python 3.11 through the
Windows `py` launcher.

## Audit Notes

- The repo-local `.venv` is intentionally ignored by git and may be deleted and
  recreated at any time.
- A broken `.venv` can happen when it points at a removed Windows Store Python
  shim. Recreate it with the commands below.
- There is no hardcoded Blender Python path in the project. Blender Python is
  useful only as an emergency syntax checker and is not a supported test runner.
- Native tests import `agent_os_core`, so the Rust extension must be installed
  into the active venv with `maturin develop`.

## One-Time Setup

Install Python 3.11 for Windows from python.org or winget, making sure the `py`
launcher is enabled.

```powershell
py -3.11 --version
cargo --version
```

If `.venv` is broken or points to a missing interpreter, remove it first:

```powershell
Remove-Item -Recurse -Force .venv
```

Create and activate a clean virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-dev.txt
maturin develop
```

## Run Tests

```powershell
pytest
$env:PYO3_PYTHON = ".\.venv\Scripts\python.exe"
cargo test
```

`PYO3_PYTHON` makes Cargo/PyO3 use the same clean venv interpreter that pytest
uses. If the venv is activated, this is usually discoverable from `PATH`, but
setting it explicitly avoids Windows launcher and Store-shim surprises.

For a quick Python-only check:

```powershell
python -m compileall kernel tests examples
```

## Helper Scripts

The scripts under `scripts/` wrap the same commands:

```powershell
.\scripts\setup_dev_windows.ps1
.\scripts\test_windows.ps1
```

To recreate a broken venv:

```powershell
.\scripts\setup_dev_windows.ps1 -Recreate
```
