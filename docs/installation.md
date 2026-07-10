# Installation

Sulcus OS uses `sulcus-os` as its future Python distribution name and
`agentos` as its backward-compatible import package. The project remains
pre-v1 and is currently installed from source; it is not published to PyPI.
The source distribution currently declares `LicenseRef-Unlicensed`; obtain a
license grant from the repository owner before redistributing it.

## Python-only installation

Python 3.10 or newer is supported; Python 3.14 is the primary development
environment used for this repository.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
python examples\public_api_quickstart.py
sulcus-check
```

Unix/macOS shells:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e .
python examples/public_api_quickstart.py
sulcus-check
```

This supports the public LLM, tools, timeline, and agent tool-loop APIs without
Rust. Native dashboard, IPC, memory, and WASM features remain unavailable until
the extension is built.

## Optional extras

```powershell
pip install -e .[openai]       # OpenAI-compatible provider SDK
pip install -e .[dashboard]    # Textual/Rich dashboard dependencies
pip install -e .[dev]          # pytest, build, and native development tools
pip install -e .[native-dev]
maturin develop                # build/install agent_os_core explicitly
```

For a full local development environment:

```powershell
pip install -e .[dev,dashboard,openai,native-dev]
maturin develop
```

## Build distributions

```powershell
python -m build
```

This produces a Python-only wheel and source distribution in `dist/`; neither
base artifact compiles or bundles the optional Rust extension. To test a wheel
from outside the source tree, create a fresh environment and install the wheel
file with `pip install dist\sulcus_os-<version>-py3-none-any.whl`.

## Troubleshooting

- `Native core: unavailable` from `sulcus-check` is expected for Python-only
  installs. Run `maturin develop` only for native features.
- If an OpenAI-compatible provider says its SDK is missing, install
  `.[openai]` and retry.
- To reset a local environment, deactivate it, remove `.venv`, recreate it,
  then repeat the editable-install commands.
