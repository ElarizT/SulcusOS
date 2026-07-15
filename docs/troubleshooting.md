# Troubleshooting

## `agent_os_core` is unavailable

This is expected in Python-only mode. Confirm with `sulcus check`; LLM, tool
loop, limits, approval, checkpoints, and the flagship demo remain available.
Build the extension only for native IPC, dashboard, memory, or WASM features.

## `maturin develop` fails or installs into the wrong environment

Activate the intended virtual environment, then run from the repository root:

```powershell
python -m pip install -e .[native-dev]
maturin develop
python -c "import agent_os_core; print('native core available')"
sulcus check
```

Ensure a supported Python, Rust toolchain, and Cargo are on `PATH`. On Windows,
see [Windows native development](windows_dev_setup.md).

## The optional OpenAI dependency is missing

The deterministic provider needs no extra. For `OpenAICompatibleProvider`:

```powershell
python -m pip install -e .[openai]
```

Set provider credentials in the environment; do not put API keys in
`sulcus.toml`.

## `sulcus.toml` is invalid

Run `sulcus config check`. Errors identify the file and field. Common causes are
unknown sections/keys, quoted booleans (`"true"` instead of `true`), negative
limits, or an unsupported provider. Sulcus reads only `sulcus.toml` in the
current working directory; it does not search parents.

## Checkpoint compatibility fails

Reconstruct the same tool names, descriptions, parameter schemas, execution
mode, and parallel-safety flags. Provider/model labels must also match when the
runtime exposes them. Checkpoints do not serialize functions or runtime
objects. There is no automatic schema migration in checkpoint format version 1.

## A checkpoint is stale, consumed, or missing

`max_age_seconds` can intentionally reject old files. A successful complete
resume renames the source to `<name>.consumed`; it cannot be replayed. Partial
decisions preserve the original file and execute nothing. Do not rename a
consumed file back as a retry mechanism—create an application-level recovery
decision instead.

## `sulcus` is not recognized after install

Confirm the virtual environment is active and use the same interpreter for
installation and execution:

```powershell
python -m pip install -e .
python -m agentos.cli --version
```

If the module command works but `sulcus` does not, reopen the shell after
activation or check that `.venv\Scripts` is on `PATH`.

## Editable install issues

Run commands from the repository root and prefer `python -m pip` so pip and
Python refer to the same environment. If metadata is stale, uninstall and
reinstall in the active environment:

```powershell
python -m pip uninstall sulcus-os
python -m pip install -e .
python scripts\verify_package.py
```

Avoid testing installed-package behavior from a different checkout whose root
shadows the environment on `sys.path`.
