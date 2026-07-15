# Sulcus OS documentation

Start here if this is your first visit:

1. [Installation](installation.md) — source install, optional extras, and native development.
2. [10-minute quickstart](quickstart.md) — an offline registered-tool loop, limits, approval, and persistence.
3. [Concepts](concepts.md) — how agents, tools, controls, checkpoints, and observability fit together.
4. [Architecture](architecture.md) — component layers, process model, IPC, and Python/native boundary.

## Reference and operations

- [Public API and stability](public_api.md)
- [Project configuration](configuration.md)
- [Persistent checkpoints](checkpoints.md)
- [Examples by purpose](examples.md)
- [Troubleshooting](troubleshooting.md)
- [Roadmap and maturity](roadmap.md)

## Runtime guides

- [Flagship Supervised Research Team demo](../examples/supervised_research_team/README.md)
- [Agent SDK quickstart](sdk_quickstart.md)
- [Supervision trees](supervision.md)
- [IPC protocol](ipc_protocol.md)
- [Persistent memory](persistent_memory.md)
- [Interactive shell and execution modes](interactive_shell.md)
- [Windows native development](windows_dev_setup.md)

The project import package is named `agentos`. New application code should use
documented `agentos.*` imports and treat `kernel.*` as internal.
