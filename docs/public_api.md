# Public Python API

Sulcus OS currently uses `agentos` as its Python import package for backward
compatibility. Step 45 establishes the intended pre-v1 public surface.

## Stability levels

- **Stable:** `agentos` and `agentos.runtime`, `agentos.tools`, `agentos.ipc`,
  and `agentos.native` are the intended v1-facing APIs.
- **Advanced:** `agentos.llm` exposes provider-neutral runtime types for
  integrations; its wider provider configuration may evolve with notice.
- **Internal:** `kernel.*` implements Sulcus OS. Existing imports remain
  supported for compatibility, but new applications should use `agentos.*`.

## Top-level `agentos`

`AgentProcess`, process lifecycle enums, structured IPC helpers, core
tool-runtime types, agent tool-loop controls, approval/checkpoint types, and
native capability inspection are available at the top level. LLM types live in
`agentos.llm` to keep the default namespace focused.

## Public submodules

- `agentos.runtime`: `AgentToolLoop`, config/result, permissions, limits, and
  resumable approval types.
- `agentos.tools`: registry, runtime, definitions, execution results, and
  tool exceptions.
- `agentos.llm`: `LLMRuntime`, messages/responses, tool-call types, and the
  deterministic provider useful for offline integrations.
- `agentos.ipc`: structured IPC envelopes and helpers.
- `agentos.native`: native capability reporting and explicit requirement
  errors. It never exposes the raw extension module.

## Migration

```python
# Existing compatibility import
from kernel.tools import ToolRegistry

# Preferred public import
from agentos.tools import ToolRegistry
```

`kernel.*` imports are not deprecated at import time: Sulcus itself uses them,
and noisy warnings would affect applications and tests. They are intentionally
internal and may receive a later, explicit compatibility-deprecation notice.

Python-only LLM and tool-loop APIs do not require `agent_os_core`. Native
dashboard, IPC, memory, and WASM runtime usage still requires it and should be
checked through `agentos.native`.
