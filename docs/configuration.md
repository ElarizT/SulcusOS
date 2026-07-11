# Project configuration

Sulcus optionally reads `sulcus.toml` from the current working directory. It
does not search parent directories and never creates or overwrites the file.
Without a file, existing runtime defaults remain unchanged.

## Format

```toml
[sulcus]
execution_mode = "sequential"       # sequential or parallel
require_tool_approval = false

[limits]
max_tool_calls_per_loop = 10
max_tool_calls_per_round = 4
tool_timeout_ms = 5000

[llm]
provider = "deterministic"           # deterministic, openai, openrouter, groq
model = "default"
```

All sections and keys are optional. Limits are non-negative integers. Unknown
sections and keys, invalid booleans, unsupported providers, and unknown
execution modes are rejected with the file path and field name.

## Precedence and environment variables

Values resolve in this order, from highest to lowest priority:

1. Explicit CLI or Python arguments
2. Environment variables
3. `sulcus.toml`
4. Existing runtime defaults

Supported variables are `SULCUS_EXECUTION_MODE`,
`SULCUS_REQUIRE_TOOL_APPROVAL`, `SULCUS_MAX_TOOL_CALLS_PER_LOOP`,
`SULCUS_MAX_TOOL_CALLS_PER_ROUND`, `SULCUS_TOOL_TIMEOUT_MS`,
`SULCUS_LLM_PROVIDER`, and `SULCUS_LLM_MODEL`. Existing
`AGENTOS_LLM_PROVIDER` and `AGENTOS_LLM_MODEL` settings remain supported as LLM
aliases. API keys remain provider environment settings and are never loaded
into or printed by project configuration.

## Commands

```powershell
sulcus config path
sulcus config check
sulcus config show
```

`path` reports the file in the current directory or says none exists. `check`
returns zero for valid/default configuration and one for invalid configuration.
`show` prints effective values after environment precedence. Its output is a
fixed sanitized schema and never includes API keys or arbitrary environment
values.

The flagship command uses configured execution mode and resource limits while
explicit `--parallel` or `--sequential` flags still win:

```powershell
sulcus demo research-team --parallel
```

Provider and model settings describe existing LLM runtime choices. The
flagship remains an offline scripted demo and does not turn those settings into
network access.

## Python API

```python
from agentos.config import SulcusConfig, load_config, resolve_config

file_config = load_config()
effective = resolve_config(file_config, explicit={"execution_mode": "parallel"})
```

Common errors include misspelled section names, quoted booleans such as
`"true"`, negative limits, and unsupported provider names. CLI errors are
concise and do not include tracebacks.
