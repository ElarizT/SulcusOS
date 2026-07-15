# 10-minute offline quickstart

This tutorial uses only public `agentos` imports. It needs no API key, network,
Rust toolchain, or native core.

## 1. Install from the repository

From the repository root in PowerShell:

```powershell
python -m pip install -e .
sulcus --version
sulcus check
```

`Native core: unavailable` is an expected result for this tutorial.

## 2. Run the smallest complete tool loop

The checked-in example is the fastest path:

```powershell
python examples\public_api_quickstart.py
```

Expected output:

```text
The answer is 42.
```

Here is the complete pattern used by that example:

```python
from agentos.llm import LLMResponse, LLMRuntime, LLMToolCall
from agentos.runtime import AgentToolLoop
from agentos.tools import ToolRegistry, ToolRuntime


class ScriptedProvider:
    """Two deterministic responses: request a tool, then finish."""

    name = "quickstart"
    default_model = "offline"

    def __init__(self):
        self.responses = [
            LLMResponse(
                content="",
                provider=self.name,
                model=self.default_model,
                tool_calls=(
                    LLMToolCall("add-1", "add_numbers", {"a": 20, "b": 22}),
                ),
            ),
            LLMResponse(
                content="The answer is 42.",
                provider=self.name,
                model=self.default_model,
            ),
        ]

    def complete(self, request):
        return self.responses.pop(0)


registry = ToolRegistry()
registry.register(
    name="add_numbers",
    description="Add two numbers.",
    parameters_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    },
    func=lambda a, b: a + b,
)

loop = AgentToolLoop(
    llm_runtime=LLMRuntime(provider=ScriptedProvider()),
    tool_runtime=ToolRuntime(registry=registry),
)
result = loop.run(
    messages=[{"role": "user", "content": "What is 20 plus 22?"}],
    tools=registry.llm_tool_definitions(),
)

assert result.completed and result.final_response is not None
print(result.final_response.content)
```

The important boundary is explicit: the registry both authorizes the local
callable and supplies the definition advertised to the LLM. `LLMRuntime` alone
never executes a tool.

## 3. Enable resource limits

Limits can be attached to the loop configuration or supplied for one run:

```python
from agentos.runtime import AgentToolLoopConfig, ToolResourceLimits

limits = ToolResourceLimits(
    max_tool_calls_per_loop=4,
    max_tool_calls_per_round=2,
    max_calls_per_tool={"add_numbers": 2},
    tool_timeout_ms=5_000,
)

loop = AgentToolLoop(
    llm_runtime=LLMRuntime(provider=ScriptedProvider()),
    tool_runtime=ToolRuntime(registry=registry),
    config=AgentToolLoopConfig(tool_resource_limits=limits),
)
```

To see a deterministic denial after the first allowed call:

```powershell
python examples\agent_tool_loop_resource_limits_demo.py
```

Requested calls count toward limits even when they are denied. A synchronous
timeout is reported only after the callable returns; it does not interrupt a
stuck Python function.

## 4. Add resumable approval

Approval pauses the loop before any pending tool executes:

```python
from agentos.runtime import ToolApprovalDecision

paused = loop.run(
    messages=[{"role": "user", "content": "What is 20 plus 22?"}],
    tools=registry.llm_tool_definitions(),
    require_tool_approval=True,
)

assert paused.reason == "approval_required"
assert paused.checkpoint is not None
pending = paused.pending_approvals[0]
print(pending.tool_call_id, pending.tool_name)

resumed = loop.resume(
    checkpoint=paused.checkpoint,
    approval_decisions=[
        ToolApprovalDecision(pending.tool_call_id, approved=True),
    ],
)
assert resumed.completed
```

Run the complete approve/deny example:

```powershell
python examples\agent_tool_loop_approval_resume_demo.py
```

An incomplete decision set remains paused and executes nothing. Applications,
not Sulcus, decide how a human or policy engine supplies decisions.

## 5. Save and resume across a restart

To persist instead of resuming in memory, start a fresh run through step 4 and
save its paused checkpoint before supplying any decisions:

```python
from agentos.checkpoints import (
    inspect_checkpoint,
    resume_checkpoint,
    save_checkpoint,
)

save_checkpoint(paused.checkpoint, "approval.checkpoint.json")
metadata = inspect_checkpoint("approval.checkpoint.json")
print(metadata.checkpoint_id, metadata.required_tools)

# In a later process, reconstruct an equivalent loop, provider, and registry.
# `fresh_loop` must have the same registered tool description and schema.
completed = resume_checkpoint(
    fresh_loop,
    "approval.checkpoint.json",
    [ToolApprovalDecision(pending.tool_call_id, approved=True)],
)
```

The complete copy-paste restart simulation below constructs `fresh_loop`, fresh
providers, and a fresh registry, then proves the original LLM request is not
repeated:

```powershell
python -m examples.agent_tool_loop_persistent_checkpoint_demo
```

A complete file-backed resume renames the source to `.consumed`. Checkpoint
files contain message content, arguments, and results; protect them as
sensitive application data. See [Persistent checkpoints](checkpoints.md).

## 6. Run the flagship workflow

```powershell
sulcus demo research-team
```

Then expose more runtime controls:

```powershell
sulcus demo research-team --parallel --tight-limits --show-timeline
sulcus demo research-team --approve-publish
```

The default run denies simulated publication. The demo uses bundled sources and
scripted decisions, so results are deterministic and offline.

## Next steps

- Read [Concepts](concepts.md) for how the pieces relate.
- Read [Architecture](architecture.md) for supervision, IPC, and native services.
- Browse [Examples](examples.md) by capability and runtime requirement.
- Add project defaults with [Configuration](configuration.md).
