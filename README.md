# Agent OS
![Python 3.14](https://img.shields.io/badge/Python-3.14-blue) ![Rust Core](https://img.shields.io/badge/Rust-Core-orange) ![83+ Tests Passing](https://img.shields.io/badge/Tests-83%2B%20Passing-brightgreen) ![Active Development](https://img.shields.io/badge/Status-Active%20Development-green)

An experimental runtime for hierarchical multi-agent systems featuring supervision, IPC, memory paging, fault tolerance, and an interactive dashboard.

<p align="center">
  <img src="docs/AgentOS_demo.gif" width="900">
</p>

A lightweight hybrid runtime that combines:

- A Rust core crate (`agent_os_core`)
- Python orchestration (`main.py`, `kernel/`)
- Optional WASM sandboxed execution for constrained agent code

## Try the demos

Launch the interactive Agent OS dashboard:

```powershell
python main.py
```

Then use the dashboard shell:

```text
AgentOS> demos
AgentOS> run examples/research_team
AgentOS> run demos/supervisor_recovery
AgentOS> run demos/memory_paging
```

| Demo | Command | Demonstrates |
|---|---|---|
| Research Team | `run examples/research_team` | Multi-agent workflow orchestration, IPC flow, and agent hierarchy visualization |
| Supervisor Recovery | `run demos/supervisor_recovery` | Child termination detection, supervisor restart, and fault tolerance |
| Memory Paging | `run demos/memory_paging` | Context/page allocation visualization |

The dashboard visualizes the Agent Tree View, IPC Mailbox Lane Monitor,
Process Registry, workflow or recovery status, and WASM isolation status.

## Project Structure

- `src/` Rust crate source
- `kernel/` Python runtime modules (processes, dashboard, toolchain, LLM integration)
- `tests/` Test suite
- `examples/` Example assets/configurations
- `docs/` Documentation
- `main.py` Python entrypoint
- `Cargo.toml` Rust crate config

## Documentation

- `docs/sdk_quickstart.md` is the 15-minute developer guide for writing agents.
- `docs/interactive_shell.md` covers the process shell and isolation modes.
- `docs/ipc_protocol.md` covers the structured Agent-to-Agent IPC protocol.
- `docs/supervision.md` covers supervisor trees and restart policies.
- `docs/persistent_memory.md` covers tiered persistent memory paging.

## External agents preview

v0.7 begins support for running user-provided agents through an `agentos.toml`
manifest. Inspect a project from the Agent OS dashboard shell:

```text
AgentOS> inspect ./examples/external_basic_agent
AgentOS> run ./examples/external_basic_agent
```

v0.7 currently supports inspecting and running local Python `basic` agents only.

## Requirements

- Python 3.10+
- Rust (stable toolchain)
- Cargo

## Quick Start

For Windows test/development setup, see `docs/windows_dev_setup.md`.

### 1) Python environment

```powershell
py -3.14 -m venv .venv
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

## Self-Healing Multi-Agent Demo

Start the dashboard with `python main.py`, then run:

```text
run examples/agent_os_demo_supervisor.py
ps
```

The demo supervisor launches a coordinator, a persistent-memory agent, an
isolated worker, and an isolated crash probe. The coordinator exercises
structured request/reply IPC, records recalled cold memory, intentionally
crashes the probe, and verifies the restarted replacement. Keep the dashboard
open to watch child PIDs, restart counts, mailbox counters, and paged memory.

For a finite headless smoke run that prints the same telemetry as JSON:

```powershell
python -m examples.run_agent_os_demo
```

## Research Team Demo

Run the deterministic multi-agent research showcase:

```powershell
python -m examples.research_team.research_team
```

It demonstrates typed IPC contracts, planner fan-out, research fan-in,
synthesis, and critic review without API keys or external services. See
`examples/research_team/README.md` for the architecture and expected output.

## Supervisor Recovery Demo

From the Agent OS dashboard shell, run:

```text
AgentOS> run demos/supervisor_recovery
```

This deterministic dashboard demo shows basic fault tolerance:

```text
child termination -> supervisor detection -> automatic restart
```

### Agent Tree View

Visualizes supervisor-to-agent relationships for active workflows.

## Notes

- Runtime behavior is configurable through environment variables used in `main.py`.
- Logs are written to files such as `agent_runtime.log` and `agent_debug.log`.

### Structured runtime events

Agent OS emits structured runtime events for dashboard observability. These
events power future filtering, metrics, replay, and debugging features.

### Runtime timeline

The dashboard includes a chronological Runtime Timeline view for structured
events, with compact lifecycle and metadata summaries.

### Agent metrics

Step 20 added an Agent Metrics Panel for compact per-agent runtime health and
activity inspection, building on structured runtime events and the runtime timeline.

### IPC inspector

Step 21 adds an IPC Inspector for visualizing communication between agents.

### Execution replay

Step 22 adds Execution Replay for deterministic playback of recorded runtime
events. Replay is intended for debugging, demonstrations, and observability and
does not influence runtime execution.

### Agent dependency graph

Step 23 adds an Agent Dependency Graph for inspecting observed workflow
structure and communication dependencies between agents. It is observability-only
and does not introduce dependency scheduling yet.

### LLM Runtime Layer

Step 24 starts the v0.8.0 LLM Runtime Layer with a provider-agnostic interface
and deterministic test provider. This is the foundation for future real
providers; it requires no API keys or network access.

Step 25 adds an optional OpenAI-compatible provider adapter for real LLM calls.
It supports OpenAI and configurable compatible APIs such as Groq or OpenRouter
through `AGENTOS_LLM_API_KEY`, `AGENTOS_LLM_BASE_URL`, `AGENTOS_LLM_MODEL`, and
`AGENTOS_LLM_PROVIDER`. The OpenAI SDK is loaded only when the adapter is used,
and tests remain offline and deterministic.

```python
from kernel.llm import LLMRuntime, OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    api_key="placeholder-key",
    base_url="https://api.openai.com/v1",
    default_model="gpt-4.1-mini",
)

runtime = LLMRuntime(provider=provider)
response = runtime.chat(messages=[{"role": "user", "content": "Hello"}])
```

Step 26 adds provider routing and fallback support for the LLM Runtime Layer.
Registry-based runtimes select providers by stable route name and try configured
fallbacks in deterministic order when a provider raises `LLMProviderError`.
Routing and fallback events contain safe metadata only.

```python
runtime = LLMRuntime(
    providers={
        "primary": OpenAICompatibleProvider(
            api_key="placeholder-openai-key",
            default_model="gpt-4.1-mini",
        ),
        "fast": OpenAICompatibleProvider(
            api_key="placeholder-groq-key",
            provider_name="groq",
            base_url="https://api.groq.com/openai/v1",
            default_model="openai/gpt-oss-20b",
        ),
    },
    default_provider="primary",
    fallback_providers=["fast"],
)

response = runtime.chat(messages=[{"role": "user", "content": "Hello"}])
```

Step 27 adds deterministic LLM retry policies and timeout support. Retries are
applied to each provider before the runtime moves to the next configured
fallback. The default policy remains one attempt with no retry.

```python
from kernel.llm import LLMRetryPolicy, LLMRuntime

runtime = LLMRuntime(
    providers={...},
    default_provider="primary",
    fallback_providers=["fast"],
    retry_policy=LLMRetryPolicy(
        max_attempts=2,
        retry_on=("timeout", "rate_limit", "transient"),
    ),
    timeout_seconds=30,
)

response = runtime.chat(messages=[{"role": "user", "content": "Hello"}])
```

Step 28 adds LLM token budget and usage guardrails. Budgets are optional, use
only known provider-reported usage, and introduce no heavy tokenizer
dependency. A budget overrun is a runtime guardrail failure, not a provider
failure, so it does not trigger retries or fallbacks.

```python
from kernel.llm import LLMRuntime, LLMTokenBudget

runtime = LLMRuntime(
    providers={...},
    default_provider="primary",
    token_budget=LLMTokenBudget(
        name="demo-budget",
        max_prompt_tokens=20_000,
        max_completion_tokens=5_000,
        max_total_tokens=25_000,
    ),
)

response = runtime.chat(messages=[{"role": "user", "content": "Hello"}])
usage = runtime.usage_snapshot()
```

Step 29 adds optional deterministic LLM response caching. Caching is disabled by
default, and the default in-memory cache adds no external dependencies. Cache
hits do not call providers or double-count provider usage. Cache events contain
only safe provider/model fields and short opaque hashes, never prompt contents.

```python
from kernel.llm import LLMResponseCache, LLMRuntime

runtime = LLMRuntime(
    providers={...},
    default_provider="primary",
    cache=LLMResponseCache(enabled=True),
)

first = runtime.chat(messages=[{"role": "user", "content": "Hello"}])
second = runtime.chat(messages=[{"role": "user", "content": "Hello"}])  # cached
stats = runtime.cache_snapshot()
```

Cache keys include the provider route, model, messages, temperature, and
supported `max_tokens` option. A fallback response is cached under the provider
route that succeeded. Set request metadata `cache=False` (or
`options={"cache": False}`) to opt out for one request.

Step 30 adds a provider-neutral LLM streaming interface. Providers may opt into
streaming support while existing completion-only providers remain compatible.
Retries and fallbacks are allowed only before the first chunk is yielded.

```python
for chunk in runtime.stream_chat(messages=[{"role": "user", "content": "Hello"}]):
    print(chunk.delta, end="")
```

Streamed text is not logged by default; stream events contain safe values such
as provider, model, chunk index, and delta character count. Streaming bypasses
the response cache for now and never caches partial chunks. Provider-reported
usage is counted once when final usage is available after stream completion.
The OpenAI-compatible adapter remains completion-only in Step 30 and fails
streaming requests with a clean unsupported-streaming error.

Step 31 adds an LLM Stream Monitor dashboard panel for safe streaming
observability. The monitor is built entirely from safe Step 30 RuntimeEvents and
shows provider, model, status, chunk and character counts, usage, and sanitized
errors. Streamed text and prompts are not displayed by default. The panel is
useful for live debugging and demos while preserving dashboard scroll behavior.

Step 32 adds optional LLM cost accounting based on configured provider/model
rates. Pricing is user-configured, missing or incomplete usage produces no cost
estimate, cache hits do not double-count, and no billing API or network access
is used. The LLM Cost Monitor dashboard panel summarizes safe cost events.

```python
from kernel.llm import LLMCostRate, LLMCostTable, LLMRuntime

cost_table = LLMCostTable([
    LLMCostRate(
        provider="openai",
        model="gpt-4.1-mini",
        prompt_per_1m_tokens=0.40,
        completion_per_1m_tokens=1.60,
    )
])

runtime = LLMRuntime(
    providers={...},
    cost_table=cost_table,
)

response = runtime.chat(messages=[...])
costs = runtime.cost_snapshot()
```

Step 33 adds provider-neutral LLM tool-calling support. Agents can describe
safe callable tools structurally with `LLMToolDefinition` and
`parameters_schema`; providers that support tool calling map those definitions
into their native schema and map returned tool-call requests back into neutral
`LLMToolCall` objects on `LLMResponse.tool_calls`.

```python
from kernel.llm import LLMRuntime, LLMToolDefinition

tool = LLMToolDefinition(
    name="get_weather",
    description="Get weather for a city.",
    parameters_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)

response = runtime.chat(
    messages=[{"role": "user", "content": "Check Warsaw weather"}],
    tools=[tool],
    tool_choice="auto",
)

for tool_call in response.tool_calls:
    print(tool_call.name, tool_call.arguments)
```

Tool calls are returned but not executed automatically in Step 33. Tool
execution will be a future runtime feature. Runtime events only expose safe
metadata such as provider, model, tool counts, and tool names; prompts, tool
arguments, API keys, headers, and raw provider responses are not logged by
default.

Step 34 adds a safe Tool Execution Runtime for registered tools. Tools must be
explicitly registered with `ToolRegistry`; Agent OS does not import tools by
name, provide shell/network/file tools by default, or automatically execute LLM
tool calls from `LLMRuntime.chat`. Execution is explicit through `ToolRuntime`,
which validates required fields and basic JSON-schema-like types before calling
the approved callable.

```python
from kernel.tools import ToolRegistry, ToolRuntime

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
    },
    func=lambda a, b: a + b,
)

runtime = ToolRuntime(registry=registry)
result = runtime.execute("add_numbers", {"a": 2, "b": 3})
```

`ToolRuntime` can also execute a Step 33 `LLMToolCall` explicitly and convert
`ToolExecutionResult` back to `LLMToolResult` for future agent tool loops. Tool
events contain safe metadata such as tool name, argument keys, duration, and
error category; argument values, stack traces, prompts, API keys, and raw
exceptions are not logged by default. Timeout fields are part of the API, with
the current dependency-free implementation checking elapsed synchronous runtime
after the callable returns rather than interrupting Python execution mid-call.

Step 35 adds a safe Agent Tool Loop for bounded LLM-tool orchestration.
`LLMRuntime.chat` still does not execute tools automatically; callers must
explicitly create and invoke `AgentToolLoop`. The loop sends tool definitions to
the LLM, detects returned tool calls, executes only registered `ToolRuntime`
tools, feeds sanitized tool results back to the LLM, and stops at a final
response, a pending approval point, a tool error, or `max_steps`.

```python
from kernel.agent_tool_loop import AgentToolLoop

loop = AgentToolLoop(
    llm_runtime=llm_runtime,
    tool_runtime=tool_runtime,
)

result = loop.run(
    messages=[
        {"role": "user", "content": "Calculate 15 + 27 using tools."}
    ],
    tools=[add_numbers_tool],
    max_steps=4,
)

print(result.final_response.content)
```

The default loop configuration is conservative and deterministic:
`max_steps=4`, no parallel tool execution, stop on tool errors, and intermediate
steps included in the structured result. Approval mode
(`require_tool_approval=True`) returns pending tool calls without executing
them, ready for a future UI approval flow. Safe `RuntimeEvent`s expose only
metadata such as step index, provider/model, tool names, tool counts, success
markers, and error categories; prompts, argument values, API keys, headers, raw
provider responses, and stack traces are not logged by default.

### Agent Tool Loop Demo

Step 36 stabilizes the Phase 6 Agent Tool Loop demo as a regression foundation.
Run the live smoke test with an OpenAI-compatible provider configured:

```powershell
$env:AGENTOS_LLM_API_KEY = "..."
python examples/agent_tool_loop_phase6_smoke_test.py
```

Expected output shows `Completed: True`, `Reason: completed`, one successful
`add_numbers` tool execution result with content `"42"`, a final LLM response
that mentions `42`, and `Phase 6 Agent Tool Loop smoke test passed.`

The demo proves the first verified Sulcus OS flow where:

```text
Agent -> LLM -> tool call -> tool runtime -> tool result -> LLM final response
```

This matters because it confirms that the LLM runtime, registered tool runtime,
and bounded agent orchestration loop can work together without letting
`LLMRuntime.chat` execute tools implicitly. Live provider calls stay in examples
and manual smoke tests. CI coverage uses deterministic providers and does not
require API keys or internet access.

For an offline multi-tool demo, run:

```powershell
python examples/agent_tool_loop_multi_tool_demo.py
```

That scripted demo verifies a single agent tool loop run can execute both
`add_numbers` and `multiply_numbers`, feed both tool execution results back to
the LLM, and produce a final LLM response containing both results.

### Agent Tool Loop Timeline

Step 37 adds structured runtime timeline events for the Agent Tool Loop. The
timeline proves that Sulcus OS observed each phase of a loop run, not just the
final answer: loop start, LLM request, LLM response, requested tool calls, tool
execution, final LLM request, final LLM response, and loop completion or
failure.

Run the offline deterministic demo:

```powershell
python examples/agent_tool_loop_multi_tool_demo.py
```

The timeline section prints a sequence like:

```text
Timeline:
1. agent_tool_loop_started
2. llm_request_started
3. llm_response_received
4. tool_execution_group_started execution_mode=sequential
5. tool_call_requested add_numbers execution_mode=sequential
6. tool_execution_started add_numbers execution_mode=sequential
7. tool_execution_completed add_numbers execution_mode=sequential
8. tool_call_requested multiply_numbers execution_mode=sequential
9. tool_execution_started multiply_numbers execution_mode=sequential
10. tool_execution_completed multiply_numbers execution_mode=sequential
11. tool_execution_group_completed execution_mode=sequential
12. llm_followup_request_started
13. llm_final_request_started final_attempt=False
14. llm_followup_response_received
15. llm_final_response_received
16. agent_tool_loop_completed
```

These events matter because dashboard and demo users can inspect the verified
flow as structured runtime history instead of inferring it from print output.
Live provider smoke tests still require `AGENTOS_LLM_API_KEY`; deterministic
timeline tests run offline without API keys or internet access.

### Multi-Round Agent Tool Loop

Step 38 extends the Agent Tool Loop to keep running across multiple LLM/tool
rounds until the LLM returns a final assistant response with no tool calls, a
tool or provider error occurs, or `max_steps` is reached. One LLM response
counts as one round, and tool calls within that response are executed
sequentially.

Run the offline deterministic multi-round demo:

```powershell
python examples/agent_tool_loop_multi_round_demo.py
```

Example flow:

```text
Round 1: LLM asks for add_numbers(20, 22)
Round 1: Tool result is 42
Round 2: LLM asks for multiply_numbers(42, 2)
Round 2: Tool result is 84
Round 3: LLM returns "The final answer is 84."
```

This matters because real agents often need more than one reasoning/tool cycle.
Sulcus OS can now supervise those cycles with structured timeline events such
as `llm_followup_request_started`, `llm_followup_response_received`, and
round-indexed tool execution events. Live provider smoke tests still require
`AGENTOS_LLM_API_KEY`, and parallel tool execution remains future work.

### Parallel Tool Execution

Sequential execution remains the default. Set
`tool_execution_mode="parallel"` to request group-level parallel execution
within one LLM response:

```python
config = AgentToolLoopConfig(tool_execution_mode="parallel")
```

Sulcus OS only executes a tool-call group concurrently when every requested
tool is registered with `parallel_safe=True`. If any requested tool is missing
from the registry or is not marked parallel-safe, the group safely falls back to
sequential execution with
`fallback_reason="not_all_tools_parallel_safe"`.

The timeline exposes both requested and effective execution mode on group and
per-tool events:

- `requested_execution_mode`
- `effective_execution_mode`
- `fallback_reason`, when a parallel request falls back
- `parallel_safe_tool_count`
- `unsafe_tool_count`

The legacy `execution_mode` metadata key is preserved and reflects the
effective mode. Tool result order remains deterministic: `result.tool_results`
and the tool messages sent to the next LLM round follow the original LLM
tool-call order even if parallel tools finish out of order.

Parallel execution is currently scoped to one tool-call group from a single LLM
response. Multi-round execution still waits for all results from the current
group before making the next LLM call. Live provider smoke tests still require
`AGENTOS_LLM_API_KEY`.

Run the offline deterministic parallel demo:

```powershell
python examples/agent_tool_loop_parallel_tool_demo.py
```

Mini example:

```text
Round 1: LLM requests add_numbers and multiply_numbers
Sulcus OS emits tool_execution_group_started with requested=parallel effective=parallel
Sulcus OS executes both calls concurrently when both tools are parallel-safe
Sulcus OS emits tool_execution_group_completed with deterministic result order
```

### Tool Permission Policies

By default, the Agent Tool Loop remains permissive for backwards compatibility.
If no policy is configured, registered tools advertised to the run behave as
they did before. A `ToolPermissionPolicy` can make tool access explicit with an
allowlist, a denylist, or both:

```python
from kernel.agent_tool_loop import AgentToolLoopConfig, ToolPermissionPolicy

permissive = ToolPermissionPolicy()

allowlist = ToolPermissionPolicy(
    default_allow=False,
    allowed_tools={"add_numbers"},
)

denylist = ToolPermissionPolicy(
    default_allow=True,
    denied_tools={"delete_file"},
)

config = AgentToolLoopConfig(tool_permission_policy=allowlist)
```

Policy semantics are intentionally small:

- `default_allow=True` allows every tool unless it appears in `denied_tools`.
- `default_allow=False` allows only tools listed in `allowed_tools`.
- `denied_tools` always wins over `allowed_tools`.
- Policies apply across all rounds and all execution modes.

Denied calls are represented as normal failed tool results with an error such
as `Tool call denied by permission policy: multiply_numbers`. The denied tool
function is not executed, and Sulcus OS emits `tool_call_denied` after the
usual `tool_call_requested` event. Group failure metadata includes
`failed_tool_count` and `denied_tool_count`, while loop start metadata includes
a safe policy summary: `tool_policy_enabled`, `policy_default_allow`,
`allowed_tool_count`, and `denied_tool_count`.

In parallel mode, denied tools are never submitted to the executor. Allowed
tools in the same group keep the existing parallel behavior and deterministic
result ordering; the group is marked failed if any tool call was denied.

Run the offline deterministic permission demo:

```powershell
python examples/agent_tool_loop_tool_permission_demo.py
```

### Tool Resource Limits

Tool usage is unlimited by default for backwards compatibility. Configure
`ToolResourceLimits` when an agent loop needs explicit runtime safety controls:

```python
from kernel.agent_tool_loop import AgentToolLoopConfig, ToolResourceLimits

config = AgentToolLoopConfig(
    tool_resource_limits=ToolResourceLimits(max_tool_calls_per_loop=4)
)

per_round = ToolResourceLimits(max_tool_calls_per_round=2)
per_tool = ToolResourceLimits(max_calls_per_tool={"search_docs": 3})
timeout = ToolResourceLimits(tool_timeout_ms=500)
```

Supported limits:

- `max_tool_calls_per_loop`: caps requested tool calls across the whole run.
- `max_tool_calls_per_round`: caps requested tool calls in one LLM response.
- `max_calls_per_tool`: caps requested calls by tool name across the run.
- `tool_timeout_ms`: caps actual tool execution duration per call.

`None` means unlimited. Zero is valid and means no calls are allowed for that
limit. Call-count limits count requested tool calls, including
permission-denied and resource-denied calls, because the agent attempted to use
runtime tool capacity. Limits apply across multi-round loops.

Resource-denied calls return normal failed `LLMToolResult` values with errors
such as `Tool call denied by resource limits: max_tool_calls_per_loop exceeded`.
They do not execute the tool and do not emit `tool_execution_started`. Sulcus OS
emits `tool_call_requested` first, then `tool_call_resource_denied`.

Timeouts happen after execution starts. A timed-out tool emits
`tool_execution_started` followed by `tool_execution_failed` with
`error_type="ToolTimeoutError"`, `error_category="timeout"`, and
`limit_name="tool_timeout_ms"`. Synchronous Python tool functions run through a
thread when an agent-loop timeout is configured; Python cannot forcibly kill a
stuck thread, so the loop returns a timeout result while the underlying function
may finish later.

Permissions and resource limits work together. Permission denial wins over
resource denial for a single call, so forbidden tools emit `tool_call_denied`
rather than `tool_call_resource_denied`. In parallel mode, permission-denied and
resource-denied calls are not submitted to the executor, while result ordering
still follows the original LLM tool-call order.

Loop start metadata includes a safe resource-limit summary:
`tool_resource_limits_enabled`, `max_tool_calls_per_loop`,
`max_tool_calls_per_round`, `max_calls_per_tool_count`, and `tool_timeout_ms`.
Group metadata includes `resource_limits_enabled`,
`resource_denied_tool_count`, and `timed_out_tool_count`.

Run the offline deterministic resource-limit demo:

```powershell
python examples/agent_tool_loop_resource_limits_demo.py
```
