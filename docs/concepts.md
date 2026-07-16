# Concepts

This guide explains relationships between the runtime pieces. It is not an API
catalog; use [Public API](public_api.md) for import boundaries.

## AgentProcess and supervision

`AgentProcess` is a process-like agent with startup, message, and shutdown
hooks. When attached to a process registry it has a PID, mailbox, runtime
services, and observable lifecycle. A supervisor owns child processes and
chooses which children to restart after failure. Restart policies determine
whether one child is eligible; supervisor strategies determine the affected
set. Restart budgets and backoff prevent unbounded crash loops.

## ToolRegistry and ToolRuntime

`ToolRegistry` is the authority for executable tool names. A registration binds
a stable name and JSON-schema-like description to one Python callable.
`ToolRuntime` looks up only registered tools, validates arguments, executes the
callable, and returns a structured result. Passing a tool definition to an LLM
advertises it; registration is what authorizes local execution.

## AgentToolLoop

`AgentToolLoop` coordinates a bounded conversation between `LLMRuntime` and
`ToolRuntime`. It sends tool definitions to the model, preflights requested
calls, executes allowed calls, returns sanitized results to the model, and
stops on a final response, approval pause, error, or step limit. Creating a
plain `LLMRuntime` never enables automatic tool execution.

## Execution modes

Tool calls run sequentially by default. Parallel execution is requested with
`tool_execution_mode="parallel"`; only tools registered as `parallel_safe` can
run concurrently, and the loop reports a safe fallback when appropriate.

Agent processes have a separate execution-mode concept: trusted in-process
`asyncio` tasks versus spawned Python child processes. Process isolation
reduces shared state but is not a security sandbox. Native WASM execution is a
different, constrained feature.

## Permission policies and resource limits

`ToolPermissionPolicy` answers whether a named tool is allowed. Explicit deny
rules win; an allowlist can change the default to deny. Permissions are checked
before execution and produce structured denials.

`ToolResourceLimits` bounds requested calls per loop, per round, or per tool,
and can flag calls that exceed a duration. Denied requests still count toward
call budgets. The dependency-free synchronous timeout check occurs after the
callable returns; it does not terminate a stuck function.

## Resumable approval

With approval enabled, allowed calls pause before execution. The result exposes
only safe pending metadata plus an in-memory checkpoint. The application must
provide explicit decisions keyed by tool-call ID. A complete set resumes the
same round; an incomplete set stays paused and executes nothing. This makes
approval an application-controlled lifecycle rather than terminal UI logic.

## Persistent checkpoints

`agentos.checkpoints` writes a paused approval checkpoint to a versioned local
JSON file. A later process reconstructs a compatible loop and registry, then
resumes without repeating the LLM request that produced the pending calls.
Compatibility is based on tool names, descriptions, schemas, execution safety,
and available provider/model identity—not serialized callables.

## Configuration

An optional `sulcus.toml` in the current working directory supplies project
defaults for loop execution mode, approval, common resource limits, and LLM
provider/model labels. Precedence is explicit Python/CLI arguments, environment
variables, file values, then runtime defaults. Configuration is validated
strictly and never stores provider API keys.

## Runtime timeline

Runtime components emit structured events to an event sink. Together those
events form a timeline of LLM requests, tool preflight and execution, approval,
limit decisions, failures, and completion. Event metadata is deliberately
sanitized. The native-backed dashboard visualizes additional process, IPC,
memory, stream, and cost signals; the event model itself is usable in Python.

## Stability map

- **Intended stable:** the compact `agentos` facade and the documented
  `agentos.runtime`, `agentos.tools`, `agentos.ipc`, and `agentos.native`
  modules.
- **Advanced:** `agentos.llm`, including provider routing, streaming, caching,
  usage budgets, and cost accounting.
- **Documented workflow APIs:** `agentos.config` and `agentos.checkpoints`.
- **Experimental/internal:** `kernel.*`, dashboard composition, native services,
  WASM toolchain, replay/dependency views, and external-agent loading.

Version 1.0.0rc1 freezes the intended stable surface for release-candidate
validation. Advanced and internal surfaces retain the qualifications above.
