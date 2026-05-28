# Architecture Diagram

```mermaid
flowchart TD
    U["User / Operator"] --> PY["Python Host Runtime (main.py)"]
    PY --> DASH["Dashboard UI (kernel/dashboard.py)"]
    PY --> PROC["Process Registry (kernel/process.py)"]
    PY --> LLM["LLM Manager (kernel/llm.py)"]
    PY --> TOOL["Toolchain Compiler (kernel/toolchain.py)"]

    PY <--> IPC["Native IPC Bus (Rust: src/ipc.rs)"]
    PY <--> MEM["Context Memory Manager (Rust: src/memory.rs)"]
    PY <--> RUST["Rust Kernel API (src/lib.rs)"]

    RUST --> WASM["WASM Sandbox Manager (src/sandbox.rs)"]
    TOOL --> WASM
    WASM --> AGENTS["Sandboxed Agent Kernels"]

    PROC --> RUNNER["Isolated Runner (kernel/process_runner.py)"]
    RUNNER --> AGENTS

    AGENTS --> LOGS["Runtime Logs (agent_runtime.log / agent_debug.log)"]
    DASH --> LOGS
```

## Runtime Boundaries

- Host orchestration: Python (`main.py`, `kernel/`)
- Performance-critical + sandbox internals: Rust (`src/`)
- Execution isolation: WASM sandbox and optional process isolation

## Key Data Flows

1. User input enters `main.py`.
2. Python host routes work to LLM/toolchain and process manager.
3. Rust core provides IPC, memory, and sandbox primitives.
4. Compiled/validated agent code runs inside WASM sandbox.
5. Results and telemetry flow to dashboard and logs.
