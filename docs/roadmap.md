# Roadmap

## Q2 2026: Stability Foundation

- Harden process lifecycle and crash recovery in `kernel/process.py` and `kernel/process_runner.py`.
- Expand test coverage for IPC and process isolation failure paths.
- Standardize structured logging format for `agent_runtime.log` and `agent_debug.log`.
- Document environment variables and defaults in a dedicated config reference.

## Q3 2026: Sandbox + Toolchain Maturity

- Improve `kernel/toolchain.py` compile diagnostics and user-facing errors.
- Add deterministic execution checks for generated sandbox code.
- Benchmark and tune WASM fuel/memory defaults (`AGENT_OS_SANDBOX_FUEL`, mailbox sizing).
- Add regression tests for unsafe/unsupported code patterns.

## Q4 2026: Developer Experience

- Add one-command local setup and validation script.
- Provide richer example agents (`examples/`) for common patterns (stateful, resilient, long-running).
- Add architecture decision records under `docs/` for major runtime choices.
- Improve dashboard observability with execution timeline and per-agent health.

## Q1 2027: Production Readiness

- Introduce metrics export hooks (latency, token usage, crash rate, retry rate).
- Add graceful degradation strategies when LLM/toolchain is unavailable.
- Define versioned agent manifest schema and migration strategy.
- Prepare release checklist and CI quality gates for reproducible builds.

## Ongoing Backlog

- Security review of host/sandbox boundary.
- Performance profiling under concurrent multi-agent load.
- Better defaults for manifest and process isolation modes.
