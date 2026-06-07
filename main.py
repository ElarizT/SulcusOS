from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import signal
import shlex
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_os_core import (
    AgentMessage,
    ContextMemoryManager,
    NativeIPCBus,
    RustKernel,
    WasmSandboxManager,
)
from kernel.dashboard import SHELL_PROMPT, AgentOSDashboard
from kernel.memory_store import PersistentMemoryManager
from kernel.process import ProcessRegistry
from kernel.shell_help import (
    DEMO_COMMANDS,
    format_demo_browser,
    format_shell_help,
    is_memory_paging_demo_path,
    is_supervisor_recovery_demo_path,
)

try:
    from kernel.llm import AsyncLLMManager, LLMConfig, normalize_code_block
except ImportError:
    AsyncLLMManager = None
    LLMConfig = None
    normalize_code_block = None

try:
    from kernel.toolchain import compile_agent_script
except ImportError:
    compile_agent_script = None


AGENTS_MANIFEST_PATH = Path(os.getenv("AGENT_OS_AGENTS_MANIFEST", "agents.json"))
AGENT_RUNTIME_LOG = Path("agent_runtime.log")
SYSTEM_AGENT_NAME = "System"
ORCHESTRATOR_AGENT_NAME = "Orchestrator"
DEFAULT_MAILBOX_SIZE = int(os.getenv("AGENT_OS_MAILBOX_SIZE", "1024"))
DEFAULT_AGENT_TOKEN_BUDGET = int(os.getenv("AGENT_OS_AGENT_TOKEN_BUDGET", "8000"))
DEFAULT_SANDBOX_FUEL = int(os.getenv("AGENT_OS_SANDBOX_FUEL", "50000"))
HOST_IDLE_SECONDS = float(os.getenv("AGENT_OS_HOST_IDLE_SECONDS", "0.25"))
HOST_RETRY_SECONDS = float(os.getenv("AGENT_OS_HOST_RETRY_SECONDS", "3.0"))
ENABLE_LEGACY_MANIFEST_AGENTS = os.getenv("AGENT_OS_ENABLE_LEGACY_AGENTS", "0") == "1"
AGENT_PROCESS_ROOT = Path(os.getenv("AGENT_OS_PROCESS_ROOT", ".")).resolve()
AGENT_PROCESS_ISOLATION = os.getenv("AGENT_OS_PROCESS_ISOLATION", "in-process")
AGENT_PROCESS_STARTUP_TIMEOUT = float(os.getenv("AGENT_OS_PROCESS_STARTUP_TIMEOUT", "5.0"))
AGENT_OS_MEMORY_DIR = Path(os.getenv("AGENT_OS_MEMORY_DIR", ".agent_os/memory"))

COMPILER_RULES = (
    "CRITICAL COMPILER RULES:\n"
    "- Return only sandbox-safe Python in a fenced ```python markdown block.\n"
    "- Define exactly one function named run with no parameters: def run():\n"
    "- Return one numeric scalar value.\n"
    "- Use only simple scalar assignments on separate lines. For example, write a = 0 then b = 1.\n"
    "- Never use tuple/list/dict/set unpacking or multiple assignment such as a, b = 0, 1.\n"
    "- Do not use imports, classes, decorators, async, comprehensions, lambdas, exceptions, or function calls.\n"
    "- Do not allocate strings, lists, dictionaries, sets, objects, or buffers in sandbox code.\n"
    "- Use synchronous byte-casting only at the host boundary; sandbox code must stay numeric and synchronous.\n"
    "- Do not use list method attachments like .append().\n"
    "- Keep loops flat: no nested if statements or early returns inside while loops.\n"
    "- Supported operations are arithmetic, comparisons, simple if/else, while loops, and numeric return.\n"
)

DEFAULT_AGENTS_MANIFEST: dict[str, Any] = {
    "agents": [
        {
            "name": "GeneralistAgent",
            "capabilities": ["general", "analysis", "planning", "reasoning"],
            "system_prompt": "You are a general-purpose runtime agent. Convert routed requests into compact sandbox-safe Python kernels when execution is useful.",
        },
        {
            "name": "CodeExecutionAgent",
            "capabilities": ["codegen", "compute", "execute", "math"],
            "system_prompt": "You specialize in producing tiny deterministic Python functions for the Agent OS WASM sandbox.",
        },
    ]
}


@dataclass(frozen=True)
class AgentSpec:
    name: str
    capabilities: tuple[str, ...]
    system_prompt: str


@dataclass(frozen=True)
class RouteDecision:
    agent_name: str
    capability: str
    score: int


class DynamicAgentRegistry:
    """Runtime view of manifest-backed agents and their routing capabilities."""

    def __init__(self, agent_specs: list[AgentSpec]) -> None:
        if not agent_specs:
            raise ValueError("at least one agent spec is required")
        self.agent_specs = agent_specs
        self._by_name = {agent.name: agent for agent in agent_specs}

    @property
    def names(self) -> list[str]:
        return [agent.name for agent in self.agent_specs]

    def get(self, agent_name: str) -> AgentSpec:
        return self._by_name[agent_name]

    def route(self, request_text: str) -> RouteDecision:
        normalized_request = normalize_route_text(request_text)
        best: RouteDecision | None = None

        for agent in self.agent_specs:
            for capability in agent.capabilities:
                score = score_capability_match(normalized_request, capability)
                if best is None or score > best.score:
                    best = RouteDecision(agent.name, capability, score)

        if best is None or best.score <= 0:
            fallback = self.agent_specs[0]
            capability = fallback.capabilities[0] if fallback.capabilities else "general"
            return RouteDecision(fallback.name, capability, 0)
        return best


def create_bus(kernel: RustKernel) -> NativeIPCBus:
    return NativeIPCBus(kernel)


def log_agent_exception(agent_name: str, stage: str) -> None:
    with AGENT_RUNTIME_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(f"LLM AGENT CRASH DETECTED [{agent_name}] during {stage}:\n")
        log_file.write(traceback.format_exc())
        log_file.write("\n" + "=" * 64 + "\n")


def build_llm_config() -> LLMConfig:
    if LLMConfig is None:
        raise RuntimeError("kernel.llm could not be imported")

    return LLMConfig(
        provider=os.getenv("AGENT_OS_LLM_PROVIDER", "openai"),
        model_name=os.getenv("AGENT_OS_LLM_MODEL", "gemini-2.5-flash-lite"),
        api_key=os.getenv("AGENT_OS_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("AGENT_OS_LLM_BASE_URL"),
    )


def load_agent_manifest(path: Path = AGENTS_MANIFEST_PATH) -> list[AgentSpec]:
    manifest = DEFAULT_AGENTS_MANIFEST
    if path.exists():
        with path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)

    if isinstance(manifest, list):
        raw_agents = manifest
    else:
        raw_agents = manifest.get("agents", [])
    if not isinstance(raw_agents, list):
        raise ValueError("agents manifest must contain an 'agents' list")

    specs: list[AgentSpec] = []
    seen_names: set[str] = set()
    for index, raw_agent in enumerate(raw_agents):
        if not isinstance(raw_agent, dict):
            raise ValueError(f"agent spec at index {index} must be an object")

        name = str(raw_agent.get("name", "")).strip()
        system_prompt = str(raw_agent.get("system_prompt", "")).strip()
        raw_capabilities = raw_agent.get("capabilities", [])

        if not name:
            raise ValueError(f"agent spec at index {index} is missing name")
        if name in {SYSTEM_AGENT_NAME, ORCHESTRATOR_AGENT_NAME}:
            raise ValueError(f"'{name}' is reserved by the Agent OS control plane")
        if name in seen_names:
            raise ValueError(f"duplicate agent name '{name}' in manifest")
        if not system_prompt:
            raise ValueError(f"agent '{name}' is missing system_prompt")
        if not isinstance(raw_capabilities, list):
            raise ValueError(f"agent '{name}' capabilities must be a list")

        capabilities = tuple(
            capability.strip().lower()
            for capability in (str(item) for item in raw_capabilities)
            if capability.strip()
        )
        if not capabilities:
            raise ValueError(f"agent '{name}' must declare at least one capability")

        specs.append(AgentSpec(name=name, capabilities=capabilities, system_prompt=system_prompt))
        seen_names.add(name)

    if not specs:
        raise ValueError("agents manifest did not define any runtime agents")
    return specs


def register_runtime_agents(
    registry: DynamicAgentRegistry,
    kernel: RustKernel,
    bus: NativeIPCBus,
    memory: ContextMemoryManager,
) -> None:
    bus.register_mailbox(SYSTEM_AGENT_NAME, DEFAULT_MAILBOX_SIZE)
    bus.register_mailbox(ORCHESTRATOR_AGENT_NAME, DEFAULT_MAILBOX_SIZE)
    kernel.register_agent_capability(ORCHESTRATOR_AGENT_NAME, "orchestration")
    memory.register_agent(SYSTEM_AGENT_NAME, DEFAULT_AGENT_TOKEN_BUDGET)
    memory.register_agent(ORCHESTRATOR_AGENT_NAME, DEFAULT_AGENT_TOKEN_BUDGET)

    for agent in registry.agent_specs:
        bus.register_mailbox(agent.name, DEFAULT_MAILBOX_SIZE)
        memory.register_agent(agent.name, DEFAULT_AGENT_TOKEN_BUDGET)
        for capability in agent.capabilities:
            kernel.register_agent_capability(agent.name, capability)


def register_control_plane(
    kernel: RustKernel,
    bus: NativeIPCBus,
    memory: ContextMemoryManager,
) -> None:
    bus.register_mailbox(SYSTEM_AGENT_NAME, DEFAULT_MAILBOX_SIZE)
    bus.register_mailbox(ORCHESTRATOR_AGENT_NAME, DEFAULT_MAILBOX_SIZE)
    kernel.register_agent_capability(ORCHESTRATOR_AGENT_NAME, "orchestration")
    memory.register_agent(SYSTEM_AGENT_NAME, DEFAULT_AGENT_TOKEN_BUDGET)
    memory.register_agent(ORCHESTRATOR_AGENT_NAME, DEFAULT_AGENT_TOKEN_BUDGET)


def normalize_route_text(text: str) -> set[str]:
    separators = "\n\t.,;:!?()[]{}<>/\\|+-*_='\"`"
    normalized = text.lower()
    for separator in separators:
        normalized = normalized.replace(separator, " ")
    return {token for token in normalized.split() if token}


def score_capability_match(request_tokens: set[str], capability: str) -> int:
    capability_tokens = normalize_route_text(capability.replace("-", " ").replace("_", " "))
    if not capability_tokens:
        return 0
    score = 0
    for token in capability_tokens:
        if token in request_tokens:
            score += 3
        elif any(token in request_token or request_token in token for request_token in request_tokens):
            score += 1
    return score


def build_agent_system_prompt(agent_name: str, system_prompt: str, capabilities: tuple[str, ...]) -> str:
    return (
        f"{system_prompt}\n\n"
        "You are running as a dynamically registered Agent OS runtime host.\n"
        f"Agent name: {agent_name}\n"
        f"Declared routing capabilities: {', '.join(capabilities)}\n\n"
        "When a task is routed to you, answer with the smallest useful sandbox program. "
        "If the request is descriptive, still provide a tiny numeric health/check kernel that represents completion.\n\n"
        f"{COMPILER_RULES}"
    )


def build_agent_context(
    message: AgentMessage,
    agent_name: str,
    capabilities: tuple[str, ...],
    memory: ContextMemoryManager,
) -> list[str]:
    try:
        payload: Any = json.loads(message.payload)
    except json.JSONDecodeError:
        payload = {"raw": message.payload}

    context = [
        json.dumps(
            {
                "mailbox_message": {
                    "sender": message.sender,
                    "receiver": message.receiver,
                    "payload": payload,
                },
                "agent_name": agent_name,
                "capabilities": list(capabilities),
                "compiler_contract": COMPILER_RULES,
            }
        )
    ]
    context.extend(memory.get_active_context(agent_name))
    return context


def execute_extracted_code(
    agent_name: str,
    response_text: str,
    code_blocks: list[str],
    memory: ContextMemoryManager,
    sandbox: WasmSandboxManager,
) -> list[dict[str, Any]]:
    if compile_agent_script is None:
        raise RuntimeError("compile_agent_script is unavailable; cannot execute LLM code")

    memory.append_context_frame(
        agent_name,
        json.dumps(
            {
                "event": "llm_response",
                "text": response_text[:2000],
                "code_blocks": len(code_blocks),
            }
        ),
        max(len(response_text) // 4, 1),
    )

    if not code_blocks:
        raise RuntimeError(f"{agent_name} LLM response contained no Python markdown code blocks")

    executions: list[dict[str, Any]] = []
    for index, code in enumerate(code_blocks):
        if normalize_code_block is not None:
            code = normalize_code_block(code)
        else:
            code = code.replace("\\\\n", "\n").replace("\\n", "\n")

        wasm_bytes = compile_agent_script(code)
        result = sandbox.execute_wasm_binary(bytes(wasm_bytes), DEFAULT_SANDBOX_FUEL)
        execution = {
            "event": "wasm_execution",
            "code_block": index,
            "success": result.success,
            "fuel_consumed": result.fuel_consumed,
            "error_message": result.error_message,
        }
        executions.append(execution)
        memory.append_context_frame(agent_name, json.dumps(execution), 64)

    return executions


async def universal_agent_host(
    agent_name: str,
    system_prompt: str,
    capabilities: tuple[str, ...],
    kernel: RustKernel,
    bus: NativeIPCBus,
    memory: ContextMemoryManager,
    sandbox: WasmSandboxManager,
    llm: AsyncLLMManager,
) -> None:
    """Generic runtime worker bound to one native Tokio-backed mailbox."""

    host_prompt = build_agent_system_prompt(agent_name, system_prompt, capabilities)

    while not kernel.is_shutting_down():
        try:
            message = await asyncio.wait_for(
                bus.recv_message(agent_name),
                timeout=HOST_IDLE_SECONDS,
            )
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
        except Exception:
            log_agent_exception(agent_name, "mailbox_receive")
            await asyncio.sleep(HOST_RETRY_SECONDS)
            continue

        try:
            context = build_agent_context(message, agent_name, capabilities, memory)
            response = await llm.generate_response(host_prompt, context)
            memory.append_context_frame(
                agent_name,
                json.dumps(
                    {
                        "event": "llm_usage",
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    }
                ),
                max(response.input_tokens + response.output_tokens, 1),
            )
            executions = execute_extracted_code(
                agent_name,
                response.text,
                response.extracted_code_blocks,
                memory,
                sandbox,
            )
            bus.send_message(
                AgentMessage(
                    agent_name,
                    ORCHESTRATOR_AGENT_NAME,
                    json.dumps(
                        {
                            "cmd": "agent_task_complete",
                            "agent": agent_name,
                            "source": message.sender,
                            "code_blocks": len(response.extracted_code_blocks),
                            "executions": executions,
                        }
                    ),
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log_agent_exception(agent_name, "universal_agent_host")
            with contextlib.suppress(Exception):
                bus.send_message(
                    AgentMessage(
                        agent_name,
                        ORCHESTRATOR_AGENT_NAME,
                        json.dumps(
                            {
                                "cmd": "agent_task_failed",
                                "agent": agent_name,
                                "source": message.sender,
                            }
                        ),
                    )
                )
            await asyncio.sleep(HOST_RETRY_SECONDS)


async def orchestration_router(
    registry: DynamicAgentRegistry,
    bus: NativeIPCBus,
    memory: ContextMemoryManager,
    stop_event: asyncio.Event,
) -> None:
    """Route top-level user/system requests to manifest-registered runtime agents."""

    while not stop_event.is_set():
        try:
            message = await asyncio.wait_for(
                bus.recv_message(ORCHESTRATOR_AGENT_NAME),
                timeout=HOST_IDLE_SECONDS,
            )
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
        except Exception:
            log_agent_exception(ORCHESTRATOR_AGENT_NAME, "router_receive")
            await asyncio.sleep(HOST_RETRY_SECONDS)
            continue

        try:
            payload = json.loads(message.payload)
        except json.JSONDecodeError:
            payload = {"task": message.payload}

        command = payload.get("cmd", "route_task")
        if command in {"agent_task_complete", "agent_task_failed"}:
            memory.append_context_frame(ORCHESTRATOR_AGENT_NAME, json.dumps(payload), 64)
            continue
        if command not in {"route_task", "user_request", "task", "codegen"}:
            memory.append_context_frame(
                ORCHESTRATOR_AGENT_NAME,
                json.dumps({"event": "ignored_message", "payload": payload}),
                32,
            )
            continue

        request_text = str(payload.get("task") or payload.get("request") or payload.get("prompt") or payload)
        route = registry.route(request_text)
        routed_payload = {
            "cmd": "execute_task",
            "task": request_text,
            "routed_capability": route.capability,
            "route_score": route.score,
            "source": message.sender,
            "original_payload": payload,
            "compiler_rules": COMPILER_RULES,
        }
        memory.append_context_frame(
            ORCHESTRATOR_AGENT_NAME,
            json.dumps({"event": "route", "target": route.agent_name, **routed_payload}),
            max(len(request_text) // 4, 1),
        )
        bus.send_message(
            AgentMessage(
                ORCHESTRATOR_AGENT_NAME,
                route.agent_name,
                json.dumps(routed_payload),
            )
        )


async def optional_stdin_ingress(bus: NativeIPCBus, stop_event: asyncio.Event) -> None:
    if os.getenv("AGENT_OS_ENABLE_STDIN", "0") != "1":
        await stop_event.wait()
        return

    while not stop_event.is_set():
        try:
            line = await asyncio.to_thread(input, f"{SHELL_PROMPT} ")
        except (EOFError, KeyboardInterrupt):
            stop_event.set()
            return
        if not line.strip():
            continue
        bus.send_message(
            AgentMessage(
                SYSTEM_AGENT_NAME,
                ORCHESTRATOR_AGENT_NAME,
                json.dumps({"cmd": "user_request", "request": line.strip()}),
            )
        )


async def drain_mailbox(bus: NativeIPCBus, agent_name: str, timeout: float = 0.02) -> int:
    drained = 0
    while True:
        try:
            await asyncio.wait_for(bus.recv_message(agent_name), timeout=timeout)
        except asyncio.TimeoutError:
            return drained
        drained += 1


def seed_boot_task(bus: NativeIPCBus, registry: DynamicAgentRegistry) -> None:
    task = os.getenv(
        "AGENT_OS_BOOT_TASK",
        "Calculate the 10th Fibonacci number using a sandbox-safe Python run function.",
    )
    bus.send_message(
        AgentMessage(
            SYSTEM_AGENT_NAME,
            ORCHESTRATOR_AGENT_NAME,
            json.dumps(
                {
                    "cmd": "route_task",
                    "task": task,
                    "available_agents": [
                        {"name": agent.name, "capabilities": list(agent.capabilities)}
                        for agent in registry.agent_specs
                    ],
                    "compiler_rules": COMPILER_RULES,
                }
            ),
        )
    )


async def main() -> None:
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    llm_manager = None

    def request_shutdown() -> None:
        stop_event.set()

    try:
        loop.add_signal_handler(signal.SIGINT, request_shutdown)
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda _signum, _frame: request_shutdown())

    agent_specs: list[AgentSpec] = []
    legacy_registry: DynamicAgentRegistry | None = None
    if ENABLE_LEGACY_MANIFEST_AGENTS:
        agent_specs = load_agent_manifest()
        legacy_registry = DynamicAgentRegistry(agent_specs)
    kernel = RustKernel()
    bus = create_bus(kernel)
    memory = PersistentMemoryManager(memory_dir=AGENT_OS_MEMORY_DIR)
    sandbox = WasmSandboxManager()
    if legacy_registry is not None:
        register_runtime_agents(legacy_registry, kernel, bus, memory)
    else:
        register_control_plane(kernel, bus, memory)

    process_registry = ProcessRegistry(
        kernel=kernel,
        bus=bus,
        memory=memory,
        mailbox_size=DEFAULT_MAILBOX_SIZE,
        token_budget=DEFAULT_AGENT_TOKEN_BUDGET,
        allowed_roots=[AGENT_PROCESS_ROOT],
        execution_mode="isolated" if AGENT_PROCESS_ISOLATION == "process" else "in-process",
        startup_timeout_seconds=AGENT_PROCESS_STARTUP_TIMEOUT,
    )
    app: AgentOSDashboard

    async def handle_shell_command(command: str) -> str:
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"could not parse command: {exc}") from exc
        if not parts:
            return ""
        verb = parts[0].lower()
        argument = parts[1] if len(parts) > 1 else ""

        if verb == "run":
            if len(parts) != 2:
                raise ValueError("usage: run <path-to-agent.py>")
            normalized_argument = argument.replace("\\", "/").rstrip("/")
            if normalized_argument == "examples/research_team":
                from examples.research_team.research_team import run_demo

                with contextlib.redirect_stdout(io.StringIO()):
                    state = await run_demo()
                app.load_research_team_snapshot(state)
                return "Research Team demo loaded: Workflow Complete | Final Score: 8.7/10"
            if is_supervisor_recovery_demo_path(argument):
                from demos.supervisor_recovery import build_demo_snapshot

                app.load_supervisor_recovery_snapshot(build_demo_snapshot())
                return "Supervisor Recovery demo loaded: Recovery Complete"
            if is_memory_paging_demo_path(argument):
                from demos.memory_paging import build_demo_snapshot

                app.load_memory_paging_snapshot(build_demo_snapshot())
                return "Memory Paging demo loaded: Memory Demo Complete"
            record = await process_registry.run_path(argument)
            return f"started PID {record.pid} ({record.name}) from {record.path}"

        if verb == "ps":
            rows = await process_registry.list_processes()
            if not rows:
                return "no active agent processes"
            lines = ["PID   NAME                 STATUS     MODE        PPID  KIDS  RST  STRATEGY      MEMORY H/P  IPC S/R/E"]
            for row in rows:
                depth = int(row.get("tree_depth", 0))
                display_name = f"{'  ' * depth}{row['name'][:20]}"
                parent_pid = "" if row.get("parent_pid") is None else str(row.get("parent_pid"))
                lines.append(
                    f"{row['pid']:<5} "
                    f"{display_name:<20} "
                    f"{row['status']:<10} "
                    f"{row['execution_mode']:<11} "
                    f"{parent_pid:<5} "
                    f"{row.get('child_count', 0):>4} "
                    f"{row.get('restart_count', 0):>3} "
                    f"{row.get('supervisor_strategy', ''):<13} "
                    f"{row.get('memory_hot_tokens', row.get('memory_tokens', 0))}/{row.get('memory_paged_count', 0):<9} "
                    f"{row.get('messages_sent', 0)}/{row.get('messages_received', 0)}/{row.get('message_errors', 0)}"
                )
            return "\n".join(lines)

        if verb == "kill":
            if len(parts) != 2:
                raise ValueError("usage: kill <PID>")
            try:
                pid = int(argument)
            except ValueError as exc:
                raise ValueError("PID must be an integer") from exc
            record = await process_registry.kill(pid)
            return f"killed PID {record.pid} ({record.name})"

        if verb in DEMO_COMMANDS:
            if len(parts) != 1:
                raise ValueError(f"usage: {verb}")
            return format_demo_browser()

        if verb in {"help", "?"}:
            return format_shell_help(AGENT_PROCESS_ROOT)

        raise ValueError("unknown command; try: help")

    app = AgentOSDashboard(
        kernel=kernel,
        bus=bus,
        memory=memory,
        sandbox=sandbox,
        command_handler=handle_shell_command,
        process_snapshot=process_registry.list_processes,
        supervision_event_snapshot=process_registry.list_supervision_events,
    )
    if legacy_registry is not None:
        seed_boot_task(bus, legacy_registry)

    tasks: list[asyncio.Task[Any]] = [
        asyncio.create_task(app.run_async(), name="tui:dashboard"),
        asyncio.create_task(optional_stdin_ingress(bus, stop_event), name="control:stdin"),
    ]
    if legacy_registry is not None:
        tasks.append(
            asyncio.create_task(
                orchestration_router(legacy_registry, bus, memory, stop_event),
                name="control:router",
            )
        )

    try:
        if AsyncLLMManager is None:
            if legacy_registry is not None:
                raise RuntimeError("AsyncLLMManager import failed")
        if legacy_registry is not None:
            llm_manager = AsyncLLMManager(build_llm_config())
            for agent in legacy_registry.agent_specs:
                tasks.append(
                    asyncio.create_task(
                        universal_agent_host(
                            agent.name,
                            agent.system_prompt,
                            agent.capabilities,
                            kernel,
                            bus,
                            memory,
                            sandbox,
                            llm_manager,
                        ),
                        name=f"agent:{agent.name}",
                    )
                )

        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc
    finally:
        stop_event.set()
        with contextlib.suppress(Exception):
            kernel.shutdown()
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        legacy_names = legacy_registry.names if legacy_registry is not None else []
        for agent_name in [SYSTEM_AGENT_NAME, ORCHESTRATOR_AGENT_NAME, *legacy_names]:
            with contextlib.suppress(Exception):
                await drain_mailbox(bus, agent_name)
        if llm_manager is not None:
            await llm_manager.aclose()

        with contextlib.suppress(NotImplementedError):
            loop.remove_signal_handler(signal.SIGINT)


if __name__ == "__main__":
    asyncio.run(main())
