from __future__ import annotations

import asyncio
import ast
import contextlib
import importlib.util
import inspect
import io
import json
import multiprocessing
import os
import queue
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any

from kernel.events import RuntimeEvent

from kernel.ipc_protocol import (
    ErrorMessage,
    EventMessage,
    IPCMessage,
    IPCProtocolError,
    TaskRequest,
    TaskResponse,
    make_error,
    make_message,
    new_message_id,
    parse_message,
)

try:
    from agent_os_core import AgentMessage
except ImportError:
    @dataclass
    class AgentMessage:  # type: ignore[no-redef]
        sender: str
        receiver: str
        payload: str

        def __post_init__(self) -> None:
            json.loads(self.payload)


class ProcessState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    KILLED = "killed"
    CRASHED = "crashed"
    EXITED = "exited"


class ExecutionMode(str, Enum):
    IN_PROCESS = "in-process"
    ISOLATED = "isolated"


class SupervisorStrategy(str, Enum):
    ONE_FOR_ONE = "one_for_one"
    ONE_FOR_ALL = "one_for_all"
    REST_FOR_ONE = "rest_for_one"


class RestartPolicy(str, Enum):
    PERMANENT = "permanent"
    TRANSIENT = "transient"
    TEMPORARY = "temporary"


PUBLIC_AGENTOS_IMPORTS = {
    "AgentProcess",
    "ControlMessage",
    "ErrorMessage",
    "EventMessage",
    "ExecutionMode",
    "HeartbeatMessage",
    "IPCMessage",
    "IPCProtocolError",
    "RestartPolicy",
    "SupervisorStrategy",
    "TaskRequest",
    "TaskResponse",
    "make_error",
    "make_message",
    "parse_message",
}


class AgentProcess:
    """Base class for standalone Agent OS process scripts."""

    name = "AgentProcess"
    mailbox_size = 1024
    token_budget = 8000
    capabilities: tuple[str, ...] = ()
    supervisor_strategy = SupervisorStrategy.ONE_FOR_ONE.value
    max_restarts = 3
    restart_window_seconds = 60.0
    restart_backoff_seconds = 0.0
    memory_restore_policy = "none"

    def __init__(self) -> None:
        self.pid: int | None = None
        self.agent_name = self.name
        self.bus: Any = None
        self.memory: Any = None
        self.kernel: Any = None
        self.registry: Any = None
        self.stop_event: asyncio.Event | None = None

    async def on_start(self) -> None:
        """Optional startup hook."""

    async def on_message(self, message: AgentMessage) -> None:
        """Handle one mailbox message."""

    async def on_stop(self) -> None:
        """Optional shutdown hook."""

    async def run(self) -> None:
        if self.stop_event is None or self.bus is None:
            raise RuntimeError("process was not attached to Agent OS")

        started = False
        try:
            await self.on_start()
            started = True
            while not self.stop_event.is_set():
                try:
                    message = await asyncio.wait_for(
                        self.bus.recv_message(self.agent_name),
                        timeout=0.25,
                    )
                except asyncio.TimeoutError:
                    continue
                await self.on_message(self._coerce_inbound_message(message))
        finally:
            if started:
                await self.on_stop()

    def send(
        self,
        target_pid: int,
        payload: Any,
        message_type: str = "task_request",
        priority: str = "normal",
        ttl: float | None = None,
    ) -> IPCMessage:
        if self.bus is None:
            raise RuntimeError("process was not attached to Agent OS")
        message = make_message(
            source_pid=self._require_pid(),
            target_pid=target_pid,
            payload=payload,
            message_type=message_type,
            priority=priority,
            ttl=ttl,
        )
        if self.registry is not None:
            return self.registry.route_ipc_message(message)
        self.bus.send_message(AgentMessage(self.agent_name, str(target_pid), message.to_json()))
        return message

    async def receive(self, timeout: float | None = None) -> IPCMessage:
        if self.bus is None:
            raise RuntimeError("process was not attached to Agent OS")
        receive = self.bus.recv_message(self.agent_name)
        raw = await asyncio.wait_for(receive, timeout=timeout) if timeout is not None else await receive
        return self._coerce_inbound_message(raw)

    async def request(self, target_pid: int, payload: Any, timeout: float | None = None) -> IPCMessage:
        correlation_id = new_message_id()
        request = make_message(
            source_pid=self._require_pid(),
            target_pid=target_pid,
            payload=payload,
            message_type=TaskRequest.message_type,
            correlation_id=correlation_id,
            ttl=timeout,
        )
        if self.registry is not None:
            routed = self.registry.route_ipc_message(request)
            if isinstance(routed, ErrorMessage):
                return routed
        else:
            self.bus.send_message(AgentMessage(self.agent_name, str(target_pid), request.to_json()))

        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            remaining = None if deadline is None else max(deadline - time.monotonic(), 0.0)
            if remaining == 0.0:
                return make_error(
                    source_pid=self._require_pid(),
                    target_pid=target_pid,
                    code="timeout",
                    message="request timed out while awaiting response",
                    correlation_id=correlation_id,
                )
            try:
                message = await self.receive(timeout=remaining)
            except asyncio.TimeoutError:
                return make_error(
                    source_pid=self._require_pid(),
                    target_pid=target_pid,
                    code="timeout",
                    message="request timed out while awaiting response",
                    correlation_id=correlation_id,
                )
            if message.correlation_id == correlation_id and message.type in {
                TaskResponse.message_type,
                ErrorMessage.message_type,
            }:
                return message
            await self.on_message(message)

    def reply(self, request_message: IPCMessage, payload: Any) -> IPCMessage:
        response = TaskResponse(
            source_pid=self._require_pid(),
            target_pid=request_message.source_pid,
            payload=payload,
            correlation_id=request_message.correlation_id,
            priority=request_message.priority,
        )
        if self.registry is not None:
            return self.registry.route_ipc_message(response)
        self.bus.send_message(AgentMessage(self.agent_name, str(request_message.source_pid), response.to_json()))
        return response

    def emit(self, event_name: str, payload: Any) -> IPCMessage:
        return self.send(
            1,
            {"event": event_name, "payload": payload},
            message_type=EventMessage.message_type,
            priority="low",
        )

    async def spawn_child(
        self,
        path: str,
        *,
        restart_policy: str = RestartPolicy.PERMANENT.value,
        execution_mode: str | None = None,
    ) -> int:
        if self.registry is None:
            raise RuntimeError("child spawning requires registry-attached trusted mode")
        record = await self.registry.spawn_child(
            self._require_pid(),
            path,
            restart_policy=restart_policy,
            execution_mode=execution_mode,
        )
        return record.pid

    def monitor_child(self, pid: int) -> None:
        if self.registry is None:
            raise RuntimeError("child monitoring requires registry-attached trusted mode")
        self.registry.monitor_child(self._require_pid(), pid)

    def list_children(self) -> list[int]:
        if self.registry is None:
            raise RuntimeError("child listing requires registry-attached trusted mode")
        return self.registry.list_children(self._require_pid())

    async def terminate_child(self, pid: int) -> None:
        if self.registry is None:
            raise RuntimeError("child termination requires registry-attached trusted mode")
        await self.registry.terminate_child(self._require_pid(), pid)

    def remember(
        self,
        content: Any,
        token_estimate: int = 1,
        *,
        importance: float = 0.5,
        tags: list[str] | tuple[str, ...] | None = None,
        source: dict[str, Any] | None = None,
    ) -> None:
        if self.memory is None:
            raise RuntimeError("process was not attached to Agent OS")
        evicted = False
        if hasattr(self.memory, "append_context_frame"):
            try:
                evicted = bool(self.memory.append_context_frame(
                    self.agent_name,
                    content,
                    max(int(token_estimate), 1),
                    importance=importance,
                    tags=tags,
                    source=source,
                ))
            except TypeError:
                evicted = bool(self.memory.append_context_frame(
                    self.agent_name,
                    json.dumps(content),
                    max(int(token_estimate), 1),
                ))
        if self.registry is not None:
            self.registry.emit_memory_event(
                self._require_pid(),
                "memory_recorded",
                {"tags": list(tags or []), "token_estimate": max(int(token_estimate), 1)},
            )
            if evicted:
                self.registry.emit_memory_event(
                    self._require_pid(),
                    "memory_evicted",
                    {"reason": "token_budget"},
                )

    def recall(
        self,
        query: str | None = None,
        *,
        tags: list[str] | tuple[str, ...] | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if self.memory is None or not hasattr(self.memory, "recall"):
            return []
        recalled = list(self.memory.recall(self.agent_name, query=query, tags=tags, limit=limit))
        if self.registry is not None:
            self.registry.emit_memory_event(
                self._require_pid(),
                "memory_recalled",
                {"query": query, "tags": list(tags or []), "count": len(recalled)},
            )
        return recalled

    def forget(self, memory_id: str) -> bool:
        if self.memory is None or not hasattr(self.memory, "forget"):
            return False
        forgotten = bool(self.memory.forget(memory_id))
        if forgotten and self.registry is not None:
            self.registry.emit_memory_event(
                self._require_pid(),
                "memory_forgotten",
                {"memory_id": memory_id},
            )
        return forgotten

    def memory_stats(self) -> dict[str, Any]:
        if self.memory is None:
            return {}
        with contextlib.suppress(Exception):
            return dict(self.memory.get_page_table_summary(self.agent_name))
        return {}

    def _require_pid(self) -> int:
        if self.pid is None:
            raise RuntimeError("process PID is not assigned")
        return self.pid

    def _coerce_inbound_message(self, message: Any) -> IPCMessage:
        if isinstance(message, IPCMessage):
            return message
        payload = getattr(message, "payload", message)
        return parse_message(payload)


@dataclass
class ProcessRecord:
    pid: int
    name: str
    path: Path
    state: ProcessState
    started_at: float
    mailbox_size: int
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS
    task: asyncio.Task[Any] | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    child_process: Any = None
    status_queue: Any = None
    shutdown_queue: Any = None
    child_inbox: Any = None
    child_outbox: Any = None
    resources_registered: bool = False
    error: str | None = None
    parent_pid: int | None = None
    supervisor_pid: int | None = None
    child_pids: list[int] = field(default_factory=list)
    supervision_strategy: SupervisorStrategy = SupervisorStrategy.ONE_FOR_ONE
    restart_policy: RestartPolicy = RestartPolicy.PERMANENT
    restart_count: int = 0
    restart_history: list[float] = field(default_factory=list)
    max_restarts: int = 3
    restart_window_seconds: float = 60.0
    restart_backoff_seconds: float = 0.0
    escalated: bool = False
    memory_restore_policy: str = "none"
    latest_snapshot_id: str | None = None
    external: bool = False

    @property
    def uptime_seconds(self) -> float:
        return max(time.monotonic() - self.started_at, 0.0)


@dataclass(frozen=True)
class ExternalAgentRunResult:
    manifest_name: str
    record: ProcessRecord
    output: str
    error: str | None = None
    events: tuple[RuntimeEvent, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.record.state is ProcessState.EXITED


class ProcessRegistry:
    """Owns process lifecycle, resource registration, and telemetry snapshots."""

    def __init__(
        self,
        *,
        kernel: Any,
        bus: Any,
        memory: Any,
        mailbox_size: int = 1024,
        token_budget: int = 8000,
        allowed_roots: list[Path | str] | None = None,
        execution_mode: str | ExecutionMode = ExecutionMode.IN_PROCESS,
        startup_timeout_seconds: float = 5.0,
    ) -> None:
        self.kernel = kernel
        self.bus = bus
        self.memory = memory
        self.mailbox_size = mailbox_size
        self.token_budget = token_budget
        raw_roots = allowed_roots or [Path.cwd()]
        self.allowed_roots = tuple(Path(root).expanduser().resolve() for root in raw_roots)
        execution_mode_value = execution_mode.value if isinstance(execution_mode, ExecutionMode) else str(execution_mode)
        normalized_mode = "isolated" if execution_mode_value == "process" else execution_mode_value
        self.execution_mode = ExecutionMode(normalized_mode)
        self.startup_timeout_seconds = startup_timeout_seconds
        self._mp_context = multiprocessing.get_context("spawn")
        self._next_pid = 100
        self._records: dict[int, ProcessRecord] = {}
        self._by_name: dict[str, int] = {}
        self._message_stats: dict[int, dict[str, int]] = {}
        self._supervision_events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def run_path(self, raw_path: str) -> ProcessRecord:
        return await self._run_path(
            raw_path,
            parent_pid=None,
            restart_policy=RestartPolicy.PERMANENT,
            execution_mode=self.execution_mode,
        )

    async def run_external_project(self, raw_path: str) -> ExternalAgentRunResult:
        """Validate and run one short-lived external basic Python agent."""
        from agentos.loader import load_external_agent

        manifest = load_external_agent(raw_path)
        events = [
            RuntimeEvent.info(
                "ExternalAgentRuntime",
                "external_agent_loaded",
                f"Loaded external agent {manifest.name}",
                {"agent": manifest.name},
            )
        ]
        self._validate_allowed_path(manifest.project_dir)
        self._validate_allowed_path(manifest.entrypoint_path)
        self._preflight_source(manifest.entrypoint_path)

        process = self._load_process(manifest.entrypoint_path)
        name = str(getattr(process, "name", "") or process.__class__.__name__).strip()
        self._validate_process(process, name)

        async with self._lock:
            if name in self._by_name:
                raise ValueError(f"process name '{name}' is already running")

            pid = self._allocate_pid()
            mailbox_size = int(getattr(process, "mailbox_size", self.mailbox_size) or self.mailbox_size)
            token_budget = int(getattr(process, "token_budget", self.token_budget) or self.token_budget)
            capabilities = tuple(str(item) for item in getattr(process, "capabilities", ()) or ())
            record = ProcessRecord(
                pid=pid,
                name=name,
                path=manifest.entrypoint_path,
                state=ProcessState.STARTING,
                started_at=time.monotonic(),
                mailbox_size=mailbox_size,
                execution_mode=ExecutionMode.IN_PROCESS,
                restart_policy=RestartPolicy.TEMPORARY,
                external=True,
            )

            try:
                self.bus.register_mailbox(name, mailbox_size)
                self.memory.register_agent(name, token_budget)
                self._bind_memory_process(name, pid)
                for capability in capabilities:
                    if capability.strip():
                        self.kernel.register_agent_capability(name, capability.strip())
                record.resources_registered = True

                process.pid = pid
                process.agent_name = name
                process.bus = self.bus
                process.memory = self.memory
                process.kernel = self.kernel
                process.registry = self
                process.stop_event = record.stop_event

                self._records[pid] = record
                self._by_name[name] = pid
            except Exception:
                await self._rollback_startup_locked(record)
                raise

        captured = io.StringIO()
        error: str | None = None
        record.state = ProcessState.RUNNING
        events.append(
            RuntimeEvent.info(
                "ExternalAgentRuntime",
                "external_agent_started",
                f"Started external agent {manifest.name}",
                {"agent": manifest.name, "pid": record.pid},
            )
        )
        try:
            with contextlib.redirect_stdout(captured):
                await process.on_start()
                await process.on_stop()
            record.state = ProcessState.EXITED
            events.append(
                RuntimeEvent.info(
                    "ExternalAgentRuntime",
                    "external_agent_completed",
                    f"Completed external agent {manifest.name}",
                    {"agent": manifest.name, "pid": record.pid},
                )
            )
        except Exception as exc:
            record.state = ProcessState.CRASHED
            record.error = traceback.format_exc(limit=8)
            error = str(exc) or exc.__class__.__name__
            events.append(
                RuntimeEvent.error(
                    "ExternalAgentRuntime",
                    "external_agent_failed",
                    f"External agent {manifest.name} failed: {error}",
                    {"agent": manifest.name, "pid": record.pid, "error": error},
                )
            )
        finally:
            async with self._lock:
                await self._cleanup_locked(record)

        return ExternalAgentRunResult(
            manifest_name=manifest.name,
            record=record,
            output=captured.getvalue().rstrip(),
            error=error,
            events=tuple(events),
        )

    async def _run_path(
        self,
        raw_path: str,
        *,
        parent_pid: int | None,
        restart_policy: str | RestartPolicy,
        execution_mode: str | ExecutionMode,
    ) -> ProcessRecord:
        if not raw_path.strip():
            raise ValueError("usage: run <path>")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()

        self._validate_allowed_path(path)
        if not path.exists():
            raise FileNotFoundError(f"agent script does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"agent script path is not a file: {path}")
        if path.suffix != ".py":
            raise ValueError("agent script must be a .py file")

        self._preflight_source(path)
        mode = self._normalize_execution_mode(execution_mode)
        policy = self._normalize_restart_policy(restart_policy)
        if mode is ExecutionMode.ISOLATED:
            return await self._run_path_isolated(path, parent_pid=parent_pid, restart_policy=policy)

        return await self._run_path_in_process(path, parent_pid=parent_pid, restart_policy=policy)

    async def _run_path_in_process(
        self,
        path: Path,
        *,
        parent_pid: int | None = None,
        restart_policy: RestartPolicy = RestartPolicy.PERMANENT,
    ) -> ProcessRecord:
        process = self._load_process(path)
        name = str(getattr(process, "name", "") or process.__class__.__name__).strip()
        self._validate_process(process, name)

        async with self._lock:
            self._validate_parent_running_locked(parent_pid)
            if name in self._by_name:
                raise ValueError(f"process name '{name}' is already running")

            pid = self._allocate_pid()
            mailbox_size = int(getattr(process, "mailbox_size", self.mailbox_size) or self.mailbox_size)
            token_budget = int(getattr(process, "token_budget", self.token_budget) or self.token_budget)
            capabilities = tuple(str(item) for item in getattr(process, "capabilities", ()) or ())

            record = ProcessRecord(
                pid=pid,
                name=name,
                path=path,
                state=ProcessState.STARTING,
                started_at=time.monotonic(),
                mailbox_size=mailbox_size,
                execution_mode=ExecutionMode.IN_PROCESS,
                parent_pid=parent_pid,
                supervisor_pid=parent_pid,
                restart_policy=restart_policy,
                supervision_strategy=self._normalize_supervisor_strategy(
                    getattr(process, "supervisor_strategy", SupervisorStrategy.ONE_FOR_ONE.value)
                ),
                max_restarts=int(getattr(process, "max_restarts", 3) or 3),
                restart_window_seconds=float(getattr(process, "restart_window_seconds", 60.0) or 60.0),
                restart_backoff_seconds=float(getattr(process, "restart_backoff_seconds", 0.0) or 0.0),
                memory_restore_policy=str(getattr(process, "memory_restore_policy", "none") or "none"),
            )

            try:
                self.bus.register_mailbox(name, mailbox_size)
                self.memory.register_agent(name, token_budget)
                self._bind_memory_process(name, pid)
                for capability in capabilities:
                    if capability.strip():
                        self.kernel.register_agent_capability(name, capability.strip())
                record.resources_registered = True

                process.pid = pid
                process.agent_name = name
                process.bus = self.bus
                process.memory = self.memory
                process.kernel = self.kernel
                process.registry = self
                process.stop_event = record.stop_event

                task = asyncio.create_task(
                    self._run_process(record, process),
                    name=f"agent-process:{pid}:{name}",
                )
                record.task = task
                self._records[pid] = record
                self._by_name[name] = pid
                if parent_pid is not None:
                    self._link_child_locked(parent_pid, pid)
                self._emit_supervision_event_locked("process_started", record)
            except Exception:
                await self._rollback_startup_locked(record)
                raise
            return record

    async def _run_path_isolated(
        self,
        path: Path,
        *,
        parent_pid: int | None = None,
        restart_policy: RestartPolicy = RestartPolicy.PERMANENT,
    ) -> ProcessRecord:
        async with self._lock:
            pid = self._allocate_pid()

        status_queue = self._mp_context.Queue()
        shutdown_queue = self._mp_context.Queue()
        child_inbox = self._mp_context.Queue(maxsize=self.mailbox_size)
        child_outbox = self._mp_context.Queue(maxsize=self.mailbox_size)
        child = self._mp_context.Process(
            target=_child_entrypoint,
            args=(
                {
                    "path": str(path),
                    "pid": pid,
                    "allowed_roots": [str(root) for root in self.allowed_roots],
                    "environment": self._minimal_child_environment(),
                },
                status_queue,
                shutdown_queue,
                child_inbox,
                child_outbox,
            ),
            name=f"agent-os-process-{pid}",
        )
        child.start()

        record = ProcessRecord(
            pid=pid,
            name=f"<starting:{pid}>",
            path=path,
            state=ProcessState.STARTING,
            started_at=time.monotonic(),
            mailbox_size=self.mailbox_size,
            execution_mode=ExecutionMode.ISOLATED,
            child_process=child,
            status_queue=status_queue,
            shutdown_queue=shutdown_queue,
            child_inbox=child_inbox,
            child_outbox=child_outbox,
            parent_pid=parent_pid,
            supervisor_pid=parent_pid,
            restart_policy=restart_policy,
        )

        try:
            message = await self._wait_for_child_message(status_queue, self.startup_timeout_seconds)
            if message.get("type") == "crashed":
                record.state = ProcessState.CRASHED
                record.error = str(message.get("error") or "child crashed during startup")
                await self._stop_child(record)
                if parent_pid is not None:
                    async with self._lock:
                        record.name = f"<crashed:{pid}>"
                        self._records[pid] = record
                        try:
                            self._validate_parent_running_locked(parent_pid)
                        except RuntimeError:
                            record.parent_pid = None
                            return record
                        else:
                            self._link_child_locked(parent_pid, pid)
                    await self._handle_terminal_process(record)
                    return record
                raise RuntimeError(f"agent process crashed during startup: {record.error}")
            if message.get("type") != "ready":
                raise RuntimeError(f"agent process sent invalid startup message: {message}")

            name = str(message.get("name", "")).strip()
            mailbox_size = int(message.get("mailbox_size", self.mailbox_size) or self.mailbox_size)
            token_budget = int(message.get("token_budget", self.token_budget) or self.token_budget)
            capabilities = tuple(str(item) for item in message.get("capabilities", ()) or ())
            self._validate_metadata(name, mailbox_size, token_budget, capabilities)

            async with self._lock:
                self._validate_parent_running_locked(parent_pid)
                if name in self._by_name:
                    raise ValueError(f"process name '{name}' is already running")
                record.name = name
                record.mailbox_size = mailbox_size
                record.supervision_strategy = self._normalize_supervisor_strategy(
                    message.get("supervisor_strategy", SupervisorStrategy.ONE_FOR_ONE.value)
                )
                record.max_restarts = int(message.get("max_restarts", 3) or 3)
                record.restart_window_seconds = float(message.get("restart_window_seconds", 60.0) or 60.0)
                record.restart_backoff_seconds = float(message.get("restart_backoff_seconds", 0.0) or 0.0)
                record.memory_restore_policy = str(message.get("memory_restore_policy", "none") or "none")
                self.bus.register_mailbox(name, mailbox_size)
                self.memory.register_agent(name, token_budget)
                self._bind_memory_process(name, pid)
                for capability in capabilities:
                    if capability.strip():
                        self.kernel.register_agent_capability(name, capability.strip())
                record.resources_registered = True
                record.state = ProcessState.RUNNING
                record.task = asyncio.create_task(
                    self._monitor_child(record),
                    name=f"agent-process-monitor:{pid}:{name}",
                )
                self._records[pid] = record
                self._by_name[name] = pid
                if parent_pid is not None:
                    self._link_child_locked(parent_pid, pid)
                self._emit_supervision_event_locked("process_started", record)
                return record
        except Exception:
            await self._stop_child(record)
            async with self._lock:
                await self._rollback_startup_locked(record)
            raise

    async def kill(self, pid: int, *, auto_restart: bool = True) -> ProcessRecord:
        supervisor: ProcessRecord | None = None
        async with self._lock:
            record = self._records.get(pid)
            if record is None:
                raise KeyError(f"unknown PID {pid}")
            if record.state in {ProcessState.KILLED, ProcessState.CRASHED, ProcessState.EXITED}:
                raise RuntimeError(f"process {pid} is already {record.state.value}")
            child_pids = list(record.child_pids)
            record.state = ProcessState.STOPPING
            record.stop_event.set()
            task = record.task
            child = record.child_process
            shutdown_queue = record.shutdown_queue

        for child_pid in child_pids:
            with contextlib.suppress(KeyError, RuntimeError):
                await self.kill(child_pid, auto_restart=False)

        if record.execution_mode is ExecutionMode.ISOLATED:
            with contextlib.suppress(Exception):
                shutdown_queue.put("stop")
            await self._join_child(child, timeout=2.0)
            if child is not None and child.is_alive():
                child.terminate()
                await self._join_child(child, timeout=2.0)
        elif task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if task is not None and record.execution_mode is ExecutionMode.ISOLATED:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        async with self._lock:
            record.state = ProcessState.KILLED
            await self._cleanup_locked(record)
            record.child_pids.clear()
            if record.parent_pid is not None:
                self._emit_supervision_event_locked("child_terminated", record)
                supervisor = self._records.get(record.parent_pid)
                if (
                    auto_restart
                    and supervisor is not None
                    and supervisor.state in {ProcessState.STARTING, ProcessState.RUNNING}
                ):
                    self._emit_supervision_event_locked("child_restart_requested", record)
            if (
                not auto_restart
                or supervisor is None
                or supervisor.state not in {ProcessState.STARTING, ProcessState.RUNNING}
            ):
                self._unlink_child_locked(record)
                return record

        replacement = await self._restart_child(supervisor, record)
        if replacement.execution_mode is ExecutionMode.IN_PROCESS:
            await asyncio.sleep(0)
        async with self._lock:
            self._unlink_child_locked(record)
            self._emit_supervision_event_locked(
                "child_restarted",
                supervisor,
                details={"old_pid": record.pid, "new_pid": replacement.pid, "child_name": replacement.name},
            )
        return record

    async def list_processes(self) -> list[dict[str, Any]]:
        await self._reap_finished_children()
        async with self._lock:
            return [self._snapshot(record) for record in sorted(self._records.values(), key=lambda item: item.pid)]

    def list_supervision_events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._supervision_events]

    async def spawn_child(
        self,
        parent_pid: int,
        raw_path: str,
        *,
        restart_policy: str | RestartPolicy = RestartPolicy.PERMANENT,
        execution_mode: str | ExecutionMode | None = None,
    ) -> ProcessRecord:
        async with self._lock:
            parent = self._records.get(parent_pid)
            if parent is None:
                raise KeyError(f"unknown parent PID {parent_pid}")
            if parent.state not in {ProcessState.STARTING, ProcessState.RUNNING}:
                raise RuntimeError(f"parent PID {parent_pid} is {parent.state.value}")
        try:
            return await self._run_path(
                raw_path,
                parent_pid=parent_pid,
                restart_policy=restart_policy,
                execution_mode=execution_mode or self.execution_mode,
            )
        except Exception:
            async with self._lock:
                parent = self._records.get(parent_pid)
                if parent is not None:
                    parent.error = traceback.format_exc(limit=8)
                    self._emit_supervision_event_locked(
                        "supervision_escalation",
                        parent,
                        details={"path": raw_path, "error": "child failed before registration"},
                    )
                    parent.escalated = True
            raise

    def monitor_child(self, parent_pid: int, child_pid: int) -> None:
        parent = self._records.get(parent_pid)
        child = self._records.get(child_pid)
        if parent is None:
            raise KeyError(f"unknown parent PID {parent_pid}")
        if child is None:
            raise KeyError(f"unknown child PID {child_pid}")
        child.parent_pid = parent_pid
        child.supervisor_pid = parent_pid
        if child_pid not in parent.child_pids:
            parent.child_pids.append(child_pid)

    def list_children(self, parent_pid: int) -> list[int]:
        record = self._records.get(parent_pid)
        if record is None:
            raise KeyError(f"unknown PID {parent_pid}")
        return list(record.child_pids)

    async def terminate_child(self, parent_pid: int, child_pid: int) -> None:
        async with self._lock:
            parent = self._records.get(parent_pid)
            if parent is None:
                raise KeyError(f"unknown parent PID {parent_pid}")
            if child_pid not in parent.child_pids:
                raise ValueError(f"PID {child_pid} is not supervised by {parent_pid}")
        await self.kill(child_pid)

    def route_ipc_message(self, message: IPCMessage) -> IPCMessage:
        try:
            message.validate()
            if message.is_expired():
                raise IPCProtocolError("timeout", "message expired before delivery")
            source = self._records.get(message.source_pid)
            target = self._records.get(message.target_pid)
            if source is None:
                raise IPCProtocolError("invalid_message", f"source PID {message.source_pid} is not registered")
            if target is None:
                raise IPCProtocolError("target_not_found", f"target PID {message.target_pid} is not registered")
            if source.state not in {ProcessState.STARTING, ProcessState.RUNNING}:
                raise IPCProtocolError("process_dead", f"source PID {message.source_pid} is {source.state.value}")
            if target.state not in {ProcessState.STARTING, ProcessState.RUNNING}:
                raise IPCProtocolError("process_dead", f"target PID {message.target_pid} is {target.state.value}")
            self.bus.send_message(AgentMessage(source.name, target.name, message.to_json()))
            self._bump_message_stat(message.source_pid, "sent")
            self._bump_message_stat(message.target_pid, "received")
            return message
        except IPCProtocolError as exc:
            return self._protocol_error(message, exc.code, str(exc))
        except Exception as exc:
            code = "mailbox_full" if isinstance(exc, (asyncio.QueueFull, queue.Full)) else "invalid_message"
            return self._protocol_error(message, code, str(exc))

    def send_ipc_message(
        self,
        source_pid: int,
        target_pid: int,
        payload: Any,
        message_type: str = TaskRequest.message_type,
        priority: str = "normal",
        ttl: float | None = None,
    ) -> IPCMessage:
        try:
            message = make_message(
                source_pid=source_pid,
                target_pid=target_pid,
                payload=payload,
                message_type=message_type,
                priority=priority,
                ttl=ttl,
            )
        except IPCProtocolError as exc:
            return make_error(
                source_pid=source_pid if isinstance(source_pid, int) and source_pid > 0 else 1,
                target_pid=target_pid if isinstance(target_pid, int) and target_pid > 0 else 1,
                code=exc.code,
                message=str(exc),
            )
        return self.route_ipc_message(message)

    def _load_process(self, path: Path) -> AgentProcess:
        module_name = f"agent_os_dynamic_{path.stem}_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"could not load Python module from {path}")

        module = importlib.util.module_from_spec(spec)
        self._install_sdk_symbol(module)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            raise RuntimeError(f"agent script failed during import: {exc}") from exc

        process = self._find_process(module)
        if process is None:
            raise ValueError(
                "agent script must define an AgentProcess subclass, for example: "
                "class MyAgent(AgentProcess): name = \"MyAgent\""
            )
        return process

    def _validate_allowed_path(self, path: Path) -> None:
        if any(_is_relative_to(path, root) for root in self.allowed_roots):
            return
        roots = ", ".join(str(root) for root in self.allowed_roots)
        raise PermissionError(f"agent script must be under an allowed workspace root: {roots}")

    def _preflight_source(self, path: Path) -> None:
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"agent script has invalid Python syntax: {exc}") from exc

        has_process_entry = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = ", ".join(alias.name for alias in node.names)
                raise ValueError(
                    f"agent script import is not allowed: import {names}. "
                    "Use 'from agentos import AgentProcess'; additional imports are disabled for agent scripts."
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = {alias.name for alias in node.names}
                allowed_names = PUBLIC_AGENTOS_IMPORTS if module == "agentos" else {"AgentProcess"}
                if module not in {"agentos", "kernel.process"} or imported_names - allowed_names:
                    raise ValueError(
                        f"agent script import is not allowed: from {module} import {', '.join(sorted(imported_names))}. "
                        "Use 'from agentos import AgentProcess'; additional imports are disabled for agent scripts."
                    )
            if isinstance(node, ast.ClassDef):
                if any(_base_name(base) == "AgentProcess" for base in node.bases):
                    has_process_entry = True
            if isinstance(node, ast.FunctionDef) and node.name == "create_process":
                has_process_entry = True

        if not has_process_entry:
            raise ValueError(
                "agent script must define an AgentProcess subclass, for example: "
                "class MyAgent(AgentProcess): name = \"MyAgent\""
            )

    def _validate_process(self, process: AgentProcess, name: str) -> None:
        if not isinstance(process, AgentProcess):
            raise TypeError("loaded object is not an AgentProcess")
        if not name or name == AgentProcess.name:
            raise ValueError('agent process must define a unique non-empty name, for example: name = "MyAgent"')
        if any(char in name for char in "\r\n\t/\\"):
            raise ValueError("agent process name contains unsupported characters")
        mailbox_size = self._positive_int_setting(process, "mailbox_size", self.mailbox_size)
        token_budget = self._positive_int_setting(process, "token_budget", self.token_budget)
        capabilities = getattr(process, "capabilities", ())
        if isinstance(capabilities, str) or not all(isinstance(item, str) for item in capabilities):
            raise ValueError("agent process capabilities must be an iterable of strings")
        self._normalize_supervisor_strategy(getattr(process, "supervisor_strategy", SupervisorStrategy.ONE_FOR_ONE.value))
        if int(getattr(process, "max_restarts", 3) or 3) < 0:
            raise ValueError("agent process max_restarts must be non-negative")
        if float(getattr(process, "restart_window_seconds", 60.0) or 60.0) <= 0:
            raise ValueError("agent process restart_window_seconds must be greater than zero")
        if float(getattr(process, "restart_backoff_seconds", 0.0) or 0.0) < 0:
            raise ValueError("agent process restart_backoff_seconds must be non-negative")
        if str(getattr(process, "memory_restore_policy", "none") or "none") not in {
            "none",
            "hot_only",
            "latest_snapshot",
            "persistent_recall",
        }:
            raise ValueError("agent process memory_restore_policy is unsupported")

    def _validate_metadata(
        self,
        name: str,
        mailbox_size: int,
        token_budget: int,
        capabilities: tuple[str, ...],
    ) -> None:
        if not name or name == AgentProcess.name:
            raise ValueError('agent process must define a unique non-empty name, for example: name = "MyAgent"')
        if any(char in name for char in "\r\n\t/\\"):
            raise ValueError("agent process name contains unsupported characters")
        if mailbox_size <= 0:
            raise ValueError("agent process mailbox_size must be greater than zero")
        if token_budget <= 0:
            raise ValueError("agent process token_budget must be greater than zero")
        if not all(isinstance(item, str) for item in capabilities):
            raise ValueError("agent process capabilities must be strings")

    def _normalize_execution_mode(self, execution_mode: str | ExecutionMode) -> ExecutionMode:
        execution_mode_value = execution_mode.value if isinstance(execution_mode, ExecutionMode) else str(execution_mode)
        normalized_mode = "isolated" if execution_mode_value == "process" else execution_mode_value
        return ExecutionMode(normalized_mode)

    def _normalize_restart_policy(self, restart_policy: str | RestartPolicy) -> RestartPolicy:
        value = restart_policy.value if isinstance(restart_policy, RestartPolicy) else str(restart_policy)
        return RestartPolicy(value)

    def _normalize_supervisor_strategy(self, strategy: str | SupervisorStrategy) -> SupervisorStrategy:
        value = strategy.value if isinstance(strategy, SupervisorStrategy) else str(strategy)
        return SupervisorStrategy(value)

    @staticmethod
    def _positive_int_setting(process: AgentProcess, setting: str, default: int) -> int:
        raw_value = getattr(process, setting, default)
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"agent process {setting} must be a positive integer; got {raw_value!r}") from exc
        if value <= 0:
            raise ValueError(f"agent process {setting} must be a positive integer; got {raw_value!r}")
        return value

    def _install_sdk_symbol(self, module: ModuleType) -> None:
        module.AgentProcess = AgentProcess

    def _find_process(self, module: ModuleType) -> AgentProcess | None:
        factory = getattr(module, "create_process", None)
        if callable(factory):
            created = factory()
            if not isinstance(created, AgentProcess):
                raise TypeError("create_process() must return an AgentProcess instance")
            return created

        for _, value in inspect.getmembers(module, inspect.isclass):
            if value is AgentProcess or not issubclass(value, AgentProcess):
                continue
            return value()
        return None

    async def _run_process(self, record: ProcessRecord, process: AgentProcess) -> None:
        record.state = ProcessState.RUNNING
        try:
            await process.run()
            if record.state not in {ProcessState.STOPPING, ProcessState.KILLED}:
                record.state = ProcessState.EXITED
        except asyncio.CancelledError:
            raise
        except Exception:
            record.state = ProcessState.CRASHED
            record.error = traceback.format_exc(limit=8)
        finally:
            terminal = record.state in {ProcessState.CRASHED, ProcessState.EXITED}
            async with self._lock:
                if terminal:
                    await self._cleanup_locked(record)
            if terminal:
                await self._handle_terminal_process(record)

    async def _monitor_child(self, record: ProcessRecord) -> None:
        try:
            while record.state == ProcessState.RUNNING:
                message = self._read_child_message(record.status_queue)
                if message is not None:
                    message_type = message.get("type")
                    if message_type == "crashed":
                        record.state = ProcessState.CRASHED
                        record.error = str(message.get("error") or "child crashed")
                        break
                    if message_type == "exited":
                        record.state = ProcessState.EXITED
                        break
                await self._bridge_child_ipc(record)

                child = record.child_process
                if child is not None and not child.is_alive():
                    exitcode = child.exitcode
                    if record.state == ProcessState.RUNNING:
                        if exitcode == 0:
                            record.state = ProcessState.EXITED
                        else:
                            record.state = ProcessState.CRASHED
                            record.error = f"child process exited unexpectedly with code {exitcode}"
                    break
                await asyncio.sleep(0.1)
        finally:
            terminal = record.state in {ProcessState.CRASHED, ProcessState.EXITED}
            async with self._lock:
                if terminal:
                    await self._cleanup_locked(record)
            if terminal:
                await self._handle_terminal_process(record)

    async def _cleanup_locked(self, record: ProcessRecord) -> None:
        self._by_name.pop(record.name, None)
        if record.resources_registered:
            with contextlib.suppress(Exception):
                self.bus.unregister_mailbox(record.name)
            with contextlib.suppress(Exception):
                self.memory.unregister_agent(record.name)
            with contextlib.suppress(Exception):
                self.kernel.unregister_agent(record.name)
            record.resources_registered = False

    async def _rollback_startup_locked(self, record: ProcessRecord) -> None:
        """Remove any partial registrations created before startup committed."""
        self._by_name.pop(record.name, None)
        self._records.pop(record.pid, None)
        self._message_stats.pop(record.pid, None)
        self._unlink_child_locked(record)
        with contextlib.suppress(Exception):
            self.bus.unregister_mailbox(record.name)
        with contextlib.suppress(Exception):
            self.memory.unregister_agent(record.name)
        with contextlib.suppress(Exception):
            self.kernel.unregister_agent(record.name)
        record.resources_registered = False

    async def _handle_terminal_process(self, record: ProcessRecord) -> None:
        abnormal = record.state is ProcessState.CRASHED
        event_name = "process_crashed" if abnormal else "process_stopped"
        if record.child_pids:
            await self._terminate_children_of_terminal_parent(record)

        async with self._lock:
            self._emit_supervision_event_locked(event_name, record)
            parent = self._records.get(record.parent_pid) if record.parent_pid is not None else None
            if parent is None or parent.state not in {ProcessState.STARTING, ProcessState.RUNNING}:
                return
            affected = self._affected_children_locked(parent, record.pid)
            candidates = [
                child
                for child_pid in affected
                if (child := self._records.get(child_pid)) is not None
                and self._restart_policy_allows(child, abnormal=abnormal)
            ]
            if not candidates:
                if record.pid in affected:
                    self._unlink_child_locked(record)
                return
            if not self._restart_budget_allows(parent, len(candidates)):
                parent.escalated = True
                self._emit_supervision_event_locked(
                    "supervision_escalation",
                    parent,
                    details={
                        "child_pid": record.pid,
                        "attempted_restarts": len(candidates),
                        "max_restarts": parent.max_restarts,
                        "restart_window_seconds": parent.restart_window_seconds,
                    },
                )
                if record.pid in affected:
                    self._unlink_child_locked(record)
                return
            self._record_restart_attempts(parent, len(candidates))

        restarted: list[tuple[int, ProcessRecord]] = []
        escalated = False
        for child in candidates:
            if parent.restart_backoff_seconds > 0:
                await asyncio.sleep(parent.restart_backoff_seconds * max(parent.restart_count, 1))
            async with self._lock:
                current_parent = self._records.get(parent.pid)
                if (
                    current_parent is None
                    or current_parent.state not in {ProcessState.STARTING, ProcessState.RUNNING}
                    or child.parent_pid != parent.pid
                ):
                    break
            if child.state in {ProcessState.STARTING, ProcessState.RUNNING, ProcessState.STOPPING}:
                with contextlib.suppress(KeyError, RuntimeError):
                    await self.kill(child.pid, auto_restart=False)
            async with self._lock:
                current_parent = self._records.get(parent.pid)
                if current_parent is None or current_parent.state not in {ProcessState.STARTING, ProcessState.RUNNING}:
                    break
            try:
                replacement = await self._restart_child(parent, child)
            except Exception as exc:
                async with self._lock:
                    parent.escalated = True
                    parent.error = f"supervision restart failed for child PID {child.pid}: {exc}"
                    self._emit_supervision_event_locked(
                        "supervision_escalation",
                        parent,
                        details={"child_pid": child.pid, "error": str(exc)},
                    )
                escalated = True
                break
            restarted.append((child.pid, replacement))

        if restarted and not escalated:
            async with self._lock:
                for old_pid, replacement in restarted:
                    self._emit_supervision_event_locked(
                        "child_restarted",
                        parent,
                        details={"old_pid": old_pid, "new_pid": replacement.pid, "child_name": replacement.name},
                    )

    async def _terminate_children_of_terminal_parent(self, record: ProcessRecord) -> None:
        for child_pid in list(record.child_pids):
            with contextlib.suppress(KeyError, RuntimeError):
                await self.kill(child_pid, auto_restart=False)
        async with self._lock:
            for child_pid in list(record.child_pids):
                child = self._records.get(child_pid)
                if child is not None:
                    child.parent_pid = None
            record.child_pids.clear()

    def _affected_children_locked(self, parent: ProcessRecord, failed_pid: int) -> list[int]:
        child_pids = list(parent.child_pids)
        if failed_pid not in child_pids:
            return [failed_pid]
        if parent.supervision_strategy is SupervisorStrategy.ONE_FOR_ALL:
            return child_pids
        if parent.supervision_strategy is SupervisorStrategy.REST_FOR_ONE:
            return child_pids[child_pids.index(failed_pid) :]
        return [failed_pid]

    def _restart_policy_allows(self, child: ProcessRecord, *, abnormal: bool) -> bool:
        if child.restart_policy is RestartPolicy.TEMPORARY:
            return False
        if child.restart_policy is RestartPolicy.TRANSIENT and not abnormal:
            return False
        return True

    def _restart_budget_allows(self, parent: ProcessRecord, attempts: int) -> bool:
        now = time.monotonic()
        parent.restart_history = [
            item for item in parent.restart_history if now - item <= parent.restart_window_seconds
        ]
        return len(parent.restart_history) + attempts <= parent.max_restarts

    def _record_restart_attempts(self, parent: ProcessRecord, attempts: int) -> None:
        now = time.monotonic()
        parent.restart_history.extend(now for _ in range(attempts))
        parent.restart_count += attempts

    async def _restart_child(self, parent: ProcessRecord, child: ProcessRecord) -> ProcessRecord:
        replacement = await self._run_path(
            str(child.path),
            parent_pid=parent.pid,
            restart_policy=child.restart_policy,
            execution_mode=child.execution_mode,
        )
        replacement.restart_count = child.restart_count + 1
        self._restore_memory_for_restarted_child(child, replacement)
        async with self._lock:
            if child.pid in parent.child_pids:
                index = parent.child_pids.index(child.pid)
                parent.child_pids[index] = replacement.pid
            elif replacement.pid not in parent.child_pids:
                parent.child_pids.append(replacement.pid)
            seen: set[int] = set()
            parent.child_pids = [
                pid for pid in parent.child_pids if not (pid in seen or seen.add(pid))
            ]
        return replacement

    def snapshot_process(self, pid: int) -> str:
        record = self._records.get(pid)
        if record is None:
            raise KeyError(f"unknown PID {pid}")
        if not hasattr(self.memory, "snapshot_process"):
            raise RuntimeError("memory manager does not support snapshots")
        snapshot_id = str(self.memory.snapshot_process(pid, record.name))
        record.latest_snapshot_id = snapshot_id
        self._emit_memory_event("memory_snapshot_created", record, {"snapshot_id": snapshot_id})
        return snapshot_id

    def restore_process_memory(self, pid: int, snapshot_id: str | None = None) -> str | None:
        record = self._records.get(pid)
        if record is None:
            raise KeyError(f"unknown PID {pid}")
        if not hasattr(self.memory, "restore_process_memory"):
            return None
        restored = self.memory.restore_process_memory(pid, record.name, snapshot_id)
        if restored is not None:
            record.latest_snapshot_id = str(restored)
            self._emit_memory_event("memory_restored", record, {"snapshot_id": str(restored)})
        return None if restored is None else str(restored)

    def emit_memory_event(self, pid: int, event_name: str, details: dict[str, Any]) -> None:
        record = self._records.get(pid)
        if record is None:
            return
        self._emit_memory_event(event_name, record, details)

    def _restore_memory_for_restarted_child(self, old: ProcessRecord, new: ProcessRecord) -> None:
        policy = new.memory_restore_policy or old.memory_restore_policy
        if policy == "none":
            return
        if policy == "latest_snapshot":
            if hasattr(self.memory, "restore_process_memory"):
                restored = self.memory.restore_process_memory(new.pid, new.name, old.latest_snapshot_id)
                if restored is not None:
                    new.latest_snapshot_id = str(restored)
                    self._emit_memory_event("memory_restored", new, {"snapshot_id": str(restored)})
        elif policy == "hot_only":
            if hasattr(self.memory, "restore_process_memory"):
                restored = self.memory.restore_process_memory(
                    new.pid,
                    new.name,
                    old.latest_snapshot_id,
                    hot_only=True,
                )
                if restored is not None:
                    new.latest_snapshot_id = str(restored)
                    self._emit_memory_event("memory_restored", new, {"snapshot_id": str(restored)})
        elif policy == "persistent_recall" and hasattr(self.memory, "recall"):
            recalled = self.memory.recall(new.name, limit=5)
            for item in recalled:
                with contextlib.suppress(Exception):
                    self.memory.append_context_frame(
                        new.name,
                        item.get("content"),
                        int(item.get("token_estimate", 1)),
                        importance=float(item.get("importance", 0.5)),
                        tags=list(item.get("tags", [])),
                        source={"restored_from": item.get("memory_id")},
                    )
            if recalled:
                self._emit_memory_event("memory_restored", new, {"records": len(recalled)})

    def _link_child_locked(self, parent_pid: int, child_pid: int) -> None:
        parent = self._records.get(parent_pid)
        child = self._records.get(child_pid)
        if parent is None or child is None:
            return
        child.parent_pid = parent_pid
        child.supervisor_pid = parent_pid
        if child_pid not in parent.child_pids:
            parent.child_pids.append(child_pid)

    def _validate_parent_running_locked(self, parent_pid: int | None) -> None:
        if parent_pid is None:
            return
        parent = self._records.get(parent_pid)
        if parent is None:
            raise RuntimeError(f"parent PID {parent_pid} is not registered")
        if parent.state not in {ProcessState.STARTING, ProcessState.RUNNING}:
            raise RuntimeError(f"parent PID {parent_pid} is {parent.state.value}")

    def _unlink_child_locked(self, child: ProcessRecord) -> None:
        if child.parent_pid is None:
            return
        parent = self._records.get(child.parent_pid)
        if parent is not None:
            parent.child_pids = [pid for pid in parent.child_pids if pid != child.pid]
        child.parent_pid = None

    def _emit_supervision_event_locked(
        self,
        event_name: str,
        record: ProcessRecord,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        parent_pid = record.parent_pid
        if event_name in {"child_restarted", "supervision_escalation"}:
            parent_pid = record.pid
        if parent_pid is None:
            return
        parent = self._records.get(parent_pid)
        if parent is None or parent.state not in {ProcessState.STARTING, ProcessState.RUNNING}:
            return
        payload = {
            "event": event_name,
            "pid": record.pid,
            "name": record.name,
            "state": record.state.value,
            "parent_pid": record.parent_pid,
            "details": details or {},
        }
        messages = {
            "child_terminated": f"Detected child termination:\n{record.name}",
            "child_restart_requested": f"Restarting child:\n{record.name}",
            "child_restarted": f"Child restarted:\n{(details or {}).get('child_name', record.name)}",
        }
        if event_name in messages:
            self._supervision_events.append(
                {
                    **payload,
                    "supervisor_pid": parent.pid,
                    "supervisor_name": parent.name,
                    "message": messages[event_name],
                }
            )
        try:
            message = EventMessage(
                source_pid=record.pid,
                target_pid=parent.pid,
                payload=payload,
                priority="high" if event_name in {"process_crashed", "supervision_escalation"} else "normal",
            )
            self.bus.send_message(AgentMessage(record.name, parent.name, message.to_json()))
            self._bump_message_stat(record.pid, "sent")
            self._bump_message_stat(parent.pid, "received")
        except Exception:
            self._bump_message_stat(record.pid, "errors")

    def _emit_memory_event(self, event_name: str, record: ProcessRecord, details: dict[str, Any]) -> None:
        self._emit_supervision_event_locked(
            event_name,
            record,
            details=details,
        )

    def _bind_memory_process(self, name: str, pid: int) -> None:
        if hasattr(self.memory, "bind_process"):
            with contextlib.suppress(Exception):
                self.memory.bind_process(name, pid)

    def _allocate_pid(self) -> int:
        pid = self._next_pid
        self._next_pid += 1
        return pid

    def _snapshot(self, record: ProcessRecord) -> dict[str, Any]:
        mailbox = self._mailbox_snapshot(record.name)
        memory = self._memory_snapshot(record.name)
        stats = self._message_stats.get(record.pid, {})
        return {
            "pid": record.pid,
            "name": record.name,
            "status": record.state.value,
            "execution_mode": record.execution_mode.value,
            "uptime_seconds": record.uptime_seconds,
            "memory_tokens": memory.get("current_active_tokens", 0),
            "memory_hot_tokens": memory.get("current_active_tokens", 0),
            "memory_paged_count": memory.get("paged_out_frames", 0),
            "memory_snapshot_count": memory.get("snapshot_count", 0),
            "memory_last_eviction_time": memory.get("last_eviction_time"),
            "memory_store_size_bytes": memory.get("memory_store_size_bytes", 0),
            "mailbox_depth": mailbox.get("queue_depth", 0),
            "mailbox_size": mailbox.get("buffer_size", record.mailbox_size),
            "messages_sent": stats.get("sent", 0),
            "messages_received": stats.get("received", 0),
            "message_errors": stats.get("errors", 0),
            "parent_pid": record.parent_pid,
            "supervisor_pid": record.supervisor_pid,
            "child_count": len(record.child_pids),
            "child_pids": list(record.child_pids),
            "tree_depth": self._tree_depth(record),
            "restart_count": record.restart_count,
            "supervisor_restart_count": record.restart_count,
            "supervisor_strategy": record.supervision_strategy.value,
            "restart_policy": record.restart_policy.value,
            "supervision_escalated": record.escalated,
            "path": str(record.path),
            "error": record.error,
            "external": record.external,
        }

    def _tree_depth(self, record: ProcessRecord) -> int:
        depth = 0
        seen: set[int] = set()
        parent_pid = record.parent_pid
        while parent_pid is not None and parent_pid not in seen:
            seen.add(parent_pid)
            parent = self._records.get(parent_pid)
            if parent is None:
                break
            depth += 1
            parent_pid = parent.parent_pid
        return depth

    async def _bridge_child_ipc(self, record: ProcessRecord) -> None:
        if record.child_inbox is None or record.child_outbox is None:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            message = await asyncio.wait_for(self.bus.recv_message(record.name), timeout=0.01)
            try:
                record.child_inbox.put_nowait(message.payload)
            except Exception:
                self._bump_message_stat(record.pid, "errors")
        while True:
            try:
                raw = record.child_outbox.get_nowait()
            except queue.Empty:
                break
            try:
                self.route_ipc_message(parse_message(raw))
            except IPCProtocolError as exc:
                self._bump_message_stat(record.pid, "errors")

    def _mailbox_snapshot(self, name: str) -> dict[str, Any]:
        for agent_name, queue_depth, buffer_size, routing_method in self.bus.get_mailbox_metrics():
            if agent_name == name:
                return {
                    "queue_depth": queue_depth,
                    "buffer_size": buffer_size,
                    "routing_method": routing_method,
                }
        return {}

    def _memory_snapshot(self, name: str) -> dict[str, Any]:
        with contextlib.suppress(Exception):
            return dict(self.memory.get_page_table_summary(name))
        return {}

    async def _wait_for_child_message(self, status_queue: Any, timeout: float) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(status_queue.get, True, timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"agent process did not report ready within {timeout:.1f}s") from exc

    def _read_child_message(self, status_queue: Any) -> dict[str, Any] | None:
        if status_queue is None:
            return None
        with contextlib.suppress(queue.Empty):
            return status_queue.get_nowait()
        return None

    async def _reap_finished_children(self) -> None:
        async with self._lock:
            records = [
                record
                for record in self._records.values()
                if record.execution_mode is ExecutionMode.ISOLATED
                and record.state == ProcessState.RUNNING
                and record.child_process is not None
                and not record.child_process.is_alive()
            ]
        for record in records:
            if record.task is not None:
                continue
            terminal = False
            async with self._lock:
                exitcode = record.child_process.exitcode
                record.state = ProcessState.EXITED if exitcode == 0 else ProcessState.CRASHED
                if record.state == ProcessState.CRASHED:
                    record.error = f"child process exited unexpectedly with code {exitcode}"
                await self._cleanup_locked(record)
                terminal = True
            if terminal:
                await self._handle_terminal_process(record)

    async def _stop_child(self, record: ProcessRecord) -> None:
        with contextlib.suppress(Exception):
            if record.shutdown_queue is not None:
                record.shutdown_queue.put("stop")
        child = record.child_process
        if child is None:
            return
        await self._join_child(child, timeout=1.0)
        if child.is_alive():
            child.terminate()
            await self._join_child(child, timeout=2.0)

    async def _join_child(self, child: Any, timeout: float) -> None:
        if child is None:
            return
        await asyncio.to_thread(child.join, timeout)

    def _minimal_child_environment(self) -> dict[str, str]:
        keys = ["PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PYTHONPATH"]
        env = {key: os.environ[key] for key in keys if key in os.environ}
        env["AGENT_OS_CHILD_PROCESS"] = "1"
        return env

    def _bump_message_stat(self, pid: int, key: str) -> None:
        self._message_stats.setdefault(pid, {"sent": 0, "received": 0, "errors": 0})[key] += 1

    def _protocol_error(self, message: IPCMessage, code: str, text: str) -> ErrorMessage:
        self._bump_message_stat(message.source_pid, "errors")
        return make_error(
            source_pid=message.target_pid,
            target_pid=message.source_pid,
            code=code,
            message=text,
            correlation_id=message.correlation_id,
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    if hasattr(path, "is_relative_to"):
        return path.is_relative_to(root)
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _child_entrypoint(
    config: dict[str, Any],
    status_queue: Any,
    shutdown_queue: Any,
    child_inbox: Any,
    child_outbox: Any,
) -> None:
    from kernel.process_runner import run_child

    run_child(config, status_queue, shutdown_queue, child_inbox, child_outbox)
