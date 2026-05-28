from __future__ import annotations

import asyncio
import ast
import contextlib
import importlib.util
import inspect
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


class AgentProcess:
    """Base class for standalone Agent OS process scripts."""

    name = "AgentProcess"
    mailbox_size = 1024
    token_budget = 8000
    capabilities: tuple[str, ...] = ()

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

    def remember(self, content: dict[str, Any], token_estimate: int = 1) -> None:
        if self.memory is None:
            raise RuntimeError("process was not attached to Agent OS")
        self.memory.append_context_frame(
            self.agent_name,
            json.dumps(content),
            max(int(token_estimate), 1),
        )

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

    @property
    def uptime_seconds(self) -> float:
        return max(time.monotonic() - self.started_at, 0.0)


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
        normalized_mode = "isolated" if str(execution_mode) == "process" else str(execution_mode)
        self.execution_mode = ExecutionMode(normalized_mode)
        self.startup_timeout_seconds = startup_timeout_seconds
        self._mp_context = multiprocessing.get_context("spawn")
        self._next_pid = 100
        self._records: dict[int, ProcessRecord] = {}
        self._by_name: dict[str, int] = {}
        self._message_stats: dict[int, dict[str, int]] = {}
        self._lock = asyncio.Lock()

    async def run_path(self, raw_path: str) -> ProcessRecord:
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
        if self.execution_mode is ExecutionMode.ISOLATED:
            return await self._run_path_isolated(path)

        return await self._run_path_in_process(path)

    async def _run_path_in_process(self, path: Path) -> ProcessRecord:
        process = self._load_process(path)
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
                path=path,
                state=ProcessState.STARTING,
                started_at=time.monotonic(),
                mailbox_size=mailbox_size,
                execution_mode=ExecutionMode.IN_PROCESS,
            )

            try:
                self.bus.register_mailbox(name, mailbox_size)
                self.memory.register_agent(name, token_budget)
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
            except Exception:
                await self._cleanup_locked(record)
                raise
            return record

    async def _run_path_isolated(self, path: Path) -> ProcessRecord:
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
        )

        try:
            message = await self._wait_for_child_message(status_queue, self.startup_timeout_seconds)
            if message.get("type") == "crashed":
                record.state = ProcessState.CRASHED
                record.error = str(message.get("error") or "child crashed during startup")
                await self._stop_child(record)
                raise RuntimeError(f"agent process crashed during startup: {record.error}")
            if message.get("type") != "ready":
                raise RuntimeError(f"agent process sent invalid startup message: {message}")

            name = str(message.get("name", "")).strip()
            mailbox_size = int(message.get("mailbox_size", self.mailbox_size) or self.mailbox_size)
            token_budget = int(message.get("token_budget", self.token_budget) or self.token_budget)
            capabilities = tuple(str(item) for item in message.get("capabilities", ()) or ())
            self._validate_metadata(name, mailbox_size, token_budget, capabilities)

            async with self._lock:
                if name in self._by_name:
                    raise ValueError(f"process name '{name}' is already running")
                record.name = name
                record.mailbox_size = mailbox_size
                self.bus.register_mailbox(name, mailbox_size)
                self.memory.register_agent(name, token_budget)
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
                return record
        except Exception:
            await self._stop_child(record)
            async with self._lock:
                await self._cleanup_locked(record)
            raise

    async def kill(self, pid: int) -> ProcessRecord:
        async with self._lock:
            record = self._records.get(pid)
            if record is None:
                raise KeyError(f"unknown PID {pid}")
            if record.state in {ProcessState.KILLED, ProcessState.CRASHED, ProcessState.EXITED}:
                raise RuntimeError(f"process {pid} is already {record.state.value}")
            record.state = ProcessState.STOPPING
            record.stop_event.set()
            task = record.task
            child = record.child_process
            shutdown_queue = record.shutdown_queue

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
            return record

    async def list_processes(self) -> list[dict[str, Any]]:
        await self._reap_finished_children()
        async with self._lock:
            return [self._snapshot(record) for record in sorted(self._records.values(), key=lambda item: item.pid)]

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
            code = "mailbox_full" if "full" in str(exc).lower() else "invalid_message"
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
            raise ValueError("agent script must define an AgentProcess subclass or create_process()")
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
                raise ValueError(f"agent script import is not allowed: import {names}")
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported_names = {alias.name for alias in node.names}
                if module != "kernel.process" or imported_names - {"AgentProcess"}:
                    raise ValueError(
                        f"agent script import is not allowed: from {module} import {', '.join(sorted(imported_names))}"
                    )
            if isinstance(node, ast.ClassDef):
                if any(_base_name(base) == "AgentProcess" for base in node.bases):
                    has_process_entry = True
            if isinstance(node, ast.FunctionDef) and node.name == "create_process":
                has_process_entry = True

        if not has_process_entry:
            raise ValueError("agent script must define an AgentProcess subclass or create_process()")

    def _validate_process(self, process: AgentProcess, name: str) -> None:
        if not isinstance(process, AgentProcess):
            raise TypeError("loaded object is not an AgentProcess")
        if not name:
            raise ValueError("agent process name must not be empty")
        if any(char in name for char in "\r\n\t/\\"):
            raise ValueError("agent process name contains unsupported characters")
        mailbox_size = int(getattr(process, "mailbox_size", self.mailbox_size) or self.mailbox_size)
        token_budget = int(getattr(process, "token_budget", self.token_budget) or self.token_budget)
        if mailbox_size <= 0:
            raise ValueError("agent process mailbox_size must be greater than zero")
        if token_budget <= 0:
            raise ValueError("agent process token_budget must be greater than zero")
        capabilities = getattr(process, "capabilities", ())
        if isinstance(capabilities, str) or not all(isinstance(item, str) for item in capabilities):
            raise ValueError("agent process capabilities must be an iterable of strings")

    def _validate_metadata(
        self,
        name: str,
        mailbox_size: int,
        token_budget: int,
        capabilities: tuple[str, ...],
    ) -> None:
        if not name:
            raise ValueError("agent process name must not be empty")
        if any(char in name for char in "\r\n\t/\\"):
            raise ValueError("agent process name contains unsupported characters")
        if mailbox_size <= 0:
            raise ValueError("agent process mailbox_size must be greater than zero")
        if token_budget <= 0:
            raise ValueError("agent process token_budget must be greater than zero")
        if not all(isinstance(item, str) for item in capabilities):
            raise ValueError("agent process capabilities must be strings")

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
            async with self._lock:
                if record.state in {ProcessState.CRASHED, ProcessState.EXITED}:
                    await self._cleanup_locked(record)

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
            async with self._lock:
                if record.state in {ProcessState.CRASHED, ProcessState.EXITED}:
                    await self._cleanup_locked(record)

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
            "mailbox_depth": mailbox.get("queue_depth", 0),
            "mailbox_size": mailbox.get("buffer_size", record.mailbox_size),
            "messages_sent": stats.get("sent", 0),
            "messages_received": stats.get("received", 0),
            "message_errors": stats.get("errors", 0),
            "path": str(record.path),
            "error": record.error,
        }

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
            async with self._lock:
                exitcode = record.child_process.exitcode
                record.state = ProcessState.EXITED if exitcode == 0 else ProcessState.CRASHED
                if record.state == ProcessState.CRASHED:
                    record.error = f"child process exited unexpectedly with code {exitcode}"
                await self._cleanup_locked(record)

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
