from __future__ import annotations

import asyncio
import ast
import importlib.util
import inspect
import json
import os
import queue
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from kernel.process import AgentMessage, AgentProcess


@dataclass
class RunnerConfig:
    path: str
    pid: int
    allowed_roots: list[str]
    environment: dict[str, str]


class IsolatedStopEvent:
    def __init__(self, shutdown_queue: Any) -> None:
        self.shutdown_queue = shutdown_queue
        self._stopped = False

    def is_set(self) -> bool:
        if self._stopped:
            return True
        try:
            message = self.shutdown_queue.get_nowait()
        except queue.Empty:
            return False
        self._stopped = message == "stop"
        return self._stopped


class IsolatedBus:
    def __init__(self, *, inbox: Any, outbox: Any) -> None:
        self.inbox = inbox
        self.outbox = outbox

    async def recv_message(self, agent_name: str) -> AgentMessage:
        try:
            payload = await asyncio.to_thread(self.inbox.get, True, 0.25)
        except queue.Empty as exc:
            raise asyncio.TimeoutError from exc
        return AgentMessage(agent_name, agent_name, payload)

    def send_message(self, message: AgentMessage) -> None:
        self.outbox.put_nowait(message.payload)


class IsolatedMemory:
    def __init__(self) -> None:
        self.active_tokens = 0

    def append_context_frame(self, _agent_name: str, content: str, token_estimate: int) -> None:
        json.loads(content)
        self.active_tokens += max(int(token_estimate), 1)


def run_child(
    config_data: dict[str, Any],
    status_queue: Any,
    shutdown_queue: Any,
    child_inbox: Any,
    child_outbox: Any,
) -> None:
    config = RunnerConfig(**config_data)
    _apply_minimal_environment(config.environment)
    try:
        path = Path(config.path).resolve()
        _validate_allowed_path(path, [Path(root).resolve() for root in config.allowed_roots])
        _preflight_source(path)
        process = _load_process(path)
        name = str(getattr(process, "name", "") or process.__class__.__name__).strip()
        _validate_process(process, name)

        process.pid = config.pid
        process.agent_name = name
        process.bus = IsolatedBus(inbox=child_inbox, outbox=child_outbox)
        process.memory = IsolatedMemory()
        process.kernel = None
        process.stop_event = IsolatedStopEvent(shutdown_queue)

        status_queue.put(
            {
                "type": "ready",
                "name": name,
                "mailbox_size": int(getattr(process, "mailbox_size", 1024) or 1024),
                "token_budget": int(getattr(process, "token_budget", 8000) or 8000),
                "capabilities": list(getattr(process, "capabilities", ()) or ()),
                "supervisor_strategy": str(getattr(process, "supervisor_strategy", "one_for_one")),
                "max_restarts": int(getattr(process, "max_restarts", 3) or 3),
                "restart_window_seconds": float(getattr(process, "restart_window_seconds", 60.0) or 60.0),
                "restart_backoff_seconds": float(getattr(process, "restart_backoff_seconds", 0.0) or 0.0),
                "memory_restore_policy": str(getattr(process, "memory_restore_policy", "none") or "none"),
            }
        )
        asyncio.run(process.run())
        status_queue.put({"type": "exited"})
    except BaseException:
        status_queue.put({"type": "crashed", "error": traceback.format_exc(limit=8)})


def _apply_minimal_environment(environment: dict[str, str]) -> None:
    os.environ.clear()
    os.environ.update(environment)


def _load_process(path: Path) -> AgentProcess:
    module_name = f"agent_os_isolated_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    module.AgentProcess = AgentProcess
    spec.loader.exec_module(module)

    process = _find_process(module)
    if process is None:
        raise ValueError("agent script must define an AgentProcess subclass or create_process()")
    return process


def _validate_allowed_path(path: Path, allowed_roots: list[Path]) -> None:
    if any(_is_relative_to(path, root) for root in allowed_roots):
        return
    roots = ", ".join(str(root) for root in allowed_roots)
    raise PermissionError(f"agent script must be under an allowed workspace root: {roots}")


def _preflight_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
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


def _find_process(module: ModuleType) -> AgentProcess | None:
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


def _validate_process(process: AgentProcess, name: str) -> None:
    if not isinstance(process, AgentProcess):
        raise TypeError("loaded object is not an AgentProcess")
    if not name:
        raise ValueError("agent process name must not be empty")
    if any(char in name for char in "\r\n\t/\\"):
        raise ValueError("agent process name contains unsupported characters")
    mailbox_size = int(getattr(process, "mailbox_size", 1024) or 1024)
    token_budget = int(getattr(process, "token_budget", 8000) or 8000)
    if mailbox_size <= 0:
        raise ValueError("agent process mailbox_size must be greater than zero")
    if token_budget <= 0:
        raise ValueError("agent process token_budget must be greater than zero")
    capabilities = getattr(process, "capabilities", ())
    if isinstance(capabilities, str) or not all(isinstance(item, str) for item in capabilities):
        raise ValueError("agent process capabilities must be an iterable of strings")


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
