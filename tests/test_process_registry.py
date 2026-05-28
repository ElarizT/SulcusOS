import asyncio
import json

import pytest

from kernel.process import AgentMessage, ProcessRegistry


class FakeKernel:
    def __init__(self) -> None:
        self.agents: set[str] = set()
        self.capabilities: dict[str, set[str]] = {}

    def register_agent_capability(self, agent_name: str, capability: str) -> None:
        self.agents.add(agent_name)
        self.capabilities.setdefault(capability, set()).add(agent_name)

    def unregister_agent(self, agent_name: str) -> None:
        self.agents.discard(agent_name)
        for agents in self.capabilities.values():
            agents.discard(agent_name)


class FakeBus:
    def __init__(self) -> None:
        self.mailboxes: dict[str, asyncio.Queue[AgentMessage]] = {}
        self.sizes: dict[str, int] = {}

    def register_mailbox(self, agent_name: str, buffer_size: int) -> None:
        if agent_name in self.mailboxes:
            raise ValueError("duplicate mailbox")
        self.mailboxes[agent_name] = asyncio.Queue(maxsize=buffer_size)
        self.sizes[agent_name] = buffer_size

    def unregister_mailbox(self, agent_name: str) -> bool:
        self.sizes.pop(agent_name, None)
        return self.mailboxes.pop(agent_name, None) is not None

    def send_message(self, message: AgentMessage) -> None:
        self.mailboxes[message.receiver].put_nowait(message)

    async def recv_message(self, agent_name: str) -> AgentMessage:
        return await self.mailboxes[agent_name].get()

    def get_mailbox_metrics(self) -> list[tuple[str, int, int, str]]:
        return [
            (name, mailbox.qsize(), self.sizes[name], "Direct")
            for name, mailbox in self.mailboxes.items()
        ]


class FakeMemory:
    def __init__(self) -> None:
        self.tables: dict[str, int] = {}

    def register_agent(self, agent_name: str, max_active_tokens: int) -> None:
        self.tables[agent_name] = 0

    def unregister_agent(self, agent_name: str) -> bool:
        return self.tables.pop(agent_name, None) is not None

    def append_context_frame(self, agent_name: str, content: str, token_estimate: int) -> None:
        json.loads(content)
        self.tables[agent_name] += token_estimate

    def get_page_table_summary(self, agent_name: str) -> dict[str, int]:
        return {"current_active_tokens": self.tables[agent_name]}


@pytest.mark.asyncio
async def test_run_ps_and_kill_process(tmp_path) -> None:
    script = tmp_path / "echo_agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class EchoAgent(AgentProcess):
    name = "EchoAgent"
    capabilities = ("echo",)

    async def on_message(self, message):
        self.remember({"received": message.payload}, 3)
""",
        encoding="utf-8",
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    record = await registry.run_path(str(script))
    assert record.pid == 100
    assert record.name == "EchoAgent"

    rows = await registry.list_processes()
    assert rows[0]["status"] in {"starting", "running"}

    await registry.kill(record.pid)
    rows = await registry.list_processes()
    assert rows[0]["status"] == "killed"
    assert rows[0]["mailbox_size"] == 1024


@pytest.mark.asyncio
async def test_duplicate_process_names_are_rejected(tmp_path) -> None:
    script = tmp_path / "duplicate_agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class DuplicateAgent(AgentProcess):
    name = "DuplicateAgent"
""",
        encoding="utf-8",
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    await registry.run_path(str(script))
    with pytest.raises(ValueError, match="already running"):
        await registry.run_path(str(script))


@pytest.mark.asyncio
async def test_invalid_path_is_rejected(tmp_path) -> None:
    registry = ProcessRegistry(
        kernel=FakeKernel(),
        bus=FakeBus(),
        memory=FakeMemory(),
        allowed_roots=[tmp_path],
    )

    with pytest.raises(FileNotFoundError):
        await registry.run_path(str(tmp_path / "missing.py"))


@pytest.mark.asyncio
async def test_path_outside_allowed_root_is_rejected(tmp_path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    script = outside / "agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class OutsideAgent(AgentProcess):
    name = "OutsideAgent"
""",
        encoding="utf-8",
    )
    registry = ProcessRegistry(
        kernel=FakeKernel(),
        bus=FakeBus(),
        memory=FakeMemory(),
        allowed_roots=[root],
    )

    with pytest.raises(PermissionError, match="allowed workspace root"):
        await registry.run_path(str(script))


@pytest.mark.asyncio
async def test_bad_script_import_is_rejected_before_registration(tmp_path) -> None:
    script = tmp_path / "bad_import.py"
    script.write_text(
        """
import os

from kernel.process import AgentProcess

class BadAgent(AgentProcess):
    name = "BadAgent"
""",
        encoding="utf-8",
    )
    bus = FakeBus()
    memory = FakeMemory()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=bus, memory=memory, allowed_roots=[tmp_path])

    with pytest.raises(ValueError, match="import is not allowed"):
        await registry.run_path(str(script))

    assert bus.mailboxes == {}
    assert memory.tables == {}


@pytest.mark.asyncio
async def test_process_crash_cleans_resources_and_keeps_status(tmp_path) -> None:
    script = tmp_path / "crash_agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class CrashAgent(AgentProcess):
    name = "CrashAgent"

    async def on_start(self):
        raise RuntimeError("boom")
""",
        encoding="utf-8",
    )
    bus = FakeBus()
    memory = FakeMemory()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=bus, memory=memory, allowed_roots=[tmp_path])

    record = await registry.run_path(str(script))
    for _ in range(20):
        rows = await registry.list_processes()
        if rows[0]["status"] == "crashed":
            break
        await asyncio.sleep(0.01)

    rows = await registry.list_processes()
    assert rows[0]["pid"] == record.pid
    assert rows[0]["status"] == "crashed"
    assert rows[0]["error"]
    assert bus.mailboxes == {}
    assert memory.tables == {}

    restarted = await registry.run_path(str(script))
    assert restarted.pid != record.pid


@pytest.mark.asyncio
async def test_kill_cleans_resources_and_registry_consistency(tmp_path) -> None:
    script = tmp_path / "long_agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class LongAgent(AgentProcess):
    name = "LongAgent"
    capabilities = ("long",)
""",
        encoding="utf-8",
    )
    kernel = FakeKernel()
    bus = FakeBus()
    memory = FakeMemory()
    registry = ProcessRegistry(kernel=kernel, bus=bus, memory=memory, allowed_roots=[tmp_path])

    record = await registry.run_path(str(script))
    await registry.kill(record.pid)

    rows = await registry.list_processes()
    assert rows[0]["status"] == "killed"
    assert bus.mailboxes == {}
    assert memory.tables == {}
    assert "LongAgent" not in kernel.agents
    assert all("LongAgent" not in agents for agents in kernel.capabilities.values())


@pytest.mark.asyncio
async def test_registration_failure_rolls_back_mailbox(tmp_path) -> None:
    script = tmp_path / "rollback_agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class RollbackAgent(AgentProcess):
    name = "RollbackAgent"
""",
        encoding="utf-8",
    )

    class FailingMemory(FakeMemory):
        def register_agent(self, agent_name: str, max_active_tokens: int) -> None:
            raise RuntimeError("memory offline")

    bus = FakeBus()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=bus, memory=FailingMemory(), allowed_roots=[tmp_path])

    with pytest.raises(RuntimeError, match="memory offline"):
        await registry.run_path(str(script))

    assert bus.mailboxes == {}
    assert await registry.list_processes() == []


@pytest.mark.asyncio
async def test_process_mode_startup_and_kill_cleanup(tmp_path) -> None:
    script = tmp_path / "isolated_agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class IsolatedAgent(AgentProcess):
    name = "IsolatedAgent"
    capabilities = ("isolated",)
""",
        encoding="utf-8",
    )
    kernel = FakeKernel()
    bus = FakeBus()
    memory = FakeMemory()
    registry = ProcessRegistry(
        kernel=kernel,
        bus=bus,
        memory=memory,
        allowed_roots=[tmp_path],
        execution_mode="isolated",
    )

    record = await registry.run_path(str(script))
    rows = await registry.list_processes()
    assert rows[0]["status"] == "running"
    assert rows[0]["execution_mode"] == "isolated"

    await registry.kill(record.pid)
    rows = await registry.list_processes()
    assert rows[0]["status"] == "killed"
    assert bus.mailboxes == {}
    assert memory.tables == {}
    assert "IsolatedAgent" not in kernel.agents


@pytest.mark.asyncio
async def test_process_mode_child_crash_cleans_resources(tmp_path) -> None:
    script = tmp_path / "crashing_child.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class CrashingChild(AgentProcess):
    name = "CrashingChild"

    async def on_start(self):
        raise RuntimeError("child boom")
""",
        encoding="utf-8",
    )
    bus = FakeBus()
    memory = FakeMemory()
    registry = ProcessRegistry(
        kernel=FakeKernel(),
        bus=bus,
        memory=memory,
        allowed_roots=[tmp_path],
        execution_mode="isolated",
    )

    record = await registry.run_path(str(script))
    for _ in range(50):
        rows = await registry.list_processes()
        if rows[0]["status"] == "crashed":
            break
        await asyncio.sleep(0.05)

    rows = await registry.list_processes()
    assert rows[0]["pid"] == record.pid
    assert rows[0]["status"] == "crashed"
    assert bus.mailboxes == {}
    assert memory.tables == {}


@pytest.mark.asyncio
async def test_process_mode_invalid_child_metadata_cleans_up(tmp_path) -> None:
    script = tmp_path / "invalid_child.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class InvalidChild(AgentProcess):
    name = "InvalidChild"
    mailbox_size = -1
""",
        encoding="utf-8",
    )
    bus = FakeBus()
    memory = FakeMemory()
    registry = ProcessRegistry(
        kernel=FakeKernel(),
        bus=bus,
        memory=memory,
        allowed_roots=[tmp_path],
        execution_mode="isolated",
    )

    with pytest.raises(RuntimeError, match="crashed during startup"):
        await registry.run_path(str(script))

    assert bus.mailboxes == {}
    assert memory.tables == {}
    assert await registry.list_processes() == []


@pytest.mark.asyncio
async def test_process_mode_timeout_cleans_stale_child(tmp_path, monkeypatch) -> None:
    script = tmp_path / "timeout_agent.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class TimeoutAgent(AgentProcess):
    name = "TimeoutAgent"
""",
        encoding="utf-8",
    )
    bus = FakeBus()
    registry = ProcessRegistry(
        kernel=FakeKernel(),
        bus=bus,
        memory=FakeMemory(),
        allowed_roots=[tmp_path],
        execution_mode="isolated",
        startup_timeout_seconds=0.01,
    )

    async def timeout(*_args, **_kwargs):
        raise TimeoutError("synthetic startup timeout")

    monkeypatch.setattr(registry, "_wait_for_child_message", timeout)

    with pytest.raises(TimeoutError, match="synthetic startup timeout"):
        await registry.run_path(str(script))

    assert bus.mailboxes == {}
    assert await registry.list_processes() == []


@pytest.mark.asyncio
async def test_process_mode_windows_safe_path_with_spaces(tmp_path) -> None:
    root = tmp_path / "root with spaces"
    root.mkdir()
    script = root / "agent with spaces.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class SpacePathAgent(AgentProcess):
    name = "SpacePathAgent"
""",
        encoding="utf-8",
    )
    registry = ProcessRegistry(
        kernel=FakeKernel(),
        bus=FakeBus(),
        memory=FakeMemory(),
        allowed_roots=[root],
        execution_mode="isolated",
    )

    record = await registry.run_path(str(script))
    assert record.name == "SpacePathAgent"
    await registry.kill(record.pid)
