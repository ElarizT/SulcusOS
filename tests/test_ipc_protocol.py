import asyncio
import time

import pytest

from kernel.ipc_protocol import ErrorMessage, IPCProtocolError, TaskRequest, TaskResponse, parse_message
from kernel.process import ProcessRecord, ProcessRegistry, ProcessState
from test_process_registry import FakeBus, FakeKernel, FakeMemory


@pytest.mark.asyncio
async def test_valid_structured_message_send(tmp_path) -> None:
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])
    registry.bus.register_mailbox("Sender", 4)
    registry.bus.register_mailbox("Receiver", 4)
    sender = ProcessRecord(100, "Sender", tmp_path / "sender.py", ProcessState.RUNNING, time.monotonic(), 4)
    receiver = ProcessRecord(101, "Receiver", tmp_path / "receiver.py", ProcessState.RUNNING, time.monotonic(), 4)
    registry._records[sender.pid] = sender
    registry._records[receiver.pid] = receiver

    message = registry.send_ipc_message(sender.pid, receiver.pid, {"cmd": "ping"})

    assert isinstance(message, TaskRequest)
    delivered = await registry.bus.recv_message("Receiver")
    parsed = parse_message(delivered.payload)
    assert parsed.source_pid == sender.pid
    assert parsed.target_pid == receiver.pid
    assert parsed.payload == {"cmd": "ping"}


@pytest.mark.asyncio
async def test_invalid_target_pid_returns_error(tmp_path) -> None:
    script = tmp_path / "sender.py"
    script.write_text(
        """
from kernel.process import AgentProcess

class Sender(AgentProcess):
    name = "Sender"
""",
        encoding="utf-8",
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])
    sender = await registry.run_path(str(script))

    error = registry.send_ipc_message(sender.pid, 9999, {"cmd": "ping"})

    assert isinstance(error, ErrorMessage)
    assert error.error_code == "target_not_found"


@pytest.mark.asyncio
async def test_request_reply_correlation(tmp_path) -> None:
    pong = tmp_path / "pong.py"
    pong.write_text(
        """
from kernel.process import AgentProcess

class Pong(AgentProcess):
    name = "Pong"

    async def on_message(self, message):
        self.reply(message, {"pong": message.payload["ping"]})
""",
        encoding="utf-8",
    )
    ping = tmp_path / "ping.py"
    ping.write_text(
        """
from kernel.process import AgentProcess

class Ping(AgentProcess):
    name = "Ping"

    async def on_start(self):
        response = await self.request(100, {"ping": "hello"}, timeout=1.0)
        self.remember({"response": response.payload, "correlation_id": response.correlation_id}, 1)
""",
        encoding="utf-8",
    )
    memory = FakeMemory()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=memory, allowed_roots=[tmp_path])

    await registry.run_path(str(pong))
    await registry.run_path(str(ping))
    for _ in range(50):
        if memory.tables.get("Ping", 0) > 0:
            break
        await asyncio.sleep(0.02)

    assert memory.tables["Ping"] > 0
    rows = await registry.list_processes()
    ping_row = next(row for row in rows if row["name"] == "Ping")
    assert ping_row["messages_sent"] >= 1
    assert ping_row["messages_received"] >= 1


@pytest.mark.asyncio
async def test_request_timeout_returns_structured_error(tmp_path) -> None:
    idle = tmp_path / "idle.py"
    idle.write_text(
        """
from kernel.process import AgentProcess

class Idle(AgentProcess):
    name = "Idle"
""",
        encoding="utf-8",
    )
    requester = tmp_path / "requester.py"
    requester.write_text(
        """
from kernel.process import AgentProcess

class Requester(AgentProcess):
    name = "Requester"

    async def on_start(self):
        response = await self.request(100, {"ping": "hello"}, timeout=0.05)
        self.remember({"code": response.payload["code"]}, 1)
""",
        encoding="utf-8",
    )
    memory = FakeMemory()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=memory, allowed_roots=[tmp_path])

    await registry.run_path(str(idle))
    await registry.run_path(str(requester))
    for _ in range(50):
        if memory.tables.get("Requester", 0) > 0:
            break
        await asyncio.sleep(0.02)

    assert memory.tables["Requester"] > 0


@pytest.mark.asyncio
async def test_mailbox_full_returns_structured_error(tmp_path) -> None:
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])
    registry.bus.register_mailbox("Sender", 1)
    registry.bus.register_mailbox("Tiny", 1)
    source = ProcessRecord(100, "Sender", tmp_path / "sender.py", ProcessState.RUNNING, time.monotonic(), 1)
    target = ProcessRecord(101, "Tiny", tmp_path / "tiny.py", ProcessState.RUNNING, time.monotonic(), 1)
    registry._records[source.pid] = source
    registry._records[target.pid] = target

    first = registry.send_ipc_message(source.pid, target.pid, {"n": 1})
    second = registry.send_ipc_message(source.pid, target.pid, {"n": 2})

    assert isinstance(first, TaskRequest)
    assert isinstance(second, ErrorMessage)
    assert second.error_code == "mailbox_full"


def test_message_validation_failures_and_serialization() -> None:
    with pytest.raises(IPCProtocolError, match="priority"):
        TaskRequest(source_pid=1, target_pid=2, payload={}, correlation_id="c", priority="urgent")
    with pytest.raises(IPCProtocolError, match="serializable"):
        TaskRequest(source_pid=1, target_pid=2, payload={"bad": object()}, correlation_id="c")

    message = TaskResponse(source_pid=2, target_pid=1, payload={"ok": True}, correlation_id="c")
    assert parse_message(message.to_json()) == message


@pytest.mark.asyncio
async def test_isolated_process_message_exchange(tmp_path) -> None:
    pong = tmp_path / "pong.py"
    pong.write_text(
        """
from kernel.process import AgentProcess

class Pong(AgentProcess):
    name = "Pong"

    async def on_message(self, message):
        self.reply(message, {"pong": True})
""",
        encoding="utf-8",
    )
    ping = tmp_path / "ping.py"
    ping.write_text(
        """
from kernel.process import AgentProcess

class Ping(AgentProcess):
    name = "Ping"

    async def on_start(self):
        response = await self.request(100, {"ping": True}, timeout=2.0)
        if response.payload.get("pong"):
            self.send(100, {"seen": True}, message_type="event")
""",
        encoding="utf-8",
    )
    registry = ProcessRegistry(
        kernel=FakeKernel(),
        bus=FakeBus(),
        memory=FakeMemory(),
        allowed_roots=[tmp_path],
        execution_mode="isolated",
    )

    await registry.run_path(str(pong))
    await registry.run_path(str(ping))
    for _ in range(80):
        rows = await registry.list_processes()
        ping_row = next(row for row in rows if row["name"] == "Ping")
        if ping_row["messages_received"] >= 1:
            break
        await asyncio.sleep(0.05)

    rows = await registry.list_processes()
    ping_row = next(row for row in rows if row["name"] == "Ping")
    assert ping_row["messages_sent"] >= 1
    assert ping_row["messages_received"] >= 1
