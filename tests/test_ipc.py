import asyncio
import json
import logging
import time

import pytest
from kernel.native_core import native_core_available, require_native_core

pytestmark = pytest.mark.requires_native_core

if not native_core_available():
    pytest.skip("requires agent_os_core native extension", allow_module_level=True)

native_core = require_native_core("native IPC tests")
AgentMessage = native_core.AgentMessage
NativeIPCBus = native_core.NativeIPCBus

LOGGER = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_standard_async_ping_pong() -> None:
    bus = NativeIPCBus()
    bus.register_mailbox("Agent_Alpha", 16)
    bus.register_mailbox("Agent_Beta", 16)

    bus.send_message(AgentMessage("Agent_Alpha", "Agent_Beta", '{"cmd": "ping"}'))
    ping = await asyncio.wait_for(bus.recv_message("Agent_Beta"), timeout=1.0)

    assert ping.sender == "Agent_Alpha"
    assert ping.receiver == "Agent_Beta"
    assert json.loads(ping.payload) == {"cmd": "ping"}

    bus.send_message(AgentMessage("Agent_Beta", "Agent_Alpha", '{"cmd": "pong"}'))
    pong = await asyncio.wait_for(bus.recv_message("Agent_Alpha"), timeout=1.0)

    assert pong.sender == "Agent_Beta"
    assert pong.receiver == "Agent_Alpha"
    assert json.loads(pong.payload) == {"cmd": "pong"}


@pytest.mark.asyncio
async def test_high_throughput_concurrency_stress(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    bus = NativeIPCBus()
    agent_count = 5
    total_messages = 10_000
    messages_per_agent = total_messages // agent_count
    agent_names = [f"Agent_{index}" for index in range(agent_count)]

    for agent_name in agent_names:
        bus.register_mailbox(agent_name, messages_per_agent + 256)

    async def agent(index: int) -> set[tuple[str, int]]:
        sender = agent_names[index]
        receiver = agent_names[(index + 1) % agent_count]

        for sequence in range(messages_per_agent):
            payload = json.dumps({"sender": sender, "sequence": sequence})
            bus.send_message(AgentMessage(sender, receiver, payload))
            if sequence % 256 == 0:
                await asyncio.sleep(0)

        received: set[tuple[str, int]] = set()
        for _ in range(messages_per_agent):
            message = await asyncio.wait_for(bus.recv_message(sender), timeout=5.0)
            payload = json.loads(message.payload)

            assert message.receiver == sender
            assert payload["sender"] == message.sender
            received.add((message.sender, payload["sequence"]))

        return received

    started = time.perf_counter()
    results = await asyncio.gather(*(agent(index) for index in range(agent_count)))
    elapsed = time.perf_counter() - started

    delivered = set().union(*results)
    assert len(delivered) == total_messages
    assert sum(len(result) for result in results) == total_messages

    LOGGER.info("Processed %s IPC messages in %.4f seconds", total_messages, elapsed)


@pytest.mark.asyncio
async def test_backpressure_bounded_mailbox_overflow() -> None:
    bus = NativeIPCBus()
    bus.register_mailbox("ThrottledAgent", 5)

    accepted = 0
    failures: list[Exception] = []

    for sequence in range(10):
        try:
            payload = json.dumps({"sequence": sequence})
            bus.send_message(AgentMessage("Producer", "ThrottledAgent", payload))
            accepted += 1
        except Exception as exc:
            failures.append(exc)

    assert accepted == 5
    assert len(failures) == 5
    assert all("full" in str(exc).lower() for exc in failures)

    drained = []
    for _ in range(accepted):
        message = await asyncio.wait_for(bus.recv_message("ThrottledAgent"), timeout=1.0)
        drained.append(json.loads(message.payload)["sequence"])

    assert drained == list(range(accepted))


def test_malformed_json_input_guard_rails() -> None:
    with pytest.raises(ValueError, match="payload must be valid JSON"):
        AgentMessage("Agent_Alpha", "Agent_Beta", "{malformed_json:")
