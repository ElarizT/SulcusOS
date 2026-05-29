import asyncio

import pytest

from kernel.memory_store import PersistentMemoryManager
from kernel.process import ProcessRegistry
from test_process_registry import FakeBus, FakeKernel


def test_remembering_memory(tmp_path) -> None:
    memory = PersistentMemoryManager(tmp_path / "memory")
    memory.register_agent("Agent", 100)
    memory.bind_process("Agent", 100)

    memory.append_context_frame("Agent", {"fact": "hello"}, 5, importance=0.8, tags=["greeting"])

    assert memory.get_page_table_summary("Agent")["active_frames"] == 1
    assert memory.recall("Agent", tags=["greeting"])[0]["content"] == {"fact": "hello"}


def test_token_budget_eviction_persists_low_importance_first(tmp_path) -> None:
    memory = PersistentMemoryManager(tmp_path / "memory")
    memory.register_agent("Agent", 10)

    memory.append_context_frame("Agent", {"fact": "keep"}, 6, importance=0.9, tags=["keep"])
    evicted = memory.append_context_frame("Agent", {"fact": "evict"}, 6, importance=0.1, tags=["drop"])

    summary = memory.get_page_table_summary("Agent")
    assert evicted is True
    assert summary["cold_frames"] == 1
    assert memory.recall("Agent", tags=["drop"])[0]["tier"] in {"warm", "cold"}
    assert memory.recall("Agent", tags=["keep"])[0]["tier"] == "hot"


def test_persistence_across_manager_reload(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    memory = PersistentMemoryManager(memory_dir)
    memory.register_agent("Agent", 5)
    memory.append_context_frame("Agent", {"fact": "persistent"}, 6, tags=["persist"])

    reloaded = PersistentMemoryManager(memory_dir)
    reloaded.register_agent("Agent", 5)

    assert reloaded.recall("Agent", tags=["persist"])[0]["content"] == {"fact": "persistent"}


def test_recall_by_tag_and_substring(tmp_path) -> None:
    memory = PersistentMemoryManager(tmp_path / "memory")
    memory.register_agent("Agent", 100)
    memory.append_context_frame("Agent", {"note": "alpha rocket"}, 3, tags=["space"])
    memory.append_context_frame("Agent", {"note": "beta ocean"}, 3, tags=["water"])

    assert memory.recall("Agent", tags=["space"])[0]["content"]["note"] == "alpha rocket"
    assert memory.recall("Agent", query="ocean")[0]["content"]["note"] == "beta ocean"


def test_forget_memory(tmp_path) -> None:
    memory = PersistentMemoryManager(tmp_path / "memory")
    memory.register_agent("Agent", 5)
    memory.append_context_frame("Agent", {"note": "remove me"}, 6, tags=["delete"])
    record = memory.recall("Agent", tags=["delete"])[0]

    assert memory.forget(record["memory_id"]) is True
    assert memory.recall("Agent", tags=["delete"]) == []


def test_snapshot_creation_and_restore(tmp_path) -> None:
    memory = PersistentMemoryManager(tmp_path / "memory")
    memory.register_agent("Agent", 100)
    memory.bind_process("Agent", 100)
    memory.append_context_frame("Agent", {"note": "snapshot me"}, 4, tags=["snap"])

    snapshot_id = memory.snapshot_process(100, "Agent")
    memory.unregister_agent("Agent")
    memory.register_agent("Agent", 100)
    restored = memory.restore_process_memory(101, "Agent", snapshot_id)

    assert restored == snapshot_id
    assert memory.recall("Agent", tags=["snap"])[0]["owner_pid"] == 101


@pytest.mark.asyncio
async def test_supervised_restart_with_latest_snapshot_policy(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "child.py"
    parent_path.write_text(
        'from kernel.process import AgentProcess\n\nclass Parent(AgentProcess):\n    name = "Parent"\n',
        encoding="utf-8",
    )
    child_path.write_text(
        'from kernel.process import AgentProcess\n\n'
        'class Child(AgentProcess):\n'
        '    name = "Child"\n'
        '    memory_restore_policy = "latest_snapshot"\n'
        '    async def on_message(self, message):\n'
        '        if message.payload.get("cmd") == "crash":\n'
        '            raise RuntimeError("boom")\n',
        encoding="utf-8",
    )
    memory = PersistentMemoryManager(tmp_path / "memory")
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=memory, allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path))
    memory.append_context_frame("Child", {"note": "restore me"}, 4, tags=["restore"])
    snapshot_id = registry.snapshot_process(child.pid)
    registry.send_ipc_message(parent.pid, child.pid, {"cmd": "crash"}, message_type="control")

    for _ in range(50):
        children = registry.list_children(parent.pid)
        if children and children[0] != child.pid:
            break
        await asyncio.sleep(0.02)

    new_child = registry.list_children(parent.pid)[0]
    assert registry._records[new_child].latest_snapshot_id == snapshot_id
    assert memory.recall("Child", tags=["restore"])[0]["owner_pid"] == new_child


def test_windows_safe_memory_paths(tmp_path) -> None:
    root = tmp_path / "memory root with spaces"
    memory = PersistentMemoryManager(root)
    memory.register_agent("Agent", 10)
    memory.append_context_frame("Agent", {"ok": True}, 1)

    assert memory.records_path.parent == root.resolve()
    assert memory.records_path.exists() is False
