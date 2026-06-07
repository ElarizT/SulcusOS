import asyncio

import pytest

from kernel.dashboard import AgentOSDashboard
from kernel.process import ProcessRegistry
from test_process_registry import FakeBus, FakeKernel, FakeMemory


async def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition was not reached before timeout")


def write_agent(path, class_body: str) -> None:
    path.write_text(
        "from kernel.process import AgentProcess\n\n" + class_body,
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_one_for_one_restarts_only_failed_child(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    crash_path = tmp_path / "crash.py"
    stable_path = tmp_path / "stable.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n    supervisor_strategy = "one_for_one"\n')
    write_agent(
        crash_path,
        'class CrashWorker(AgentProcess):\n'
        '    name = "CrashWorker"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("boom")\n',
    )
    write_agent(stable_path, 'class StableWorker(AgentProcess):\n    name = "StableWorker"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    failed = await registry.spawn_child(parent.pid, str(crash_path))
    stable = await registry.spawn_child(parent.pid, str(stable_path))
    registry.send_ipc_message(parent.pid, failed.pid, {"cmd": "crash"}, message_type="control")

    await wait_for(lambda: failed.pid not in registry.list_children(parent.pid))
    children = registry.list_children(parent.pid)
    assert stable.pid in children
    assert failed.pid not in children
    assert len(children) == 2


@pytest.mark.asyncio
async def test_one_for_all_restarts_all_children(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    crash_path = tmp_path / "crash.py"
    stable_path = tmp_path / "stable.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n    supervisor_strategy = "one_for_all"\n')
    write_agent(
        crash_path,
        'class CrashWorker(AgentProcess):\n'
        '    name = "CrashWorker"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("boom")\n',
    )
    write_agent(stable_path, 'class StableWorker(AgentProcess):\n    name = "StableWorker"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    failed = await registry.spawn_child(parent.pid, str(crash_path))
    stable = await registry.spawn_child(parent.pid, str(stable_path))
    original_children = {failed.pid, stable.pid}
    registry.send_ipc_message(parent.pid, failed.pid, {"cmd": "crash"}, message_type="control")

    await wait_for(lambda: set(registry.list_children(parent.pid)).isdisjoint(original_children))
    children = set(registry.list_children(parent.pid))
    assert children.isdisjoint(original_children)
    assert len(children) == 2


@pytest.mark.asyncio
async def test_rest_for_one_restarts_failed_child_and_later_children(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    first_path = tmp_path / "first.py"
    crash_path = tmp_path / "crash.py"
    later_path = tmp_path / "later.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n    supervisor_strategy = "rest_for_one"\n')
    write_agent(first_path, 'class FirstWorker(AgentProcess):\n    name = "FirstWorker"\n')
    write_agent(
        crash_path,
        'class CrashWorker(AgentProcess):\n'
        '    name = "CrashWorker"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("boom")\n',
    )
    write_agent(later_path, 'class LaterWorker(AgentProcess):\n    name = "LaterWorker"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    first = await registry.spawn_child(parent.pid, str(first_path))
    failed = await registry.spawn_child(parent.pid, str(crash_path))
    later = await registry.spawn_child(parent.pid, str(later_path))
    registry.send_ipc_message(parent.pid, failed.pid, {"cmd": "crash"}, message_type="control")

    await wait_for(
        lambda: failed.pid not in registry.list_children(parent.pid)
        and later.pid not in registry.list_children(parent.pid)
        and len(registry.list_children(parent.pid)) == 3
    )

    children = set(registry.list_children(parent.pid))
    assert first.pid in children
    assert failed.pid not in children
    assert later.pid not in children
    assert len(children) == 3


@pytest.mark.asyncio
async def test_transient_policy_does_not_restart_normal_exit(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "exiting.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(child_path, 'class Exiting(AgentProcess):\n    name = "Exiting"\n    async def run(self):\n        return\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path), restart_policy="transient")

    await wait_for(lambda: registry._records[child.pid].state.value == "exited")
    await asyncio.sleep(0.05)

    assert registry._records[parent.pid].restart_count == 0
    assert child.pid not in registry.list_children(parent.pid)


@pytest.mark.asyncio
async def test_restart_threshold_escalates(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "crashing.py"
    write_agent(
        parent_path,
        'class Parent(AgentProcess):\n'
        '    name = "Parent"\n'
        '    max_restarts = 1\n'
        '    restart_window_seconds = 10.0\n',
    )
    write_agent(
        child_path,
        'class Crashing(AgentProcess):\n'
        '    name = "Crashing"\n'
        '    async def on_start(self):\n'
        '        raise RuntimeError("startup boom")\n',
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    await registry.spawn_child(parent.pid, str(child_path), restart_policy="permanent")

    await wait_for(lambda: registry._records[parent.pid].escalated)

    assert registry._records[parent.pid].restart_count == 1


@pytest.mark.asyncio
async def test_child_cleanup_after_crash(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "temporary.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(
        child_path,
        'class Temporary(AgentProcess):\n'
        '    name = "Temporary"\n'
        '    async def on_start(self):\n'
        '        raise RuntimeError("no restart")\n',
    )
    bus = FakeBus()
    memory = FakeMemory()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=bus, memory=memory, allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path), restart_policy="temporary")

    await wait_for(lambda: registry._records[child.pid].state.value == "crashed")

    assert not registry._records[child.pid].resources_registered
    assert "Temporary" not in bus.mailboxes
    assert "Temporary" not in memory.tables


@pytest.mark.asyncio
async def test_parent_termination_cascades_to_children(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "child.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(child_path, 'class Child(AgentProcess):\n    name = "Child"\n')
    bus = FakeBus()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=bus, memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path))
    await registry.kill(parent.pid)

    assert registry._records[parent.pid].state.value == "killed"
    assert registry._records[child.pid].state.value == "killed"
    assert bus.mailboxes == {}


@pytest.mark.asyncio
async def test_killed_child_notifies_supervisor_and_restarts(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "child.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(child_path, 'class Child(AgentProcess):\n    name = "Child"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path))
    await registry.kill(child.pid)

    rows = await registry.list_processes()
    child_row = next(row for row in rows if row["pid"] == child.pid)
    replacement_row = next(row for row in rows if row["name"] == "Child" and row["status"] == "running")
    parent_row = next(row for row in rows if row["pid"] == parent.pid)
    events = registry.list_supervision_events()

    assert child_row["status"] == "killed"
    assert child_row["supervisor_pid"] == parent.pid
    assert replacement_row["pid"] != child.pid
    assert replacement_row["restart_count"] == 1
    assert parent_row["child_count"] == 1
    assert parent_row["child_pids"] == [replacement_row["pid"]]
    assert parent_row["restart_count"] == 0
    assert events == [
        {
            "event": "child_terminated",
            "pid": child.pid,
            "name": "Child",
            "state": "killed",
            "parent_pid": parent.pid,
            "details": {},
            "supervisor_pid": parent.pid,
            "supervisor_name": "Parent",
            "message": "Detected child termination:\nChild",
        },
        {
            "event": "child_restart_requested",
            "pid": child.pid,
            "name": "Child",
            "state": "killed",
            "parent_pid": parent.pid,
            "details": {},
            "supervisor_pid": parent.pid,
            "supervisor_name": "Parent",
            "message": "Restarting child:\nChild",
        },
        {
            "event": "child_restarted",
            "pid": parent.pid,
            "name": "Parent",
            "state": "running",
            "parent_pid": None,
            "details": {
                "old_pid": child.pid,
                "new_pid": replacement_row["pid"],
                "child_name": "Child",
            },
            "supervisor_pid": parent.pid,
            "supervisor_name": "Parent",
            "message": "Child restarted:\nChild",
        },
    ]
    assert [record.name for record in registry._records.values()].count("Child") == 2


@pytest.mark.asyncio
async def test_dashboard_reflects_restarted_supervised_child(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "child.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(child_path, 'class Child(AgentProcess):\n    name = "Child"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])
    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path))
    await registry.kill(child.pid)
    rows = await registry.list_processes()

    hierarchy = AgentOSDashboard._hierarchy_from_process_rows(rows)
    tree = AgentOSDashboard._format_agent_tree(hierarchy)

    assert hierarchy == {"supervisor": "Parent", "children": ["Child (restarted)"]}
    assert AgentOSDashboard._display_process_status("killed") == "TERMINATED"
    assert "Child (restarted)" in tree
    assert "Child (terminated)" not in tree


@pytest.mark.asyncio
async def test_killed_unsupervised_process_does_not_restart(tmp_path) -> None:
    child_path = tmp_path / "child.py"
    write_agent(child_path, 'class Child(AgentProcess):\n    name = "Child"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    child = await registry.run_path(str(child_path))
    await registry.kill(child.pid)

    rows = await registry.list_processes()
    assert [(row["pid"], row["status"]) for row in rows] == [(child.pid, "killed")]
    assert registry.list_supervision_events() == []


@pytest.mark.asyncio
async def test_isolated_child_restart(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "isolated_crash.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(
        child_path,
        'class IsolatedCrash(AgentProcess):\n'
        '    name = "IsolatedCrash"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("isolated boom")\n',
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path), execution_mode="isolated")
    registry.send_ipc_message(parent.pid, child.pid, {"cmd": "crash"}, message_type="control")

    await wait_for(lambda: registry._records[parent.pid].restart_count >= 1, timeout=5.0)

    assert registry._records[parent.pid].child_pids


@pytest.mark.asyncio
async def test_supervisor_registry_consistency(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "child.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(child_path, 'class Child(AgentProcess):\n    name = "Child"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path), restart_policy="temporary")
    rows = await registry.list_processes()
    child_row = next(row for row in rows if row["pid"] == child.pid)
    parent_row = next(row for row in rows if row["pid"] == parent.pid)

    assert child_row["parent_pid"] == parent.pid
    assert child_row["restart_policy"] == "temporary"
    assert parent_row["child_count"] == 1
    assert child.pid in parent_row["child_pids"]


@pytest.mark.asyncio
async def test_parent_killed_while_child_restart_is_backing_off(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "crash.py"
    write_agent(
        parent_path,
        'class Parent(AgentProcess):\n'
        '    name = "Parent"\n'
        '    restart_backoff_seconds = 0.3\n',
    )
    write_agent(
        child_path,
        'class Crash(AgentProcess):\n'
        '    name = "Crash"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("boom")\n',
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path))
    registry.send_ipc_message(parent.pid, child.pid, {"cmd": "crash"}, message_type="control")
    await wait_for(lambda: registry._records[parent.pid].restart_count == 1)

    await registry.kill(parent.pid)
    await asyncio.sleep(0.4)

    assert registry._records[parent.pid].state.value == "killed"
    assert registry._records[parent.pid].child_pids == []
    assert [record.name for record in registry._records.values()].count("Crash") == 1


@pytest.mark.asyncio
async def test_supervisor_crash_terminates_live_children(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "child.py"
    write_agent(
        parent_path,
        'class Parent(AgentProcess):\n'
        '    name = "Parent"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("supervisor boom")\n',
    )
    write_agent(child_path, 'class Child(AgentProcess):\n    name = "Child"\n')
    bus = FakeBus()
    registry = ProcessRegistry(kernel=FakeKernel(), bus=bus, memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path))
    registry.send_ipc_message(child.pid, parent.pid, {"cmd": "crash"}, message_type="control")

    await wait_for(lambda: registry._records[parent.pid].state.value == "crashed")

    assert registry._records[child.pid].state.value == "killed"
    assert registry._records[parent.pid].child_pids == []
    assert bus.mailboxes == {}


@pytest.mark.asyncio
async def test_one_for_all_threshold_does_not_partially_restart(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    crash_path = tmp_path / "crash.py"
    stable_path = tmp_path / "stable.py"
    write_agent(
        parent_path,
        'class Parent(AgentProcess):\n'
        '    name = "Parent"\n'
        '    supervisor_strategy = "one_for_all"\n'
        '    max_restarts = 1\n',
    )
    write_agent(
        crash_path,
        'class Crash(AgentProcess):\n'
        '    name = "Crash"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("boom")\n',
    )
    write_agent(stable_path, 'class Stable(AgentProcess):\n    name = "Stable"\n')
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    failed = await registry.spawn_child(parent.pid, str(crash_path))
    stable = await registry.spawn_child(parent.pid, str(stable_path))
    registry.send_ipc_message(parent.pid, failed.pid, {"cmd": "crash"}, message_type="control")

    await wait_for(lambda: registry._records[parent.pid].escalated)

    assert registry._records[parent.pid].restart_count == 0
    assert stable.pid in registry.list_children(parent.pid)
    assert registry._records[stable.pid].state.value == "running"
    assert [record.name for record in registry._records.values()].count("Crash") == 1


@pytest.mark.asyncio
async def test_temporary_startup_crash_removes_stale_child_pid(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "crash.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n')
    write_agent(
        child_path,
        'class Crash(AgentProcess):\n'
        '    name = "Crash"\n'
        '    async def on_start(self):\n'
        '        raise RuntimeError("startup boom")\n',
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path), restart_policy="temporary")
    await wait_for(lambda: registry._records[child.pid].state.value == "crashed")

    assert child.pid not in registry.list_children(parent.pid)
    rows = await registry.list_processes()
    parent_row = next(row for row in rows if row["pid"] == parent.pid)
    assert parent_row["child_count"] == 0


@pytest.mark.asyncio
async def test_supervision_event_delivery_failure_does_not_block_restart(tmp_path) -> None:
    parent_path = tmp_path / "parent.py"
    child_path = tmp_path / "crash.py"
    write_agent(parent_path, 'class Parent(AgentProcess):\n    name = "Parent"\n    mailbox_size = 1\n')
    write_agent(
        child_path,
        'class Crash(AgentProcess):\n'
        '    name = "Crash"\n'
        '    async def on_message(self, message):\n'
        '        raise RuntimeError("boom")\n',
    )
    registry = ProcessRegistry(kernel=FakeKernel(), bus=FakeBus(), memory=FakeMemory(), allowed_roots=[tmp_path])

    parent = await registry.run_path(str(parent_path))
    child = await registry.spawn_child(parent.pid, str(child_path))
    registry.send_ipc_message(parent.pid, child.pid, {"cmd": "crash"}, message_type="control")

    await wait_for(lambda: child.pid not in registry.list_children(parent.pid))

    assert registry._records[parent.pid].restart_count == 1
    assert registry._records[parent.pid].child_pids
