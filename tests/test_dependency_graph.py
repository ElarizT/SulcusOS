from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import Static

from kernel.dashboard import AgentOSDashboard
from kernel.dependency_graph import (
    DependencyEdge,
    DependencyGraphSnapshot,
    DependencyNode,
    build_dependency_graph,
    format_dependency_edge,
    format_dependency_node,
    render_dependency_graph,
)
from kernel.events import RuntimeEvent


BASE_TIME = datetime(2026, 6, 10, 14, 0, 0, tzinfo=timezone.utc)


def lifecycle(name: str, action: str, seconds: int, pid: int | None = None) -> RuntimeEvent:
    metadata = {"agent": name}
    if pid is not None:
        metadata["pid"] = pid
    return RuntimeEvent(
        BASE_TIME + timedelta(seconds=seconds),
        "ERROR" if action == "failed" else "INFO",
        "ExternalAgentRuntime",
        f"external_agent_{action}",
        action,
        metadata,
    )


def make_dashboard() -> AgentOSDashboard:
    return AgentOSDashboard(kernel=object(), bus=object(), memory=object(), sandbox=object())


def test_build_graph_constructs_nodes_from_processes_events_and_ipc_endpoints() -> None:
    snapshot = build_dependency_graph(
        process_rows=[{"name": "planner", "pid": 10, "status": "running"}],
        events=[lifecycle("researcher", "loaded", 0, 11)],
        ipc_records=[{"sender": "researcher", "receiver": "synthesizer", "message_type": "result"}],
    )

    assert [node.name for node in snapshot.nodes] == ["planner", "researcher", "synthesizer"]
    assert snapshot.nodes[0] == DependencyNode("planner", "running", 10)
    assert snapshot.nodes[1].status == "loaded"
    assert snapshot.nodes[1].pid == 11


def test_build_graph_aggregates_duplicate_ipc_edges_and_orders_deterministically() -> None:
    snapshot = build_dependency_graph(
        ipc_records=[
            {"sender": "researcher", "receiver": "synthesizer", "message_type": "result"},
            {"sender": "planner", "receiver": "researcher", "message_type": "assignment"},
            {"sender": "planner", "receiver": "researcher", "message_type": "assignment"},
        ]
    )

    assert snapshot.edges == (
        DependencyEdge("planner", "researcher", "assignment", 2, "assignment"),
        DependencyEdge("researcher", "synthesizer", "result", 1, "result"),
    )


def test_build_graph_adds_declared_supervision_edge_without_inventing_others() -> None:
    snapshot = build_dependency_graph(
        process_rows=[
            {"pid": 1, "name": "supervisor", "status": "running"},
            {"pid": 2, "name": "worker", "status": "running", "supervisor_pid": 1},
        ]
    )

    assert snapshot.edges == (DependencyEdge("supervisor", "worker", "supervision"),)


def test_lifecycle_events_derive_status_and_timestamps() -> None:
    snapshot = build_dependency_graph(
        events=[
            lifecycle("planner", "completed", 2, 10),
            lifecycle("critic", "failed", 3, 12),
            lifecycle("planner", "started", 1, 10),
        ]
    )

    critic, planner = snapshot.nodes
    assert critic.status == "failed"
    assert critic.failed_at == BASE_TIME + timedelta(seconds=3)
    assert planner.status == "complete"
    assert planner.started_at == BASE_TIME + timedelta(seconds=1)
    assert planner.completed_at == BASE_TIME + timedelta(seconds=2)


def test_dependency_graph_rendering_is_compact_and_has_empty_state() -> None:
    node = DependencyNode("planner", "complete", 1234)
    edge = DependencyEdge("planner", "researcher", "assignment", 1, "assignment")

    assert "planner" in format_dependency_node(node)
    assert "complete" in format_dependency_node(node)
    assert "pid=1234" in format_dependency_node(node)
    assert "planner" in format_dependency_edge(edge)
    assert "msgs=1" in format_dependency_edge(edge)
    assert render_dependency_graph(DependencyGraphSnapshot()) == ["No dependency graph available yet."]
    assert render_dependency_graph(DependencyGraphSnapshot((node,), (edge,)))[0] == "Nodes:"


@pytest.mark.asyncio
async def test_dashboard_renders_dependency_graph_with_observability_panels() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]
    dashboard._process_rows = [{"name": "planner", "pid": 10, "status": "running"}]
    dashboard._demo_ipc_records = [
        {"sender": "planner", "receiver": "researcher", "message_type": "assignment"}
    ]
    dashboard._runtime_events = [lifecycle("planner", "started", 1, 10)]

    async with dashboard.run_test(size=(120, 42)) as pilot:
        dashboard._render_dependency_graph()
        dashboard._render_timeline()
        dashboard._render_agent_metrics()
        dashboard._render_ipc_inspector([])
        dashboard._render_replay()
        await pilot.pause(0)

        graph = str(dashboard.query_one("#dependency-graph", Static).render())
        assert "Agent Dependency Graph" in str(
            dashboard.query_one("#dependency-graph-title", Static).render()
        )
        assert "planner" in graph
        assert "researcher" in graph
        assert "assignment" in graph
        assert "external_agent" in str(dashboard.query_one("#runtime-timeline", Static).render())
        assert "planner" in str(dashboard.query_one("#agent-metrics", Static).render())
        assert "msgs=1" in str(dashboard.query_one("#ipc-inspector", Static).render())
        assert "[001] planner" in str(dashboard.query_one("#execution-replay", Static).render())


@pytest.mark.asyncio
async def test_dashboard_dependency_graph_empty_state_and_scroll_preservation() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(100, 30)) as pilot:
        dashboard._render_dependency_graph()
        await pilot.pause(0)
        graph = dashboard.query_one("#dependency-graph", Static)
        assert "No dependency graph available yet." in str(graph.render())
        assert graph.size.height >= 1

        dashboard._demo_ipc_records = [
            {"sender": f"agent_{index}", "receiver": "sink", "message_type": "result"}
            for index in range(30)
        ]
        dashboard._render_dependency_graph()
        await pilot.pause(0)
        graph.scroll_to(y=4, animate=False, force=True)
        await pilot.pause(0)
        before = graph.scroll_y

        dashboard._render_dependency_graph()
        dashboard._demo_ipc_records.append(
            {"sender": "agent_30", "receiver": "sink", "message_type": "result"}
        )
        dashboard._render_dependency_graph()
        await pilot.pause(0)

        assert graph.scroll_y == before
