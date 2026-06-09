from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import Static

from kernel.dashboard import AgentOSDashboard
from kernel.events import RuntimeEvent
from kernel.metrics import AgentMetrics, build_agent_metrics_snapshot, format_agent_metric, render_agent_metrics


STARTED_AT = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)


def lifecycle_event(event_type: str, *, seconds: float, metadata: dict) -> RuntimeEvent:
    return RuntimeEvent(
        STARTED_AT + timedelta(seconds=seconds),
        "ERROR" if event_type.endswith("failed") else "INFO",
        "ExternalAgentRuntime",
        event_type,
        event_type,
        metadata,
    )


def make_dashboard() -> AgentOSDashboard:
    return AgentOSDashboard(kernel=object(), bus=object(), memory=object(), sandbox=object())


def test_format_agent_metric_is_compact_and_stable() -> None:
    metric = AgentMetrics(
        name="planner",
        status="running",
        pid=1234,
        runtime_seconds=2.31,
        messages_sent=3,
        messages_received=1,
        restart_count=0,
    )

    row = format_agent_metric(metric)

    assert "planner" in row
    assert "running" in row
    assert "pid=1234" in row
    assert "sent=3" in row
    assert "recv=1" in row
    assert "restarts=0" in row
    assert "uptime=2.31s" in row


def test_format_agent_metric_omits_unavailable_optional_fields() -> None:
    row = format_agent_metric(AgentMetrics(name="planner", status="loaded"))

    assert "planner" in row
    assert "loaded" in row
    assert "pid=" not in row
    assert "sent=" not in row
    assert "runtime=" not in row
    assert "error=" not in row


def test_build_metrics_uses_process_message_counters() -> None:
    snapshot = build_agent_metrics_snapshot(
        [
            {
                "name": "planner",
                "status": "running",
                "pid": 1234,
                "messages_sent": 3,
                "messages_received": 1,
                "restart_count": 0,
                "uptime_seconds": 2.31,
            }
        ]
    )

    assert snapshot.metrics == (
        AgentMetrics(
            name="planner",
            status="running",
            pid=1234,
            runtime_seconds=2.31,
            messages_sent=3,
            messages_received=1,
            restart_count=0,
        ),
    )


def test_build_metrics_derives_lifecycle_status_and_runtime_from_events() -> None:
    events = [
        lifecycle_event("external_agent_completed", seconds=1.84, metadata={"agent": "researcher", "pid": 8}),
        lifecycle_event("external_agent_loaded", seconds=0, metadata={"agent": "researcher"}),
        lifecycle_event("external_agent_started", seconds=1, metadata={"agent": "researcher", "pid": 8}),
    ]

    metric = build_agent_metrics_snapshot(events=events).metrics[0]

    assert metric.name == "researcher"
    assert metric.status == "complete"
    assert metric.pid == 8
    assert metric.runtime_seconds == pytest.approx(0.84)


def test_build_metrics_marks_failed_lifecycle_and_uses_available_exit_code() -> None:
    events = [
        lifecycle_event("external_agent_started", seconds=1, metadata={"agent": "critic", "pid": 9}),
        lifecycle_event(
            "external_agent_failed",
            seconds=2,
            metadata={"agent": "critic", "pid": 9, "exit_code": 1},
        ),
    ]

    metric = build_agent_metrics_snapshot(events=events).metrics[0]

    assert metric.status == "failed"
    assert metric.exit_code == 1
    assert metric.error is True
    assert "error=true" in format_agent_metric(metric)


def test_render_agent_metrics_accepts_snapshot_and_iterable() -> None:
    metric = AgentMetrics(name="planner", status="running")
    snapshot = build_agent_metrics_snapshot([{"name": "planner", "status": "running"}])

    assert render_agent_metrics(snapshot) == render_agent_metrics([metric])


@pytest.mark.asyncio
async def test_dashboard_renders_agent_metrics_and_runtime_timeline() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]
    dashboard._process_rows = [
        {
            "name": "planner",
            "status": "running",
            "pid": 1234,
            "messages_sent": 3,
            "messages_received": 1,
            "uptime_seconds": 2.31,
        }
    ]
    dashboard._runtime_events = [
        lifecycle_event("external_agent_started", seconds=1, metadata={"agent": "planner", "pid": 1234})
    ]

    async with dashboard.run_test(size=(120, 40)) as pilot:
        dashboard._render_agent_metrics()
        dashboard._render_timeline()
        await pilot.pause(0)

        metrics_title = str(dashboard.query_one("#metrics-title", Static).render())
        metrics = str(dashboard.query_one("#agent-metrics", Static).render())
        timeline = str(dashboard.query_one("#runtime-timeline", Static).render())

        assert "Agent Metrics" in metrics_title
        assert "planner" in metrics
        assert "sent=3" in metrics
        assert "recv=1" in metrics
        assert "Runtime Timeline" not in metrics
        assert "external_agent" in timeline


@pytest.mark.asyncio
async def test_dashboard_agent_metrics_empty_state() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 40)) as pilot:
        dashboard._render_agent_metrics()
        await pilot.pause(0)

        metrics = str(dashboard.query_one("#agent-metrics", Static).render())
        assert "No agent metrics available yet." in metrics
