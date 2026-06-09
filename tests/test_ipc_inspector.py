from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import Static

from kernel.dashboard import AgentOSDashboard, MailboxMetric
from kernel.events import RuntimeEvent
from kernel.ipc_inspector import IPCConnection, build_ipc_snapshot, format_ipc_connection, render_ipc_inspector


BASE_TIME = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


def make_dashboard() -> AgentOSDashboard:
    return AgentOSDashboard(kernel=object(), bus=object(), memory=object(), sandbox=object())


def test_build_ipc_snapshot_aggregates_repeated_connections() -> None:
    snapshot = build_ipc_snapshot(
        [
            {"sender": "planner", "receiver": "researcher", "message_type": "assignment"},
            {"sender": "planner", "receiver": "researcher", "message_type": "assignment"},
            {"sender": "planner", "receiver": "researcher", "message_type": "follow_up"},
        ]
    )

    assert snapshot.connections == (
        IPCConnection(
            sender="planner",
            receiver="researcher",
            message_count=3,
            latest_message_type="follow_up",
        ),
    )


def test_build_ipc_snapshot_orders_connections_deterministically() -> None:
    snapshot = build_ipc_snapshot(
        [
            {"sender": "synthesizer", "receiver": "critic", "message_type": "report"},
            {"sender": "planner", "receiver": "researcher_b", "message_type": "assignment"},
            {"sender": "planner", "receiver": "researcher_a", "message_type": "assignment"},
        ]
    )

    assert [(item.sender, item.receiver) for item in snapshot.connections] == [
        ("planner", "researcher_a"),
        ("planner", "researcher_b"),
        ("synthesizer", "critic"),
    ]


def test_build_ipc_snapshot_uses_latest_timestamp_and_pending_mailbox_depth() -> None:
    snapshot = build_ipc_snapshot(
        [
            {
                "sender": "planner",
                "receiver": "researcher",
                "message_type": "follow_up",
                "timestamp": BASE_TIME + timedelta(seconds=2),
            },
            {
                "sender": "planner",
                "receiver": "researcher",
                "message_type": "assignment",
                "timestamp": BASE_TIME,
            },
        ],
        mailbox_metrics=[MailboxMetric("researcher", 4, 10, "Direct")],
    )

    connection = snapshot.connections[0]
    assert connection.latest_message_type == "follow_up"
    assert connection.latest_timestamp == BASE_TIME + timedelta(seconds=2)
    assert connection.pending_mailbox_size == 4


def test_partial_record_does_not_erase_known_latest_timestamp() -> None:
    snapshot = build_ipc_snapshot(
        [
            {
                "sender": "planner",
                "receiver": "researcher",
                "message_type": "assignment",
                "timestamp": BASE_TIME,
            },
            {"sender": "planner", "receiver": "researcher", "message_type": "unknown_time"},
        ]
    )

    connection = snapshot.connections[0]
    assert connection.latest_message_type == "assignment"
    assert connection.latest_timestamp == BASE_TIME


def test_build_ipc_snapshot_resolves_pid_endpoints_and_handles_partial_records() -> None:
    snapshot = build_ipc_snapshot(
        [
            {"source_pid": 100, "target_pid": 101, "message_type": "task_request"},
            {"sender": "missing_receiver"},
            "legacy log",
        ],
        process_rows=[{"pid": 100, "name": "planner"}, {"pid": 101, "name": "researcher"}],
    )

    assert snapshot.connections == (
        IPCConnection("planner", "researcher", 1, "task_request"),
    )


def test_build_ipc_snapshot_accepts_structured_runtime_events() -> None:
    event = RuntimeEvent.info(
        "IPC",
        "ipc_message_sent",
        "Sent assignment",
        {"sender": "planner", "receiver": "researcher", "topic": "assignment"},
    )

    connection = build_ipc_snapshot([event]).connections[0]

    assert connection.latest_message_type == "assignment"
    assert connection.latest_timestamp == event.timestamp


def test_format_and_render_ipc_connection_are_compact() -> None:
    connection = IPCConnection(
        "planner",
        "researcher",
        20,
        "assignment",
        BASE_TIME,
        3,
    )

    row = format_ipc_connection(connection)

    assert "planner" in row
    assert "-> researcher" in row
    assert "msgs=20" in row
    assert "latest=assignment" in row
    assert "at=12:00:00.000" in row
    assert "pending=3" in row
    assert render_ipc_inspector([connection]) == [row]


@pytest.mark.asyncio
async def test_dashboard_renders_ipc_inspector_with_timeline_and_metrics() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]
    dashboard._demo_ipc_records = [
        {"sender": "planner", "receiver": "researcher", "message_type": "assignment"},
        {"sender": "planner", "receiver": "researcher", "message_type": "assignment"},
    ]
    dashboard._process_rows = [{"name": "planner", "status": "running", "messages_sent": 2}]
    dashboard._runtime_events = [
        RuntimeEvent.info(
            "IPC",
            "ipc_message_sent",
            "Sent assignment",
            {"sender": "planner", "receiver": "researcher", "topic": "assignment"},
        )
    ]

    async with dashboard.run_test(size=(120, 40)) as pilot:
        dashboard._render_ipc_inspector([MailboxMetric("researcher", 1, 10, "Direct")])
        dashboard._render_agent_metrics()
        dashboard._render_timeline()
        await pilot.pause(0)

        title = str(dashboard.query_one("#ipc-inspector-title", Static).render())
        inspector = str(dashboard.query_one("#ipc-inspector", Static).render())
        metrics = str(dashboard.query_one("#agent-metrics", Static).render())
        timeline = str(dashboard.query_one("#runtime-timeline", Static).render())

        assert "IPC Inspector" in title
        assert "msgs=3" in inspector
        assert "pending=1" in inspector
        assert "planner" in metrics
        assert "ipc_message_sent" in timeline


@pytest.mark.asyncio
async def test_dashboard_ipc_inspector_empty_state() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 40)) as pilot:
        dashboard._render_ipc_inspector([])
        await pilot.pause(0)

        inspector = str(dashboard.query_one("#ipc-inspector", Static).render())
        assert "No IPC activity yet." in inspector
