from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import Static

from kernel.dashboard import AgentOSDashboard
from kernel.events import RuntimeEvent, RuntimeEventLog
from kernel.replay import (
    ReplayRecord,
    ReplaySession,
    build_replay_session,
    format_replay_record,
    render_replay,
    render_replay_session,
    replay_events,
)


BASE_TIME = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


def event(seconds: int, action: str, *, agent: str = "planner", metadata: dict | None = None) -> RuntimeEvent:
    values = {"agent": agent}
    values.update(metadata or {})
    return RuntimeEvent(
        BASE_TIME + timedelta(seconds=seconds),
        "INFO",
        "ExternalAgentRuntime",
        f"external_agent_{action}",
        action,
        values,
    )


def make_dashboard() -> AgentOSDashboard:
    return AgentOSDashboard(kernel=object(), bus=object(), memory=object(), sandbox=object())


def test_build_replay_session_from_runtime_event_log_orders_records_deterministically() -> None:
    log = RuntimeEventLog()
    log.extend([event(2, "completed"), event(0, "loaded"), event(1, "started")])

    session = build_replay_session(log)

    assert [record.action for record in session.records] == [
        "external_agent_loaded",
        "external_agent_started",
        "external_agent_completed",
    ]
    assert session.snapshot().records == session.records


def test_replay_iterator_supports_full_offset_and_limit_without_delays() -> None:
    session = build_replay_session([event(index, action) for index, action in enumerate(("loaded", "started", "completed"))])

    assert list(replay_events(session)) == list(session.records)
    assert [record.action for record in replay_events(session, offset=1)] == [
        "external_agent_started",
        "external_agent_completed",
    ]
    assert [record.action for record in replay_events(session, offset=1, limit=1)] == [
        "external_agent_started"
    ]
    assert list(replay_events(session, limit=0)) == []


def test_format_replay_record_is_stable_compact_and_omits_nested_metadata() -> None:
    record = ReplayRecord(
        BASE_TIME,
        "INFO",
        "ExternalAgentRuntime",
        "external_agent_completed",
        {"agent": "Planner", "pid": 1234, "exit_code": 0, "message_count": 3, "nested": {"x": 1}},
    )

    row = format_replay_record(record, 4)

    assert row.startswith("[004] planner")
    assert "completed" in row
    assert "pid=1234 exit=0 msgs=3" in row
    assert "nested" not in row


def test_render_replay_session_preserves_original_indexes_with_offset_and_limit() -> None:
    session = build_replay_session([event(0, "loaded"), event(1, "started"), event(2, "completed")])

    assert render_replay_session(session, offset=1, limit=1)[0].startswith("[002]")
    assert render_replay(session.records) == render_replay_session(session)
    assert render_replay_session(ReplaySession()) == []


@pytest.mark.asyncio
async def test_dashboard_renders_replay_empty_state_and_keeps_other_panels_compatible() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 40)) as pilot:
        dashboard._render_replay()
        dashboard._render_timeline()
        dashboard._render_agent_metrics()
        dashboard._render_ipc_inspector([])
        await pilot.pause(0)

        assert "Execution Replay" in str(dashboard.query_one("#replay-title", Static).render())
        assert "No replay data available." in str(dashboard.query_one("#execution-replay", Static).render())
        assert "No runtime events yet." in str(dashboard.query_one("#runtime-timeline", Static).render())
        assert "No agent metrics available yet." in str(dashboard.query_one("#agent-metrics", Static).render())
        assert "No IPC activity yet." in str(dashboard.query_one("#ipc-inspector", Static).render())


@pytest.mark.asyncio
async def test_dashboard_replay_avoids_unchanged_updates_and_preserves_scroll() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]
    dashboard._runtime_events = [event(index, "started", agent=f"agent_{index}") for index in range(30)]

    async with dashboard.run_test(size=(100, 30)) as pilot:
        dashboard._render_replay()
        await pilot.pause(0)
        replay = dashboard.query_one("#execution-replay", Static)
        replay.scroll_to(y=4, animate=False, force=True)
        await pilot.pause(0)
        before = replay.scroll_y

        dashboard._render_replay()
        dashboard._runtime_events.append(event(31, "completed", agent="agent_31"))
        dashboard._render_replay()
        await pilot.pause(0)

        content = str(replay.render())
        assert replay.scroll_y == before
        assert "[001] agent_0" in content
        assert "[031] agent_31" in content
