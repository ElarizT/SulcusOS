from datetime import datetime, timedelta, timezone

import pytest
from textual.widgets import Static

from kernel.dashboard import AgentOSDashboard
from kernel.events import RuntimeEvent
from kernel.timeline import format_timeline_event, render_runtime_timeline


def event(
    second: int,
    source: str,
    event_type: str,
    message: str,
    metadata: dict | None = None,
) -> RuntimeEvent:
    return RuntimeEvent(
        datetime(2026, 6, 9, 12, 1, second, 123000, tzinfo=timezone.utc),
        "INFO",
        source,
        event_type,
        message,
        metadata or {},
    )


def make_dashboard() -> AgentOSDashboard:
    return AgentOSDashboard(kernel=object(), bus=object(), memory=object(), sandbox=object())


def test_format_timeline_event_is_compact_and_includes_milliseconds() -> None:
    row = format_timeline_event(
        event(
            15,
            "ExternalAgentRuntime",
            "external_agent_loaded",
            "Loaded",
            {"agent": "Planner"},
        )
    )

    assert row.startswith("12:01:15.123")
    assert "external_agent" in row
    assert "planner" in row
    assert row.endswith("loaded")


def test_render_runtime_timeline_orders_structured_events_chronologically() -> None:
    completed = event(17, "ExternalAgentRuntime", "external_agent_completed", "Completed")
    loaded = event(15, "ExternalAgentRuntime", "external_agent_loaded", "Loaded")
    started = event(16, "ExternalAgentRuntime", "external_agent_started", "Started")

    rows = render_runtime_timeline([completed, loaded, started])

    assert [row[:12] for row in rows] == ["12:01:15.123", "12:01:16.123", "12:01:17.123"]


def test_timeline_metadata_summary_includes_scalars_and_omits_nested_values() -> None:
    row = format_timeline_event(
        event(
            15,
            "ExternalAgentRuntime",
            "external_agent_completed",
            "Completed",
            {
                "agent": "Planner",
                "pid": 1234,
                "exit_code": 0,
                "duration_ms": 842,
                "details": {"large": ["nested"]},
            },
        )
    )

    assert "pid=1234" in row
    assert "exit_code=0" in row
    assert "duration_ms=842" in row
    assert "details" not in row
    assert "nested" not in row


def test_timeline_formats_agent_tool_loop_tool_events_clearly() -> None:
    row = format_timeline_event(
        event(
            15,
            "AgentToolLoop",
            "tool_execution_completed",
            "Tool execution completed",
            {
                "tool_name": "add_numbers",
                "round_index": 1,
                "success": True,
                "tool_call_id": "call_1",
            },
        )
    )

    assert "agent_tool_loop" in row
    assert "add_numbers" in row
    assert "tool_execution_completed" in row
    assert "round_index=1" in row
    assert "success=True" in row
    assert "tool_call_id" not in row


def test_timeline_formats_tool_execution_group_mode_metadata() -> None:
    row = format_timeline_event(
        event(
            15,
            "AgentToolLoop",
            "tool_execution_group_completed",
            "Tool execution group completed",
            {
                "execution_mode": "sequential",
                "requested_execution_mode": "parallel",
                "effective_execution_mode": "sequential",
                "fallback_reason": "not_all_tools_parallel_safe",
                "round_index": 0,
                "tool_call_count": 2,
                "parallel_safe_tool_count": 1,
                "unsafe_tool_count": 1,
                "successful_tool_count": 2,
                "failed_tool_count": 0,
                "tool_names": ("add_numbers", "multiply_numbers"),
            },
        )
    )

    assert "tool_execution_group_completed" in row
    assert "execution_mode=sequential" in row
    assert "requested_execution_mode=parallel" in row
    assert "effective_execution_mode=sequential" in row
    assert "fallback_reason=not_all_tools_parallel_safe" in row
    assert "tool_call_count=2" in row
    assert "parallel_safe_tool_count=1" in row
    assert "unsafe_tool_count=1" in row
    assert "successful_tool_count=2" in row
    assert "failed_tool_count=0" in row
    assert "tool_names" not in row


def test_timeline_formats_tool_permission_metadata() -> None:
    row = format_timeline_event(
        event(
            15,
            "AgentToolLoop",
            "tool_call_denied",
            "Tool call denied",
            {
                "tool_name": "multiply_numbers",
                "round_index": 1,
                "reason": "permission_policy",
                "policy_default_allow": False,
                "matched_rule": "not_in_allowed_tools",
                "tool_call_id": "call_secret",
            },
        )
    )

    assert "tool_call_denied" in row
    assert "multiply_numbers" in row
    assert "round_index=1" in row
    assert "reason=permission_policy" in row
    assert "policy_default_allow=False" in row
    assert "matched_rule=not_in_allowed_tools" in row
    assert "tool_call_id" not in row


def test_timeline_safely_renders_legacy_strings_and_dictionaries() -> None:
    rows = render_runtime_timeline(
        [
            "legacy log string",
            {"event": "child_restarted", "message": "Child restarted", "nested": {"ignored": True}},
        ]
    )

    assert rows == ["legacy log string", "child_restarted Child restarted"]


def test_timeline_limit_returns_latest_rows_after_ordering() -> None:
    base = event(10, "Supervisor", "child_started", "Started")
    events = [
        RuntimeEvent(
            base.timestamp + timedelta(seconds=index),
            base.level,
            base.source,
            base.event_type,
            base.message,
            base.metadata,
        )
        for index in range(4)
    ]

    rows = render_runtime_timeline(events, limit=2)

    assert [row[:12] for row in rows] == ["12:01:12.123", "12:01:13.123"]


@pytest.mark.asyncio
async def test_dashboard_renders_visible_runtime_timeline() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]
    dashboard._runtime_events = [
        event(
            15,
            "ExternalAgentRuntime",
            "external_agent_loaded",
            "Loaded",
            {"agent": "Planner", "pid": 1234},
        )
    ]

    async with dashboard.run_test(size=(120, 40)) as pilot:
        dashboard._render_timeline()
        await pilot.pause(0)

        title = str(dashboard.query_one("#timeline-title", Static).render())
        timeline = str(dashboard.query_one("#runtime-timeline", Static).render())

        assert "Runtime Timeline" in title
        assert "12:01:15.123" in timeline
        assert "external_agent" in timeline
        assert "planner" in timeline
        assert "pid=1234" in timeline
