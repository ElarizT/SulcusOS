from datetime import datetime, timezone

import pytest
from textual.widgets import RichLog

from kernel.dashboard import AgentOSDashboard
from kernel.events import RuntimeEvent, RuntimeEventLog, render_runtime_event


def test_runtime_event_helper_uses_timezone_aware_utc_timestamp() -> None:
    event = RuntimeEvent.info("Supervisor", "child_restarted", "Child restarted")

    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset() == timezone.utc.utcoffset(event.timestamp)


def test_runtime_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        RuntimeEvent(datetime.now(), "INFO", "Supervisor", "test", "Test", {})


def test_runtime_event_log_append_extend_latest_and_filters() -> None:
    info = RuntimeEvent.info("MemorySupervisor", "page_allocated", "Allocated Page 0")
    warning = RuntimeEvent.warning("MemorySupervisor", "page_evicted", "Evicted Page 1")
    error = RuntimeEvent.error("ExternalAgentRuntime", "external_agent_failed", "Agent failed")
    log = RuntimeEventLog()

    log.append(info)
    log.extend([warning, error])

    assert log.latest() == [info, warning, error]
    assert log.latest(2) == [warning, error]
    assert log.latest(0) == []
    assert log.by_level("WARNING") == [warning]
    assert log.by_type("page_allocated") == [info]


def test_runtime_event_renderer_includes_core_fields() -> None:
    event = RuntimeEvent(
        datetime(2026, 6, 9, 15, 58, 31, tzinfo=timezone.utc),
        "INFO",
        "MemorySupervisor",
        "page_allocated",
        "Allocated Page 0",
        {"page": 0},
    )

    rendered = render_runtime_event(event)

    assert "15:58:31" in rendered
    assert "INFO" in rendered
    assert "MemorySupervisor" in rendered
    assert "Allocated Page 0" in rendered


@pytest.mark.asyncio
async def test_dashboard_renders_runtime_events_and_string_logs() -> None:
    dashboard = AgentOSDashboard(kernel=object(), bus=object(), memory=object(), sandbox=object())
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 30)) as pilot:
        dashboard._write_execution_log("legacy string log")
        dashboard._write_execution_log(
            RuntimeEvent.info("MemorySupervisor", "page_allocated", "Allocated Page 0")
        )
        await pilot.pause(0)

        log = dashboard.query_one("#wasm-log", RichLog)
        output = "\n".join(line.text for line in log.lines)

        assert "legacy string log" in output
        assert "INFO" in output
        assert "MemorySupervisor" in output
        assert "Allocated Page 0" in output
