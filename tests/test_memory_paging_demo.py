from types import SimpleNamespace

import pytest

from demos.memory_paging import build_demo_snapshot
from kernel.dashboard import AgentOSDashboard
from kernel.events import RuntimeEvent
from kernel.shell_help import MEMORY_PAGING_DEMO_PATH, format_demo_browser, is_memory_paging_demo_path
from textual.widgets import RichLog, Static


class EmptyTelemetry:
    pass


def make_dashboard() -> AgentOSDashboard:
    return AgentOSDashboard(
        kernel=EmptyTelemetry(),
        bus=EmptyTelemetry(),
        memory=EmptyTelemetry(),
        sandbox=EmptyTelemetry(),
    )


def test_memory_paging_demo_path_is_runnable_convention() -> None:
    assert MEMORY_PAGING_DEMO_PATH == "demos/memory_paging"
    assert is_memory_paging_demo_path("demos/memory_paging")
    assert is_memory_paging_demo_path("demos\\memory_paging\\")


def test_demo_browser_lists_memory_paging() -> None:
    output = format_demo_browser()

    assert "memory_paging" in output
    assert "page allocation, page eviction, and context visualization" in output
    assert f"run {MEMORY_PAGING_DEMO_PATH}" in output


def test_memory_paging_snapshot_populates_deterministic_dashboard_state() -> None:
    dashboard = make_dashboard()
    dashboard.load_memory_paging_snapshot(build_demo_snapshot())

    assert dashboard._demo_status == "Memory Demo Complete"
    assert [row["name"] for row in dashboard._demo_process_rows] == [
        "MemorySupervisor",
        "AgentA",
        "AgentB",
    ]
    assert dashboard._demo_page_tables == [
        {
            "agent": "Agent A",
            "pages": [
                {"page": 0, "state": "active"},
                {"page": 1, "state": "evicted"},
            ],
        },
        {
            "agent": "Agent B",
            "pages": [
                {"page": 2, "state": "active"},
                {"page": 3, "state": "active"},
            ],
        },
    ]


def test_memory_paging_event_log_contains_allocation_and_eviction_sequence() -> None:
    dashboard = make_dashboard()
    dashboard.load_memory_paging_snapshot(build_demo_snapshot())

    assert all(isinstance(event, RuntimeEvent) for event in dashboard._demo_supervision_events)
    assert [event.message for event in dashboard._demo_supervision_events] == [
        "Allocated Page 0",
        "Allocated Page 1",
        "Allocated Page 2",
        "Evicted Page 1",
        "Allocated Page 3",
    ]
    assert [event.event_type for event in dashboard._demo_supervision_events] == [
        "page_allocated",
        "page_allocated",
        "page_allocated",
        "page_evicted",
        "page_allocated",
    ]


@pytest.mark.asyncio
async def test_memory_paging_renders_page_table_event_log_and_status() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 30)) as pilot:
        dashboard.load_memory_paging_snapshot(build_demo_snapshot())
        dashboard._render_memory([])
        dashboard._render_supervision_events()
        dashboard._render_status([])
        await pilot.pause(0)

        memory = str(dashboard.query_one("#memory-bars", Static).render())
        status = str(dashboard.query_one("#status-bar", Static).render())
        log = dashboard.query_one("#wasm-log", RichLog)
        events = "\n".join(line.text for line in log.lines)

        assert "Agent A" in memory
        assert "Page 0" in memory
        assert "Page 1  Evicted" in memory
        assert "Agent B" in memory
        assert "Page 3" in memory
        assert "Allocated Page 0" in events
        assert "Evicted Page 1" in events
        assert "Memory Demo Complete" in status


def test_existing_demo_loaders_clear_memory_paging_visualization() -> None:
    dashboard = make_dashboard()
    dashboard.load_memory_paging_snapshot(build_demo_snapshot())

    dashboard.load_supervisor_recovery_snapshot(
        {
            "status": "Recovery Complete",
            "process_rows": [],
            "hierarchy": {"supervisor": "RecoverySupervisor", "children": []},
            "events": [],
        }
    )

    assert dashboard._demo_page_tables is None

    dashboard.load_memory_paging_snapshot(build_demo_snapshot())
    dashboard.load_research_team_snapshot(
        {
            "critic_review": SimpleNamespace(score=8.7),
            "hierarchy": {"supervisor": "ResearchTeamSupervisor", "children": []},
        }
    )

    assert dashboard._demo_page_tables is None
