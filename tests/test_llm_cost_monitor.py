from __future__ import annotations

from datetime import datetime, timezone

import pytest
from textual.widgets import Static

from kernel.dashboard import AgentOSDashboard
from kernel.events import RuntimeEvent
from kernel.llm_cost_monitor import (
    build_llm_cost_snapshot,
    render_llm_cost_snapshot,
)


def cost_event(
    provider: str,
    model: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    total_tokens: int = 15,
    total_cost: str = "0.0012",
) -> RuntimeEvent:
    return RuntimeEvent(
        datetime(2026, 6, 15, tzinfo=timezone.utc),
        "INFO",
        "LLMRuntime",
        "llm.cost_recorded",
        "safe event",
        {
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "currency": "USD",
        },
    )


def make_dashboard() -> AgentOSDashboard:
    return AgentOSDashboard(kernel=object(), bus=object(), memory=object(), sandbox=object())


def test_cost_monitor_aggregates_safe_recorded_events() -> None:
    snapshot = build_llm_cost_snapshot(
        [cost_event("openai", "model"), cost_event("openai", "model")]
    )

    assert render_llm_cost_snapshot(snapshot) == [
        "total: $0.0024 USD",
        "openai/model  calls=2  tokens=30  cost=$0.0024 USD",
    ]


def test_cost_monitor_ignores_unsafe_and_malformed_metadata() -> None:
    event = cost_event("openai", "model")
    event.metadata.update(
        {"prompt": "private prompt", "api_key": "private key", "content": "private text"}
    )
    malformed = cost_event("openai", "bad", total_cost="not-a-number")

    rendered = repr(render_llm_cost_snapshot(build_llm_cost_snapshot([event, malformed])))

    assert "openai/model" in rendered
    assert "private prompt" not in rendered
    assert "private key" not in rendered
    assert "private text" not in rendered
    assert "openai/bad" not in rendered


@pytest.mark.asyncio
async def test_dashboard_cost_monitor_coexists_preserves_scroll_and_avoids_rebuilds() -> None:
    dashboard = make_dashboard()
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]
    dashboard._runtime_events = [
        cost_event(f"provider-{index}", "model") for index in range(30)
    ]

    async with dashboard.run_test(size=(120, 42)) as pilot:
        dashboard._render_llm_cost_monitor()
        dashboard._render_llm_stream_monitor()
        await pilot.pause(0)
        monitor = dashboard.query_one("#llm-cost-monitor", Static)
        monitor.scroll_to(y=4, animate=False, force=True)
        await pilot.pause(0)
        before = monitor.scroll_y
        content_before = dashboard._scrollable_content["#llm-cost-monitor"]

        dashboard._render_llm_cost_monitor()
        dashboard._runtime_events.append(cost_event("provider-30", "model"))
        dashboard._render_llm_cost_monitor()
        await pilot.pause(0)

        assert "LLM Cost Monitor" in str(
            dashboard.query_one("#llm-cost-title", Static).render()
        )
        assert dashboard._scrollable_content["#llm-cost-monitor"] != content_before
        assert monitor.scroll_y == before
        assert dashboard.query_one("#llm-stream-monitor", Static)
