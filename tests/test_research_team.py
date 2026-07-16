import pytest
from rich.text import Text
from textual.widgets import DataTable, Static

from examples.research_team.agents import PlannerAgent, SynthesizerAgent
from examples.research_team.contracts import ResearchResult
from examples.research_team.data import BENEFITS, CRITIC_REVIEW, MARKET_TRENDS, RISKS, TOPIC
from examples.research_team.research_team import run_demo
from kernel.dashboard import SHELL_PROMPT, AgentOSDashboard
from kernel.events import RuntimeEvent


class EmptyTelemetry:
    pass


def test_dashboard_uses_branded_shell_prompt() -> None:
    assert SHELL_PROMPT == "Sulcus>"


def test_empty_dashboard_tree_shows_placeholder() -> None:
    tree = AgentOSDashboard._format_agent_tree(None)

    assert "No active hierarchy" in tree


def test_agent_tree_uses_consistent_state_markers() -> None:
    tree = AgentOSDashboard._format_agent_tree(
        {
            "supervisor": "Supervisor",
            "children": ["Running", "Restarted", "Failed", "Completed", "Terminated"],
        },
        states={
            "Supervisor": "running",
            "Running": "running",
            "Restarted": "restarted",
            "Failed": "failed",
            "Completed": "completed",
            "Terminated": "terminated",
        },
    )
    plain = Text.from_markup(tree).plain

    assert "[SUP] [>] Supervisor" in plain
    assert "[>] Running" in plain
    assert "[R] Restarted" in plain
    assert "[!] Failed" in plain
    assert "[+] Completed" in plain
    assert "[X] Terminated" in plain


def test_planner_creates_expected_assignments() -> None:
    assignments = PlannerAgent().create_assignments()

    assert [(item.topic, item.focus_area, item.destination) for item in assignments] == [
        (TOPIC, "Benefits", "ResearchBenefitsAgent"),
        (TOPIC, "Risks", "ResearchRisksAgent"),
        (TOPIC, "Market Trends", "ResearchMarketAgent"),
    ]


@pytest.mark.asyncio
async def test_planner_assignments_are_delivered() -> None:
    state = await run_demo()

    assert [agent.assignment_received.focus_area for agent in state["research_agents"]] == [
        "Benefits",
        "Risks",
        "Market Trends",
    ]


@pytest.mark.asyncio
async def test_research_results_are_delivered_to_synthesizer() -> None:
    state = await run_demo()

    assert {
        focus_area: result.findings
        for focus_area, result in state["synthesizer"].results.items()
    } == {
        "Benefits": BENEFITS,
        "Risks": RISKS,
        "Market Trends": MARKET_TRENDS,
    }


def test_synthesizer_waits_for_all_required_results() -> None:
    synthesizer = SynthesizerAgent()
    synthesizer.results["Benefits"] = ResearchResult("Benefits", BENEFITS)
    synthesizer.results["Risks"] = ResearchResult("Risks", RISKS)

    assert synthesizer.create_report() is None


@pytest.mark.asyncio
async def test_synthesized_report_is_delivered_to_critic() -> None:
    state = await run_demo()

    assert state["synthesizer"].report_sent == state["critic"].report_received
    assert state["critic"].report_received.topic == TOPIC
    assert state["critic"].report_received.benefits == BENEFITS
    assert state["critic"].report_received.risks == RISKS
    assert state["critic"].report_received.market == MARKET_TRENDS
    assert state["critic"].report_received.summary == (
        "AI in healthcare is progressing rapidly, with strong potential benefits, "
        "meaningful risks, and growing market adoption."
    )


@pytest.mark.asyncio
async def test_critic_generates_expected_review() -> None:
    state = await run_demo()

    assert state["critic_review"] == CRITIC_REVIEW
    assert state["critic_review"].score == 8.7
    assert state["critic_review"].strengths == CRITIC_REVIEW.strengths
    assert state["critic_review"].weaknesses == CRITIC_REVIEW.weaknesses
    assert state["critic_review"].final_note == CRITIC_REVIEW.final_note


@pytest.mark.asyncio
async def test_full_workflow_returns_major_artifacts() -> None:
    state = await run_demo()

    assert [assignment.focus_area for assignment in state["assignments"]] == [
        "Benefits",
        "Risks",
        "Market Trends",
    ]
    assert set(state["research_results"]) == {"Benefits", "Risks", "Market Trends"}
    assert state["synthesized_report"] == state["synthesizer"].report_sent
    assert state["critic_review"] == state["critic"].review


@pytest.mark.asyncio
async def test_research_team_records_real_workflow_events_in_execution_order() -> None:
    state = await run_demo()
    events = state["events"]

    assert all(isinstance(event, RuntimeEvent) for event in events)
    assert [(event.event_type, event.metadata["agent"]) for event in events] == [
        ("workflow_started", "ResearchTeamSupervisor"),
        ("agent_work_started", "PlannerAgent"),
        ("agent_work_completed", "PlannerAgent"),
        ("agent_work_started", "ResearchBenefitsAgent"),
        ("agent_work_completed", "ResearchBenefitsAgent"),
        ("agent_work_started", "ResearchRisksAgent"),
        ("agent_work_completed", "ResearchRisksAgent"),
        ("agent_work_started", "ResearchMarketAgent"),
        ("agent_work_completed", "ResearchMarketAgent"),
        ("agent_work_started", "SynthesizerAgent"),
        ("agent_work_completed", "SynthesizerAgent"),
        ("agent_work_started", "CriticAgent"),
        ("agent_work_completed", "CriticAgent"),
        ("workflow_completed", "ResearchTeamSupervisor"),
    ]
    assert [event.timestamp for event in events] == sorted(event.timestamp for event in events)
    assert all(event.timestamp.utcoffset() is not None for event in events)
    assert all(set(event.metadata) == {"agent"} for event in events)


@pytest.mark.asyncio
async def test_dashboard_snapshot_shows_completed_research_team_workflow() -> None:
    state = await run_demo()
    dashboard = AgentOSDashboard(
        kernel=EmptyTelemetry(),
        bus=EmptyTelemetry(),
        memory=EmptyTelemetry(),
        sandbox=EmptyTelemetry(),
    )

    dashboard.load_research_team_snapshot(state)

    assert dashboard._demo_status == "Workflow Complete  Final Score: 8.7/10"
    assert dashboard._demo_supervision_events == state["events"]
    assert dashboard._observable_events() == state["events"]
    assert [row["name"] for row in dashboard._demo_process_rows] == [
        "PlannerAgent",
        "ResearchBenefitsAgent",
        "ResearchRisksAgent",
        "ResearchMarketAgent",
        "SynthesizerAgent",
        "CriticAgent",
    ]
    assert [(metric.agent_name, metric.queue_depth) for metric in dashboard._demo_mailboxes] == [
        ("PlannerAgent", 3),
        ("ResearchAgents", 3),
        ("SynthesizerAgent", 1),
        ("CriticAgent", 1),
    ]


@pytest.mark.asyncio
async def test_research_team_dashboard_renders_timeline_without_duplicate_events() -> None:
    state = await run_demo()
    dashboard = AgentOSDashboard(
        kernel=EmptyTelemetry(),
        bus=EmptyTelemetry(),
        memory=EmptyTelemetry(),
        sandbox=EmptyTelemetry(),
    )
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 36)) as pilot:
        dashboard.load_research_team_snapshot(state)
        dashboard._render_timeline()
        dashboard._render_timeline()
        dashboard._render_agent_tree()
        dashboard._render_processes(dashboard._demo_process_rows)
        dashboard._render_mailboxes(dashboard._demo_mailboxes)
        dashboard._render_status(dashboard._demo_mailboxes)
        await pilot.pause(0)

        timeline = str(dashboard.query_one("#runtime-timeline", Static).render())
        tree = str(dashboard.query_one("#agent-tree", Static).render())
        processes = dashboard.query_one("#process-table", DataTable)
        ipc = dashboard.query_one("#ipc-table", DataTable)
        status = str(dashboard.query_one("#status-bar", Static).render())

        assert timeline.count("workflow_started") == 1
        assert timeline.count("workflow_completed") == 1
        assert timeline.count("agent_work_started") == 6
        assert timeline.count("agent_work_completed") == 6
        assert "research_team_supervisor" in timeline
        assert "planner_agent" in timeline
        assert "ResearchTeamSupervisor" in tree
        assert processes.row_count == 6
        assert ipc.row_count == 4
        assert "Workflow Complete" in status


@pytest.mark.asyncio
async def test_research_team_demo_populates_dashboard_hierarchy() -> None:
    state = await run_demo()
    dashboard = AgentOSDashboard(
        kernel=EmptyTelemetry(),
        bus=EmptyTelemetry(),
        memory=EmptyTelemetry(),
        sandbox=EmptyTelemetry(),
    )

    dashboard.load_research_team_snapshot(state)

    assert dashboard._demo_hierarchy == {
        "supervisor": "ResearchTeamSupervisor",
        "children": [
            "PlannerAgent",
            "ResearchBenefitsAgent",
            "ResearchRisksAgent",
            "ResearchMarketAgent",
            "SynthesizerAgent",
            "CriticAgent",
        ],
    }


@pytest.mark.asyncio
async def test_dashboard_tree_rendering_includes_expected_agent_names() -> None:
    state = await run_demo()
    dashboard = AgentOSDashboard(
        kernel=EmptyTelemetry(),
        bus=EmptyTelemetry(),
        memory=EmptyTelemetry(),
        sandbox=EmptyTelemetry(),
    )

    dashboard.load_research_team_snapshot(state)
    tree = AgentOSDashboard._format_agent_tree(dashboard._demo_hierarchy)

    for name in [
        "ResearchTeamSupervisor",
        "PlannerAgent",
        "ResearchBenefitsAgent",
        "ResearchRisksAgent",
        "ResearchMarketAgent",
        "SynthesizerAgent",
        "CriticAgent",
    ]:
        assert name in tree


@pytest.mark.asyncio
async def test_dashboard_tree_panel_has_room_for_rendered_hierarchy() -> None:
    state = await run_demo()
    dashboard = AgentOSDashboard(
        kernel=EmptyTelemetry(),
        bus=EmptyTelemetry(),
        memory=EmptyTelemetry(),
        sandbox=EmptyTelemetry(),
    )
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 30)) as pilot:
        dashboard.load_research_team_snapshot(state)
        dashboard._render_agent_tree()
        await pilot.pause(0)

        tree = AgentOSDashboard._format_agent_tree(dashboard._demo_hierarchy)
        tree_widget = dashboard.query_one("#agent-tree")
        rendered_tree = str(tree_widget.render())

        assert tree_widget.size.height >= len(tree.splitlines())
        for name in [
            "ResearchTeamSupervisor",
            "PlannerAgent",
            "ResearchBenefitsAgent",
            "ResearchRisksAgent",
            "ResearchMarketAgent",
            "SynthesizerAgent",
            "CriticAgent",
        ]:
            assert name in rendered_tree
