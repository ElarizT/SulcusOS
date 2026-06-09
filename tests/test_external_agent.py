from pathlib import Path

import pytest

from kernel.dashboard import AgentOSDashboard
from kernel.process import ProcessRegistry
from main import format_external_agent_run, is_external_agent_project_path
from test_process_registry import FakeBus, FakeKernel, FakeMemory
from textual.widgets import Static


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PROJECT = PROJECT_ROOT / "examples" / "external_basic_agent"


def make_registry(root: Path) -> ProcessRegistry:
    return ProcessRegistry(
        kernel=FakeKernel(),
        bus=FakeBus(),
        memory=FakeMemory(),
        allowed_roots=[root],
    )


def write_external_project(tmp_path, source: str, *, runtime: str = "python") -> Path:
    project = tmp_path / "external_agent"
    project.mkdir()
    (project / "agentos.toml").write_text(
        (
            'name = "external_test"\n'
            'type = "basic"\n'
            'entrypoint = "agent.py"\n'
            f'runtime = "{runtime}"\n'
        ),
        encoding="utf-8",
    )
    (project / "agent.py").write_text(source, encoding="utf-8")
    return project


def test_sample_run_path_is_recognized_as_external_project() -> None:
    assert is_external_agent_project_path(str(SAMPLE_PROJECT))


@pytest.mark.asyncio
async def test_valid_external_basic_agent_runs_and_records_completion() -> None:
    registry = make_registry(PROJECT_ROOT)

    result = await registry.run_external_project(str(SAMPLE_PROJECT))
    rows = await registry.list_processes()

    assert result.succeeded
    assert "[ExternalBasicAgent] Started" in result.output
    assert rows[0]["name"] == "external_basic_agent"
    assert rows[0]["status"] == "exited"
    assert rows[0]["external"] is True
    assert [event.event_type for event in result.events] == [
        "external_agent_loaded",
        "external_agent_started",
        "external_agent_completed",
    ]

    shell_output = format_external_agent_run(result)
    assert "External agent loaded:" in shell_output
    assert "external_basic_agent" in shell_output
    assert "External agent completed:" in shell_output


@pytest.mark.asyncio
async def test_invalid_external_manifest_does_not_execute(tmp_path) -> None:
    marker = tmp_path / "executed.txt"
    project = write_external_project(
        tmp_path,
        (
            "from agentos import AgentProcess\n\n"
            f"open({str(marker)!r}, 'w').write('executed')\n\n"
            "class InvalidAgent(AgentProcess):\n"
            '    name = "InvalidAgent"\n'
        ),
        runtime="node",
    )

    with pytest.raises(ValueError, match="unsupported external agent runtime"):
        await make_registry(tmp_path).run_external_project(str(project))

    assert not marker.exists()


@pytest.mark.asyncio
async def test_external_agent_runtime_exception_is_captured(tmp_path) -> None:
    project = write_external_project(
        tmp_path,
        (
            "from agentos import AgentProcess\n\n"
            "class FailingExternalAgent(AgentProcess):\n"
            '    name = "FailingExternalAgent"\n\n'
            "    async def on_start(self):\n"
            '        print("[FailingExternalAgent] Started")\n'
            '        raise RuntimeError("startup boom")\n'
        ),
    )
    registry = make_registry(tmp_path)

    result = await registry.run_external_project(str(project))
    rows = await registry.list_processes()

    assert not result.succeeded
    assert result.error == "startup boom"
    assert "[FailingExternalAgent] Started" in result.output
    assert rows[0]["status"] == "crashed"
    assert [event.event_type for event in result.events] == [
        "external_agent_loaded",
        "external_agent_started",
        "external_agent_failed",
    ]
    assert "startup boom" in format_external_agent_run(result)


def test_dashboard_external_completion_status_clears_demo_snapshots() -> None:
    dashboard = AgentOSDashboard(
        kernel=object(),
        bus=object(),
        memory=object(),
        sandbox=object(),
    )
    dashboard._demo_process_rows = [{"name": "OldDemo"}]

    dashboard.load_external_agent_result(succeeded=True)

    assert dashboard._demo_status == "External Agent Complete"
    assert dashboard._demo_process_rows is None
    assert dashboard._display_process_status("exited", external=True) == "COMPLETED"
    assert dashboard._display_process_status("killed") == "TERMINATED"


@pytest.mark.asyncio
async def test_dashboard_status_bar_shows_external_agent_completion() -> None:
    dashboard = AgentOSDashboard(
        kernel=object(),
        bus=object(),
        memory=object(),
        sandbox=object(),
    )
    dashboard.refresh_metrics = lambda: None  # type: ignore[method-assign]

    async with dashboard.run_test(size=(120, 30)) as pilot:
        dashboard.load_external_agent_result(succeeded=True)
        dashboard._render_status([])
        await pilot.pause(0)

        status = str(dashboard.query_one("#status-bar", Static).render())
        assert "External Agent Complete" in status
