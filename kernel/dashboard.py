from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

from kernel.dependency_graph import build_dependency_graph, render_dependency_graph
from kernel.events import RuntimeEvent, render_runtime_event
from kernel.ipc_inspector import build_ipc_snapshot, render_ipc_inspector
from kernel.llm_cost_monitor import build_llm_cost_snapshot, render_llm_cost_snapshot
from kernel.llm_stream_monitor import build_llm_stream_snapshot, render_llm_stream_snapshot
from kernel.metrics import build_agent_metrics_snapshot, render_agent_metrics
from kernel.replay import build_replay_session, render_replay_session
from kernel.timeline import render_runtime_timeline

SHELL_PROMPT = "Sulcus>"


@dataclass
class MailboxMetric:
    agent_name: str
    queue_depth: int
    buffer_size: int
    routing_method: str


@dataclass
class WasmRunMetric:
    timestamp: int
    success: bool
    fuel_consumed: int
    error_message: str | None


class AgentOSDashboard(App[None]):
    """Real-time terminal dashboard for Sulcus OS kernel telemetry."""

    TITLE = "Sulcus OS"
    SUB_TITLE = "Runtime Dashboard"

    CSS = """
    Screen {
        background: #080b0f;
        color: #d7dde8;
    }

    #status-bar {
        height: 3;
        padding: 0 2;
        content-align: left middle;
        background: #101722;
        border-bottom: solid #263245;
    }

    #main-grid {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 2fr 3fr;
        grid-rows: 4fr 5fr;
        height: 1fr;
    }

    .pane {
        border: solid #263245;
        padding: 0 1;
        min-height: 7;
    }

    .primary-title {
        background: #101722;
        color: #c9eeff;
        padding: 0 1;
    }

    .secondary-title {
        color: #7086a6;
    }

    .section-title {
        background: #0d121b;
        color: #93a8c6;
        padding: 0 1;
    }

    .optional-monitor {
        display: none;
    }

    #process-pane {
        column-span: 1;
        row-span: 1;
    }

    #agent-tree-pane {
        column-span: 1;
        row-span: 1;
    }

    .pane-title {
        height: 1;
        color: #8bd5ff;
        text-style: bold;
    }

    #llm-stream-title {
        height: 1;
        color: #8bd5ff;
        text-style: bold;
    }

    #llm-cost-title {
        height: 1;
        color: #8bd5ff;
        text-style: bold;
    }

    #ipc-table {
        height: 3;
        min-height: 3;
    }

    #memory-bars {
        height: 3;
        overflow: auto;
    }

    #wasm-log {
        height: 1fr;
        min-height: 3;
        background: #080b0f;
        overflow: auto;
    }

    #console-log {
        height: 2;
        min-height: 2;
        background: #0b1018;
        overflow: auto;
    }

    #llm-stream-monitor {
        height: 3;
        overflow: auto;
    }

    #llm-cost-monitor {
        height: 1;
        overflow: auto;
    }

    #process-table {
        height: 1fr;
    }

    #agent-tree {
        height: 7;
        min-height: 7;
        overflow: auto;
    }

    #dependency-graph {
        height: 1fr;
        overflow: auto;
    }

    #runtime-timeline {
        height: 2fr;
        min-height: 4;
        overflow: auto;
    }

    #agent-metrics {
        height: 3;
        overflow: auto;
    }

    #ipc-inspector {
        height: 3;
        overflow: auto;
    }

    #execution-replay {
        height: 1fr;
        min-height: 3;
        overflow: auto;
    }

    #shell-input {
        dock: bottom;
        height: 3;
        padding: 0 1;
        border-top: solid #263245;
        background: #101722;
    }
    """

    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        *,
        kernel: Any,
        bus: Any,
        memory: Any,
        sandbox: Any,
        command_handler: Callable[[str], Awaitable[str] | str] | None = None,
        process_snapshot: Callable[[], Awaitable[list[dict[str, Any]]] | list[dict[str, Any]]] | None = None,
        supervision_event_snapshot: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.bus = bus
        self.memory = memory
        self.sandbox = sandbox
        self.command_handler = command_handler
        self.process_snapshot = process_snapshot
        self.supervision_event_snapshot = supervision_event_snapshot
        self._last_pending_evictions: dict[str, int] = {}
        self._logged_wasm_runs = 0
        self._logged_supervision_events = 0
        self._logged_runtime_events = 0
        self._process_rows: list[dict[str, Any]] = []
        self._demo_mailboxes: list[MailboxMetric] | None = None
        self._demo_process_rows: list[dict[str, Any]] | None = None
        self._demo_hierarchy: dict[str, Any] | None = None
        self._demo_supervision_events: list[dict[str, Any] | RuntimeEvent] | None = None
        self._runtime_events: list[RuntimeEvent] = []
        self._demo_page_tables: list[dict[str, Any]] | None = None
        self._demo_ipc_records: list[Any] | None = None
        self._demo_status: str | None = None
        self._mailbox_signature: tuple[tuple[str, int, int, str], ...] | None = None
        self._process_signature: tuple[tuple[Any, ...], ...] | None = None
        self._scrollable_content: dict[str, str] = {}
        self._timeline_signature: tuple[str, ...] | None = None
        self._metrics_signature: tuple[str, ...] | None = None
        self._ipc_inspector_signature: tuple[str, ...] | None = None
        self._replay_signature: tuple[str, ...] | None = None
        self._dependency_graph_signature: tuple[str, ...] | None = None
        self._llm_stream_signature: tuple[str, ...] | None = None
        self._llm_cost_signature: tuple[str, ...] | None = None
        self._wasm_placeholder_logged = False

    def load_research_team_snapshot(self, state: dict[str, Any]) -> None:
        self._logged_supervision_events = 0
        self._demo_supervision_events = list(state.get("events") or ())
        self._demo_page_tables = None
        assignments = state.get("assignments", ())
        research_agents = state.get("research_agents", ())
        self._demo_ipc_records = [
            {"sender": "PlannerAgent", "receiver": assignment.destination, "message_type": "assignment"}
            for assignment in assignments
        ]
        self._demo_ipc_records.extend(
            {"sender": agent.name, "receiver": "SynthesizerAgent", "message_type": "result"}
            for agent in research_agents
        )
        if state.get("synthesized_report") is not None:
            self._demo_ipc_records.append(
                {"sender": "SynthesizerAgent", "receiver": "CriticAgent", "message_type": "report"}
            )
        review = state["critic_review"]
        self._demo_status = f"Workflow Complete  Final Score: {review.score}/10"
        self._demo_mailboxes = [
            MailboxMetric("PlannerAgent", 3, 3, "Assignments Sent"),
            MailboxMetric("ResearchAgents", 3, 3, "Results Sent"),
            MailboxMetric("SynthesizerAgent", 1, 1, "Report Sent"),
            MailboxMetric("CriticAgent", 1, 1, "Review Complete"),
        ]
        names = [
            "PlannerAgent",
            "ResearchBenefitsAgent",
            "ResearchRisksAgent",
            "ResearchMarketAgent",
            "SynthesizerAgent",
            "CriticAgent",
        ]
        self._demo_process_rows = [
            {
                "pid": 100 + index,
                "name": name,
                "status": "exited",
                "execution_mode": "demo",
                "messages_sent": (3, 1, 1, 1, 1, 1)[index],
                "messages_received": (0, 1, 1, 1, 3, 1)[index],
                "message_errors": 0,
            }
            for index, name in enumerate(names)
        ]
        self._demo_hierarchy = state.get(
            "hierarchy",
            {
                "supervisor": "ResearchTeamSupervisor",
                "children": names,
            },
        )

    def load_supervisor_recovery_snapshot(self, state: dict[str, Any]) -> None:
        self._logged_supervision_events = 0
        self._demo_page_tables = None
        self._demo_ipc_records = None
        self._demo_status = str(state["status"])
        self._demo_mailboxes = [
            MailboxMetric("RecoverySupervisor", 3, 3, "Supervisor Events"),
            MailboxMetric("RecoveryWorkerAgent", 1, 1, "Restarted"),
        ]
        self._demo_process_rows = list(state["process_rows"])
        self._demo_hierarchy = dict(state["hierarchy"])
        self._demo_supervision_events = list(state["events"])

    def load_memory_paging_snapshot(self, state: dict[str, Any]) -> None:
        self._logged_supervision_events = 0
        self._demo_ipc_records = None
        self._demo_status = str(state["status"])
        self._demo_mailboxes = [
            MailboxMetric("AgentA", 0, 1, "Context Loaded"),
            MailboxMetric("AgentB", 0, 1, "Context Loaded"),
        ]
        self._demo_process_rows = list(state["process_rows"])
        self._demo_hierarchy = dict(state["hierarchy"])
        self._demo_page_tables = list(state["page_tables"])
        self._demo_supervision_events = list(state["events"])

    def load_external_agent_result(
        self, *, succeeded: bool, events: list[RuntimeEvent] | tuple[RuntimeEvent, ...] = ()
    ) -> None:
        """Return dashboard panels to live state and show external run status."""
        self._demo_mailboxes = None
        self._demo_process_rows = None
        self._demo_hierarchy = None
        self._demo_supervision_events = None
        self._demo_page_tables = None
        self._demo_ipc_records = None
        self._demo_status = "External Agent Complete" if succeeded else "External Agent Failed"
        self._runtime_events.extend(events)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="status-bar")
        with Container(id="main-grid"):
            with Vertical(id="agent-tree-pane", classes="pane"):
                yield Static("Agent Tree", classes="pane-title primary-title")
                yield Static(id="agent-tree")
                yield Static(
                    "Agent Dependency Graph",
                    id="dependency-graph-title",
                    classes="pane-title secondary-title optional-monitor",
                )
                yield Static(id="dependency-graph", classes="optional-monitor")
            with Vertical(id="timeline-pane", classes="pane"):
                yield Static(
                    "Runtime Timeline", id="timeline-title", classes="pane-title primary-title"
                )
                yield Static(id="runtime-timeline")
                yield Static(
                    "Execution Replay",
                    id="replay-title",
                    classes="pane-title secondary-title optional-monitor",
                )
                yield Static(id="execution-replay", classes="optional-monitor")
            with Vertical(id="process-pane", classes="pane"):
                yield Static("Processes / IPC", classes="pane-title primary-title")
                yield DataTable(id="process-table")
                yield Static(
                    "[dim]No processes registered. Use the command bar to launch an agent.[/]",
                    id="process-empty",
                )
                yield Static(
                    "Agent Metrics",
                    id="metrics-title",
                    classes="pane-title section-title optional-monitor",
                )
                yield Static(id="agent-metrics", classes="optional-monitor")
                yield Static(
                    "IPC Queues", id="ipc-title", classes="pane-title section-title optional-monitor"
                )
                yield DataTable(id="ipc-table", classes="optional-monitor")
                yield Static(
                    "IPC Inspector",
                    id="ipc-inspector-title",
                    classes="pane-title section-title optional-monitor",
                )
                yield Static(id="ipc-inspector", classes="optional-monitor")
            with Vertical(id="wasm-pane", classes="pane"):
                yield Static("Tool / LLM Activity", classes="pane-title primary-title")
                yield RichLog(id="wasm-log", markup=True, wrap=True, highlight=True, auto_scroll=False)
                yield Static("Console", id="console-title", classes="pane-title section-title")
                yield RichLog(
                    id="console-log",
                    markup=True,
                    wrap=True,
                    highlight=True,
                    auto_scroll=False,
                )
                yield Static(
                    "LLM Stream Monitor",
                    id="llm-stream-title",
                    classes="pane-title secondary-title optional-monitor",
                )
                yield Static(id="llm-stream-monitor", classes="optional-monitor")
                yield Static(
                    "LLM Cost Monitor",
                    id="llm-cost-title",
                    classes="pane-title secondary-title optional-monitor",
                )
                yield Static(id="llm-cost-monitor", classes="optional-monitor")
                yield Static(
                    "Memory Context",
                    id="memory-title",
                    classes="pane-title secondary-title optional-monitor",
                )
                yield Static(id="memory-bars", classes="optional-monitor")
        yield Input(
            placeholder=f"{SHELL_PROMPT} run <path> | inspect <path> | demos | ps | kill <PID>",
            id="shell-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#ipc-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Agent Name", "Queue Depth", "Routing Method")
        process_table = self.query_one("#process-table", DataTable)
        process_table.cursor_type = "row"
        process_table.add_columns(
            "PID", "Agent", "State", "Mode", "Supervision", "Memory", "IPC"
        )
        self.set_interval(0.1, self.refresh_metrics)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        event.input.value = ""
        if not command:
            return
        self._write_console_log(f"[bold #8bd5ff]{SHELL_PROMPT}[/] {command}")

        if self.command_handler is None:
            self._write_console_log("[yellow]No command handler is attached.[/]")
            return

        try:
            result = self.command_handler(command)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment,misc]
        except Exception as exc:
            self._write_console_log(f"[bold red]error:[/] {exc}")
            return

        if result:
            self._write_console_log(str(result))

    def refresh_metrics(self) -> None:
        mailboxes = self._read_mailboxes()
        memory_agents = self._read_memory_agents()
        wasm_runs = self._read_wasm_runs()

        self._render_status(mailboxes)
        self._render_mailboxes(mailboxes)
        self._render_ipc_inspector(mailboxes)
        self._render_memory(memory_agents)
        self._render_agent_tree()
        self._render_dependency_graph()
        self._render_wasm_log(wasm_runs)
        self._render_supervision_events()
        self._render_runtime_events()
        self._render_llm_stream_monitor()
        self._render_llm_cost_monitor()
        self._render_timeline()
        self._render_replay()
        self.run_worker(self._refresh_process_rows(), exclusive=True, group="process-refresh")

    async def _refresh_process_rows(self) -> None:
        if self._demo_process_rows is not None:
            self._process_rows = self._demo_process_rows
            self._render_processes(self._process_rows)
            self._render_agent_metrics()
            return
        if self.process_snapshot is None:
            self._process_rows = []
            self._render_processes([])
            self._render_agent_metrics()
            return
        try:
            rows = self.process_snapshot()
            if hasattr(rows, "__await__"):
                rows = await rows  # type: ignore[assignment,misc]
            self._process_rows = list(rows)
        except Exception:
            self._process_rows = []
        self._render_processes(self._process_rows)
        self._render_agent_metrics()

    def _render_status(self, mailboxes: list[MailboxMetric]) -> None:
        process_rows = (
            self._demo_process_rows
            if self._demo_process_rows is not None
            else self._process_rows
        )
        observed_agents = max(len(process_rows), len(mailboxes))
        total_agents = (
            len(self._demo_process_rows)
            if self._demo_process_rows is not None
            else max(
                int(
                    self._safe_call(
                        self.kernel, "total_registered_agents", default=observed_agents
                    )
                ),
                observed_agents,
            )
        )
        active_tokens = self._safe_call(self.memory, "get_global_active_token_count", default=0)

        active_agents = sum(
            str(row.get("status", "")).lower() in {"running", "starting"}
            for row in process_rows
        )
        tool_calls = self._tool_call_count(self._observable_events())
        health = self._health_state(process_rows, self._demo_status)
        health_style = {
            "HEALTHY": "bold #8cffb5",
            "PARTIAL": "bold #ffd166",
            "DEGRADED": "bold #ff9f43",
            "FAILED": "bold #ff6b6b",
        }[health]

        self.query_one("#status-bar", Static).update(
            f"[bold #8bd5ff]SULCUS OS[/]   "
            f"[dim]HEALTH[/] [{health_style}]{health:<8}[/]  [#3b4b62]|[/]  "
            f"[dim]AGENTS[/] [bold]{active_agents}/{total_agents}[/]  [#3b4b62]|[/]  "
            f"[dim]TOOL CALLS[/] [bold]{tool_calls}[/]  [#3b4b62]|[/]  "
            f"[dim]TOKENS[/] [bold]{active_tokens}[/]"
            f"{'  [#3b4b62]|[/]  [bold #c9eeff]' + self._demo_status + '[/]' if self._demo_status else ''}"
        )

    @staticmethod
    def _health_state(process_rows: list[dict[str, Any]], demo_status: str | None) -> str:
        """Map observed presentation state to one stable dashboard health label."""
        statuses = [str(row.get("status", "")).lower() for row in process_rows]
        failed = sum(status in {"crashed", "failed"} for status in statuses)
        active = sum(status in {"running", "starting"} for status in statuses)
        restarted = any(int(row.get("restart_count", 0) or 0) > 0 for row in process_rows)
        explicitly_failed = bool(demo_status and "failed" in demo_status.lower())

        if explicitly_failed or (statuses and failed == len(statuses)):
            return "FAILED"
        if failed and not active:
            return "DEGRADED"
        if failed or restarted or "stopping" in statuses:
            return "PARTIAL"
        return "HEALTHY"

    @staticmethod
    def _tool_call_count(events: list[Any]) -> int:
        call_ids: set[str] = set()
        anonymous = 0
        request_types = {"tool.execution_requested", "tool_call_requested"}
        for event in events:
            if not isinstance(event, RuntimeEvent) or event.event_type not in request_types:
                continue
            call_id = event.metadata.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                call_ids.add(call_id)
            else:
                anonymous += 1
        return len(call_ids) + anonymous

    def _render_mailboxes(self, mailboxes: list[MailboxMetric]) -> None:
        signature = tuple(
            (metric.agent_name, metric.queue_depth, metric.buffer_size, metric.routing_method)
            for metric in mailboxes
        )
        if signature == self._mailbox_signature:
            return
        self._mailbox_signature = signature
        self._set_optional_monitor("#ipc-title", "#ipc-table", visible=bool(mailboxes))
        table = self.query_one("#ipc-table", DataTable)
        with self._preserve_scroll(table):
            table.clear()

            for metric in mailboxes:
                ratio = metric.queue_depth / metric.buffer_size if metric.buffer_size else 0.0
                queue_style = "bold red" if ratio >= 0.8 else "green"
                route_style = "yellow" if metric.routing_method != "Direct" else "cyan"

                table.add_row(
                    metric.agent_name,
                    Text(f"{metric.queue_depth}/{metric.buffer_size}", style=queue_style),
                    Text(metric.routing_method, style=route_style),
                )

    def _render_memory(self, agents: list[str]) -> None:
        if self._demo_page_tables is not None:
            self._set_optional_monitor("#memory-title", "#memory-bars", visible=True)
            self._update_scrollable_static(
                "#memory-bars",
                self._format_demo_page_tables(self._demo_page_tables),
            )
            return

        self._set_optional_monitor("#memory-title", "#memory-bars", visible=bool(agents))
        lines: list[str] = []

        if not agents:
            lines.append("[dim]No page tables registered yet.[/]")

        for agent_name in agents:
            summary = self._safe_call(self.memory, "get_page_table_summary", agent_name, default={})
            current = int(summary.get("current_active_tokens", 0))
            maximum = max(int(summary.get("max_active_tokens", 1)), 1)
            pending = int(summary.get("pending_evictions", 0))
            ratio = min(current / maximum, 1.0)
            filled = int(ratio * 24)
            bar = "#" * filled + "-" * (24 - filled)
            color = "red" if ratio >= 0.85 else "yellow" if ratio >= 0.65 else "green"

            previous_pending = self._last_pending_evictions.get(agent_name, pending)
            swapping = pending > previous_pending or (pending and int(time.time() * 5) % 2 == 0)
            self._last_pending_evictions[agent_name] = pending

            suffix = " [bold blink red]SWAPPING...[/]" if swapping else ""
            lines.append(
                f"[bold]{agent_name}[/] [{color}]{bar}[/] "
                f"{current}/{maximum} tokens "
                f"active={summary.get('active_frames', 0)} "
                f"paged={summary.get('paged_out_frames', 0)}{suffix}"
            )

        self._update_scrollable_static("#memory-bars", "\n".join(lines))

    @staticmethod
    def _format_demo_page_tables(page_tables: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for table in page_tables:
            lines.append(f"[bold]{table['agent']}[/]")
            for page in table["pages"]:
                state = str(page["state"])
                style = "bold red" if state == "evicted" else "green"
                label = "Evicted" if state == "evicted" else "Active"
                lines.append(f"  [{style}]Page {page['page']}  {label}[/]")
        return "\n".join(lines)

    def _render_wasm_log(self, runs: list[WasmRunMetric]) -> None:
        if not runs and not self._wasm_placeholder_logged:
            self._write_execution_log(
                "[dim]Sandboxes Active: 0 | Executions: 0 | Isolation: Ready[/]",
                scroll_end=False,
            )
            self._wasm_placeholder_logged = True
        for run in runs[self._logged_wasm_runs :]:
            status = "[green]Success[/]" if run.success else "[bold dark_red]Trapped[/]"
            error = ""
            if run.error_message:
                error = f" [dark_red]{run.error_message}[/]"
            self._write_execution_log(
                f"{run.timestamp} | Last Run Executed | {status} | "
                f"Fuel Consumed: [bold]{run.fuel_consumed}[/]{error}",
                scroll_end=False,
            )
        self._logged_wasm_runs = len(runs)

    def _render_supervision_events(self) -> None:
        if self._demo_supervision_events is not None:
            events = self._demo_supervision_events
        elif self.supervision_event_snapshot is not None:
            events = self.supervision_event_snapshot()
        else:
            return

        for event in events[self._logged_supervision_events :]:
            if isinstance(event, RuntimeEvent):
                self._write_execution_log(event, scroll_end=False)
                continue
            style = {
                "child_terminated": "bold red",
                "child_restart_requested": "bold yellow",
                "child_restarted": "bold green",
                "page_allocated": "bold green",
                "page_evicted": "bold red",
            }.get(str(event.get("event")))
            if style is not None:
                if str(event.get("event", "")).startswith("page_"):
                    message = f"[Memory] {event.get('message', '')}"
                else:
                    message = f"[Supervisor]\n{event.get('message', '')}"
                self._write_execution_log(Text(message, style=style), scroll_end=False)
        self._logged_supervision_events = len(events)

    def _render_runtime_events(self) -> None:
        for event in self._runtime_events[self._logged_runtime_events :]:
            self._write_execution_log(event, scroll_end=False)
        self._logged_runtime_events = len(self._runtime_events)

    def _render_timeline(self) -> None:
        events = self._observable_events()
        rows = render_runtime_timeline(events)
        signature = tuple(rows)
        if signature == self._timeline_signature:
            return
        self._timeline_signature = signature
        content = "\n".join(rows) if rows else "[dim]No runtime events yet.[/]"
        self._update_scrollable_static_follow_end("#runtime-timeline", content)

    def _render_replay(self) -> None:
        session = build_replay_session(self._observable_events())
        rows = render_replay_session(session)
        signature = tuple(rows)
        if signature == self._replay_signature:
            return
        self._replay_signature = signature
        self._set_optional_monitor("#replay-title", "#execution-replay", visible=bool(rows))
        content = "\n".join(rows) if rows else "[dim]No replay data available.[/]"
        self._update_scrollable_static_follow_end("#execution-replay", content)

    def _render_llm_stream_monitor(self) -> None:
        snapshot = build_llm_stream_snapshot(self._observable_events())
        rows = render_llm_stream_snapshot(snapshot)
        signature = tuple(rows)
        if signature == self._llm_stream_signature:
            return
        self._llm_stream_signature = signature
        self._set_optional_monitor(
            "#llm-stream-title", "#llm-stream-monitor", visible=bool(snapshot.metrics)
        )
        self._update_scrollable_static_follow_end("#llm-stream-monitor", "\n".join(rows))

    def _render_llm_cost_monitor(self) -> None:
        snapshot = build_llm_cost_snapshot(self._observable_events())
        rows = render_llm_cost_snapshot(snapshot)
        signature = tuple(rows)
        if signature == self._llm_cost_signature:
            return
        self._llm_cost_signature = signature
        self._set_optional_monitor(
            "#llm-cost-title",
            "#llm-cost-monitor",
            visible=bool(snapshot.ledger.records),
        )
        self._update_scrollable_static_follow_end("#llm-cost-monitor", "\n".join(rows))

    def _observable_events(self) -> list[Any]:
        events: list[Any] = []
        if self._demo_supervision_events is not None:
            events.extend(self._demo_supervision_events)
        elif self.supervision_event_snapshot is not None:
            events.extend(self.supervision_event_snapshot())
        events.extend(self._runtime_events)
        return events

    def _render_agent_metrics(self) -> None:
        events: list[Any] = []
        if self._demo_supervision_events is not None:
            events.extend(self._demo_supervision_events)
        events.extend(self._runtime_events)
        snapshot = build_agent_metrics_snapshot(self._process_rows, events)
        rows = render_agent_metrics(snapshot)
        signature = tuple(rows)
        if signature == self._metrics_signature:
            return
        self._metrics_signature = signature
        self._set_optional_monitor("#metrics-title", "#agent-metrics", visible=bool(rows))
        content = "\n".join(rows) if rows else "[dim]No agent metrics available yet.[/]"
        self._update_scrollable_static_follow_end("#agent-metrics", content)

    def _render_ipc_inspector(self, mailboxes: list[MailboxMetric] | None = None) -> None:
        snapshot = build_ipc_snapshot(
            self._ipc_records(),
            process_rows=self._process_rows,
            mailbox_metrics=mailboxes or self._read_mailboxes(),
        )
        rows = render_ipc_inspector(snapshot)
        signature = tuple(rows)
        if signature == self._ipc_inspector_signature:
            return
        self._ipc_inspector_signature = signature
        self._set_optional_monitor(
            "#ipc-inspector-title", "#ipc-inspector", visible=bool(rows)
        )
        content = "\n".join(rows) if rows else "[dim]No IPC activity yet.[/]"
        self._update_scrollable_static_follow_end("#ipc-inspector", content)

    def _render_dependency_graph(self) -> None:
        snapshot = build_dependency_graph(
            process_rows=self._process_rows,
            events=self._observable_events(),
            ipc_records=self._ipc_records(),
        )
        rows = render_dependency_graph(snapshot)
        signature = tuple(rows)
        if signature == self._dependency_graph_signature:
            return
        self._dependency_graph_signature = signature
        self._set_optional_monitor(
            "#dependency-graph-title",
            "#dependency-graph",
            visible=True,
        )
        self._update_scrollable_static_follow_end("#dependency-graph", "\n".join(rows))

    def _ipc_records(self) -> list[Any]:
        records: list[Any] = list(self._demo_ipc_records or ())
        records.extend(
            event
            for event in self._runtime_events
            if ("sender" in event.metadata or "source" in event.metadata)
            and ("receiver" in event.metadata or "target" in event.metadata)
        )
        return records

    def _render_processes(self, rows: list[dict[str, Any]]) -> None:
        signature = tuple(
            (
                row.get("pid"),
                row.get("name"),
                row.get("status"),
                row.get("execution_mode"),
                row.get("supervisor_pid"),
                row.get("child_count"),
                row.get("restart_count"),
                row.get("supervisor_strategy"),
                row.get("memory_hot_tokens", row.get("memory_tokens", 0)),
                row.get("memory_paged_count", 0),
                row.get("messages_sent", 0),
                row.get("messages_received", 0),
                row.get("message_errors", 0),
                row.get("external"),
            )
            for row in rows
        )
        if signature == self._process_signature:
            return
        self._process_signature = signature
        table = self.query_one("#process-table", DataTable)
        table.display = bool(rows)
        self.query_one("#process-empty", Static).display = not rows
        with self._preserve_scroll(table):
            table.clear()
            for row in rows:
                status = str(row.get("status", "unknown"))
                restart_count = int(row.get("restart_count", 0) or 0)
                display_status = self._display_process_status(
                    status, external=bool(row.get("external"))
                )
                if status == "running" and restart_count:
                    display_status = "RESTARTED"
                status_style = {
                    "RUNNING": "bold #8cffb5",
                    "RESTARTED": "bold #8bd5ff",
                    "STARTING": "bold #ffd166",
                    "STOPPING": "#ffd166",
                    "TERMINATED": "dim #ff8a8a",
                    "CRASHED": "bold #ff6b6b",
                    "FAILED": "bold #ff6b6b",
                    "COMPLETED": "#8cffb5",
                    "EXITED": "dim",
                }.get(display_status, "white")
                depth = int(row.get("tree_depth", 0))
                display_name = f"{'|  ' * depth}{row.get('name', '')}"
                table.add_row(
                    str(row.get("pid", "")),
                    display_name,
                    Text(display_status, style=status_style),
                    str(row.get("execution_mode", "")),
                    self._format_supervision(row),
                    f"{row.get('memory_hot_tokens', row.get('memory_tokens', 0))}/{row.get('memory_paged_count', 0)}",
                    f"{row.get('messages_sent', 0)}/{row.get('messages_received', 0)}/{row.get('message_errors', 0)}",
                )

    @staticmethod
    def _format_supervision(row: dict[str, Any]) -> str:
        parts: list[str] = []
        if row.get("supervisor_pid") is not None:
            parts.append(f"parent={row['supervisor_pid']}")
        child_count = int(row.get("child_count", 0) or 0)
        if child_count:
            parts.append(f"children={child_count}")
        restart_count = int(row.get("restart_count", 0) or 0)
        if restart_count:
            parts.append(f"restarts={restart_count}")
        strategy = str(row.get("supervisor_strategy", "")).strip()
        if strategy:
            parts.append(strategy)
        return "  ".join(parts) or "-"

    @staticmethod
    def _display_process_status(status: str, *, external: bool = False) -> str:
        if external and status == "exited":
            return "COMPLETED"
        return "TERMINATED" if status == "killed" else status.upper()

    def _render_agent_tree(self) -> None:
        rows = (
            self._demo_process_rows
            if self._demo_process_rows is not None
            else self._process_rows
        )
        hierarchy = self._demo_hierarchy or self._hierarchy_from_process_rows(rows)
        states = {
            str(row.get("name", "")): self._agent_tree_state(row)
            for row in rows
            if row.get("name")
        }
        self._update_scrollable_static(
            "#agent-tree", self._format_agent_tree(hierarchy, states=states)
        )

    def _update_scrollable_static(self, selector: str, content: str) -> None:
        if self._scrollable_content.get(selector) == content:
            return
        self._scrollable_content[selector] = content
        widget = self.query_one(selector, Static)
        with self._preserve_scroll(widget):
            widget.update(content)

    def _set_optional_monitor(
        self, title_selector: str, content_selector: str, *, visible: bool
    ) -> None:
        """Show secondary telemetry only when it contributes useful information."""
        self.query_one(title_selector).display = visible
        self.query_one(content_selector).display = visible

    def _update_scrollable_static_follow_end(self, selector: str, content: str) -> None:
        if self._scrollable_content.get(selector) == content:
            return
        widget = self.query_one(selector, Static)
        was_at_end = widget.is_vertical_scroll_end
        if was_at_end:
            self._scrollable_content[selector] = content
            widget.update(content)
            widget.scroll_end(animate=False)
            return
        self._update_scrollable_static(selector, content)

    def _write_execution_log(self, content: Any, *, scroll_end: bool | None = None) -> None:
        log = self.query_one("#wasm-log", RichLog)
        if isinstance(content, RuntimeEvent):
            content = render_runtime_event(content)
        if scroll_end is None:
            scroll_end = log.is_vertical_scroll_end
        log.write(content, scroll_end=scroll_end)

    def _write_console_log(self, content: Any, *, scroll_end: bool | None = None) -> None:
        """Write interactive shell output without mixing it into runtime telemetry."""
        log = self.query_one("#console-log", RichLog)
        if scroll_end is None:
            scroll_end = log.is_vertical_scroll_end
        log.write(content, scroll_end=scroll_end)

    @contextmanager
    def _preserve_scroll(self, widget: Any) -> Any:
        scroll_x = getattr(widget, "scroll_x", 0)
        scroll_y = getattr(widget, "scroll_y", 0)
        try:
            yield
        finally:
            try:
                widget.scroll_to(x=scroll_x, y=scroll_y, animate=False, force=True)
            except Exception:
                pass

    @staticmethod
    def _hierarchy_from_process_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
        by_pid = {row.get("pid"): row for row in rows}
        supervised = [row for row in rows if row.get("supervisor_pid") in by_pid]
        if not supervised:
            return None
        supervisor_pid = supervised[0]["supervisor_pid"]
        supervisor = by_pid[supervisor_pid]
        children_by_name: dict[str, dict[str, Any]] = {}
        for row in supervised:
            if row.get("supervisor_pid") != supervisor_pid:
                continue
            name = str(row.get("name", ""))
            current = children_by_name.get(name)
            if current is None or current.get("status") == "killed":
                children_by_name[name] = row
        children = []
        for name, row in children_by_name.items():
            if row.get("status") == "killed":
                suffix = " (terminated)"
            elif row.get("status") in {"crashed", "failed"}:
                suffix = " (failed)"
            elif int(row.get("restart_count", 0)) > 0:
                suffix = " (restarted)"
            else:
                suffix = ""
            children.append(f"{name}{suffix}")
        return {"supervisor": supervisor.get("name", ""), "children": children}

    @staticmethod
    def _format_agent_tree(
        hierarchy: dict[str, Any] | None, *, states: dict[str, str] | None = None
    ) -> str:
        if not hierarchy:
            return "[dim]No active hierarchy. Running agents will appear here.[/]"

        supervisor = str(hierarchy.get("supervisor", "")).strip()
        children = [str(child) for child in hierarchy.get("children", [])]
        if not supervisor:
            return "[dim]No active hierarchy. Running agents will appear here.[/]"

        states = states or {}
        markers = {
            "running": "[#8cffb5]\\[>][/]",
            "restarted": "[bold #8bd5ff]\\[R][/]",
            "failed": "[bold #ff6b6b]\\[!][/]",
            "completed": "[#8cffb5]\\[+][/]",
            "terminated": "[dim #ff8a8a]\\[X][/]",
            "unknown": "[dim]\\[?][/]",
        }
        supervisor_state = states.get(supervisor, "running")
        lines = [
            f"[bold #c9eeff]\\[SUP][/] {markers.get(supervisor_state, markers['unknown'])} "
            f"[bold]{supervisor}[/]"
        ]
        for index, child in enumerate(children):
            connector = "`--" if index == len(children) - 1 else "|--"
            name = child
            if child.endswith(" (restarted)"):
                name = child.removesuffix(" (restarted)")
                fallback_state = "restarted"
            elif child.endswith(" (terminated)"):
                name = child.removesuffix(" (terminated)")
                fallback_state = "terminated"
            elif child.endswith(" (failed)") or child.endswith(" (crashed)"):
                name = child.rsplit(" (", 1)[0]
                fallback_state = "failed"
            elif child.endswith(" (completed)"):
                name = child.removesuffix(" (completed)")
                fallback_state = "completed"
            else:
                fallback_state = "running"
            state = states.get(name, fallback_state)
            marker = markers.get(state, markers["unknown"])
            lines.append(f"{connector} {marker} {child}")
        return "\n".join(lines)

    @staticmethod
    def _agent_tree_state(row: dict[str, Any]) -> str:
        status = str(row.get("status", "unknown")).lower()
        if status in {"crashed", "failed"}:
            return "failed"
        if status == "killed":
            return "terminated"
        if status in {"exited", "completed", "complete"}:
            return "completed"
        if status == "running" and int(row.get("restart_count", 0) or 0) > 0:
            return "restarted"
        if status in {"running", "starting", "stopping"}:
            return "running"
        return "unknown"

    def _read_mailboxes(self) -> list[MailboxMetric]:
        if self._demo_mailboxes is not None:
            return self._demo_mailboxes
        raw_metrics = self._safe_call(self.bus, "get_mailbox_metrics", default=[])
        metrics: list[MailboxMetric] = []
        for raw in raw_metrics:
            if isinstance(raw, dict):
                metrics.append(
                    MailboxMetric(
                        str(raw.get("agent_name", "")),
                        int(raw.get("queue_depth", 0)),
                        int(raw.get("buffer_size", 0)),
                        str(raw.get("routing_method", "Direct")),
                    )
                )
            else:
                agent_name, queue_depth, buffer_size, routing_method = raw
                metrics.append(
                    MailboxMetric(
                        str(agent_name),
                        int(queue_depth),
                        int(buffer_size),
                        str(routing_method),
                    )
                )
        return metrics

    def _read_memory_agents(self) -> list[str]:
        agents = self._safe_call(self.memory, "list_agents", default=[])
        return [str(agent) for agent in agents]

    def _read_wasm_runs(self) -> list[WasmRunMetric]:
        raw_runs = self._safe_call(self.sandbox, "get_execution_metrics", default=[])
        runs: list[WasmRunMetric] = []
        for raw in raw_runs:
            timestamp, success, fuel_consumed, error_message = raw
            runs.append(
                WasmRunMetric(
                    int(timestamp),
                    bool(success),
                    int(fuel_consumed),
                    None if error_message is None else str(error_message),
                )
            )
        return runs

    @staticmethod
    def _safe_call(obj: Any, method_name: str, *args: Any, default: Any) -> Any:
        method = getattr(obj, method_name, None)
        if method is None:
            return default
        try:
            return method(*args)
        except Exception:
            return default
