from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.widgets import DataTable, Footer, Header, Input, RichLog, Static

SHELL_PROMPT = "AgentOS>"


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
    """Real-time terminal dashboard for Agent OS kernel telemetry."""

    CSS = """
    Screen {
        background: #080b0f;
        color: #d7dde8;
    }

    #status-bar {
        height: 3;
        padding: 0 1;
        content-align: left middle;
        background: #101722;
        border-bottom: solid #263245;
    }

    #main-grid {
        layout: grid;
        grid-size: 2 3;
        grid-columns: 1fr 1fr;
        grid-rows: 2fr 1fr 3fr;
        height: 1fr;
    }

    .pane {
        border: solid #263245;
        padding: 0 1;
        min-height: 8;
    }

    #ipc-pane {
        column-span: 1;
        row-span: 1;
    }

    #memory-pane {
        column-span: 1;
        row-span: 1;
    }

    #wasm-pane {
        column-span: 2;
        row-span: 1;
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

    #ipc-table {
        height: 1fr;
    }

    #memory-bars {
        height: 1fr;
        padding-top: 1;
    }

    #wasm-log {
        height: 1fr;
        background: #080b0f;
    }

    #process-table {
        height: 1fr;
    }

    #agent-tree {
        height: 1fr;
    }

    #shell-input {
        dock: bottom;
        height: 3;
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
        self._heartbeat_index = 0
        self._last_pending_evictions: dict[str, int] = {}
        self._logged_wasm_runs = 0
        self._logged_supervision_events = 0
        self._process_rows: list[dict[str, Any]] = []
        self._demo_mailboxes: list[MailboxMetric] | None = None
        self._demo_process_rows: list[dict[str, Any]] | None = None
        self._demo_hierarchy: dict[str, Any] | None = None
        self._demo_supervision_events: list[dict[str, Any]] | None = None
        self._demo_page_tables: list[dict[str, Any]] | None = None
        self._demo_status: str | None = None
        self._wasm_placeholder_logged = False

    def load_research_team_snapshot(self, state: dict[str, Any]) -> None:
        self._logged_supervision_events = 0
        self._demo_supervision_events = None
        self._demo_page_tables = None
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
        self._demo_status = str(state["status"])
        self._demo_mailboxes = [
            MailboxMetric("AgentA", 0, 1, "Context Loaded"),
            MailboxMetric("AgentB", 0, 1, "Context Loaded"),
        ]
        self._demo_process_rows = list(state["process_rows"])
        self._demo_hierarchy = dict(state["hierarchy"])
        self._demo_page_tables = list(state["page_tables"])
        self._demo_supervision_events = list(state["events"])

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(id="status-bar")
        with Container(id="main-grid"):
            with Vertical(id="ipc-pane", classes="pane"):
                yield Static("IPC Mailbox Lane Monitor", classes="pane-title")
                yield DataTable(id="ipc-table")
            with Vertical(id="memory-pane", classes="pane"):
                yield Static("Page Table Context Visualizer", classes="pane-title")
                yield Static(id="memory-bars")
            with Vertical(id="wasm-pane", classes="pane"):
                yield Static("WASM Execution Shield Matrix", classes="pane-title")
                yield RichLog(id="wasm-log", markup=True, wrap=True, highlight=True)
            with Vertical(id="agent-tree-pane", classes="pane"):
                yield Static("Agent Tree View", classes="pane-title")
                yield Static(id="agent-tree")
            with Vertical(id="process-pane", classes="pane"):
                yield Static("Process Registry", classes="pane-title")
                yield DataTable(id="process-table")
        yield Input(placeholder=f"{SHELL_PROMPT} run <path> | demos | ps | kill <PID>", id="shell-input")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#ipc-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Agent Name", "Queue Depth", "Routing Method")
        process_table = self.query_one("#process-table", DataTable)
        process_table.cursor_type = "row"
        process_table.add_columns(
            "PID", "Name", "Status", "Mode", "Parent", "Kids", "Restarts", "Strategy", "Memory", "IPC"
        )
        self.set_interval(0.1, self.refresh_metrics)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        command = event.value.strip()
        event.input.value = ""
        if not command:
            return
        log = self.query_one("#wasm-log", RichLog)
        log.write(f"[bold #8bd5ff]{SHELL_PROMPT}[/] {command}")

        if self.command_handler is None:
            log.write("[yellow]No command handler is attached.[/]")
            return

        try:
            result = self.command_handler(command)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[assignment,misc]
        except Exception as exc:
            log.write(f"[bold red]error:[/] {exc}")
            return

        if result:
            log.write(str(result))

    def refresh_metrics(self) -> None:
        mailboxes = self._read_mailboxes()
        memory_agents = self._read_memory_agents()
        wasm_runs = self._read_wasm_runs()

        self._render_status(mailboxes)
        self._render_mailboxes(mailboxes)
        self._render_memory(memory_agents)
        self._render_agent_tree()
        self._render_wasm_log(wasm_runs)
        self._render_supervision_events()
        self.run_worker(self._refresh_process_rows(), exclusive=True, group="process-refresh")

    async def _refresh_process_rows(self) -> None:
        if self._demo_process_rows is not None:
            self._process_rows = self._demo_process_rows
            self._render_processes(self._process_rows)
            return
        if self.process_snapshot is None:
            self._process_rows = []
            self._render_processes([])
            return
        try:
            rows = self.process_snapshot()
            if hasattr(rows, "__await__"):
                rows = await rows  # type: ignore[assignment,misc]
            self._process_rows = list(rows)
        except Exception:
            self._process_rows = []
        self._render_processes(self._process_rows)

    def _render_status(self, mailboxes: list[MailboxMetric]) -> None:
        heartbeats = ["|", "/", "-", "\\"]
        self._heartbeat_index = (self._heartbeat_index + 1) % len(heartbeats)
        heartbeat = heartbeats[self._heartbeat_index]

        total_agents = (
            len(self._demo_process_rows)
            if self._demo_process_rows is not None
            else self._safe_call(self.kernel, "total_registered_agents", default=len(mailboxes))
        )
        active_tokens = self._safe_call(self.memory, "get_global_active_token_count", default=0)

        self.query_one("#status-bar", Static).update(
            f"[bold #8bd5ff]Agent OS[/]  "
            f"Agents: [bold]{total_agents}[/]  "
            f"Active Tokens: [bold]{active_tokens}[/]  "
            f"Health: [bold #8cffb5]{heartbeat} ONLINE[/]  "
            f"{'[bold #8cffb5]' + self._demo_status + '[/]' if self._demo_status else ''}"
        )

    def _render_mailboxes(self, mailboxes: list[MailboxMetric]) -> None:
        table = self.query_one("#ipc-table", DataTable)
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
            self.query_one("#memory-bars", Static).update(self._format_demo_page_tables(self._demo_page_tables))
            return

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

        self.query_one("#memory-bars", Static).update("\n".join(lines))

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
        log = self.query_one("#wasm-log", RichLog)
        if not runs and not self._wasm_placeholder_logged:
            log.write("[dim]Sandboxes Active: 0 | Executions: 0 | Isolation: Ready[/]")
            self._wasm_placeholder_logged = True
        for run in runs[self._logged_wasm_runs :]:
            status = "[green]Success[/]" if run.success else "[bold dark_red]Trapped[/]"
            error = ""
            if run.error_message:
                error = f" [dark_red]{run.error_message}[/]"
            log.write(
                f"{run.timestamp} | Last Run Executed | {status} | "
                f"Fuel Consumed: [bold]{run.fuel_consumed}[/]{error}"
            )
        self._logged_wasm_runs = len(runs)

    def _render_supervision_events(self) -> None:
        if self._demo_supervision_events is not None:
            events = self._demo_supervision_events
        elif self.supervision_event_snapshot is not None:
            events = self.supervision_event_snapshot()
        else:
            return
        log = self.query_one("#wasm-log", RichLog)
        for event in events[self._logged_supervision_events :]:
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
                log.write(Text(message, style=style))
        self._logged_supervision_events = len(events)

    def _render_processes(self, rows: list[dict[str, Any]]) -> None:
        table = self.query_one("#process-table", DataTable)
        table.clear()
        for row in rows:
            status = str(row.get("status", "unknown"))
            status_style = {
                "running": "green",
                "starting": "yellow",
                "stopping": "yellow",
                "killed": "dim",
                "crashed": "bold red",
                "exited": "dim",
            }.get(status, "white")
            display_status = self._display_process_status(status)
            depth = int(row.get("tree_depth", 0))
            display_name = f"{'  ' * depth}{row.get('name', '')}"
            table.add_row(
                str(row.get("pid", "")),
                display_name,
                Text(display_status, style=status_style),
                str(row.get("execution_mode", "")),
                "" if row.get("supervisor_pid") is None else str(row.get("supervisor_pid")),
                str(row.get("child_count", 0)),
                str(row.get("restart_count", 0)),
                str(row.get("supervisor_strategy", "")),
                f"{row.get('memory_hot_tokens', row.get('memory_tokens', 0))}/{row.get('memory_paged_count', 0)}",
                f"{row.get('messages_sent', 0)}/{row.get('messages_received', 0)}/{row.get('message_errors', 0)}",
            )

    @staticmethod
    def _display_process_status(status: str) -> str:
        return "TERMINATED" if status == "killed" else status

    def _render_agent_tree(self) -> None:
        hierarchy = self._demo_hierarchy or self._hierarchy_from_process_rows(self._process_rows)
        self.query_one("#agent-tree", Static).update(self._format_agent_tree(hierarchy))

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
            elif int(row.get("restart_count", 0)) > 0:
                suffix = " (restarted)"
            else:
                suffix = ""
            children.append(f"{name}{suffix}")
        return {"supervisor": supervisor.get("name", ""), "children": children}

    @staticmethod
    def _format_agent_tree(hierarchy: dict[str, Any] | None) -> str:
        if not hierarchy:
            return "[dim]No active hierarchy[/]"

        supervisor = str(hierarchy.get("supervisor", "")).strip()
        children = [str(child) for child in hierarchy.get("children", [])]
        if not supervisor:
            return "[dim]No active hierarchy[/]"

        lines = [f"[bold]{supervisor}[/]"]
        for index, child in enumerate(children):
            connector = "└──" if index == len(children) - 1 else "├──"
            lines.append(f"{connector} {child}")
        return "\n".join(lines)

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
