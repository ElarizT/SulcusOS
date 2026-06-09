"""Deterministic per-agent runtime metrics helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from kernel.events import RuntimeEvent


@dataclass(frozen=True)
class AgentMetrics:
    name: str
    status: str
    pid: int | None = None
    runtime_seconds: float | None = None
    messages_sent: int | None = None
    messages_received: int | None = None
    restart_count: int | None = None
    exit_code: int | None = None
    error: bool | None = None


@dataclass(frozen=True)
class AgentMetricsSnapshot:
    metrics: tuple[AgentMetrics, ...]


def build_agent_metrics_snapshot(
    process_rows: Iterable[Mapping[str, Any]] = (),
    events: Iterable[Any] = (),
) -> AgentMetricsSnapshot:
    """Build one metrics snapshot from existing process rows and runtime events."""
    by_name: dict[str, AgentMetrics] = {}
    for row in process_rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        status = _process_status(str(row.get("status", "unknown")), external=bool(row.get("external")))
        by_name[name] = AgentMetrics(
            name=name,
            status=status,
            pid=_optional_int(row, "pid"),
            runtime_seconds=_optional_float(row, "uptime_seconds"),
            messages_sent=_optional_int(row, "messages_sent"),
            messages_received=_optional_int(row, "messages_received"),
            restart_count=_optional_int(row, "restart_count"),
            exit_code=_optional_int(row, "exit_code"),
            error=_row_error(row),
        )

    lifecycle: dict[str, dict[str, Any]] = {}
    for event in sorted(
        (item for item in events if isinstance(item, RuntimeEvent)),
        key=lambda item: item.timestamp,
    ):
        if not event.event_type.startswith("external_agent_"):
            continue
        name = str(event.metadata.get("agent", "")).strip()
        if not name:
            continue
        state = lifecycle.setdefault(name, {})
        action = event.event_type.removeprefix("external_agent_")
        if action == "loaded":
            state["status"] = "loaded"
        elif action == "started":
            state["status"] = "running"
            state["started_at"] = event.timestamp
        elif action in {"completed", "failed"}:
            state["status"] = "complete" if action == "completed" else "failed"
            state["ended_at"] = event.timestamp
        if "pid" in event.metadata:
            state["pid"] = event.metadata["pid"]
        if "exit_code" in event.metadata:
            state["exit_code"] = event.metadata["exit_code"]
        if action == "failed":
            state["error"] = True

    for name, state in lifecycle.items():
        current = by_name.get(name, AgentMetrics(name=name, status="unknown"))
        runtime_seconds = current.runtime_seconds
        if state.get("started_at") is not None and state.get("ended_at") is not None:
            runtime_seconds = max((state["ended_at"] - state["started_at"]).total_seconds(), 0.0)
        by_name[name] = AgentMetrics(
            name=name,
            status=str(state.get("status", current.status)),
            pid=_coerce_int(state.get("pid"), current.pid),
            runtime_seconds=runtime_seconds,
            messages_sent=current.messages_sent,
            messages_received=current.messages_received,
            restart_count=current.restart_count,
            exit_code=_coerce_int(state.get("exit_code"), current.exit_code),
            error=state.get("error", current.error),
        )

    return AgentMetricsSnapshot(tuple(sorted(by_name.values(), key=lambda metric: (metric.name, metric.pid or -1))))


def format_agent_metric(metric: AgentMetrics) -> str:
    """Format one compact, stable agent metrics row."""
    parts = [f"{metric.name:<18}", f"{metric.status:<10}"]
    if metric.pid is not None:
        parts.append(f"pid={metric.pid}")
    if metric.messages_sent is not None:
        parts.append(f"sent={metric.messages_sent}")
    if metric.messages_received is not None:
        parts.append(f"recv={metric.messages_received}")
    if metric.restart_count is not None:
        parts.append(f"restarts={metric.restart_count}")
    if metric.exit_code is not None:
        parts.append(f"exit={metric.exit_code}")
    if metric.runtime_seconds is not None:
        label = "uptime" if metric.status in {"running", "starting"} else "runtime"
        parts.append(f"{label}={metric.runtime_seconds:.2f}s")
    if metric.error:
        parts.append("error=true")
    return "  ".join(parts)


def render_agent_metrics(
    snapshot: AgentMetricsSnapshot | Iterable[AgentMetrics],
) -> list[str]:
    metrics = snapshot.metrics if isinstance(snapshot, AgentMetricsSnapshot) else tuple(snapshot)
    return [format_agent_metric(metric) for metric in metrics]


def _process_status(status: str, *, external: bool) -> str:
    if status == "crashed":
        return "failed"
    if external and status == "exited":
        return "complete"
    return status


def _optional_int(values: Mapping[str, Any], key: str) -> int | None:
    return _coerce_int(values.get(key)) if key in values else None


def _optional_float(values: Mapping[str, Any], key: str) -> float | None:
    if key not in values or values[key] is None:
        return None
    try:
        return float(values[key])
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _row_error(row: Mapping[str, Any]) -> bool | None:
    if row.get("error"):
        return True
    if int(row.get("message_errors", 0) or 0) > 0:
        return True
    return None
