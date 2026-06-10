"""Deterministic observed agent dependency graph helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from kernel.events import RuntimeEvent
from kernel.ipc_inspector import IPCSnapshot, build_ipc_snapshot


@dataclass(frozen=True)
class DependencyNode:
    name: str
    status: str = "unknown"
    pid: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failed_at: datetime | None = None


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str
    relation: str | None = None
    message_count: int | None = None
    latest_message_type: str | None = None


@dataclass(frozen=True)
class DependencyGraphSnapshot:
    nodes: tuple[DependencyNode, ...] = ()
    edges: tuple[DependencyEdge, ...] = ()


def build_dependency_graph(
    *,
    process_rows: Iterable[Mapping[str, Any]] = (),
    events: Iterable[Any] = (),
    ipc_records: Iterable[Any] = (),
    ipc_snapshot: IPCSnapshot | None = None,
) -> DependencyGraphSnapshot:
    """Build a stable graph from observed runtime state and declared supervision."""
    rows = tuple(process_rows)
    runtime_events = tuple(event for event in events if isinstance(event, RuntimeEvent))
    snapshot = ipc_snapshot or build_ipc_snapshot(ipc_records, process_rows=rows)
    nodes: dict[str, DependencyNode] = {}

    pid_names: dict[int, str] = {}
    for row in rows:
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        pid = _optional_int(row.get("pid"))
        if pid is not None:
            pid_names[pid] = name
        nodes[name] = DependencyNode(
            name=name,
            status=_process_status(str(row.get("status", "unknown")), external=bool(row.get("external"))),
            pid=pid,
        )

    for event in sorted(runtime_events, key=lambda item: item.timestamp):
        if not event.event_type.startswith("external_agent_"):
            continue
        name = str(event.metadata.get("agent", "")).strip()
        if not name:
            continue
        current = nodes.get(name, DependencyNode(name))
        action = event.event_type.removeprefix("external_agent_")
        status = current.status
        started_at = current.started_at
        completed_at = current.completed_at
        failed_at = current.failed_at
        if action == "loaded":
            status = "loaded"
        elif action == "started":
            status = "running"
            started_at = event.timestamp
        elif action == "completed":
            status = "complete"
            completed_at = event.timestamp
        elif action == "failed":
            status = "failed"
            failed_at = event.timestamp
        event_pid = _optional_int(event.metadata.get("pid"))
        nodes[name] = DependencyNode(
            name=name,
            status=status,
            pid=current.pid if event_pid is None else event_pid,
            started_at=started_at,
            completed_at=completed_at,
            failed_at=failed_at,
        )

    edges: dict[tuple[str, str, str | None], DependencyEdge] = {}
    for connection in snapshot.connections:
        relation = connection.latest_message_type
        key = (connection.sender, connection.receiver, relation)
        edges[key] = DependencyEdge(
            source=connection.sender,
            target=connection.receiver,
            relation=relation,
            message_count=connection.message_count,
            latest_message_type=connection.latest_message_type,
        )
        nodes.setdefault(connection.sender, DependencyNode(connection.sender))
        nodes.setdefault(connection.receiver, DependencyNode(connection.receiver))

    for row in rows:
        target = str(row.get("name", "")).strip()
        supervisor_pid = _optional_int(row.get("supervisor_pid"))
        source = pid_names.get(supervisor_pid) if supervisor_pid is not None else None
        if not source or not target:
            continue
        key = (source, target, "supervision")
        edges.setdefault(key, DependencyEdge(source, target, "supervision"))

    return DependencyGraphSnapshot(
        nodes=tuple(sorted(nodes.values(), key=lambda node: (node.name, node.pid or -1))),
        edges=tuple(
            edges[key]
            for key in sorted(edges, key=lambda item: (item[0], item[1], item[2] or ""))
        ),
    )


def format_dependency_node(node: DependencyNode) -> str:
    """Format one compact graph node row."""
    row = f"  {node.name:<18} {node.status:<10}"
    return f"{row} pid={node.pid}" if node.pid is not None else row.rstrip()


def format_dependency_edge(edge: DependencyEdge) -> str:
    """Format one compact graph edge row."""
    parts = [f"  {edge.source:<18} -> {edge.target:<18}"]
    if edge.relation:
        parts.append(f"type={edge.relation}")
    if edge.message_count is not None:
        parts.append(f"msgs={edge.message_count}")
    if edge.latest_message_type:
        parts.append(f"latest={edge.latest_message_type}")
    return "  ".join(parts)


def render_dependency_graph(snapshot: DependencyGraphSnapshot) -> list[str]:
    """Render a stable node and edge list, including a deterministic empty state."""
    if not snapshot.nodes and not snapshot.edges:
        return ["No dependency graph available yet."]
    rows = ["Nodes:"]
    rows.extend(format_dependency_node(node) for node in snapshot.nodes)
    rows.append("Edges:")
    rows.extend(format_dependency_edge(edge) for edge in snapshot.edges)
    return rows


def _process_status(status: str, *, external: bool) -> str:
    if status == "crashed":
        return "failed"
    if external and status == "exited":
        return "complete"
    return status if status in {"loaded", "running", "complete", "completed", "failed"} else "unknown"


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
