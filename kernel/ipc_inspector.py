"""Compact, read-only IPC communication inspection helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kernel.events import RuntimeEvent


@dataclass(frozen=True)
class IPCConnection:
    sender: str
    receiver: str
    message_count: int
    latest_message_type: str | None = None
    latest_timestamp: datetime | float | None = None
    pending_mailbox_size: int | None = None


@dataclass(frozen=True)
class IPCSnapshot:
    connections: tuple[IPCConnection, ...]


def build_ipc_snapshot(
    records: Iterable[Any] = (),
    *,
    process_rows: Iterable[Mapping[str, Any]] = (),
    mailbox_metrics: Iterable[Any] = (),
) -> IPCSnapshot:
    """Aggregate available IPC records into deterministic sender/receiver edges."""
    pid_names = {
        int(row["pid"]): str(row["name"])
        for row in process_rows
        if row.get("pid") is not None and row.get("name")
    }
    pending = _pending_mailboxes(mailbox_metrics)
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}

    for index, record in enumerate(records):
        values = _record_values(record, pid_names)
        if values is None:
            continue
        sender, receiver, message_type, timestamp, count = values
        current = aggregated.setdefault(
            (sender, receiver),
            {"count": 0, "latest_type": None, "latest_timestamp": None, "latest_order": -1},
        )
        current["count"] += count
        if _is_later(timestamp, index, current["latest_timestamp"], current["latest_order"]):
            current["latest_type"] = message_type
            current["latest_timestamp"] = timestamp
            current["latest_order"] = index

    return IPCSnapshot(
        tuple(
            IPCConnection(
                sender=sender,
                receiver=receiver,
                message_count=values["count"],
                latest_message_type=values["latest_type"],
                latest_timestamp=values["latest_timestamp"],
                pending_mailbox_size=pending.get(receiver),
            )
            for (sender, receiver), values in sorted(aggregated.items())
        )
    )


def format_ipc_connection(connection: IPCConnection) -> str:
    """Format one stable, compact IPC connection row."""
    parts = [
        f"{connection.sender:<18} -> {connection.receiver:<18}",
        f"msgs={connection.message_count}",
    ]
    if connection.latest_message_type:
        parts.append(f"latest={connection.latest_message_type}")
    if connection.latest_timestamp is not None:
        parts.append(f"at={_format_timestamp(connection.latest_timestamp)}")
    if connection.pending_mailbox_size is not None:
        parts.append(f"pending={connection.pending_mailbox_size}")
    return "  ".join(parts)


def render_ipc_inspector(snapshot: IPCSnapshot | Iterable[IPCConnection]) -> list[str]:
    connections = snapshot.connections if isinstance(snapshot, IPCSnapshot) else tuple(snapshot)
    return [format_ipc_connection(connection) for connection in connections]


def _record_values(
    record: Any,
    pid_names: Mapping[int, str],
) -> tuple[str, str, str | None, datetime | float | None, int] | None:
    if isinstance(record, RuntimeEvent):
        values: Mapping[str, Any] = record.metadata
        sender = values.get("sender", values.get("source"))
        receiver = values.get("receiver", values.get("target"))
        message_type = values.get("message_type", values.get("topic", record.event_type))
        timestamp: datetime | float | None = record.timestamp
        count = values.get("message_count", 1)
    elif isinstance(record, Mapping):
        values = record
        sender = values.get("sender", values.get("source", values.get("source_pid")))
        receiver = values.get("receiver", values.get("target", values.get("target_pid")))
        message_type = values.get("message_type", values.get("topic", values.get("type")))
        timestamp = values.get("timestamp")
        count = values.get("message_count", values.get("count", 1))
    else:
        sender = getattr(record, "sender", getattr(record, "source_pid", None))
        receiver = getattr(record, "receiver", getattr(record, "target_pid", None))
        message_type = getattr(record, "message_type", getattr(record, "type", None))
        timestamp = getattr(record, "timestamp", None)
        count = 1

    sender_name = _endpoint_name(sender, pid_names)
    receiver_name = _endpoint_name(receiver, pid_names)
    if not sender_name or not receiver_name:
        return None
    try:
        message_count = int(count)
    except (TypeError, ValueError):
        message_count = 1
    if message_count <= 0:
        return None
    return sender_name, receiver_name, _optional_text(message_type), _valid_timestamp(timestamp), message_count


def _pending_mailboxes(metrics: Iterable[Any]) -> dict[str, int]:
    pending: dict[str, int] = {}
    for metric in metrics:
        if isinstance(metric, Mapping):
            name = metric.get("agent_name", metric.get("name"))
            depth = metric.get("queue_depth", metric.get("pending"))
        else:
            name = getattr(metric, "agent_name", None)
            depth = getattr(metric, "queue_depth", None)
            if name is None and isinstance(metric, (tuple, list)) and len(metric) >= 2:
                name, depth = metric[0], metric[1]
        if name is None or depth is None:
            continue
        try:
            pending[str(name)] = int(depth)
        except (TypeError, ValueError):
            continue
    return pending


def _endpoint_name(value: Any, pid_names: Mapping[int, str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return pid_names.get(value, str(value))
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return pid_names.get(int(text), text)
    return text


def _is_later(
    timestamp: datetime | float | None,
    order: int,
    current_timestamp: datetime | float | None,
    current_order: int,
) -> bool:
    if timestamp is None and current_timestamp is not None:
        return False
    if timestamp is not None and current_timestamp is None:
        return True
    if timestamp is None:
        return order > current_order
    return _timestamp_key(timestamp) >= _timestamp_key(current_timestamp)


def _timestamp_key(timestamp: datetime | float) -> float:
    return timestamp.timestamp() if isinstance(timestamp, datetime) else float(timestamp)


def _valid_timestamp(value: Any) -> datetime | float | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_timestamp(timestamp: datetime | float) -> str:
    value = timestamp if isinstance(timestamp, datetime) else datetime.fromtimestamp(timestamp, timezone.utc)
    return value.astimezone(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
