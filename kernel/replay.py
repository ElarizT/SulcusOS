"""Deterministic logical replay of recorded runtime events."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from kernel.events import RuntimeEvent, RuntimeEventLog


_SUBJECT_KEYS = ("agent", "name", "process", "child", "worker")
_METADATA_KEYS = (
    ("pid", "pid"),
    ("exit_code", "exit"),
    ("message_count", "msgs"),
    ("messages", "msgs"),
    ("duration_ms", "duration_ms"),
)


@dataclass(frozen=True)
class ReplayRecord:
    timestamp: datetime
    category: str
    source: str
    action: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplaySnapshot:
    records: tuple[ReplayRecord, ...] = ()


@dataclass(frozen=True)
class ReplaySession:
    records: tuple[ReplayRecord, ...] = ()

    def snapshot(self) -> ReplaySnapshot:
        return ReplaySnapshot(self.records)


def record_runtime_event(event: RuntimeEvent) -> ReplayRecord:
    """Convert an existing structured runtime event into a replay record."""
    if not isinstance(event, RuntimeEvent):
        raise TypeError("replay records require RuntimeEvent objects")
    return ReplayRecord(
        timestamp=event.timestamp,
        category=event.level,
        source=event.source,
        action=event.event_type,
        metadata=dict(event.metadata),
    )


def build_replay_session(
    events: RuntimeEventLog | Iterable[Any],
) -> ReplaySession:
    """Build a deterministic replay session from existing event storage."""
    source = events.events if isinstance(events, RuntimeEventLog) else events
    records = [
        (index, record_runtime_event(event))
        for index, event in enumerate(source)
        if isinstance(event, RuntimeEvent)
    ]
    records.sort(key=lambda item: (item[1].timestamp, item[0]))
    return ReplaySession(tuple(record for _, record in records))


def replay_events(
    session: ReplaySession | ReplaySnapshot,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> Iterator[ReplayRecord]:
    """Yield replay records in logical order without wall-clock delays."""
    start = max(offset, 0)
    if limit is not None and limit <= 0:
        return
    stop = None if limit is None else start + limit
    yield from session.records[start:stop]


def format_replay_record(record: ReplayRecord, index: int = 1) -> str:
    """Format one replay record as a compact, stable row."""
    source = _record_subject(record)
    action = _short_action(record.action)
    metadata = _metadata_summary(record.metadata)
    row = f"[{index:03d}] {source:<18} {action}"
    return f"{row}  {metadata}" if metadata else row


def render_replay_session(
    session: ReplaySession | ReplaySnapshot,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[str]:
    """Render replay rows while retaining their original session indexes."""
    start = max(offset, 0)
    return [
        format_replay_record(record, start + index)
        for index, record in enumerate(replay_events(session, offset=start, limit=limit), start=1)
    ]


def render_replay(
    events: RuntimeEventLog | ReplaySession | ReplaySnapshot | Iterable[Any],
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[str]:
    """Build and render a replay session from existing runtime events."""
    if isinstance(events, (ReplaySession, ReplaySnapshot)):
        session = events
    else:
        values = tuple(events.events if isinstance(events, RuntimeEventLog) else events)
        if all(isinstance(value, ReplayRecord) for value in values):
            session = ReplaySession(values)
        else:
            session = build_replay_session(values)
    return render_replay_session(session, offset=offset, limit=limit)


def _record_subject(record: ReplayRecord) -> str:
    for key in _SUBJECT_KEYS:
        value = record.metadata.get(key)
        if _is_short_scalar(value):
            return _normalize(str(value))
    return _normalize(record.source)


def _short_action(action: str) -> str:
    prefixes = ("external_agent_", "child_", "page_")
    for prefix in prefixes:
        if action.startswith(prefix):
            return _normalize(action.removeprefix(prefix))
    return _normalize(action)


def _metadata_summary(metadata: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, label in _METADATA_KEYS:
        value = metadata.get(key)
        if _is_short_scalar(value):
            parts.append(f"{label}={str(value).replace(chr(10), ' ')}")
    return " ".join(parts)


def _is_short_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and len(str(value)) <= 120


def _normalize(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower() or "-"
