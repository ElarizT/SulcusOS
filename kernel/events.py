"""Structured runtime events for Agent OS observability."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


EVENT_LEVELS = frozenset({"INFO", "WARNING", "ERROR", "DEBUG"})


@dataclass(frozen=True)
class RuntimeEvent:
    timestamp: datetime
    level: str
    source: str
    event_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("runtime event timestamp must be timezone-aware")
        if self.level not in EVENT_LEVELS:
            raise ValueError(f"unsupported runtime event level: {self.level}")

    @classmethod
    def info(
        cls,
        source: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls.now("INFO", source, event_type, message, metadata)

    @classmethod
    def warning(
        cls,
        source: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls.now("WARNING", source, event_type, message, metadata)

    @classmethod
    def error(
        cls,
        source: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls.now("ERROR", source, event_type, message, metadata)

    @classmethod
    def debug(
        cls,
        source: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls.now("DEBUG", source, event_type, message, metadata)

    @classmethod
    def now(
        cls,
        level: str,
        source: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            timestamp=datetime.now(timezone.utc),
            level=level,
            source=source,
            event_type=event_type,
            message=message,
            metadata=dict(metadata or {}),
        )


class RuntimeEventLog:
    """Small append-only in-memory runtime event container."""

    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    def append(self, event: RuntimeEvent) -> None:
        self.events.append(event)

    def extend(self, events: Iterable[RuntimeEvent]) -> None:
        self.events.extend(events)

    def latest(self, limit: int | None = None) -> list[RuntimeEvent]:
        if limit is None:
            return list(self.events)
        if limit <= 0:
            return []
        return list(self.events[-limit:])

    def by_level(self, level: str) -> list[RuntimeEvent]:
        return [event for event in self.events if event.level == level]

    def by_type(self, event_type: str) -> list[RuntimeEvent]:
        return [event for event in self.events if event.event_type == event_type]


def render_runtime_event(event: RuntimeEvent) -> str:
    """Render an event as one readable UTC dashboard log line."""
    timestamp = event.timestamp.astimezone(timezone.utc).strftime("%H:%M:%S")
    return f"{timestamp} {event.level:<8} {event.source:<18} {event.message}"

