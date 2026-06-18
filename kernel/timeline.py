"""Compact runtime timeline rendering helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import timezone
from typing import Any

from kernel.events import RuntimeEvent


_SUBJECT_KEYS = (
    "agent",
    "agent_name",
    "agent_id",
    "tool_name",
    "name",
    "process",
    "child",
    "worker",
)
_METADATA_KEYS = (
    "round_index",
    "step_index",
    "success",
    "reason",
    "tool_count",
    "tool_call_count",
    "tool_result_count",
    "duration_ms",
    "error_type",
    "error_category",
    "pid",
    "exit_code",
    "page",
    "error",
)


def format_timeline_event(event: RuntimeEvent) -> str:
    """Format one structured event as a compact UTC timeline row."""
    timestamp = event.timestamp.astimezone(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    category, action = _event_parts(event)
    subject = _event_subject(event)
    metadata = _metadata_summary(event.metadata)
    row = f"{timestamp}  {category:<16} {subject:<18} {action}"
    return f"{row}  {metadata}" if metadata else row


def render_runtime_timeline(events: Iterable[Any], limit: int | None = None) -> list[str]:
    """Render structured events chronologically and legacy values safely."""
    structured: list[tuple[int, RuntimeEvent]] = []
    legacy: list[tuple[int, Any]] = []
    for index, event in enumerate(events):
        if isinstance(event, RuntimeEvent):
            structured.append((index, event))
        else:
            legacy.append((index, event))

    structured.sort(key=lambda item: (item[1].timestamp, item[0]))
    rows = [format_timeline_event(event) for _, event in structured]
    rows.extend(_format_legacy_event(event) for _, event in legacy)
    if limit is None:
        return rows
    if limit <= 0:
        return []
    return rows[-limit:]


def _event_parts(event: RuntimeEvent) -> tuple[str, str]:
    event_type = event.event_type
    prefixes = (
        ("external_agent_", "external_agent"),
        ("page_", "memory"),
        ("child_", "supervisor"),
    )
    for prefix, category in prefixes:
        if event_type.startswith(prefix):
            return category, event_type[len(prefix) :]
    return _normalize(event.source), _normalize(event_type)


def _event_subject(event: RuntimeEvent) -> str:
    for key in _SUBJECT_KEYS:
        value = event.metadata.get(key)
        if _is_short_scalar(value):
            return _normalize(str(value))
    return _normalize(event.source)


def _metadata_summary(metadata: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in _METADATA_KEYS:
        value = metadata.get(key)
        if key in _SUBJECT_KEYS or not _is_short_scalar(value):
            continue
        text = str(value).replace("\n", " ")
        if len(text) > 40:
            text = f"{text[:37]}..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def _format_legacy_event(event: Any) -> str:
    if isinstance(event, str):
        return event
    if isinstance(event, Mapping):
        event_type = event.get("event", event.get("event_type", "legacy"))
        message = event.get("message", "")
        return " ".join(part for part in (str(event_type), str(message)) if part).strip()
    try:
        return str(event)
    except Exception:
        return f"<{type(event).__name__}>"


def _is_short_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and len(str(value)) <= 120


def _normalize(value: str) -> str:
    value = re.sub(r"(?<!^)(?=[A-Z])", "_", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower() or "-"
