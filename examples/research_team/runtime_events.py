from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from kernel.events import RuntimeEvent


EventSink = Callable[[RuntimeEvent], None]


def emit_demo_event(
    sink: EventSink,
    event_type: str,
    subject: str,
    *,
    level: str = "INFO",
    error_type: str | None = None,
) -> None:
    """Record one timestamped research-demo milestone with safe metadata."""
    metadata = {"agent": subject}
    if error_type is not None:
        metadata["error_type"] = error_type
    sink(
        RuntimeEvent.now(
            level,
            "ResearchTeamDemo",
            event_type,
            event_type.replace("_", " ").capitalize(),
            metadata,
        )
    )


@contextmanager
def record_agent_work(agent: Any) -> Iterator[None]:
    """Record the real execution boundary of one demo agent's unit of work."""
    sink = getattr(agent, "runtime_event_sink", None)
    if not callable(sink):
        yield
        return

    subject = type(agent).__name__
    emit_demo_event(sink, "agent_work_started", subject)
    try:
        yield
    except Exception as exc:
        emit_demo_event(
            sink,
            "agent_work_failed",
            subject,
            level="ERROR",
            error_type=type(exc).__name__,
        )
        raise
    else:
        emit_demo_event(sink, "agent_work_completed", subject)
