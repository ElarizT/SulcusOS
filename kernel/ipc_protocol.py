from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar


PROTOCOL_VERSION = "0.1"
MAX_MESSAGE_BYTES = 64 * 1024
PRIORITIES = {"low", "normal", "high", "critical"}


class IPCProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def now_timestamp() -> float:
    return time.time()


def new_message_id() -> str:
    return str(uuid.uuid4())


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise IPCProtocolError("invalid_message", "payload must be JSON serializable") from exc


def validate_pid(pid: int, field_name: str) -> None:
    if not isinstance(pid, int) or pid <= 0:
        raise IPCProtocolError("invalid_message", f"{field_name} must be a positive integer PID")


@dataclass(frozen=True)
class IPCMessage:
    message_type: ClassVar[str] = "message"

    source_pid: int
    target_pid: int
    payload: Any
    message_id: str = field(default_factory=new_message_id)
    correlation_id: str | None = None
    timestamp: float = field(default_factory=now_timestamp)
    priority: str = "normal"
    expires_at: float | None = None
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def type(self) -> str:
        return self.message_type

    def validate(self) -> None:
        validate_pid(self.source_pid, "source_pid")
        validate_pid(self.target_pid, "target_pid")
        if self.priority not in PRIORITIES:
            raise IPCProtocolError("invalid_message", f"unsupported priority: {self.priority}")
        if self.protocol_version != PROTOCOL_VERSION:
            raise IPCProtocolError("invalid_message", f"unsupported protocol version: {self.protocol_version}")
        if not isinstance(self.message_id, str) or not self.message_id:
            raise IPCProtocolError("invalid_message", "message_id is required")
        if self.correlation_id is not None and not isinstance(self.correlation_id, str):
            raise IPCProtocolError("invalid_message", "correlation_id must be a string")
        if self.expires_at is not None and not isinstance(self.expires_at, (int, float)):
            raise IPCProtocolError("invalid_message", "expires_at must be a timestamp")
        size = self.size_bytes()
        if size > MAX_MESSAGE_BYTES:
            raise IPCProtocolError("payload_too_large", f"message payload exceeds {MAX_MESSAGE_BYTES} bytes")

    def is_expired(self) -> bool:
        return self.expires_at is not None and now_timestamp() >= self.expires_at

    def size_bytes(self) -> int:
        return _json_size(self.to_dict(validate_size=False))

    def to_dict(self, *, validate_size: bool = True) -> dict[str, Any]:
        data = asdict(self)
        data["message_type"] = self.message_type
        if validate_size:
            _json_size(data)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class TaskRequest(IPCMessage):
    message_type: ClassVar[str] = "task_request"

    def validate(self) -> None:
        super().validate()
        if not self.correlation_id:
            raise IPCProtocolError("invalid_message", "correlation_id is required for task_request")


@dataclass(frozen=True)
class TaskResponse(IPCMessage):
    message_type: ClassVar[str] = "task_response"

    def validate(self) -> None:
        super().validate()
        if not self.correlation_id:
            raise IPCProtocolError("invalid_message", "correlation_id is required for task_response")


@dataclass(frozen=True)
class EventMessage(IPCMessage):
    message_type: ClassVar[str] = "event"


@dataclass(frozen=True)
class ErrorMessage(IPCMessage):
    message_type: ClassVar[str] = "error"

    @property
    def error_code(self) -> str:
        if isinstance(self.payload, dict):
            return str(self.payload.get("code", "unknown_error"))
        return "unknown_error"


@dataclass(frozen=True)
class HeartbeatMessage(IPCMessage):
    message_type: ClassVar[str] = "heartbeat"


@dataclass(frozen=True)
class ControlMessage(IPCMessage):
    message_type: ClassVar[str] = "control"


MESSAGE_TYPES: dict[str, type[IPCMessage]] = {
    TaskRequest.message_type: TaskRequest,
    TaskResponse.message_type: TaskResponse,
    EventMessage.message_type: EventMessage,
    ErrorMessage.message_type: ErrorMessage,
    HeartbeatMessage.message_type: HeartbeatMessage,
    ControlMessage.message_type: ControlMessage,
}


def message_class(message_type: str) -> type[IPCMessage]:
    try:
        return MESSAGE_TYPES[message_type]
    except KeyError as exc:
        raise IPCProtocolError("invalid_message", f"unsupported message_type: {message_type}") from exc


def make_message(
    *,
    source_pid: int,
    target_pid: int,
    payload: Any,
    message_type: str = TaskRequest.message_type,
    priority: str = "normal",
    correlation_id: str | None = None,
    ttl: float | None = None,
) -> IPCMessage:
    cls = message_class(message_type)
    if correlation_id is None and cls in {TaskRequest, TaskResponse}:
        correlation_id = new_message_id()
    expires_at = now_timestamp() + ttl if ttl is not None else None
    return cls(
        source_pid=source_pid,
        target_pid=target_pid,
        payload=payload,
        priority=priority,
        correlation_id=correlation_id,
        expires_at=expires_at,
    )


def parse_message(raw: str | dict[str, Any] | IPCMessage) -> IPCMessage:
    if isinstance(raw, IPCMessage):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IPCProtocolError("invalid_message", "message payload must be valid JSON") from exc
    else:
        data = dict(raw)
    message_type = data.pop("message_type", data.pop("type", None))
    if not isinstance(message_type, str):
        raise IPCProtocolError("invalid_message", "message_type is required")
    cls = message_class(message_type)
    return cls(**data)


def make_error(
    *,
    source_pid: int,
    target_pid: int,
    code: str,
    message: str,
    correlation_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> ErrorMessage:
    return ErrorMessage(
        source_pid=source_pid,
        target_pid=target_pid,
        correlation_id=correlation_id,
        payload={
            "code": code,
            "message": message,
            "details": details or {},
        },
    )
