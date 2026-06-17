"""Provider-neutral tool execution data structures."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any

from kernel.llm.types import LLMToolCall, LLMToolDefinition, LLMToolResult


class ToolRuntimeError(RuntimeError):
    """Base error raised by the tool execution runtime."""


class ToolRegistrationError(ToolRuntimeError):
    """Raised when a tool cannot be registered."""


class ToolValidationError(ToolRuntimeError):
    """Raised when a tool call cannot be validated."""


@dataclass(frozen=True)
class ToolDefinition:
    """Approved callable tool registered with Agent OS."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    func: Callable[..., Any]
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)
        if not isinstance(self.description, str):
            raise ValueError("tool description must be a string")
        if not isinstance(self.parameters_schema, Mapping):
            raise ValueError("tool parameters_schema must be a mapping")
        if not callable(self.func):
            raise ValueError("tool func must be callable")
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise ValueError("tool timeout_seconds must be a positive number")
            if self.timeout_seconds <= 0:
                raise ValueError("tool timeout_seconds must be positive")
        object.__setattr__(
            self,
            "parameters_schema",
            deepcopy(dict(self.parameters_schema)),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_llm_tool_definition(self) -> LLMToolDefinition:
        """Return the provider-neutral LLM tool definition for this tool."""
        return LLMToolDefinition(
            name=self.name,
            description=self.description,
            parameters_schema=deepcopy(self.parameters_schema),
        )


@dataclass(frozen=True)
class ToolCallRequest:
    """Explicit request to execute one registered tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool call arguments must be a mapping")
        if self.tool_call_id is not None and (
            not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip()
        ):
            raise ValueError("tool_call_id must be a nonempty string")
        object.__setattr__(self, "arguments", deepcopy(dict(self.arguments)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_llm_tool_call(cls, tool_call: LLMToolCall) -> ToolCallRequest:
        """Create an execution request from a Step 33 LLM tool call."""
        if not isinstance(tool_call, LLMToolCall):
            raise TypeError("tool_call must be an LLMToolCall")
        metadata = dict(tool_call.metadata)
        if tool_call.provider:
            metadata["provider"] = tool_call.provider
        if tool_call.model:
            metadata["model"] = tool_call.model
        return cls(
            name=tool_call.name,
            arguments=tool_call.arguments,
            tool_call_id=tool_call.id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured result of an explicit registered tool execution."""

    name: str
    content: Any = None
    success: bool = True
    tool_call_id: str | None = None
    error: str | None = None
    error_type: str | None = None
    error_category: str | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_tool_name(self.name)
        if not isinstance(self.success, bool):
            raise ValueError("tool execution success must be a boolean")
        if self.tool_call_id is not None and (
            not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip()
        ):
            raise ValueError("tool_call_id must be a nonempty string")
        for field_name in ("error", "error_type", "error_category"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
        if self.duration_ms is not None:
            if (
                isinstance(self.duration_ms, bool)
                or not isinstance(self.duration_ms, int)
                or self.duration_ms < 0
            ):
                raise ValueError("duration_ms must be a nonnegative integer")
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_llm_tool_result(self) -> LLMToolResult:
        """Convert the execution result into the Step 33 LLM tool result shape."""
        return LLMToolResult(
            tool_call_id=self.tool_call_id or self.name,
            name=self.name,
            content=_content_to_text(self.content),
            success=self.success,
            error=self.error,
        )


def tool_definition_from_llm(
    tool: LLMToolDefinition,
    *,
    func: Callable[..., Any],
    timeout_seconds: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ToolDefinition:
    """Attach an approved callable to an LLM tool definition."""
    if not isinstance(tool, LLMToolDefinition):
        raise TypeError("tool must be an LLMToolDefinition")
    return ToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters_schema=tool.parameters_schema,
        func=func,
        timeout_seconds=timeout_seconds,
        metadata=dict(metadata or {}),
    )


def tool_call_request_from_llm(tool_call: LLMToolCall) -> ToolCallRequest:
    """Create a tool execution request from an LLM tool call."""
    return ToolCallRequest.from_llm_tool_call(tool_call)


def _validate_tool_name(name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("tool name must not be empty")
    normalized = name.strip()
    if normalized != name:
        raise ValueError("tool name must not contain surrounding whitespace")
    if not all(character.isalnum() or character in {"_", "-", "."} for character in name):
        raise ValueError("tool name contains unsupported characters")


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=True, sort_keys=True)
    except TypeError:
        pass
    return str(content)
