"""Provider-neutral data structures for LLM requests and responses."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class LLMTokenBudget:
    """Optional cumulative token limits for one LLM runtime."""

    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None
    name: str | None = None
    strict: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_total_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name.strip():
                raise ValueError("token budget name must be a nonempty string")
        if not isinstance(self.strict, bool):
            raise ValueError("token budget strict must be a boolean")


@dataclass(frozen=True)
class LLMRetryPolicy:
    """Deterministic retry settings applied to each provider attempt."""

    max_attempts: int = 1
    retry_on: tuple[str, ...] = ("timeout", "rate_limit", "transient")
    backoff_seconds: float = 0.0
    max_backoff_seconds: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.retry_on, str):
            raise ValueError("retry_on must be a sequence of categories")
        object.__setattr__(self, "retry_on", tuple(self.retry_on))
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if isinstance(self.backoff_seconds, bool) or not isinstance(
            self.backoff_seconds, (int, float)
        ):
            raise ValueError("backoff_seconds must be a nonnegative number")
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must not be negative")
        if self.max_backoff_seconds is not None:
            if isinstance(self.max_backoff_seconds, bool) or not isinstance(
                self.max_backoff_seconds, (int, float)
            ):
                raise ValueError("max_backoff_seconds must be a nonnegative number")
            if self.max_backoff_seconds < 0:
                raise ValueError("max_backoff_seconds must not be negative")
        if any(
            not isinstance(category, str) or not category.strip()
            for category in self.retry_on
        ):
            raise ValueError("retry_on categories must not be empty")


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("LLM message role must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class LLMToolParameter:
    """Provider-neutral description of one tool parameter."""

    name: str
    type: str
    description: str = ""
    required: bool = False
    enum: tuple[Any, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LLM tool parameter name must not be empty")
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("LLM tool parameter type must not be empty")
        if not isinstance(self.description, str):
            raise ValueError("LLM tool parameter description must be a string")
        if not isinstance(self.required, bool):
            raise ValueError("LLM tool parameter required must be a boolean")
        if self.enum is not None:
            if isinstance(self.enum, (str, bytes)):
                raise ValueError("LLM tool parameter enum must be a sequence")
            try:
                object.__setattr__(self, "enum", tuple(self.enum))
            except TypeError:
                raise ValueError("LLM tool parameter enum must be a sequence") from None
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LLMToolDefinition:
    """Provider-neutral structured tool definition exposed to an LLM."""

    name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LLM tool name must not be empty")
        if not isinstance(self.description, str):
            raise ValueError("LLM tool description must be a string")
        if not isinstance(self.parameters_schema, Mapping):
            raise ValueError("LLM tool parameters_schema must be a mapping")
        object.__setattr__(self, "parameters_schema", deepcopy(dict(self.parameters_schema)))


@dataclass(frozen=True)
class LLMToolCall:
    """Provider-neutral structured tool-call request returned by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    provider: str = ""
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("LLM tool call id must not be empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LLM tool call name must not be empty")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("LLM tool call arguments must be a mapping")
        for field_name in ("provider", "model"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"LLM tool call {field_name} must be a string")
        object.__setattr__(self, "arguments", deepcopy(dict(self.arguments)))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LLMToolResult:
    """Provider-neutral result shape for future explicit tool execution."""

    tool_call_id: str
    name: str
    content: str
    success: bool = True
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
            raise ValueError("LLM tool result tool_call_id must not be empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("LLM tool result name must not be empty")
        if not isinstance(self.content, str):
            raise ValueError("LLM tool result content must be a string")
        if not isinstance(self.success, bool):
            raise ValueError("LLM tool result success must be a boolean")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("LLM tool result error must be a string")


@dataclass(frozen=True)
class LLMUsageLedger:
    """Cumulative provider-reported token usage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def apply_usage_to_ledger(
    ledger: LLMUsageLedger,
    usage: LLMUsage | None,
) -> LLMUsageLedger:
    """Return a new ledger containing only known provider-reported usage."""
    if usage is None:
        return ledger
    return LLMUsageLedger(
        prompt_tokens=ledger.prompt_tokens + (usage.prompt_tokens or 0),
        completion_tokens=ledger.completion_tokens + (usage.completion_tokens or 0),
        total_tokens=ledger.total_tokens + (usage.total_tokens or 0),
    )


def check_token_budget(
    budget: LLMTokenBudget,
    ledger: LLMUsageLedger,
) -> tuple[str, ...]:
    """Return the cumulative budget categories exceeded by a ledger."""
    exceeded: list[str] = []
    if (
        budget.max_prompt_tokens is not None
        and ledger.prompt_tokens > budget.max_prompt_tokens
    ):
        exceeded.append("prompt")
    if (
        budget.max_completion_tokens is not None
        and ledger.completion_tokens > budget.max_completion_tokens
    ):
        exceeded.append("completion")
    if budget.max_total_tokens is not None and ledger.total_tokens > budget.max_total_tokens:
        exceeded.append("total")
    return tuple(exceeded)


def format_usage_ledger(ledger: LLMUsageLedger) -> str:
    """Format cumulative known usage deterministically."""
    return (
        f"prompt={ledger.prompt_tokens}, "
        f"completion={ledger.completion_tokens}, "
        f"total={ledger.total_tokens}"
    )


@dataclass(frozen=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    model: str
    temperature: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float | None = None
    tools: tuple[LLMToolDefinition, ...] = ()
    tool_choice: str | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "tools", tuple(self.tools))
        if not self.messages:
            raise ValueError("LLM request must contain at least one message")
        if not self.model.strip():
            raise ValueError("LLM request model must not be empty")
        for tool in self.tools:
            if not isinstance(tool, LLMToolDefinition):
                raise ValueError("LLM request tools must be LLMToolDefinition objects")
        if self.tool_choice is not None:
            if isinstance(self.tool_choice, str):
                if not self.tool_choice.strip():
                    raise ValueError("LLM request tool_choice must not be empty")
            elif isinstance(self.tool_choice, Mapping):
                object.__setattr__(self, "tool_choice", deepcopy(dict(self.tool_choice)))
            else:
                raise ValueError("LLM request tool_choice must be a string or mapping")
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                raise ValueError("timeout_seconds must be a positive number")
            if self.timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: tuple[LLMToolCall, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if not self.model.strip():
            raise ValueError("LLM response model must not be empty")
        if not self.provider.strip():
            raise ValueError("LLM response provider must not be empty")
        for tool_call in self.tool_calls:
            if not isinstance(tool_call, LLMToolCall):
                raise ValueError("LLM response tool_calls must be LLMToolCall objects")


@dataclass(frozen=True)
class LLMStreamChunk:
    """One immutable partial response from a streaming LLM provider."""

    delta: str
    index: int
    provider: str | None = None
    model: str | None = None
    done: bool = False
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.delta, str):
            raise ValueError("LLM stream chunk delta must be a string")
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("LLM stream chunk index must be a nonnegative integer")
        for name in ("provider", "model"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"LLM stream chunk {name} must not be empty")
        if not isinstance(self.done, bool):
            raise ValueError("LLM stream chunk done must be a boolean")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class LLMStreamResult:
    """Optional collected representation of a completed LLM stream."""

    content: str
    chunks: tuple[LLMStreamChunk, ...]
    model: str
    provider: str
    usage: LLMUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunks", tuple(self.chunks))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.model.strip():
            raise ValueError("LLM stream result model must not be empty")
        if not self.provider.strip():
            raise ValueError("LLM stream result provider must not be empty")
