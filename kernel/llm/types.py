"""Provider-neutral data structures for LLM requests and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.messages:
            raise ValueError("LLM request must contain at least one message")
        if not self.model.strip():
            raise ValueError("LLM request model must not be empty")
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.model.strip():
            raise ValueError("LLM response model must not be empty")
        if not self.provider.strip():
            raise ValueError("LLM response provider must not be empty")


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
