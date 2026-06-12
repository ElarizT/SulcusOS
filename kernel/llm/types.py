"""Provider-neutral data structures for LLM requests and responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
