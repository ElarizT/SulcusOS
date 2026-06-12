"""Provider protocol and deterministic provider implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kernel.llm.types import LLMRequest, LLMResponse, LLMUsage


class LLMRuntimeError(RuntimeError):
    """Base error raised by the provider-neutral LLM runtime."""


class LLMProviderError(LLMRuntimeError):
    """Raised when an LLM provider cannot complete a request."""

    def __init__(self, *args: object, category: str = "provider") -> None:
        super().__init__(*args)
        self.category = category


def classify_llm_error(error: Exception) -> str:
    """Classify provider failures without exposing their message contents."""
    category = getattr(error, "category", None)
    if isinstance(category, str) and category:
        return category

    error_name = error.__class__.__name__.lower()
    if isinstance(error, TimeoutError) or "timeout" in error_name:
        return "timeout"
    if "ratelimit" in error_name or "rate_limit" in error_name:
        return "rate_limit"
    if isinstance(error, ImportError) or any(
        marker in error_name
        for marker in ("authentication", "configuration", "dependency")
    ):
        return "configuration"
    if isinstance(error, ConnectionError) or any(
        marker in error_name for marker in ("connection", "temporary", "transient")
    ):
        return "transient"
    if isinstance(error, LLMProviderError):
        return "provider"
    return "unknown"


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface implemented by LLM providers."""

    name: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete one structured LLM request."""


class DeterministicLLMProvider:
    """Predictable provider for tests, demos, and offline development."""

    name = "deterministic"

    def __init__(
        self,
        content: str = "Deterministic response.",
        *,
        fail: bool = False,
    ) -> None:
        self.content = content
        self.fail = fail
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.fail:
            raise LLMProviderError("deterministic provider failure")

        prompt_tokens = sum(_token_count(message.content) for message in request.messages)
        completion_tokens = _token_count(self.content)
        return LLMResponse(
            content=self.content,
            model=request.model,
            provider=self.name,
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metadata={"deterministic": True},
        )


class EchoLLMProvider(DeterministicLLMProvider):
    """Deterministic provider that returns the final message content."""

    name = "echo"

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.content = request.messages[-1].content
        return super().complete(request)


def _token_count(content: str) -> int:
    """Return a deliberately simple, deterministic token estimate."""
    return len(content.split())
