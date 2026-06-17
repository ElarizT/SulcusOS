"""Provider protocol and deterministic provider implementations."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Protocol, runtime_checkable

from kernel.llm.types import LLMRequest, LLMResponse, LLMStreamChunk, LLMUsage


class LLMRuntimeError(RuntimeError):
    """Base error raised by the provider-neutral LLM runtime."""


class LLMBudgetExceededError(LLMRuntimeError):
    """Raised when an LLM token budget guardrail is exceeded."""


class LLMProviderError(LLMRuntimeError):
    """Raised when an LLM provider cannot complete a request."""

    def __init__(self, *args: object, category: str = "provider") -> None:
        super().__init__(*args)
        self.category = category


class LLMStreamingUnsupportedError(LLMProviderError):
    """Raised when a provider does not opt into streaming support."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args, category="unsupported")


def classify_llm_error(error: Exception) -> str:
    """Classify provider failures without exposing their message contents."""
    category = getattr(error, "category", None)
    if isinstance(category, str) and category:
        return category

    status_code = _error_status_code(error)
    if status_code is not None:
        if status_code in {408, 504}:
            return "timeout"
        if status_code == 429:
            return "rate_limit"
        if status_code in {401, 403}:
            return "configuration"
        if 400 <= status_code < 500:
            return "request"
        if 500 <= status_code < 600:
            return "transient"

    error_name = error.__class__.__name__.lower()
    if isinstance(error, TimeoutError) or "timeout" in error_name:
        return "timeout"
    if "ratelimit" in error_name or "rate_limit" in error_name:
        return "rate_limit"
    if isinstance(error, ImportError) or any(
        marker in error_name
        for marker in (
            "authentication",
            "configuration",
            "dependency",
            "permission",
            "unauthorized",
        )
    ):
        return "configuration"
    if any(
        marker in error_name
        for marker in (
            "badrequest",
            "notfound",
            "not_found",
            "invalidrequest",
            "invalid_request",
            "invalid",
            "unprocessable",
            "unsupported",
        )
    ):
        return "request"
    if isinstance(error, ConnectionError) or any(
        marker in error_name for marker in ("connection", "temporary", "transient")
    ):
        return "transient"
    if isinstance(error, LLMProviderError):
        return "provider"
    return "unknown"


def _error_status_code(error: Exception) -> int | None:
    for value in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value > 0:
            return value
    return None


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface implemented by LLM providers."""

    name: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete one structured LLM request."""


@runtime_checkable
class StreamingLLMProvider(Protocol):
    """Optional interface implemented by providers that support streaming."""

    name: str

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        """Stream one structured LLM request."""


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


class DeterministicStreamingLLMProvider(DeterministicLLMProvider):
    """Predictable streaming provider for tests and offline development."""

    supports_streaming = True

    def __init__(
        self,
        chunks: Sequence[str] = ("Hello", " ", "world"),
        *,
        fail: bool = False,
    ) -> None:
        self.chunks = tuple(chunks)
        super().__init__("".join(self.chunks), fail=fail)

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        self.requests.append(request)
        if self.fail:
            raise LLMProviderError("deterministic streaming provider failure")

        for index, delta in enumerate(self.chunks):
            yield LLMStreamChunk(
                delta=delta,
                index=index,
                provider=self.name,
                model=request.model,
                metadata={"deterministic": True},
            )

        prompt_tokens = sum(_token_count(message.content) for message in request.messages)
        completion_tokens = _token_count(self.content)
        yield LLMStreamChunk(
            delta="",
            index=len(self.chunks),
            provider=self.name,
            model=request.model,
            done=True,
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            metadata={"deterministic": True},
        )


def _token_count(content: str) -> int:
    """Return a deliberately simple, deterministic token estimate."""
    return len(content.split())
