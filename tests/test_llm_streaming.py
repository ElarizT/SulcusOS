from __future__ import annotations

from collections.abc import Iterator

import pytest

from kernel.events import RuntimeEventLog
from kernel.llm import (
    DeterministicLLMProvider,
    DeterministicStreamingLLMProvider,
    LLMBudgetExceededError,
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMResponseCache,
    LLMRetryPolicy,
    LLMRuntime,
    LLMStreamChunk,
    LLMStreamResult,
    LLMStreamingUnsupportedError,
    LLMTokenBudget,
    LLMUsage,
    LLMUsageLedger,
    OpenAICompatibleProvider,
    StreamingLLMProvider,
)


class RecordingStreamingProvider:
    supports_streaming = True

    def __init__(
        self,
        name: str,
        chunks: tuple[LLMStreamChunk, ...] = (),
        *,
        fail_before: Exception | None = None,
        fail_after: Exception | None = None,
    ) -> None:
        self.name = name
        self.chunks = chunks
        self.fail_before = fail_before
        self.fail_after = fail_after
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest):  # pragma: no cover - streaming only
        raise AssertionError("complete must not be used for streaming")

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        self.requests.append(request)
        if self.fail_before is not None:
            raise self.fail_before
        yield from self.chunks
        if self.fail_after is not None:
            raise self.fail_after


class RetryStreamingProvider(RecordingStreamingProvider):
    def __init__(self, name: str, outcomes: list[Exception | tuple[LLMStreamChunk, ...]]):
        super().__init__(name)
        self.outcomes = list(outcomes)

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        yield from outcome


def test_streaming_types_and_public_protocol_are_stable() -> None:
    chunk = LLMStreamChunk("Hello", 0, metadata={"safe": True})
    result = LLMStreamResult("Hello", (chunk,), "model", "provider")
    provider = DeterministicStreamingLLMProvider()

    assert result.chunks == (chunk,)
    assert isinstance(provider, StreamingLLMProvider)


def test_stream_chat_yields_ordered_deterministic_chunks_and_final_usage() -> None:
    provider = DeterministicStreamingLLMProvider(("Hello", " ", "world"))
    runtime = LLMRuntime(provider)

    chunks = list(
        runtime.stream_chat([LLMMessage("user", "private prompt")], model="model")
    )

    assert [chunk.delta for chunk in chunks] == ["Hello", " ", "world", ""]
    assert [chunk.index for chunk in chunks] == [0, 1, 2, 3]
    assert chunks[-1].done is True
    assert chunks[-1].usage == LLMUsage(2, 2, 4)
    assert runtime.usage_snapshot() == LLMUsageLedger(2, 2, 4)
    assert provider.requests[0].model == "model"


def test_provider_without_streaming_support_fails_cleanly() -> None:
    runtime = LLMRuntime(DeterministicLLMProvider())

    with pytest.raises(LLMStreamingUnsupportedError, match="does not support streaming"):
        list(runtime.stream_chat([LLMMessage("user", "private")], model="model"))


def test_openai_compatible_streaming_is_explicitly_unsupported_offline() -> None:
    provider = OpenAICompatibleProvider(client=object())
    runtime = LLMRuntime(provider)

    with pytest.raises(LLMStreamingUnsupportedError, match="does not support streaming"):
        list(runtime.stream_chat([LLMMessage("user", "private")], model="model"))


def test_stream_lifecycle_events_are_safe_and_do_not_log_text() -> None:
    prompt = "private-stream-prompt"
    delta = "private-stream-delta"
    usage = LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    provider = RecordingStreamingProvider(
        "primary",
        (
            LLMStreamChunk(delta, 0),
            LLMStreamChunk("", 1, done=True, usage=usage),
        ),
    )
    events = RuntimeEventLog()
    runtime = LLMRuntime(provider, events)

    list(runtime.stream_chat([LLMMessage("user", prompt)], model="model"))

    assert [event.event_type for event in events.events] == [
        "llm.stream_requested",
        "llm.stream_started",
        "llm.stream_chunk",
        "llm.stream_chunk",
        "llm.stream_completed",
    ]
    assert events.events[2].metadata == {
        "provider": "primary",
        "model": "model",
        "chunk_index": 0,
        "delta_chars": len(delta),
        "done": False,
    }
    assert events.events[3].metadata == {
        "provider": "primary",
        "model": "model",
        "chunk_index": 1,
        "delta_chars": 0,
        "done": True,
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert prompt not in repr(events.events)
    assert delta not in repr(events.events)


def test_stream_timeout_is_passed_through_runtime_and_request_override() -> None:
    provider = RecordingStreamingProvider("primary")
    runtime = LLMRuntime(provider, timeout_seconds=30)

    list(runtime.stream_chat([LLMMessage("user", "one")], model="model"))
    list(
        runtime.stream_chat(
            [LLMMessage("user", "two")],
            model="model",
            timeout_seconds=5,
        )
    )

    assert [request.timeout_seconds for request in provider.requests] == [30, 5]


def test_stream_routing_and_fallback_before_first_chunk() -> None:
    primary = RecordingStreamingProvider(
        "primary-provider",
        fail_before=LLMProviderError("private failure"),
    )
    fallback = RecordingStreamingProvider(
        "fallback-provider",
        (LLMStreamChunk("fallback", 0), LLMStreamChunk("", 1, done=True)),
    )
    runtime = LLMRuntime(
        providers={"primary": primary, "fallback": fallback},
        default_provider="primary",
        fallback_providers=["fallback"],
    )

    chunks = list(
        runtime.stream_chat([LLMMessage("user", "private")], model="model")
    )

    assert [chunk.delta for chunk in chunks] == ["fallback", ""]
    assert len(primary.requests) == 1
    assert len(fallback.requests) == 1


def test_stream_retries_when_provider_fails_before_first_chunk() -> None:
    provider = RetryStreamingProvider(
        "primary",
        [
            TimeoutError("private failure"),
            (LLMStreamChunk("success", 0), LLMStreamChunk("", 1, done=True)),
        ],
    )
    runtime = LLMRuntime(provider, retry_policy=LLMRetryPolicy(max_attempts=2))

    chunks = list(
        runtime.stream_chat([LLMMessage("user", "private")], model="model")
    )

    assert [chunk.delta for chunk in chunks] == ["success", ""]
    assert len(provider.requests) == 2


def test_stream_does_not_retry_or_fallback_after_first_chunk() -> None:
    primary = RecordingStreamingProvider(
        "primary-provider",
        (LLMStreamChunk("partial", 0),),
        fail_after=TimeoutError("private failure"),
    )
    fallback = RecordingStreamingProvider(
        "fallback-provider",
        (LLMStreamChunk("unused", 0),),
    )
    runtime = LLMRuntime(
        providers={"primary": primary, "fallback": fallback},
        default_provider="primary",
        fallback_providers=["fallback"],
        retry_policy=LLMRetryPolicy(max_attempts=3),
    )
    stream = runtime.stream_chat([LLMMessage("user", "private")], model="model")

    assert next(stream).delta == "partial"
    with pytest.raises(LLMProviderError, match="after output started"):
        next(stream)

    assert len(primary.requests) == 1
    assert fallback.requests == []


def test_stream_failure_event_is_sanitized() -> None:
    prompt = "private failure prompt"
    detail = "private provider detail"
    provider = RecordingStreamingProvider(
        "primary",
        fail_before=ValueError(detail),
    )
    events = RuntimeEventLog()
    runtime = LLMRuntime(provider, events)

    with pytest.raises(LLMProviderError, match="LLM stream failed for provider"):
        list(runtime.stream_chat([LLMMessage("user", prompt)], model="model"))

    assert events.events[-1].event_type == "llm.stream_failed"
    assert prompt not in repr(events.events)
    assert detail not in repr(events.events)


def test_stream_final_usage_updates_ledger_once() -> None:
    usage = LLMUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    provider = RecordingStreamingProvider(
        "primary",
        (
            LLMStreamChunk("one", 0, usage=usage),
            LLMStreamChunk("two", 1),
            LLMStreamChunk("", 2, done=True, usage=usage),
        ),
    )
    runtime = LLMRuntime(provider)

    list(runtime.stream_chat([LLMMessage("user", "private")], model="model"))

    assert runtime.usage_snapshot() == LLMUsageLedger(3, 2, 5)


def test_stream_budget_exceeded_after_completion_updates_usage_once() -> None:
    usage = LLMUsage(total_tokens=6)
    provider = RecordingStreamingProvider(
        "primary",
        (LLMStreamChunk("complete", 0), LLMStreamChunk("", 1, done=True, usage=usage)),
    )
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        provider,
        events,
        token_budget=LLMTokenBudget(max_total_tokens=5),
    )

    with pytest.raises(LLMBudgetExceededError, match="total"):
        list(runtime.stream_chat([LLMMessage("user", "private")], model="model"))

    assert runtime.usage_snapshot() == LLMUsageLedger(total_tokens=6)
    assert len(provider.requests) == 1
    event_types = [event.event_type for event in events.events]
    assert event_types.index("llm.stream_completed") < event_types.index(
        "llm.budget_exceeded"
    )
    assert "llm.stream_failed" not in event_types


def test_stream_request_budget_blocks_provider_before_stream_starts() -> None:
    provider = RecordingStreamingProvider("primary")
    runtime = LLMRuntime(
        provider,
        token_budget=LLMTokenBudget(max_completion_tokens=5),
    )

    with pytest.raises(LLMBudgetExceededError, match="completion"):
        runtime.stream_chat(
            [LLMMessage("user", "private")],
            model="model",
            metadata={"max_tokens": 6},
        )

    assert provider.requests == []


def test_streaming_bypasses_response_cache() -> None:
    provider = DeterministicStreamingLLMProvider(("same",))
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        provider,
        events,
        cache=LLMResponseCache(enabled=True),
    )

    list(runtime.stream_chat([LLMMessage("user", "same")], model="model"))
    list(runtime.stream_chat([LLMMessage("user", "same")], model="model"))

    assert len(provider.requests) == 2
    assert runtime.cache_snapshot().hits == 0
    assert runtime.cache_snapshot().misses == 0
    assert runtime.cache_snapshot().stores == 0
    assert [event.event_type for event in events.by_type("llm.cache_bypassed")] == [
        "llm.cache_bypassed",
        "llm.cache_bypassed",
    ]
