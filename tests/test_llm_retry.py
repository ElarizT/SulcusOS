from __future__ import annotations

from typing import Any

import pytest

from kernel.events import RuntimeEventLog
from kernel.llm import (
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMRetryPolicy,
    LLMRuntime,
    OpenAICompatibleProvider,
    classify_llm_error,
)


class SequenceProvider:
    def __init__(self, name: str, outcomes: list[Exception | str]) -> None:
        self.name = name
        self.outcomes = list(outcomes)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(outcome, request.model, self.name)


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        return type(
            "Completion",
            (),
            {
                "choices": [
                    type("Choice", (), {"message": type("Message", (), {"content": "ok"})()})()
                ],
                "model": "model",
                "usage": None,
            },
        )()


def test_default_retry_policy_preserves_single_attempt() -> None:
    provider = SequenceProvider("primary", [TimeoutError("private"), "unused"])
    runtime = LLMRuntime(provider)

    with pytest.raises(LLMProviderError, match="LLM provider 'primary' failed"):
        runtime.chat([LLMMessage("user", "private")], model="model")

    assert runtime.retry_policy == LLMRetryPolicy()
    assert len(provider.requests) == 1


def test_retry_succeeds_on_second_attempt_and_emits_safe_events() -> None:
    secret = "private-prompt-and-error"
    provider = SequenceProvider("primary", [TimeoutError(secret), "success"])
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        provider,
        events,
        retry_policy=LLMRetryPolicy(max_attempts=2),
    )

    response = runtime.chat([LLMMessage("user", secret)], model="model")

    assert response.content == "success"
    assert len(provider.requests) == 2
    assert [event.event_type for event in events.events] == [
        "llm.requested",
        "llm.retry_scheduled",
        "llm.retry_started",
        "llm.retry_succeeded",
        "llm.completed",
    ]
    assert events.events[1].metadata == {
        "provider": "primary",
        "model": "model",
        "attempt": 2,
        "max_attempts": 2,
        "error_category": "timeout",
        "error_type": "TimeoutError",
        "backoff_seconds": 0.0,
    }
    assert secret not in repr(events.events)


def test_retry_exhaustion_moves_to_fallback_provider() -> None:
    primary = SequenceProvider(
        "primary-provider",
        [TimeoutError("first"), TimeoutError("second")],
    )
    fallback = SequenceProvider("fallback-provider", ["fallback success"])
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        providers={"primary": primary, "fallback": fallback},
        default_provider="primary",
        fallback_providers=["fallback"],
        retry_policy=LLMRetryPolicy(max_attempts=2),
        event_sink=events,
    )

    response = runtime.chat([LLMMessage("user", "private")], model="model")

    assert response.content == "fallback success"
    assert len(primary.requests) == 2
    assert len(fallback.requests) == 1
    event_types = [event.event_type for event in events.events]
    assert event_types.index("llm.retry_exhausted") < event_types.index("llm.fallback_started")
    assert "llm.fallback_succeeded" in event_types


def test_retry_exhaustion_error_and_events_do_not_leak_secrets() -> None:
    prompt = "private retry prompt"
    api_key = "private retry key"
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        providers={
            "primary": SequenceProvider(
                "primary-provider",
                [TimeoutError(f"{prompt}: {api_key}"), TimeoutError(api_key)],
            ),
            "fallback": SequenceProvider(
                "fallback-provider",
                [TimeoutError(prompt), TimeoutError(f"{api_key}: {prompt}")],
            ),
        },
        default_provider="primary",
        fallback_providers=["fallback"],
        retry_policy=LLMRetryPolicy(max_attempts=2),
        event_sink=events,
    )

    with pytest.raises(LLMProviderError, match="All LLM providers failed") as error:
        runtime.chat([LLMMessage("user", prompt)], model="model")

    rendered = repr((error.value, events.events))
    assert prompt not in rendered
    assert api_key not in rendered


def test_non_retryable_configuration_error_does_not_retry() -> None:
    provider = SequenceProvider(
        "primary",
        [LLMProviderError("configuration failed", category="configuration"), "unused"],
    )
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        provider,
        events,
        retry_policy=LLMRetryPolicy(max_attempts=3, retry_on=("configuration",)),
    )

    with pytest.raises(LLMProviderError, match="configuration failed"):
        runtime.chat([LLMMessage("user", "private")], model="model")

    assert len(provider.requests) == 1
    assert not any(event.event_type.startswith("llm.retry_") for event in events.events)


def test_unknown_provider_error_remains_clean_and_does_not_fallback() -> None:
    secret = "private unknown failure"
    primary = SequenceProvider("primary-provider", [ValueError(secret)])
    fallback = SequenceProvider("fallback-provider", ["unused"])
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        providers={"primary": primary, "fallback": fallback},
        default_provider="primary",
        fallback_providers=["fallback"],
        retry_policy=LLMRetryPolicy(max_attempts=3),
        event_sink=events,
    )

    with pytest.raises(LLMProviderError, match="LLM provider 'primary' failed") as error:
        runtime.chat([LLMMessage("user", "private")], model="model")

    assert error.value.__cause__ is None
    assert len(primary.requests) == 1
    assert fallback.requests == []
    assert secret not in repr((error.value, events.events))


def test_backoff_uses_injected_sleeper_and_cap() -> None:
    sleeps: list[float] = []
    provider = SequenceProvider(
        "primary",
        [TimeoutError(), TimeoutError(), "success"],
    )
    runtime = LLMRuntime(
        provider,
        retry_policy=LLMRetryPolicy(
            max_attempts=3,
            backoff_seconds=2.0,
            max_backoff_seconds=3.0,
        ),
        sleeper=sleeps.append,
    )

    runtime.chat([LLMMessage("user", "private")], model="model")

    assert sleeps == [2.0, 3.0]


def test_timeout_is_passed_through_runtime_and_request_override() -> None:
    provider = SequenceProvider("primary", ["first", "second"])
    runtime = LLMRuntime(provider, timeout_seconds=30)

    runtime.chat([LLMMessage("user", "one")], model="model")
    runtime.chat(
        [LLMMessage("user", "two")],
        model="model",
        timeout_seconds=5,
    )

    assert [request.timeout_seconds for request in provider.requests] == [30, 5]


def test_openai_compatible_provider_accepts_request_timeout_offline() -> None:
    completions = FakeCompletions()
    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": completions})()},
    )()
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        timeout_seconds=12,
        client=client,
    )

    provider.complete(
        LLMRequest(
            (LLMMessage("user", "private"),),
            "model",
            timeout_seconds=4,
        )
    )

    assert provider.timeout_seconds == 12
    assert completions.calls[0]["timeout"] == 4


def test_error_classification_is_deterministic() -> None:
    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    class APIStatusError(Exception):
        status_code = 400

    class ServerStatusError(Exception):
        status_code = 503

    assert classify_llm_error(TimeoutError()) == "timeout"
    assert classify_llm_error(RateLimitError()) == "rate_limit"
    assert classify_llm_error(BadRequestError()) == "request"
    assert classify_llm_error(APIStatusError()) == "request"
    assert classify_llm_error(ServerStatusError()) == "transient"
    assert classify_llm_error(ConnectionError()) == "transient"
    assert (
        classify_llm_error(LLMProviderError("safe", category="configuration"))
        == "configuration"
    )
    assert classify_llm_error(ModuleNotFoundError()) == "configuration"
    assert classify_llm_error(ValueError()) == "unknown"
