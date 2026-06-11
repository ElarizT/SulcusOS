from __future__ import annotations

from typing import Any

import pytest

from kernel.events import RuntimeEventLog
from kernel.llm import (
    DeterministicLLMProvider,
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMRuntime,
    LLMRuntimeError,
    LLMUsage,
)


class RoutingProvider:
    def __init__(
        self,
        name: str,
        *,
        content: str = "success",
        fail: bool = False,
        failure_detail: str = "private provider failure",
        usage: LLMUsage | None = None,
        attempts: list[str] | None = None,
    ) -> None:
        self.name = name
        self.content = content
        self.fail = fail
        self.failure_detail = failure_detail
        self.usage = usage
        self.attempts = attempts if attempts is not None else []
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.attempts.append(self.name)
        self.requests.append(request)
        if self.fail:
            raise LLMProviderError(self.failure_detail)
        return LLMResponse(
            content=self.content,
            model=request.model,
            provider=self.name,
            usage=self.usage,
        )


def test_single_provider_runtime_remains_backwards_compatible() -> None:
    provider = DeterministicLLMProvider("single response")
    runtime = LLMRuntime(provider=provider)

    response = runtime.chat([{"role": "user", "content": "hello"}], model="model")

    assert runtime.provider is provider
    assert runtime.providers == {}
    assert response.content == "single response"


def test_multiple_provider_construction_and_default_selection() -> None:
    primary = RoutingProvider("primary-provider", content="primary response")
    fast = RoutingProvider("fast-provider", content="fast response")
    runtime = LLMRuntime(
        providers={"primary": primary, "fast": fast},
        default_provider="primary",
    )

    response = runtime.chat([LLMMessage("user", "hello")], model="model")

    assert runtime.provider is primary
    assert runtime.providers == {"primary": primary, "fast": fast}
    assert response.content == "primary response"
    assert len(primary.requests) == 1
    assert fast.requests == []


def test_explicit_provider_selection() -> None:
    primary = RoutingProvider("primary-provider")
    fast = RoutingProvider("fast-provider", content="fast response")
    runtime = LLMRuntime(
        providers={"primary": primary, "fast": fast},
        default_provider="primary",
    )

    response = runtime.chat(
        [{"role": "user", "content": "hello"}],
        model="model",
        provider="fast",
    )

    assert response.content == "fast response"
    assert primary.requests == []
    assert len(fast.requests) == 1


def test_unknown_or_missing_provider_selection_is_clean() -> None:
    runtime = LLMRuntime(providers={"primary": RoutingProvider("primary-provider")})

    with pytest.raises(LLMRuntimeError, match="provider must be specified"):
        runtime.chat([{"role": "user", "content": "private"}], model="model")
    with pytest.raises(LLMRuntimeError, match="unknown requested LLM provider 'missing'"):
        runtime.chat(
            [{"role": "user", "content": "private"}],
            model="model",
            provider="missing",
        )


def test_fallback_success_returns_successful_provider_usage_and_events() -> None:
    attempts: list[str] = []
    primary = RoutingProvider("primary-provider", fail=True, attempts=attempts)
    usage = LLMUsage(prompt_tokens=10, completion_tokens=4, total_tokens=14)
    fast = RoutingProvider(
        "fast-provider",
        content="fallback response",
        usage=usage,
        attempts=attempts,
    )
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        providers={"primary": primary, "fast": fast},
        default_provider="primary",
        fallback_providers=["fast"],
        event_sink=events,
    )

    response = runtime.chat([{"role": "user", "content": "private"}], model="model")

    assert attempts == ["primary-provider", "fast-provider"]
    assert response.content == "fallback response"
    assert response.usage == usage
    assert [event.event_type for event in events.events] == [
        "llm.provider_selected",
        "llm.requested",
        "llm.provider_failed",
        "llm.fallback_started",
        "llm.fallback_succeeded",
        "llm.completed",
    ]
    assert events.events[0].metadata == {
        "provider": "primary",
        "model": "model",
        "attempt": 1,
        "success": True,
    }
    assert events.events[2].metadata == {
        "provider": "primary",
        "model": "model",
        "attempt": 1,
        "error_type": "LLMProviderError",
        "success": False,
    }
    assert events.events[-2].metadata == {
        "provider": "primary",
        "fallback_provider": "fast",
        "model": "model",
        "attempt": 2,
        "success": True,
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }


def test_duplicate_fallback_names_and_selected_provider_are_skipped() -> None:
    attempts: list[str] = []
    primary = RoutingProvider("primary-provider", fail=True, attempts=attempts)
    fast = RoutingProvider("fast-provider", fail=True, attempts=attempts)
    local = RoutingProvider("local-provider", attempts=attempts)
    runtime = LLMRuntime(
        providers={"primary": primary, "fast": fast, "local": local},
        default_provider="primary",
        fallback_providers=["primary", "fast", "fast", "local"],
    )

    runtime.chat([{"role": "user", "content": "private"}], model="model")

    assert attempts == ["primary-provider", "fast-provider", "local-provider"]


def test_fallback_exhaustion_is_sanitized_and_emits_safe_events() -> None:
    prompt = "private prompt"
    api_key = "private api key"
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        providers={
            "primary": RoutingProvider(
                "primary-provider",
                fail=True,
                failure_detail=f"{prompt}: {api_key}",
            ),
            "fast": RoutingProvider(
                "fast-provider",
                fail=True,
                failure_detail=f"{api_key}: {prompt}",
            ),
        },
        default_provider="primary",
        fallback_providers=["fast"],
        event_sink=events,
    )

    with pytest.raises(
        LLMProviderError,
        match="All LLM providers failed after 2 attempts: primary, fast",
    ) as error:
        runtime.chat([{"role": "user", "content": prompt}], model="model")

    assert error.value.__cause__ is None
    assert [event.event_type for event in events.events] == [
        "llm.provider_selected",
        "llm.requested",
        "llm.provider_failed",
        "llm.fallback_started",
        "llm.provider_failed",
        "llm.fallback_exhausted",
        "llm.failed",
    ]
    assert events.events[-2].metadata == {
        "provider": "primary",
        "model": "model",
        "attempt": 2,
        "success": False,
    }
    rendered = repr((error.value, events.events))
    assert prompt not in rendered
    assert api_key not in rendered


def test_invalid_registry_configuration_is_rejected() -> None:
    provider = RoutingProvider("provider")

    with pytest.raises(LLMRuntimeError, match="either provider or providers"):
        LLMRuntime(provider=provider, providers={"primary": provider})
    with pytest.raises(LLMRuntimeError, match="unknown default"):
        LLMRuntime(providers={"primary": provider}, default_provider="missing")
    with pytest.raises(LLMRuntimeError, match="unknown fallback"):
        LLMRuntime(
            providers={"primary": provider},
            default_provider="primary",
            fallback_providers=["missing"],
        )
