from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest

from kernel.events import RuntimeEventLog
from kernel.llm import (
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMRuntime,
    LLMToolCall,
    LLMUsage,
    OpenAICompatibleProvider,
)


class FakeCompletions:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.response


def fake_client(response: Any = None, error: Exception | None = None) -> Any:
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions(response, error))
    )


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHTTPClient:
    instances: list["FakeHTTPClient"] = []

    def __init__(self, **options: Any) -> None:
        self.options = options
        self.posts: list[dict[str, Any]] = []
        FakeHTTPClient.instances.append(self)

    def post(self, path: str, **kwargs: Any) -> FakeHTTPResponse:
        self.posts.append({"path": path, **kwargs})
        return FakeHTTPResponse(
            {
                "id": "chatcmpl-http-id",
                "created": 67890,
                "model": "http-model",
                "choices": [{"message": {"content": "http response"}}],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            }
        )


def fake_response(
    content: str = "provider response",
    *,
    model: str = "provider-model",
) -> Any:
    return SimpleNamespace(
        id="chatcmpl-safe-id",
        created=12345,
        model=model,
        system_fingerprint="fp_safe",
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=4,
            completion_tokens=2,
            total_tokens=6,
        ),
    )


def test_constructor_configuration_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTOS_LLM_API_KEY", "environment-key")
    monkeypatch.setenv("AGENTOS_LLM_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("AGENTOS_LLM_MODEL", "environment-model")
    monkeypatch.setenv("AGENTOS_LLM_PROVIDER", "environment-provider")

    provider = OpenAICompatibleProvider(
        api_key="constructor-key",
        base_url="https://constructor.example/v1",
        default_model="constructor-model",
        provider_name="constructor-provider",
        timeout_seconds=12.5,
        client=fake_client(fake_response()),
    )

    assert isinstance(provider, LLMProvider)
    assert provider.api_key == "constructor-key"
    assert provider.base_url == "https://constructor.example/v1"
    assert provider.default_model == "constructor-model"
    assert provider.name == "constructor-provider"
    assert provider.timeout_seconds == 12.5


def test_configuration_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTOS_LLM_API_KEY", "environment-key")
    monkeypatch.setenv("AGENTOS_LLM_BASE_URL", "https://environment.example/v1")
    monkeypatch.setenv("AGENTOS_LLM_MODEL", "environment-model")
    monkeypatch.setenv("AGENTOS_LLM_PROVIDER", "openrouter")

    provider = OpenAICompatibleProvider(client=fake_client(fake_response()))

    assert provider.api_key == "environment-key"
    assert provider.base_url == "https://environment.example/v1"
    assert provider.default_model == "environment-model"
    assert provider.name == "openrouter"


def test_missing_api_key_is_reported_only_when_provider_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTOS_LLM_API_KEY", raising=False)
    provider = OpenAICompatibleProvider(client=fake_client(fake_response()))

    with pytest.raises(LLMProviderError, match="requires an API key"):
        provider.complete(LLMRequest((LLMMessage("user", "private"),), "test-model"))


def test_missing_openai_dependency_uses_http_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def missing_openai(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ModuleNotFoundError("No module named 'openai'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_openai)
    import httpx

    FakeHTTPClient.instances.clear()
    monkeypatch.setattr(httpx, "Client", FakeHTTPClient)
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        base_url="https://api.example/v1",
    )
    response = provider.complete(
        LLMRequest((LLMMessage("user", "private"),), "test-model", timeout_seconds=5)
    )

    assert response.content == "http response"
    assert response.model == "http-model"
    assert response.usage == LLMUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    assert FakeHTTPClient.instances[0].options["base_url"] == "https://api.example/v1/"
    assert FakeHTTPClient.instances[0].posts == [
        {
            "path": "chat/completions",
            "json": {
                "model": "test-model",
                "messages": [{"role": "user", "content": "private"}],
                "temperature": 0.0,
            },
            "timeout": 5,
        }
    ]


def test_request_response_and_usage_mapping() -> None:
    client = fake_client(fake_response())
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        provider_name="groq",
        client=client,
    )
    request = LLMRequest(
        (
            LLMMessage("system", "private instructions", {"ignored": "message metadata"}),
            LLMMessage("user", "private prompt"),
        ),
        "requested-model",
        temperature=0.25,
        metadata={
            "options": {"max_tokens": 128, "unsupported": "ignored"},
            "private": "ignored",
        },
    )

    response = provider.complete(request)

    assert client.chat.completions.calls == [
        {
            "model": "requested-model",
            "messages": [
                {"role": "system", "content": "private instructions"},
                {"role": "user", "content": "private prompt"},
            ],
            "temperature": 0.25,
            "max_tokens": 128,
        }
    ]
    assert response == LLMResponse(
        content="provider response",
        model="provider-model",
        provider="groq",
        usage=LLMUsage(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        metadata={
            "response_id": "chatcmpl-safe-id",
            "created": 12345,
            "system_fingerprint": "fp_safe",
        },
    )


def test_tool_feedback_messages_map_to_openai_compatible_payload() -> None:
    client = fake_client(fake_response(content="final"))
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        provider_name="openai",
        client=client,
    )
    tool_call = LLMToolCall(
        id="call_1",
        name="add_numbers",
        arguments={"a": 15, "b": 27},
    )
    request = LLMRequest(
        (
            LLMMessage("user", "private prompt"),
            LLMMessage("assistant", "", {"tool_calls": (tool_call,)}),
            LLMMessage(
                "tool",
                "42",
                {"tool_call_id": "call_1", "name": "add_numbers"},
            ),
        ),
        "requested-model",
    )

    provider.complete(request)

    assert client.chat.completions.calls[0]["messages"] == [
        {"role": "user", "content": "private prompt"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "arguments": '{"a":15,"b":27}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "42", "tool_call_id": "call_1"},
    ]


def test_provider_failure_is_wrapped_without_leaking_sensitive_text() -> None:
    secret = "placeholder-key-private-prompt"
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        client=fake_client(error=RuntimeError(secret)),
    )

    with pytest.raises(LLMProviderError, match="provider request failed") as error:
        provider.complete(LLMRequest((LLMMessage("user", secret),), "test-model"))

    assert secret not in repr(error.value)
    assert error.value.__cause__ is None


def test_malformed_response_is_wrapped() -> None:
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        client=fake_client(SimpleNamespace(choices=[])),
    )

    with pytest.raises(LLMProviderError, match="malformed response"):
        provider.complete(LLMRequest((LLMMessage("user", "private"),), "test-model"))


def test_runtime_chat_uses_default_model_and_emits_safe_events() -> None:
    secret = "private-prompt-and-key"
    events = RuntimeEventLog()
    client = fake_client(fake_response(model="resolved-model"))
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        default_model="default-model",
        provider_name="openrouter",
        client=client,
    )

    response = LLMRuntime(provider, events).chat(
        [{"role": "user", "content": secret}],
        metadata={"private": secret},
    )

    assert response.model == "resolved-model"
    assert client.chat.completions.calls[0]["model"] == "default-model"
    assert [event.event_type for event in events.events] == [
        "llm.requested",
        "llm.completed",
    ]
    assert events.events[0].metadata == {
        "provider": "openrouter",
        "model": "default-model",
    }
    assert events.events[1].metadata == {
        "provider": "openrouter",
        "model": "resolved-model",
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
    }
    assert secret not in repr(events.events)


def test_runtime_failure_does_not_leak_provider_key_or_prompt() -> None:
    api_key = "private-api-key"
    prompt = "private-prompt"
    events = RuntimeEventLog()
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        provider_name="openai",
        client=fake_client(error=RuntimeError(f"{api_key}: {prompt}")),
    )

    with pytest.raises(LLMProviderError) as error:
        LLMRuntime(provider, events).chat(
            [{"role": "user", "content": prompt}],
            model="test-model",
        )

    rendered = repr((error.value, events.events))
    assert api_key not in rendered
    assert prompt not in rendered
    assert events.events[-1].metadata == {
        "provider": "openai",
        "model": "test-model",
        "error": True,
        "error_type": "LLMProviderError",
        "error_category": "unknown",
    }
