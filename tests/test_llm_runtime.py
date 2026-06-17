import pytest

from kernel.events import RuntimeEventLog
from kernel.llm import (
    DeterministicLLMProvider,
    EchoLLMProvider,
    LLMMessage,
    LLMProvider,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMRuntime,
    LLMRuntimeError,
    LLMUsage,
    OpenAICompatibleProvider,
)


def test_llm_request_response_and_usage_structures() -> None:
    message = LLMMessage("user", "hello", {"trace": "local"})
    request = LLMRequest((message,), "test-model", metadata={"purpose": "test"})
    usage = LLMUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    response = LLMResponse("done", "test-model", "test-provider", usage, {"cached": False})

    assert request.messages == (message,)
    assert request.temperature == 0.0
    assert response.content == "done"
    assert response.usage == usage
    assert response.metadata == {"cached": False}


def test_llm_structures_reject_invalid_core_values() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        LLMRequest((), "test-model")
    with pytest.raises(ValueError, match="must not be negative"):
        LLMUsage(prompt_tokens=-1)


def test_deterministic_provider_returns_predictable_content_and_usage() -> None:
    provider = DeterministicLLMProvider("fixed output")
    request = LLMRequest(
        (LLMMessage("system", "one two"), LLMMessage("user", "three")),
        "test-model",
    )

    response = provider.complete(request)

    assert isinstance(provider, LLMProvider)
    assert provider.requests == [request]
    assert response == LLMResponse(
        content="fixed output",
        model="test-model",
        provider="deterministic",
        usage=LLMUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        metadata={"deterministic": True},
    )


def test_echo_provider_returns_final_message() -> None:
    runtime = LLMRuntime(EchoLLMProvider())

    response = runtime.chat(
        [LLMMessage("system", "instructions"), {"role": "user", "content": "echo this"}],
        model="test-model",
    )

    assert response.content == "echo this"
    assert response.provider == "echo"


def test_runtime_chat_emits_requested_and_completed_events_without_prompt_content() -> None:
    secret = "do-not-leak-this-prompt"
    events = RuntimeEventLog()
    runtime = LLMRuntime(DeterministicLLMProvider("safe response"), events)

    response = runtime.chat(
        [{"role": "user", "content": secret}],
        model="test-model",
        temperature=0.0,
        metadata={"private_context": secret},
    )

    assert response.content == "safe response"
    assert [event.event_type for event in events.events] == ["llm.requested", "llm.completed"]
    assert events.events[0].metadata == {"provider": "deterministic", "model": "test-model"}
    assert events.events[1].metadata == {
        "provider": "deterministic",
        "model": "test-model",
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert secret not in repr(events.events)


def test_runtime_chat_failure_is_safe_and_emits_failure_event() -> None:
    secret = "provider-secret-detail"
    events = RuntimeEventLog()

    class FailingProvider:
        name = "failing"

        def complete(self, request: LLMRequest) -> LLMResponse:
            raise ValueError(secret)

    runtime = LLMRuntime(FailingProvider(), events)

    with pytest.raises(LLMProviderError, match="LLM provider 'failing' failed") as error:
        runtime.chat([LLMMessage("user", "private prompt")], model="test-model")

    assert isinstance(error.value, LLMRuntimeError)
    assert [event.event_type for event in events.events] == ["llm.requested", "llm.failed"]
    assert events.events[-1].level == "ERROR"
    assert events.events[-1].metadata == {
        "provider": "failing",
        "model": "test-model",
        "error": True,
        "error_type": "ValueError",
        "error_category": "unknown",
    }
    assert secret not in repr(events.events)


def test_public_llm_runtime_imports_are_stable() -> None:
    from kernel.llm import AsyncLLMManager, LLMConfig
    from kernel.llm.providers import DeterministicLLMProvider as ProviderImport
    from kernel.llm.runtime import LLMRuntime as RuntimeImport
    from kernel.llm.types import LLMMessage as MessageImport

    assert AsyncLLMManager.__name__ == "AsyncLLMManager"
    assert LLMConfig.__name__ == "LLMConfig"
    assert ProviderImport is DeterministicLLMProvider
    assert RuntimeImport is LLMRuntime
    assert MessageImport is LLMMessage
    assert OpenAICompatibleProvider.__name__ == "OpenAICompatibleProvider"
