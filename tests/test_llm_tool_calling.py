from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from kernel.events import RuntimeEventLog
from kernel.llm import (
    DeterministicLLMProvider,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRuntime,
    LLMToolCall,
    LLMToolDefinition,
    LLMToolParameter,
    LLMToolResult,
    OpenAICompatibleProvider,
)


class FakeCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **payload: Any) -> Any:
        self.calls.append(payload)
        return self.response


def fake_client(response: Any) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(response)))


def fake_tool_call(
    tool_call_id: str,
    name: str,
    arguments: str,
) -> Any:
    return SimpleNamespace(
        id=tool_call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def fake_response(*tool_calls: Any, content: str | None = "") -> Any:
    return SimpleNamespace(
        id="chatcmpl-safe-id",
        created=12345,
        model="provider-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=list(tool_calls))
            )
        ],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
    )


def weather_tool() -> LLMToolDefinition:
    return LLMToolDefinition(
        name="get_weather",
        description="Get weather for a city.",
        parameters_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    )


def test_tool_definition_construction_is_deterministic() -> None:
    parameter = LLMToolParameter(
        name="city",
        type="string",
        description="City name",
        required=True,
        enum=("Warsaw", "Berlin"),
    )
    tool = weather_tool()
    result = LLMToolResult(
        tool_call_id="call_1",
        name="get_weather",
        content='{"temperature": 21}',
    )

    assert parameter.name == "city"
    assert parameter.enum == ("Warsaw", "Berlin")
    assert tool.parameters_schema["required"] == ["city"]
    assert result.success is True


def test_runtime_request_accepts_tools_without_executing_them() -> None:
    executed = False

    class ToolCallingProvider:
        name = "fake-tools"
        default_model = "fake-model"

        def __init__(self) -> None:
            self.requests: list[LLMRequest] = []

        def complete(self, request: LLMRequest) -> LLMResponse:
            self.requests.append(request)
            return LLMResponse(
                "",
                request.model,
                self.name,
                tool_calls=(
                    LLMToolCall(
                        id="call_1",
                        name="get_weather",
                        arguments={"city": "Warsaw"},
                        provider=self.name,
                        model=request.model,
                    ),
                ),
            )

    def local_tool() -> None:
        nonlocal executed
        executed = True

    provider = ToolCallingProvider()
    response = LLMRuntime(provider).chat(
        [LLMMessage("user", "Use a tool")],
        tools=[weather_tool()],
        tool_choice="auto",
    )

    assert local_tool
    assert executed is False
    assert provider.requests[0].tools == (weather_tool(),)
    assert provider.requests[0].tool_choice == "auto"
    assert response.tool_calls[0].name == "get_weather"


def test_openai_compatible_request_mapping_with_tools() -> None:
    client = fake_client(fake_response(content="plain"))
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        provider_name="openai",
        client=client,
    )
    request = LLMRequest(
        (LLMMessage("user", "private prompt"),),
        "requested-model",
        tools=(weather_tool(),),
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    provider.complete(request)

    assert client.chat.completions.calls == [
        {
            "model": "requested-model",
            "messages": [{"role": "user", "content": "private prompt"}],
            "temperature": 0.0,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "parameters": weather_tool().parameters_schema,
                    },
                }
            ],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"},
            },
        }
    ]


def test_openai_compatible_response_mapping_with_tool_call() -> None:
    response = fake_response(
        fake_tool_call("call_1", "get_weather", '{"city": "Warsaw"}'),
        content=None,
    )
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        provider_name="openai",
        client=fake_client(response),
    )

    mapped = provider.complete(
        LLMRequest((LLMMessage("user", "private prompt"),), "requested-model")
    )

    assert mapped.content == ""
    assert mapped.tool_calls == (
        LLMToolCall(
            id="call_1",
            name="get_weather",
            arguments={"city": "Warsaw"},
            provider="openai",
            model="provider-model",
            metadata={"index": 0},
        ),
    )


def test_multiple_tool_calls_are_preserved_in_order() -> None:
    response = fake_response(
        fake_tool_call("call_1", "get_weather", '{"city": "Warsaw"}'),
        fake_tool_call("call_2", "get_time", '{"timezone": "Europe/Warsaw"}'),
    )
    provider = OpenAICompatibleProvider(
        api_key="placeholder-key",
        provider_name="openai",
        client=fake_client(response),
    )

    mapped = provider.complete(
        LLMRequest((LLMMessage("user", "private prompt"),), "requested-model")
    )

    assert [tool_call.id for tool_call in mapped.tool_calls] == ["call_1", "call_2"]
    assert [tool_call.name for tool_call in mapped.tool_calls] == [
        "get_weather",
        "get_time",
    ]


def test_normal_chat_still_works_without_tools() -> None:
    events = RuntimeEventLog()
    provider = DeterministicLLMProvider("safe response")

    response = LLMRuntime(provider, events).chat(
        [LLMMessage("user", "hello")],
        model="test-model",
    )

    assert response.content == "safe response"
    assert provider.requests[0].tools == ()
    assert provider.requests[0].tool_choice is None
    assert [event.event_type for event in events.events] == [
        "llm.requested",
        "llm.completed",
    ]


def test_runtime_tool_events_are_safe_and_do_not_leak_arguments_or_prompts() -> None:
    secret_prompt = "private prompt"
    secret_argument = "private-city-secret"
    api_key = "private-api-key"
    events = RuntimeEventLog()
    response = fake_response(
        fake_tool_call("call_1", "get_weather", f'{{"city": "{secret_argument}"}}'),
        content=None,
    )
    provider = OpenAICompatibleProvider(
        api_key=api_key,
        provider_name="openai",
        default_model="requested-model",
        client=fake_client(response),
    )

    mapped = LLMRuntime(provider, events).chat(
        [LLMMessage("user", secret_prompt)],
        tools=[weather_tool()],
    )

    assert mapped.tool_calls[0].arguments == {"city": secret_argument}
    assert [event.event_type for event in events.events] == [
        "llm.tools_available",
        "llm.requested",
        "llm.tool_call_requested",
        "llm.completed",
    ]
    assert events.by_type("llm.tools_available")[0].metadata == {
        "provider": "openai",
        "model": "requested-model",
        "tool_count": 1,
    }
    assert events.by_type("llm.tool_call_requested")[0].metadata == {
        "provider": "openai",
        "model": "provider-model",
        "tool_name": "get_weather",
        "tool_call_count": 1,
    }
    rendered_events = repr(events.events)
    assert secret_prompt not in rendered_events
    assert secret_argument not in rendered_events
    assert api_key not in rendered_events

