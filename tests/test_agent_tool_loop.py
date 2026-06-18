from __future__ import annotations

from typing import Any

from kernel.agent_tool_loop import AgentToolLoop
from kernel.events import RuntimeEventLog
from kernel.llm import LLMMessage, LLMRequest, LLMResponse, LLMRuntime, LLMToolCall, LLMToolDefinition
from kernel.tools import ToolRegistry, ToolRuntime


class ScriptedProvider:
    name = "fake-agent-llm"
    default_model = "fake-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


def add_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }


def add_tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
    )


def multiply_tool_definition() -> LLMToolDefinition:
    return LLMToolDefinition(
        name="multiply_numbers",
        description="Multiply two numbers.",
        parameters_schema=add_schema(),
    )


def tool_call(
    tool_call_id: str,
    name: str = "add_numbers",
    arguments: dict[str, Any] | None = None,
) -> LLMToolCall:
    return LLMToolCall(
        id=tool_call_id,
        name=name,
        arguments=dict(arguments or {"a": 15, "b": 27}),
        provider="fake-agent-llm",
        model="fake-model",
    )


def tool_response(*calls: LLMToolCall) -> LLMResponse:
    return LLMResponse(
        content="",
        model="fake-model",
        provider="fake-agent-llm",
        tool_calls=tuple(calls),
    )


def final_response(content: str = "final answer") -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake-model",
        provider="fake-agent-llm",
    )


def add_registry(executions: list[tuple[float, float]] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    def add_numbers(a: float, b: float) -> float:
        if executions is not None:
            executions.append((a, b))
        return a + b

    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=add_numbers,
    )
    return registry


def build_loop(
    provider: ScriptedProvider,
    registry: ToolRegistry,
    events: RuntimeEventLog | None = None,
) -> AgentToolLoop:
    return AgentToolLoop(
        llm_runtime=LLMRuntime(provider, events),
        tool_runtime=ToolRuntime(registry=registry, event_sink=events),
    )


def test_final_response_with_no_tool_calls_completes() -> None:
    events = RuntimeEventLog()
    provider = ScriptedProvider([final_response("done")])
    loop = build_loop(provider, ToolRegistry(), events)

    result = loop.run([{"role": "user", "content": "hello"}], tools=[])

    assert result.completed is True
    assert result.reason == "completed"
    assert result.final_response == final_response("done")
    assert result.pending_tool_calls == ()
    assert provider.requests[0].tools == ()
    assert events.by_type("agent_tool_loop.completed")


def test_one_tool_call_then_final_response() -> None:
    executions: list[tuple[float, float]] = []
    call = tool_call("call_1")
    provider = ScriptedProvider([tool_response(call), final_response("42")])
    loop = build_loop(provider, add_registry(executions))

    result = loop.run(
        [LLMMessage("user", "Calculate 15 + 27 using tools.")],
        tools=[add_tool_definition()],
    )

    assert result.completed is True
    assert result.final_response == final_response("42")
    assert executions == [(15, 27)]
    assert len(provider.requests) == 2
    assert [message.role for message in provider.requests[1].messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert provider.requests[1].messages[1].metadata["tool_calls"] == (call,)
    assert provider.requests[1].messages[2].content == "42"
    assert provider.requests[1].messages[2].metadata["tool_call_id"] == "call_1"
    assert result.tool_results[0].content == "42"


def test_multiple_tool_calls_in_one_llm_response_then_final_response() -> None:
    executions: list[tuple[str, float, float]] = []
    registry = ToolRegistry()

    def add_numbers(a: float, b: float) -> float:
        executions.append(("add_numbers", a, b))
        return a + b

    def multiply_numbers(a: float, b: float) -> float:
        executions.append(("multiply_numbers", a, b))
        return a * b

    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=add_numbers,
    )
    registry.register(
        name="multiply_numbers",
        description="Multiply two numbers.",
        parameters_schema=add_schema(),
        func=multiply_numbers,
    )
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_add", arguments={"a": 15, "b": 27}),
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 9},
                ),
            ),
            final_response("The sum is 42 and the product is 54."),
        ]
    )
    loop = build_loop(provider, registry)

    result = loop.run(
        [{"role": "user", "content": "Use both arithmetic tools."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is True
    assert result.reason == "completed"
    assert result.final_response == final_response(
        "The sum is 42 and the product is 54."
    )
    assert executions == [
        ("add_numbers", 15, 27),
        ("multiply_numbers", 6, 9),
    ]
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "54"]
    assert [tool_result.success for tool_result in result.tool_results] == [True, True]
    assert len(provider.requests) == 2
    assert [message.role for message in provider.requests[1].messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    assert provider.requests[1].messages[2].metadata["name"] == "add_numbers"
    assert provider.requests[1].messages[2].content == "42"
    assert provider.requests[1].messages[3].metadata["name"] == "multiply_numbers"
    assert provider.requests[1].messages[3].content == "54"


def test_multiple_sequential_tool_calls() -> None:
    executions: list[tuple[float, float]] = []
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_1", arguments={"a": 1, "b": 2})),
            tool_response(tool_call("call_2", arguments={"a": 3, "b": 4})),
            final_response("done"),
        ]
    )
    loop = build_loop(provider, add_registry(executions))

    result = loop.run(
        [{"role": "user", "content": "Use tools twice."}],
        tools=[add_tool_definition()],
    )

    assert result.completed is True
    assert executions == [(1, 2), (3, 4)]
    assert [tool_result.content for tool_result in result.tool_results] == ["3", "7"]
    assert len(provider.requests) == 3


def test_unknown_tool_failure_is_sanitized() -> None:
    events = RuntimeEventLog()
    secret_argument = "do-not-leak-tool-argument"
    unknown_tool = LLMToolDefinition(
        name="missing_tool",
        description="Missing tool.",
        parameters_schema={"type": "object"},
    )
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_missing",
                    name="missing_tool",
                    arguments={"secret": secret_argument},
                )
            )
        ]
    )
    loop = build_loop(provider, ToolRegistry(), events)

    result = loop.run(
        [{"role": "user", "content": "private prompt"}],
        tools=[unknown_tool],
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert result.tool_results[0].success is False
    assert result.steps[-1].error_type == "UnknownToolError"
    assert result.steps[-1].error_category == "unknown_tool"
    assert secret_argument not in repr(events.events)


def test_tool_validation_error_stops_before_execution() -> None:
    executions: list[tuple[float, float]] = []
    provider = ScriptedProvider(
        [tool_response(tool_call("call_1", arguments={"a": 2}))]
    )
    loop = build_loop(provider, add_registry(executions))

    result = loop.run(
        [{"role": "user", "content": "private prompt"}],
        tools=[add_tool_definition()],
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert executions == []
    assert len(provider.requests) == 1
    assert result.steps[-1].error_category == "validation"
    assert result.tool_results[0].error == "missing required argument: b"


def test_stop_on_tool_error_true_does_not_execute_later_tool_calls() -> None:
    executions: list[tuple[float, float]] = []
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_bad", arguments={"a": 2}),
                tool_call("call_good", arguments={"a": 3, "b": 4}),
            )
        ]
    )
    loop = build_loop(provider, add_registry(executions))

    result = loop.run(
        [{"role": "user", "content": "private prompt"}],
        tools=[add_tool_definition()],
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert executions == []
    assert len(result.tool_results) == 1
    assert result.steps[-1].tool_calls == (tool_call("call_bad", arguments={"a": 2}),)


def test_stop_on_tool_error_false_feeds_error_back_to_llm() -> None:
    executions: list[tuple[float, float]] = []
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_1", arguments={"a": 2})),
            final_response("I need both numbers."),
        ]
    )
    loop = build_loop(provider, add_registry(executions))

    result = loop.run(
        [{"role": "user", "content": "private prompt"}],
        tools=[add_tool_definition()],
        stop_on_tool_error=False,
    )

    assert result.completed is True
    assert executions == []
    assert len(provider.requests) == 2
    assert provider.requests[1].messages[-1].role == "tool"
    assert provider.requests[1].messages[-1].content == "missing required argument: b"
    assert result.final_response == final_response("I need both numbers.")


def test_max_steps_exceeded_does_not_execute_unbounded_tools() -> None:
    executions: list[tuple[float, float]] = []
    first = tool_call("call_1", arguments={"a": 1, "b": 2})
    second = tool_call("call_2", arguments={"a": 3, "b": 4})
    provider = ScriptedProvider([tool_response(first), tool_response(second)])
    events = RuntimeEventLog()
    loop = build_loop(provider, add_registry(executions), events)

    result = loop.run(
        [{"role": "user", "content": "keep using tools"}],
        tools=[add_tool_definition()],
        max_steps=2,
    )

    assert result.completed is False
    assert result.reason == "max_steps_exceeded"
    assert executions == [(1, 2)]
    assert result.pending_tool_calls == (second,)
    assert [event.event_type for event in events.by_type("agent_tool_loop.max_steps_exceeded")]
    assert events.by_type("agent_tool_loop.failed")[0].metadata["error_category"] == (
        "max_steps_exceeded"
    )


def test_require_tool_approval_returns_pending_without_executing() -> None:
    executions: list[tuple[float, float]] = []
    call = tool_call("call_1")
    provider = ScriptedProvider([tool_response(call)])
    events = RuntimeEventLog()
    loop = build_loop(provider, add_registry(executions), events)

    result = loop.run(
        [{"role": "user", "content": "calculate"}],
        tools=[add_tool_definition()],
        require_tool_approval=True,
    )

    assert result.completed is False
    assert result.reason == "approval_required"
    assert result.pending_tool_calls == (call,)
    assert executions == []
    assert len(provider.requests) == 1
    assert events.by_type("agent_tool_loop.approval_required")


def test_llm_runtime_chat_still_does_not_execute_tools() -> None:
    executed = False

    def add_numbers(a: float, b: float) -> float:
        nonlocal executed
        executed = True
        return a + b

    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=add_numbers,
    )
    provider = ScriptedProvider([tool_response(tool_call("call_1"))])

    response = LLMRuntime(provider).chat(
        [{"role": "user", "content": "use a tool"}],
        tools=registry.llm_tool_definitions(),
    )

    assert response.tool_calls[0].name == "add_numbers"
    assert executed is False


def test_events_do_not_leak_prompts_arguments_or_keys() -> None:
    prompt = "private-prompt-value"
    secret_argument = "private-argument-value"
    api_key = "private-api-key"
    result_secret = "private-result-value"
    events = RuntimeEventLog()
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema={
            **add_schema(),
            "additionalProperties": True,
        },
        func=lambda a, b, **kwargs: {"answer": a + b, "secret": result_secret},
    )
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_1",
                    arguments={"a": 15, "b": 27, "secret": secret_argument},
                )
            )
        ]
    )
    loop = build_loop(provider, registry, events)

    loop.run(
        [{"role": "user", "content": prompt}],
        tools=[
            LLMToolDefinition(
                name="add_numbers",
                description="Add two numbers.",
                parameters_schema={
                    **add_schema(),
                    "additionalProperties": True,
                },
            )
        ],
        metadata={"api_key": api_key},
    )

    rendered_events = repr(events.events)
    assert prompt not in rendered_events
    assert secret_argument not in rendered_events
    assert api_key not in rendered_events
    assert result_secret not in rendered_events
    assert "agent_tool_loop.tool_execution_started" in rendered_events
    assert "agent_tool_loop.tool_execution_completed" in rendered_events
