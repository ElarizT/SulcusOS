from __future__ import annotations

from kernel.events import RuntimeEventLog
from kernel.llm import LLMMessage, LLMResponse, LLMRuntime, LLMToolCall, LLMToolDefinition
from kernel.tools import (
    ToolCallRequest,
    ToolExecutionResult,
    ToolRegistry,
    ToolRegistrationError,
    ToolRuntime,
    tool_call_request_from_llm,
    tool_definition_from_llm,
)


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


def test_tool_registration_and_safe_registered_event() -> None:
    events = RuntimeEventLog()
    registry = ToolRegistry(events)

    definition = registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a + b,
        metadata={"owner": "tests"},
    )

    assert definition.name == "add_numbers"
    assert registry.get("add_numbers") == definition
    assert [event.event_type for event in events.events] == ["tool.registered"]
    assert events.events[0].metadata == {
        "tool_name": "add_numbers",
        "success": True,
        "parallel_safe": False,
        "property_count": 2,
        "required_count": 2,
    }


def test_duplicate_name_registration_is_rejected() -> None:
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a + b,
    )

    try:
        registry.register(
            name="add_numbers",
            description="Add two numbers.",
            parameters_schema=add_schema(),
            func=lambda a, b: a + b,
        )
    except ToolRegistrationError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate registration should fail")


def test_invalid_tool_name_is_rejected() -> None:
    registry = ToolRegistry()

    try:
        registry.register(
            name="bad tool",
            description="Bad tool.",
            parameters_schema={"type": "object"},
            func=lambda: None,
        )
    except ToolRegistrationError as exc:
        assert "unsupported characters" in str(exc)
    else:
        raise AssertionError("invalid name should fail")


def test_registry_listing_and_snapshot_are_deterministic() -> None:
    registry = ToolRegistry()
    registry.register(
        name="z_tool",
        description="Last.",
        parameters_schema={"type": "object"},
        func=lambda: "z",
    )
    registry.register(
        name="a_tool",
        description="First.",
        parameters_schema={"type": "object"},
        func=lambda: "a",
        timeout_seconds=3,
    )

    assert registry.names() == ("a_tool", "z_tool")
    assert [definition.name for definition in registry.list()] == ["a_tool", "z_tool"]
    assert registry.snapshot() == (
        {
            "name": "a_tool",
            "description": "First.",
            "parameters_schema": {"type": "object"},
            "timeout_seconds": 3,
            "parallel_safe": False,
            "metadata": {},
        },
        {
            "name": "z_tool",
            "description": "Last.",
            "parameters_schema": {"type": "object"},
            "timeout_seconds": None,
            "parallel_safe": False,
            "metadata": {},
        },
    )


def test_successful_tool_execution_emits_safe_events() -> None:
    events = RuntimeEventLog()
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a + b,
    )

    result = ToolRuntime(registry=registry, event_sink=events).execute(
        "add_numbers",
        {"a": 2, "b": 3},
    )

    assert result == ToolExecutionResult(
        name="add_numbers",
        content=5,
        success=True,
        duration_ms=result.duration_ms,
    )
    assert [event.event_type for event in events.events] == [
        "tool.execution_requested",
        "tool.execution_started",
        "tool.execution_completed",
    ]
    assert events.events[-1].metadata["success"] is True
    assert events.events[-1].metadata["argument_keys"] == ("a", "b")
    for event in events.events:
        assert 2 not in event.metadata.values()
        assert 3 not in event.metadata.values()


def test_unknown_tool_returns_sanitized_failed_result() -> None:
    events = RuntimeEventLog()
    result = ToolRuntime(registry=ToolRegistry(), event_sink=events).execute(
        "missing_tool",
        {"secret": "do-not-leak"},
    )

    assert result.success is False
    assert result.error == "Tool execution rejected"
    assert result.error_type == "UnknownToolError"
    assert result.error_category == "unknown_tool"
    assert [event.event_type for event in events.events] == [
        "tool.execution_requested",
        "tool.execution_rejected",
    ]
    assert "do-not-leak" not in repr(events.events)
    assert events.events[-1].metadata["argument_keys"] == ("secret",)


def test_callable_exception_is_sanitized_into_failed_result() -> None:
    secret = "private failure detail"
    events = RuntimeEventLog()
    registry = ToolRegistry()

    def fail() -> None:
        raise RuntimeError(secret)

    registry.register(
        name="explode",
        description="Fail safely.",
        parameters_schema={"type": "object"},
        func=fail,
    )

    result = ToolRuntime(registry=registry, event_sink=events).execute("explode", {})

    assert result.success is False
    assert result.error == "Tool execution failed"
    assert result.error_type == "RuntimeError"
    assert secret not in repr(result)
    assert secret not in repr(events.events)
    assert events.events[-1].event_type == "tool.execution_failed"
    assert events.events[-1].metadata["error_category"] == "exception"


def test_required_argument_validation_rejects_before_execution() -> None:
    executed = False
    registry = ToolRegistry()

    def add_numbers(a: int, b: int) -> int:
        nonlocal executed
        executed = True
        return a + b

    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=add_numbers,
    )

    result = ToolRuntime(registry=registry).execute("add_numbers", {"a": 2})

    assert executed is False
    assert result.success is False
    assert result.error == "missing required argument: b"
    assert result.error_category == "validation"


def test_basic_type_validation_rejects_before_execution() -> None:
    executed = False
    registry = ToolRegistry()

    def add_numbers(a: int, b: int) -> int:
        nonlocal executed
        executed = True
        return a + b

    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=add_numbers,
    )

    result = ToolRuntime(registry=registry).execute(
        "add_numbers",
        {"a": 2, "b": "private-value"},
    )

    assert executed is False
    assert result.success is False
    assert result.error == "invalid type for argument: b"
    assert "private-value" not in repr(result)


def test_additional_properties_false_rejects_unknown_fields() -> None:
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a + b,
    )

    result = ToolRuntime(registry=registry).execute(
        "add_numbers",
        {"a": 2, "b": 3, "secret": "do-not-leak"},
    )

    assert result.success is False
    assert result.error == "unknown argument: secret"
    assert "do-not-leak" not in repr(result)


def test_llm_tool_call_conversion_and_execution() -> None:
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a + b,
    )
    llm_call = LLMToolCall(
        id="call_1",
        name="add_numbers",
        arguments={"a": 2, "b": 3},
        provider="fake-provider",
        model="fake-model",
    )

    request = tool_call_request_from_llm(llm_call)
    result = ToolRuntime(registry=registry).execute(request)

    assert request == ToolCallRequest(
        name="add_numbers",
        arguments={"a": 2, "b": 3},
        tool_call_id="call_1",
        metadata={"provider": "fake-provider", "model": "fake-model"},
    )
    assert result.content == 5
    assert result.tool_call_id == "call_1"


def test_tool_execution_result_to_llm_tool_result() -> None:
    result = ToolExecutionResult(
        name="add_numbers",
        content=5,
        success=True,
        tool_call_id="call_1",
    )

    llm_result = result.to_llm_tool_result()

    assert llm_result.tool_call_id == "call_1"
    assert llm_result.name == "add_numbers"
    assert llm_result.content == "5"
    assert llm_result.success is True


def test_llm_tool_definition_conversion_requires_explicit_callable() -> None:
    llm_tool = LLMToolDefinition(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
    )
    definition = tool_definition_from_llm(llm_tool, func=lambda a, b: a + b)

    assert definition.name == "add_numbers"
    assert definition.to_llm_tool_definition() == llm_tool


def test_llm_runtime_chat_does_not_automatically_execute_tools() -> None:
    executed = False

    class ToolCallingProvider:
        name = "fake-llm"
        default_model = "fake-model"

        def complete(self, request):
            return LLMResponse(
                "",
                request.model,
                self.name,
                tool_calls=(
                    LLMToolCall(
                        id="call_1",
                        name="add_numbers",
                        arguments={"a": 2, "b": 3},
                    ),
                ),
            )

    registry = ToolRegistry()

    def add_numbers(a: int, b: int) -> int:
        nonlocal executed
        executed = True
        return a + b

    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=add_numbers,
    )

    response = LLMRuntime(ToolCallingProvider()).chat(
        [LLMMessage("user", "call add_numbers")],
        tools=registry.llm_tool_definitions(),
    )

    assert response.tool_calls[0].name == "add_numbers"
    assert executed is False
