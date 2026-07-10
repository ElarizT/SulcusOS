from __future__ import annotations

import time
from typing import Any

import pytest

from kernel.agent_tool_loop import (
    AgentToolLoop,
    AgentToolLoopConfig,
    ToolApprovalDecision,
    ToolPermissionPolicy,
    ToolResourceLimits,
)
from kernel.events import RuntimeEventLog
from kernel.llm import LLMMessage, LLMRequest, LLMResponse, LLMRuntime, LLMToolCall, LLMToolDefinition
from kernel.tools import ToolRegistry, ToolRuntime


class ScriptedProvider:
    name = "fake-agent-llm"
    default_model = "fake-model"

    def __init__(self, responses: list[LLMResponse | Exception]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FailingProvider:
    name = "failing-agent-llm"
    default_model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        raise RuntimeError("provider failed")


def event_types(events: RuntimeEventLog) -> list[str]:
    return [event.event_type for event in events.events]


def timeline_event_types(events: RuntimeEventLog) -> list[str]:
    return [
        event.event_type
        for event in events.events
        if not event.event_type.startswith(("agent_tool_loop.", "llm.", "tool."))
    ]


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


def arithmetic_registry(
    executions: list[tuple[str, float, float]] | None = None,
    *,
    parallel_safe: bool = False,
) -> ToolRegistry:
    registry = ToolRegistry()

    def add_numbers(a: float, b: float) -> float:
        if executions is not None:
            executions.append(("add_numbers", a, b))
        return a + b

    def multiply_numbers(a: float, b: float) -> float:
        if executions is not None:
            executions.append(("multiply_numbers", a, b))
        return a * b

    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=add_numbers,
        parallel_safe=parallel_safe,
    )
    registry.register(
        name="multiply_numbers",
        description="Multiply two numbers.",
        parameters_schema=add_schema(),
        func=multiply_numbers,
        parallel_safe=parallel_safe,
    )
    return registry


def slow_parallel_arithmetic_registry() -> ToolRegistry:
    registry = ToolRegistry()
    schema = add_schema()
    raw_properties = schema["properties"]
    assert isinstance(raw_properties, dict)
    properties = dict(raw_properties)
    schema_with_delay = {
        **schema,
        "properties": {
            **properties,
            "delay_ms": {"type": "integer"},
        },
    }

    def slow_add_numbers(a: float, b: float, delay_ms: int = 80) -> float:
        time.sleep(delay_ms / 1000)
        return a + b

    def slow_multiply_numbers(a: float, b: float, delay_ms: int = 10) -> float:
        time.sleep(delay_ms / 1000)
        return a * b

    registry.register(
        name="slow_add_numbers",
        description="Slowly add two numbers.",
        parameters_schema=schema_with_delay,
        func=slow_add_numbers,
        parallel_safe=True,
    )
    registry.register(
        name="slow_multiply_numbers",
        description="Slowly multiply two numbers.",
        parameters_schema=schema_with_delay,
        func=slow_multiply_numbers,
        parallel_safe=True,
    )
    return registry


def build_loop(
    provider: ScriptedProvider,
    registry: ToolRegistry,
    events: RuntimeEventLog | None = None,
    config: AgentToolLoopConfig | None = None,
) -> AgentToolLoop:
    return AgentToolLoop(
        llm_runtime=LLMRuntime(provider, events),
        tool_runtime=ToolRuntime(registry=registry, event_sink=events),
        config=config,
    )


def test_default_tool_execution_mode_is_sequential() -> None:
    config = AgentToolLoopConfig()

    assert config.tool_execution_mode == "sequential"


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
    assert events.by_type("agent_tool_loop_started")[0].metadata["execution_mode"] == (
        "sequential"
    )
    assert events.by_type("agent_tool_loop_started")[0].metadata[
        "tool_policy_enabled"
    ] is False
    assert events.by_type("agent_tool_loop_started")[0].metadata[
        "policy_default_allow"
    ] is True
    assert events.by_type("agent_tool_loop_started")[0].metadata[
        "allowed_tool_count"
    ] == 0
    assert events.by_type("agent_tool_loop_started")[0].metadata[
        "denied_tool_count"
    ] == 0
    assert timeline_event_types(events) == [
        "agent_tool_loop_started",
        "llm_request_started",
        "llm_response_received",
        "agent_tool_loop_completed",
    ]
    assert result.tool_results == ()


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


def test_successful_one_tool_loop_emits_timeline_events() -> None:
    events = RuntimeEventLog()
    call = tool_call("call_1")
    provider = ScriptedProvider([tool_response(call), final_response("42")])
    loop = build_loop(provider, add_registry(), events)

    result = loop.run(
        [{"role": "user", "content": "Calculate 15 + 27 using tools."}],
        tools=[add_tool_definition()],
    )

    assert result.completed is True
    assert timeline_event_types(events) == [
        "agent_tool_loop_started",
        "llm_request_started",
        "llm_response_received",
        "tool_execution_group_started",
        "tool_call_requested",
        "tool_execution_started",
        "tool_execution_completed",
        "tool_execution_group_completed",
        "llm_followup_request_started",
        "llm_final_request_started",
        "llm_followup_response_received",
        "llm_final_response_received",
        "agent_tool_loop_completed",
    ]
    requested = events.by_type("tool_call_requested")[0]
    assert requested.metadata["tool_call_id"] == "call_1"
    assert requested.metadata["tool_name"] == "add_numbers"
    assert requested.metadata["argument_keys"] == ("a", "b")
    assert "15" not in repr(requested.metadata)
    assert events.by_type("tool_execution_completed")[0].metadata["output_preview"] == "42"
    assert events.by_type("tool_execution_group_started")[0].metadata[
        "execution_mode"
    ] == "sequential"
    assert events.by_type("tool_execution_group_completed")[0].metadata[
        "successful_tool_count"
    ] == 1
    completed = events.by_type("agent_tool_loop_completed")[0]
    assert completed.metadata["completed"] is True
    assert completed.metadata["reason"] == "completed"
    assert completed.metadata["tool_result_count"] == 1
    assert completed.metadata["round_index"] == 1


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


def test_explicit_sequential_mode_preserves_multi_tool_result_order() -> None:
    events = RuntimeEventLog()
    registry = arithmetic_registry()
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
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(tool_execution_mode="sequential"),
    )

    result = loop.run(
        [{"role": "user", "content": "Use both arithmetic tools."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is True
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "54"]
    assert [
        event.metadata["execution_mode"]
        for event in events.by_type("tool_execution_completed")
    ] == ["sequential", "sequential"]


def test_parallel_mode_with_safe_tools_preserves_request_order() -> None:
    events = RuntimeEventLog()
    registry = slow_parallel_arithmetic_registry()
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_slow_add",
                    name="slow_add_numbers",
                    arguments={"a": 20, "b": 22, "delay_ms": 80},
                ),
                tool_call(
                    "call_fast_multiply",
                    name="slow_multiply_numbers",
                    arguments={"a": 6, "b": 7, "delay_ms": 10},
                ),
            ),
            final_response("Both results are available."),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(tool_execution_mode="parallel"),
    )

    result = loop.run(
        [{"role": "user", "content": "Use both slow arithmetic tools."}],
        tools=registry.llm_tool_definitions(),
    )

    assert result.completed is True
    assert [tool_result.name for tool_result in result.tool_results] == [
        "slow_add_numbers",
        "slow_multiply_numbers",
    ]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "42"]
    group_started = events.by_type("tool_execution_group_started")[0]
    assert group_started.metadata["requested_execution_mode"] == "parallel"
    assert group_started.metadata["effective_execution_mode"] == "parallel"
    assert group_started.metadata["execution_mode"] == "parallel"
    assert group_started.metadata["parallel_safe_tool_count"] == 2
    assert group_started.metadata["unsafe_tool_count"] == 0
    assert "fallback_reason" not in group_started.metadata
    assert [
        message.metadata["name"] for message in provider.requests[1].messages[2:]
    ] == ["slow_add_numbers", "slow_multiply_numbers"]
    assert [message.content for message in provider.requests[1].messages[2:]] == [
        "42",
        "42",
    ]


def test_parallel_mode_falls_back_to_sequential_when_tool_is_not_safe() -> None:
    events = RuntimeEventLog()
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
        parallel_safe=True,
    )
    registry.register(
        name="multiply_numbers",
        description="Multiply two numbers.",
        parameters_schema=add_schema(),
        func=multiply_numbers,
        parallel_safe=False,
    )
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_add", arguments={"a": 20, "b": 22}),
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 7},
                ),
            ),
            final_response("The results are 42 and 42."),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(tool_execution_mode="parallel"),
    )

    result = loop.run(
        [{"role": "user", "content": "Use both arithmetic tools."}],
        tools=registry.llm_tool_definitions(),
    )

    assert result.completed is True
    assert executions == [("add_numbers", 20, 22), ("multiply_numbers", 6, 7)]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "42"]
    group_started = events.by_type("tool_execution_group_started")[0]
    assert group_started.metadata["requested_execution_mode"] == "parallel"
    assert group_started.metadata["effective_execution_mode"] == "sequential"
    assert group_started.metadata["execution_mode"] == "sequential"
    assert group_started.metadata["fallback_reason"] == "not_all_tools_parallel_safe"
    assert group_started.metadata["parallel_safe_tool_count"] == 1
    assert group_started.metadata["unsafe_tool_count"] == 1


def test_parallel_group_failure_collects_completed_tool_results() -> None:
    events = RuntimeEventLog()
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a + b,
        parallel_safe=True,
    )
    registry.register(
        name="fail_numbers",
        description="Fail while handling numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: (_ for _ in ()).throw(RuntimeError("boom")),
        parallel_safe=True,
    )
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_add", arguments={"a": 20, "b": 22}),
                tool_call(
                    "call_fail",
                    name="fail_numbers",
                    arguments={"a": 6, "b": 7},
                ),
            )
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(tool_execution_mode="parallel"),
    )

    result = loop.run(
        [{"role": "user", "content": "Use both tools."}],
        tools=registry.llm_tool_definitions(),
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "fail_numbers",
    ]
    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    assert "tool_execution_failed" in event_types(events)
    group_failure = events.by_type("tool_execution_group_failed")[0]
    assert group_failure.metadata["requested_execution_mode"] == "parallel"
    assert group_failure.metadata["effective_execution_mode"] == "parallel"
    assert group_failure.metadata["successful_tool_count"] == 1
    assert group_failure.metadata["failed_tool_count"] == 1


def test_default_tool_permission_policy_preserves_tool_execution() -> None:
    executions: list[tuple[str, float, float]] = []
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_add", arguments={"a": 15, "b": 27})),
            final_response("42"),
        ]
    )
    loop = build_loop(
        provider,
        arithmetic_registry(executions),
        events,
        AgentToolLoopConfig(tool_permission_policy=ToolPermissionPolicy()),
    )

    result = loop.run(
        [{"role": "user", "content": "Use add_numbers."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is True
    assert executions == [("add_numbers", 15, 27)]
    started = events.by_type("agent_tool_loop_started")[0]
    assert started.metadata["tool_policy_enabled"] is True
    assert started.metadata["policy_default_allow"] is True
    assert started.metadata["allowed_tool_count"] == 0
    assert started.metadata["denied_tool_count"] == 0


def test_allowlist_tool_permission_policy_permits_listed_tool() -> None:
    executions: list[tuple[str, float, float]] = []
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_add", arguments={"a": 15, "b": 27})),
            final_response("42"),
        ]
    )
    loop = build_loop(
        provider,
        arithmetic_registry(executions),
        config=AgentToolLoopConfig(
            tool_permission_policy=ToolPermissionPolicy(
                default_allow=False,
                allowed_tools={"add_numbers"},
            )
        ),
    )

    result = loop.run(
        [{"role": "user", "content": "Use add_numbers."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is True
    assert result.reason == "completed"
    assert executions == [("add_numbers", 15, 27)]
    assert result.tool_results[0].success is True


def test_allowlist_tool_permission_policy_denies_unlisted_tool() -> None:
    executions: list[tuple[str, float, float]] = []
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 9},
                )
            )
        ]
    )
    loop = build_loop(
        provider,
        arithmetic_registry(executions),
        events,
        AgentToolLoopConfig(
            tool_permission_policy=ToolPermissionPolicy(
                default_allow=False,
                allowed_tools={"add_numbers"},
            )
        ),
    )

    result = loop.run(
        [{"role": "user", "content": "Use multiply_numbers."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert executions == []
    assert result.tool_results[0].success is False
    assert result.tool_results[0].name == "multiply_numbers"
    assert "Tool call denied by permission policy: multiply_numbers" == (
        result.tool_results[0].error
    )
    denied = events.by_type("tool_call_denied")[0]
    assert denied.metadata["tool_call_id"] == "call_multiply"
    assert denied.metadata["tool_name"] == "multiply_numbers"
    assert denied.metadata["reason"] == "permission_policy"
    assert denied.metadata["round_index"] == 0
    assert denied.metadata["requested_execution_mode"] == "sequential"
    assert denied.metadata["effective_execution_mode"] == "sequential"
    assert denied.metadata["policy_default_allow"] is False
    assert denied.metadata["matched_rule"] == "not_in_allowed_tools"
    assert events.by_type("tool_execution_started") == []
    group_failure = events.by_type("tool_execution_group_failed")[0]
    assert group_failure.metadata["failed_tool_count"] == 1
    assert group_failure.metadata["denied_tool_count"] == 1


def test_denylist_tool_permission_policy_blocks_tool() -> None:
    executions: list[tuple[str, float, float]] = []
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 9},
                )
            )
        ]
    )
    loop = build_loop(
        provider,
        arithmetic_registry(executions),
        events,
        AgentToolLoopConfig(
            tool_permission_policy=ToolPermissionPolicy(
                default_allow=True,
                denied_tools={"multiply_numbers"},
            )
        ),
    )

    result = loop.run(
        [{"role": "user", "content": "Use multiply_numbers."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is False
    assert executions == []
    assert result.tool_results[0].error == (
        "Tool call denied by permission policy: multiply_numbers"
    )
    denied = events.by_type("tool_call_denied")[0]
    assert denied.metadata["matched_rule"] == "denied_tools"
    assert denied.metadata["policy_default_allow"] is True


def test_denylist_tool_permission_policy_wins_over_allowlist() -> None:
    executions: list[tuple[str, float, float]] = []
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 9},
                )
            )
        ]
    )
    loop = build_loop(
        provider,
        arithmetic_registry(executions),
        events,
        AgentToolLoopConfig(
            tool_permission_policy=ToolPermissionPolicy(
                allowed_tools={"multiply_numbers"},
                denied_tools={"multiply_numbers"},
            )
        ),
    )

    result = loop.run(
        [{"role": "user", "content": "Use multiply_numbers."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is False
    assert executions == []
    assert result.tool_results[0].success is False
    assert events.by_type("tool_call_denied")[0].metadata["matched_rule"] == "denied_tools"


def test_multi_round_parallel_group_waits_before_next_llm_round() -> None:
    events = RuntimeEventLog()
    registry = slow_parallel_arithmetic_registry()
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_slow_add",
                    name="slow_add_numbers",
                    arguments={"a": 20, "b": 22, "delay_ms": 40},
                ),
                tool_call(
                    "call_fast_multiply",
                    name="slow_multiply_numbers",
                    arguments={"a": 6, "b": 7, "delay_ms": 5},
                ),
            ),
            final_response("The parallel group produced 42 and 42."),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(tool_execution_mode="parallel"),
    )

    result = loop.run(
        [{"role": "user", "content": "Use both slow arithmetic tools."}],
        tools=registry.llm_tool_definitions(),
        max_steps=3,
    )

    assert result.completed is True
    assert len(provider.requests) == 2
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "42"]
    assert events.by_type("tool_execution_group_started")[0].metadata["round_index"] == 0
    assert events.by_type("tool_execution_group_completed")[0].metadata["round_index"] == 0
    assert events.by_type("llm_followup_request_started")[0].metadata["round_index"] == 1


def test_multi_tool_loop_emits_timeline_events_per_tool_call() -> None:
    events = RuntimeEventLog()
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a + b,
    )
    registry.register(
        name="multiply_numbers",
        description="Multiply two numbers.",
        parameters_schema=add_schema(),
        func=lambda a, b: a * b,
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
    loop = build_loop(provider, registry, events)

    result = loop.run(
        [{"role": "user", "content": "Use both arithmetic tools."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is True
    assert [event.metadata["tool_name"] for event in events.by_type("tool_call_requested")] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [
        event.metadata["tool_name"] for event in events.by_type("tool_execution_started")
    ] == ["add_numbers", "multiply_numbers"]
    assert [
        event.metadata["tool_name"] for event in events.by_type("tool_execution_completed")
    ] == ["add_numbers", "multiply_numbers"]
    assert [event.metadata["output_preview"] for event in events.by_type("tool_execution_completed")] == [
        "42",
        "54",
    ]
    group_started = events.by_type("tool_execution_group_started")
    group_completed = events.by_type("tool_execution_group_completed")
    assert len(group_started) == 1
    assert len(group_completed) == 1
    assert group_started[0].metadata["tool_call_count"] == 2
    assert group_started[0].metadata["tool_names"] == (
        "add_numbers",
        "multiply_numbers",
    )
    assert group_started[0].metadata["execution_mode"] == "sequential"
    assert group_completed[0].metadata["tool_call_count"] == 2
    assert group_completed[0].metadata["successful_tool_count"] == 2
    assert group_completed[0].metadata["failed_tool_count"] == 0
    assert group_completed[0].metadata["execution_mode"] == "sequential"


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


def test_multi_round_two_tool_loop_reaches_final_response() -> None:
    executions: list[tuple[str, float, float]] = []
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_add", arguments={"a": 20, "b": 22})),
            tool_response(
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 42, "b": 2},
                )
            ),
            final_response("The final answer is 84."),
        ]
    )
    events = RuntimeEventLog()
    loop = build_loop(provider, arithmetic_registry(executions), events)

    result = loop.run(
        [{"role": "user", "content": "Compute a multi-round answer."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
        max_steps=4,
    )

    assert result.completed is True
    assert result.reason == "completed"
    assert result.final_response is not None
    assert "84" in result.final_response.content
    assert executions == [
        ("add_numbers", 20, 22),
        ("multiply_numbers", 42, 2),
    ]
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "84"]
    assert [tool_result.success for tool_result in result.tool_results] == [True, True]
    assert len(provider.requests) == 3
    assert [message.role for message in provider.requests[1].messages] == [
        "user",
        "assistant",
        "tool",
    ]
    assert provider.requests[1].messages[2].content == "42"
    assert [message.role for message in provider.requests[2].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert provider.requests[2].messages[2].content == "42"
    assert provider.requests[2].messages[4].content == "84"
    assert [event.metadata["round_index"] for event in events.by_type("llm_followup_request_started")] == [
        1,
        2,
    ]
    assert events.by_type("llm_final_response_received")[0].metadata["round_index"] == 2
    assert [event.metadata["execution_mode"] for event in events.by_type("tool_execution_group_started")] == [
        "sequential",
        "sequential",
    ]
    assert [event.metadata["round_index"] for event in events.by_type("tool_execution_group_completed")] == [
        0,
        1,
    ]


def test_multi_round_max_steps_stops_before_unbounded_tool_execution() -> None:
    executions: list[tuple[float, float]] = []
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_1", arguments={"a": 1, "b": 2})),
            tool_response(tool_call("call_2", arguments={"a": 3, "b": 4})),
        ]
    )
    events = RuntimeEventLog()
    loop = build_loop(provider, add_registry(executions), events)

    result = loop.run(
        [{"role": "user", "content": "keep calling tools"}],
        tools=[add_tool_definition()],
        max_steps=2,
    )

    assert result.completed is False
    assert result.reason == "max_steps_exceeded"
    assert executions == [(1, 2)]
    assert result.pending_tool_calls == (tool_call("call_2", arguments={"a": 3, "b": 4}),)
    failed = events.by_type("agent_tool_loop_failed")[-1]
    assert failed.metadata["reason"] == "max_steps_exceeded"
    assert failed.metadata["round_index"] == 1


def test_multi_round_tool_failure_preserves_existing_failure_semantics() -> None:
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_1", arguments={"a": 20, "b": 22})),
            tool_response(
                tool_call(
                    "call_bad",
                    name="multiply_numbers",
                    arguments={"a": 42},
                )
            ),
        ]
    )
    loop = build_loop(provider, arithmetic_registry(), events)

    result = loop.run(
        [{"role": "user", "content": "fail in the second round"}],
        tools=[add_tool_definition(), multiply_tool_definition()],
        max_steps=4,
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert result.tool_results[-1].success is False
    assert events.by_type("tool_execution_failed")[0].metadata["round_index"] == 1
    group_failure = events.by_type("tool_execution_group_failed")[0]
    assert group_failure.metadata["execution_mode"] == "sequential"
    assert group_failure.metadata["round_index"] == 1
    assert group_failure.metadata["tool_call_count"] == 1
    assert group_failure.metadata["successful_tool_count"] == 0
    assert group_failure.metadata["failed_tool_count"] == 1
    assert events.by_type("agent_tool_loop_failed")[-1].metadata["reason"] == "tool_error"


def test_multi_round_tool_permission_policy_enforces_each_round() -> None:
    executions: list[tuple[str, float, float]] = []
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_add", arguments={"a": 20, "b": 22})),
            tool_response(
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 42, "b": 2},
                )
            ),
        ]
    )
    loop = build_loop(
        provider,
        arithmetic_registry(executions),
        events,
        AgentToolLoopConfig(
            tool_permission_policy=ToolPermissionPolicy(
                default_allow=False,
                allowed_tools={"add_numbers"},
            )
        ),
    )

    result = loop.run(
        [{"role": "user", "content": "Compute a multi-round answer."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
        max_steps=4,
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert executions == [("add_numbers", 20, 22)]
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    denied = events.by_type("tool_call_denied")[0]
    assert denied.metadata["round_index"] == 1
    assert denied.metadata["tool_name"] == "multiply_numbers"
    assert events.by_type("tool_execution_group_failed")[0].metadata["round_index"] == 1


def test_parallel_tool_permission_policy_denies_without_scheduling_tool() -> None:
    executions: list[tuple[str, float, float]] = []
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_add", arguments={"a": 20, "b": 22}),
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 7},
                ),
            )
        ]
    )
    loop = build_loop(
        provider,
        arithmetic_registry(executions, parallel_safe=True),
        events,
        AgentToolLoopConfig(
            tool_execution_mode="parallel",
            tool_permission_policy=ToolPermissionPolicy(
                default_allow=False,
                allowed_tools={"add_numbers"},
            ),
        ),
    )

    result = loop.run(
        [{"role": "user", "content": "Use both tools."}],
        tools=[add_tool_definition(), multiply_tool_definition()],
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert executions == [("add_numbers", 20, 22)]
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    assert [
        event.metadata["tool_name"] for event in events.by_type("tool_execution_started")
    ] == ["add_numbers"]
    denied = events.by_type("tool_call_denied")[0]
    assert denied.metadata["tool_name"] == "multiply_numbers"
    assert denied.metadata["requested_execution_mode"] == "parallel"
    assert denied.metadata["effective_execution_mode"] == "parallel"
    group_failure = events.by_type("tool_execution_group_failed")[0]
    assert group_failure.metadata["successful_tool_count"] == 1
    assert group_failure.metadata["failed_tool_count"] == 1
    assert group_failure.metadata["denied_tool_count"] == 1


def test_tool_resource_limits_validate_nonnegative_values() -> None:
    ToolResourceLimits(max_tool_calls_per_loop=0)

    for kwargs in (
        {"max_tool_calls_per_loop": -1},
        {"max_tool_calls_per_round": -1},
        {"tool_timeout_ms": -1},
        {"max_calls_per_tool": {"add_numbers": -1}},
    ):
        try:
            ToolResourceLimits(**kwargs)  # type: ignore[arg-type]
        except ValueError as exc:
            assert "nonnegative integer" in str(exc)
        else:
            raise AssertionError(f"invalid limits should fail: {kwargs}")


def test_max_tool_calls_per_loop_allows_within_limit_across_rounds() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[str, float, float]] = []
    registry = arithmetic_registry(executions)
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_add")),
            tool_response(
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 7},
                )
            ),
            final_response("done"),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_resource_limits=ToolResourceLimits(max_tool_calls_per_loop=2)
        ),
    )

    result = loop.run([{"role": "user", "content": "Use two tools."}])

    assert result.completed is True
    assert [tool_result.success for tool_result in result.tool_results] == [True, True]
    assert [execution[0] for execution in executions] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert events.by_type("tool_call_resource_denied") == []


def test_max_tool_calls_per_loop_denies_when_exceeded() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[str, float, float]] = []
    registry = arithmetic_registry(executions)
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_add"),
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 7},
                ),
            ),
            final_response("not reached"),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_resource_limits=ToolResourceLimits(max_tool_calls_per_loop=1)
        ),
    )

    result = loop.run([{"role": "user", "content": "Use two tools."}])

    assert result.completed is False
    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    assert result.tool_results[1].error == (
        "Tool call denied by resource limits: max_tool_calls_per_loop exceeded"
    )
    assert [execution[0] for execution in executions] == ["add_numbers"]
    denied = events.by_type("tool_call_resource_denied")[0]
    assert denied.metadata["limit_name"] == "max_tool_calls_per_loop"
    assert denied.metadata["limit_value"] == 1
    assert denied.metadata["current_count"] == 2
    assert [
        event.metadata["tool_name"]
        for event in events.by_type("tool_execution_started")
    ] == ["add_numbers"]
    group_failure = events.by_type("tool_execution_group_failed")[0]
    assert group_failure.metadata["resource_denied_tool_count"] == 1


def test_max_tool_calls_per_round_denies_second_call_in_oversized_group() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[str, float, float]] = []
    registry = arithmetic_registry(executions)
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_add"),
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 7},
                ),
            ),
            final_response("not reached"),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_resource_limits=ToolResourceLimits(max_tool_calls_per_round=1)
        ),
    )

    result = loop.run([{"role": "user", "content": "Use two tools."}])

    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    assert result.tool_results[1].error == (
        "Tool call denied by resource limits: max_tool_calls_per_round exceeded"
    )
    assert [execution[0] for execution in executions] == ["add_numbers"]
    assert events.by_type("tool_call_resource_denied")[0].metadata["limit_name"] == (
        "max_tool_calls_per_round"
    )


def test_max_calls_per_tool_denies_repeated_tool_across_rounds() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[float, float]] = []
    registry = add_registry(executions)
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_add_1")),
            tool_response(tool_call("call_add_2")),
            final_response("not reached"),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_resource_limits=ToolResourceLimits(
                max_calls_per_tool={"add_numbers": 1}
            )
        ),
    )

    result = loop.run([{"role": "user", "content": "Use add twice."}])

    assert result.completed is False
    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    assert result.tool_results[1].error == (
        "Tool call denied by resource limits: max_calls_per_tool exceeded for add_numbers"
    )
    assert executions == [(15, 27)]
    denied = events.by_type("tool_call_resource_denied")[0]
    assert denied.metadata["round_index"] == 1
    assert denied.metadata["limit_name"] == "max_calls_per_tool"


def test_zero_tool_call_limit_denies_all_tool_calls() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[float, float]] = []
    registry = add_registry(executions)
    provider = ScriptedProvider([tool_response(tool_call("call_add"))])
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_resource_limits=ToolResourceLimits(max_tool_calls_per_loop=0)
        ),
    )

    result = loop.run([{"role": "user", "content": "Add numbers."}])

    assert result.completed is False
    assert result.tool_results[0].success is False
    assert executions == []
    assert events.by_type("tool_execution_started") == []
    assert events.by_type("tool_call_resource_denied")[0].metadata["current_count"] == 1


def test_tool_resource_limits_can_be_passed_as_run_override() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[float, float]] = []
    registry = add_registry(executions)
    provider = ScriptedProvider([tool_response(tool_call("call_add"))])
    loop = build_loop(provider, registry, events)

    result = loop.run(
        [{"role": "user", "content": "Add numbers."}],
        tool_resource_limits=ToolResourceLimits(max_tool_calls_per_loop=0),
    )

    assert result.tool_results[0].success is False
    assert executions == []
    assert events.by_type("tool_call_resource_denied")[0].metadata["limit_name"] == (
        "max_tool_calls_per_loop"
    )


def test_permission_denial_wins_over_resource_denial() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[float, float]] = []
    registry = add_registry(executions)
    provider = ScriptedProvider([tool_response(tool_call("call_add"))])
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_permission_policy=ToolPermissionPolicy(
                default_allow=False,
                allowed_tools=frozenset(),
            ),
            tool_resource_limits=ToolResourceLimits(max_tool_calls_per_loop=0),
        ),
    )

    result = loop.run([{"role": "user", "content": "Add numbers."}])

    assert result.tool_results[0].error == (
        "Tool call denied by permission policy: add_numbers"
    )
    assert events.by_type("tool_call_denied")
    assert events.by_type("tool_call_resource_denied") == []
    assert executions == []


def test_parallel_resource_denial_preserves_result_order_without_scheduling_denied_tool() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[str, float, float]] = []
    registry = arithmetic_registry(executions, parallel_safe=True)
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call("call_add"),
                tool_call(
                    "call_multiply",
                    name="multiply_numbers",
                    arguments={"a": 6, "b": 7},
                ),
            ),
            final_response("not reached"),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_execution_mode="parallel",
            tool_resource_limits=ToolResourceLimits(max_tool_calls_per_round=1),
        ),
    )

    result = loop.run([{"role": "user", "content": "Use two tools."}])

    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    assert [execution[0] for execution in executions] == ["add_numbers"]
    assert events.by_type("tool_call_resource_denied")[0].metadata["tool_name"] == (
        "multiply_numbers"
    )


def test_tool_timeout_high_enough_allows_fast_tool() -> None:
    events = RuntimeEventLog()
    executions: list[tuple[float, float]] = []
    registry = add_registry(executions)
    provider = ScriptedProvider([tool_response(tool_call("call_add")), final_response()])
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_resource_limits=ToolResourceLimits(tool_timeout_ms=500)
        ),
    )

    result = loop.run([{"role": "user", "content": "Add numbers."}])

    assert result.completed is True
    assert result.tool_results[0].success is True
    assert events.by_type("tool_execution_failed") == []


def test_tool_timeout_failure_emits_execution_failed_metadata() -> None:
    events = RuntimeEventLog()
    registry = ToolRegistry()

    def slow_tool(label: str) -> str:
        time.sleep(0.05)
        return label

    registry.register(
        name="slow_tool",
        description="Sleep briefly.",
        parameters_schema={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        },
        func=slow_tool,
        parallel_safe=True,
    )
    provider = ScriptedProvider(
        [
            tool_response(
                tool_call(
                    "call_slow",
                    name="slow_tool",
                    arguments={"label": "too late"},
                )
            ),
            final_response("not reached"),
        ]
    )
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_execution_mode="parallel",
            tool_resource_limits=ToolResourceLimits(tool_timeout_ms=1),
        ),
    )

    result = loop.run([{"role": "user", "content": "Use slow tool."}])

    assert result.completed is False
    assert result.tool_results[0].success is False
    assert result.tool_results[0].error == "Tool execution timed out after 1ms"
    failed = events.by_type("tool_execution_failed")[0]
    assert failed.metadata["error_type"] == "ToolTimeoutError"
    assert failed.metadata["error_category"] == "timeout"
    assert failed.metadata["limit_name"] == "tool_timeout_ms"
    assert failed.metadata["limit_value"] == 1
    group_failure = events.by_type("tool_execution_group_failed")[0]
    assert group_failure.metadata["timed_out_tool_count"] == 1


def test_loop_start_and_group_metadata_include_resource_limit_summary() -> None:
    events = RuntimeEventLog()
    registry = add_registry()
    provider = ScriptedProvider([tool_response(tool_call("call_add"))])
    loop = build_loop(
        provider,
        registry,
        events,
        AgentToolLoopConfig(
            tool_resource_limits=ToolResourceLimits(
                max_tool_calls_per_loop=4,
                max_tool_calls_per_round=2,
                max_calls_per_tool={"add_numbers": 3},
                tool_timeout_ms=500,
            )
        ),
    )

    loop.run([{"role": "user", "content": "Add numbers."}])

    started = events.by_type("agent_tool_loop_started")[0]
    assert started.metadata["tool_resource_limits_enabled"] is True
    assert started.metadata["max_tool_calls_per_loop"] == 4
    assert started.metadata["max_tool_calls_per_round"] == 2
    assert started.metadata["max_calls_per_tool_count"] == 1
    assert started.metadata["tool_timeout_ms"] == 500
    group_started = events.by_type("tool_execution_group_started")[0]
    assert group_started.metadata["resource_limits_enabled"] is True
    assert group_started.metadata["max_tool_calls_per_loop"] == 4
    assert group_started.metadata["max_tool_calls_per_round"] == 2
    assert group_started.metadata["tool_timeout_ms"] == 500


def test_multi_round_llm_provider_failure_in_followup_round() -> None:
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [
            tool_response(tool_call("call_1", arguments={"a": 20, "b": 22})),
            RuntimeError("follow-up provider failed"),
        ]
    )
    loop = build_loop(provider, add_registry(), events)

    result = loop.run(
        [{"role": "user", "content": "fail after one tool"}],
        tools=[add_tool_definition()],
        max_steps=4,
    )

    assert result.completed is False
    assert result.reason == "llm_error"
    assert [tool_result.content for tool_result in result.tool_results] == ["42"]
    assert len(provider.requests) == 2
    failed = events.by_type("agent_tool_loop_failed")[-1]
    assert failed.metadata["reason"] == "llm_error"
    assert failed.metadata["round_index"] == 1


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


def test_tool_failure_emits_failed_timeline_events_without_changing_semantics() -> None:
    events = RuntimeEventLog()
    provider = ScriptedProvider(
        [tool_response(tool_call("call_1", arguments={"a": 2}))]
    )
    loop = build_loop(provider, add_registry(), events)

    result = loop.run(
        [{"role": "user", "content": "private prompt"}],
        tools=[add_tool_definition()],
    )

    assert result.completed is False
    assert result.reason == "tool_error"
    assert "tool_execution_failed" in event_types(events)
    assert "agent_tool_loop_failed" in event_types(events)
    failure = events.by_type("tool_execution_failed")[0]
    assert failure.metadata["tool_call_id"] == "call_1"
    assert failure.metadata["tool_name"] == "add_numbers"
    assert failure.metadata["success"] is False
    assert failure.metadata["error_type"] == "ToolValidationError"
    assert failure.metadata["error_category"] == "validation"
    assert "missing required argument: b" in failure.metadata["error_preview"]
    group_failure = events.by_type("tool_execution_group_failed")[0]
    assert group_failure.metadata["execution_mode"] == "sequential"
    assert group_failure.metadata["tool_call_count"] == 1
    assert group_failure.metadata["failed_tool_count"] == 1


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


def test_approval_resume_executes_without_repeating_llm_request() -> None:
    executions: list[tuple[float, float]] = []
    call = tool_call("approve-1")
    provider = ScriptedProvider([tool_response(call), final_response("done")])
    events = RuntimeEventLog()
    loop = build_loop(provider, add_registry(executions), events)

    paused = loop.run(
        [{"role": "user", "content": "calculate"}], [add_tool_definition()],
        require_tool_approval=True,
    )
    assert paused.checkpoint is not None
    assert paused.pending_approvals[0].tool_call_id == "approve-1"
    resumed = loop.resume(
        checkpoint=paused.checkpoint,
        approval_decisions=[ToolApprovalDecision("approve-1", approved=True)],
    )

    assert resumed.completed is True
    assert resumed.final_response == final_response("done")
    assert executions == [(15, 27)]
    assert len(provider.requests) == 2
    event_types = [event.event_type for event in events.events]
    assert event_types.index("tool_approval_requested") < event_types.index("agent_tool_loop_paused")
    assert event_types.index("agent_tool_loop_paused") < event_types.index("agent_tool_loop_resumed")
    assert event_types.index("agent_tool_loop_resumed") < event_types.index("tool_approval_granted")


def test_approval_denial_returns_tool_result_and_consumes_checkpoint() -> None:
    executions: list[tuple[float, float]] = []
    call = tool_call("deny-1")
    provider = ScriptedProvider([tool_response(call), final_response("done")])
    loop = build_loop(provider, add_registry(executions))
    paused = loop.run([{"role": "user", "content": "calculate"}], [add_tool_definition()], require_tool_approval=True)
    assert paused.checkpoint is not None
    resumed = loop.resume(checkpoint=paused.checkpoint, approval_decisions=[ToolApprovalDecision("deny-1", False)])
    assert resumed.completed is False  # default stop_on_tool_error
    assert resumed.reason == "tool_error"
    assert executions == []
    assert resumed.tool_results[0].error == "Tool call denied by approval: add_numbers"
    with pytest.raises(ValueError, match="already been consumed"):
        loop.resume(checkpoint=paused.checkpoint, approval_decisions=[ToolApprovalDecision("deny-1", False)])


def test_partial_approval_keeps_checkpoint_pending() -> None:
    first = tool_call("first")
    second = tool_call("second")
    provider = ScriptedProvider([tool_response(first, second)])
    loop = build_loop(provider, arithmetic_registry())
    paused = loop.run([{"role": "user", "content": "calculate"}], [add_tool_definition(), multiply_tool_definition()], require_tool_approval=True)
    assert paused.checkpoint is not None
    still_paused = loop.resume(checkpoint=paused.checkpoint, approval_decisions=[ToolApprovalDecision("first", True)])
    assert still_paused.checkpoint == paused.checkpoint
    assert still_paused.reason == "approval_required"


def test_approval_preflight_permission_and_limits_do_not_become_pending() -> None:
    allowed = tool_call("allowed")
    denied = tool_call("denied", name="multiply_numbers")
    provider = ScriptedProvider([tool_response(allowed, denied)])
    loop = build_loop(provider, arithmetic_registry())
    paused = loop.run(
        [{"role": "user", "content": "calculate"}],
        [add_tool_definition(), multiply_tool_definition()], require_tool_approval=True,
        tool_permission_policy=ToolPermissionPolicy(denied_tools={"multiply_numbers"}),
    )
    assert [item.tool_call_id for item in paused.pending_approvals] == ["allowed"]


def test_approval_resume_does_not_charge_resource_limit_twice() -> None:
    executions: list[tuple[float, float]] = []
    call = tool_call("limited")
    provider = ScriptedProvider([tool_response(call), final_response("done")])
    loop = build_loop(provider, add_registry(executions))
    paused = loop.run(
        [{"role": "user", "content": "calculate"}], [add_tool_definition()],
        require_tool_approval=True,
        tool_resource_limits=ToolResourceLimits(max_tool_calls_per_loop=1),
    )
    assert paused.checkpoint is not None
    resumed = loop.resume(checkpoint=paused.checkpoint, approval_decisions=[ToolApprovalDecision("limited", True)])
    assert resumed.completed is True
    assert executions == [(15, 27)]


def test_approval_parallel_resume_preserves_requested_order() -> None:
    executions: list[tuple[str, float, float]] = []
    first = tool_call("parallel-add")
    second = tool_call("parallel-multiply", name="multiply_numbers")
    provider = ScriptedProvider([tool_response(first, second), final_response("done")])
    loop = build_loop(provider, arithmetic_registry(executions, parallel_safe=True))
    paused = loop.run(
        [{"role": "user", "content": "calculate"}], [add_tool_definition(), multiply_tool_definition()],
        require_tool_approval=True, tool_execution_mode="parallel",
    )
    assert paused.checkpoint is not None
    assert {item.effective_execution_mode for item in paused.pending_approvals} == {"parallel"}
    resumed = loop.resume(checkpoint=paused.checkpoint, approval_decisions=[
        ToolApprovalDecision("parallel-add", True), ToolApprovalDecision("parallel-multiply", True),
    ])
    assert [result.tool_call_id for result in resumed.tool_results] == ["parallel-add", "parallel-multiply"]


def test_llm_provider_failure_emits_failed_timeline_events() -> None:
    events = RuntimeEventLog()
    provider = FailingProvider()
    loop = build_loop(provider, ToolRegistry(), events)  # type: ignore[arg-type]

    result = loop.run([{"role": "user", "content": "hello"}], tools=[])

    assert result.completed is False
    assert result.reason == "llm_error"
    assert timeline_event_types(events) == [
        "agent_tool_loop_started",
        "llm_request_started",
        "agent_tool_loop_failed",
    ]
    failed = events.by_type("agent_tool_loop_failed")[0]
    assert failed.metadata["completed"] is False
    assert failed.metadata["reason"] == "llm_error"
    assert failed.metadata["error_type"] == "LLMProviderError"


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


def test_parallel_tool_execution_mode_is_supported() -> None:
    config = AgentToolLoopConfig(tool_execution_mode="parallel")

    assert config.tool_execution_mode == "parallel"


def test_unsupported_tool_execution_mode_fails_clearly() -> None:
    try:
        AgentToolLoopConfig(tool_execution_mode="nonsense")  # type: ignore[arg-type]
    except ValueError as exc:
        assert str(exc) == (
            "Unsupported tool_execution_mode: nonsense. "
            "Supported modes: sequential, parallel."
        )
    else:
        raise AssertionError("expected unsupported tool execution mode to fail")


def test_unsupported_tool_execution_mode_run_override_fails_clearly() -> None:
    provider = ScriptedProvider([final_response("done")])
    loop = build_loop(provider, ToolRegistry())

    try:
        loop.run(
            [{"role": "user", "content": "hello"}],
            tools=[],
            tool_execution_mode="nonsense",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Unsupported tool_execution_mode: nonsense. "
            "Supported modes: sequential, parallel."
        )
    else:
        raise AssertionError("expected unsupported tool execution mode to fail")
