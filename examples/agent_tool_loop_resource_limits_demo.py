from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.agent_tool_loop import AgentToolLoop, AgentToolLoopConfig, ToolResourceLimits
from kernel.events import RuntimeEvent, RuntimeEventLog
from kernel.llm import LLMRequest, LLMResponse, LLMRuntime, LLMToolCall
from kernel.tools import ToolRegistry, ToolRuntime


TIMELINE_EVENT_TYPES = {
    "agent_tool_loop_started",
    "llm_request_started",
    "llm_response_received",
    "tool_execution_group_started",
    "tool_execution_group_failed",
    "tool_call_requested",
    "tool_call_resource_denied",
    "tool_execution_started",
    "tool_execution_completed",
    "tool_execution_failed",
    "agent_tool_loop_failed",
}


class ScriptedResourceLimitsProvider:
    name = "scripted-agent-tool-loop-resource-limits-demo"
    default_model = "scripted-model"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.responses = [
            LLMResponse(
                content="",
                provider=self.name,
                model=self.default_model,
                tool_calls=(
                    LLMToolCall(
                        id="call_add",
                        name="add_numbers",
                        arguments={"a": 20, "b": 22},
                        provider=self.name,
                        model=self.default_model,
                    ),
                    LLMToolCall(
                        id="call_multiply",
                        name="multiply_numbers",
                        arguments={"a": 6, "b": 7},
                        provider=self.name,
                        model=self.default_model,
                    ),
                ),
            )
        ]

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected LLM call")
        return self.responses.pop(0)


def number_pair_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }


def build_tool_registry(executions: list[str]) -> ToolRegistry:
    registry = ToolRegistry()

    def add_numbers(a: float, b: float) -> float:
        executions.append("add_numbers")
        return a + b

    def multiply_numbers(a: float, b: float) -> float:
        executions.append("multiply_numbers")
        return a * b

    registry.register(
        name="add_numbers",
        description="Add two numbers together.",
        parameters_schema=number_pair_schema(),
        func=add_numbers,
        parallel_safe=True,
    )
    registry.register(
        name="multiply_numbers",
        description="Multiply two numbers together.",
        parameters_schema=number_pair_schema(),
        func=multiply_numbers,
        parallel_safe=True,
    )
    return registry


def print_timeline(events: RuntimeEventLog) -> None:
    print("\nTimeline:")
    timeline_events = [
        event
        for event in events.events
        if isinstance(event, RuntimeEvent) and event.event_type in TIMELINE_EVENT_TYPES
    ]
    for index, event in enumerate(timeline_events, start=1):
        metadata = event.metadata
        tool_name = metadata.get("tool_name")
        tool_suffix = f" {tool_name}" if isinstance(tool_name, str) else ""
        limit_name = metadata.get("limit_name")
        limit_suffix = f" limit={limit_name}" if isinstance(limit_name, str) else ""
        current_count = metadata.get("current_count")
        count_suffix = (
            f" current_count={current_count}" if isinstance(current_count, int) else ""
        )
        print(f"{index}. {event.event_type}{tool_suffix}{limit_suffix}{count_suffix}")


def main() -> None:
    events = RuntimeEventLog()
    executions: list[str] = []
    provider = ScriptedResourceLimitsProvider()
    registry = build_tool_registry(executions)
    limits = ToolResourceLimits(max_tool_calls_per_loop=1)
    loop = AgentToolLoop(
        llm_runtime=LLMRuntime(provider=provider, event_sink=events),
        tool_runtime=ToolRuntime(registry=registry, event_sink=events),
        config=AgentToolLoopConfig(tool_resource_limits=limits),
        event_sink=events,
    )

    result = loop.run(
        messages=[
            {
                "role": "user",
                "content": "Add 20 + 22, then multiply 6 * 7 with limited tools.",
            }
        ],
        tools=registry.llm_tool_definitions(),
        max_steps=4,
        temperature=0.0,
        tool_choice="auto",
    )

    print("Agent Tool Loop resource limits demo")
    print("Completed:", result.completed)
    print("Reason:", result.reason)
    print("Tool results:")
    for index, tool_result in enumerate(result.tool_results, start=1):
        print(
            "  "
            f"{index}. {tool_result.name}: "
            f"success={tool_result.success}, "
            f"content={tool_result.content!r}, "
            f"error={tool_result.error!r}"
        )
    denied_results = [tool_result for tool_result in result.tool_results if not tool_result.success]
    if denied_results:
        print("Denied result error:", denied_results[0].error)
    print_timeline(events)

    assert result.completed is False
    assert result.reason == "tool_error"
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.success for tool_result in result.tool_results] == [True, False]
    assert result.tool_results[1].error == (
        "Tool call denied by resource limits: max_tool_calls_per_loop exceeded"
    )
    assert executions == ["add_numbers"]
    assert [
        event.metadata["tool_name"]
        for event in events.by_type("tool_call_resource_denied")
    ] == ["multiply_numbers"]
    assert [
        event.metadata["tool_name"] for event in events.by_type("tool_execution_started")
    ] == ["add_numbers"]

    print("\nAgent Tool Loop resource limits demo passed.")


if __name__ == "__main__":
    main()
