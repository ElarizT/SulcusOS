from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.agent_tool_loop import AgentToolLoop
from kernel.events import RuntimeEvent, RuntimeEventLog
from kernel.llm import LLMRequest, LLMResponse, LLMRuntime, LLMToolCall
from kernel.tools import ToolRegistry, ToolRuntime


TIMELINE_EVENT_TYPES = {
    "agent_tool_loop_started",
    "llm_request_started",
    "llm_response_received",
    "tool_call_requested",
    "tool_execution_started",
    "tool_execution_completed",
    "tool_execution_failed",
    "llm_final_request_started",
    "llm_final_response_received",
    "agent_tool_loop_completed",
    "agent_tool_loop_failed",
}


class ScriptedMultiToolProvider:
    name = "scripted-agent-tool-loop-demo"
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
                        arguments={"a": 15, "b": 27},
                        provider=self.name,
                        model=self.default_model,
                    ),
                    LLMToolCall(
                        id="call_multiply",
                        name="multiply_numbers",
                        arguments={"a": 6, "b": 9},
                        provider=self.name,
                        model=self.default_model,
                    ),
                ),
            ),
            LLMResponse(
                content="The sum is 42 and the product is 54.",
                provider=self.name,
                model=self.default_model,
            ),
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


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers together.",
        parameters_schema=number_pair_schema(),
        func=lambda a, b: a + b,
    )
    registry.register(
        name="multiply_numbers",
        description="Multiply two numbers together.",
        parameters_schema=number_pair_schema(),
        func=lambda a, b: a * b,
    )
    return registry


def assert_multi_tool_demo_result(result, provider: ScriptedMultiToolProvider) -> None:
    assert result.completed is True
    assert result.reason == "completed"
    assert result.final_response is not None
    assert "42" in result.final_response.content
    assert "54" in result.final_response.content
    assert len(result.tool_results) == 2
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "54"]
    assert all(tool_result.success for tool_result in result.tool_results)
    assert len(provider.requests) == 2

    follow_up_roles = [message.role for message in provider.requests[1].messages]
    assert follow_up_roles == ["user", "assistant", "tool", "tool"]


def print_timeline(events: RuntimeEventLog) -> None:
    print("\nTimeline:")
    timeline_events = [
        event
        for event in events.events
        if isinstance(event, RuntimeEvent) and event.event_type in TIMELINE_EVENT_TYPES
    ]
    for index, event in enumerate(timeline_events, start=1):
        tool_name = event.metadata.get("tool_name")
        suffix = f" {tool_name}" if isinstance(tool_name, str) else ""
        print(f"{index}. {event.event_type}{suffix}")


def main() -> None:
    events = RuntimeEventLog()
    provider = ScriptedMultiToolProvider()
    registry = build_tool_registry()
    loop = AgentToolLoop(
        llm_runtime=LLMRuntime(provider=provider, event_sink=events),
        tool_runtime=ToolRuntime(registry=registry, event_sink=events),
        event_sink=events,
    )

    result = loop.run(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use add_numbers for 15 + 27 and multiply_numbers for 6 * 9, "
                    "then answer with both results."
                ),
            }
        ],
        tools=registry.llm_tool_definitions(),
        max_steps=4,
        temperature=0.0,
        tool_choice="auto",
    )

    print("Agent Tool Loop multi-tool demo")
    print("Completed:", result.completed)
    print("Reason:", result.reason)
    print("Tool execution results:")
    for tool_execution_result in result.tool_results:
        print(
            "  "
            f"{tool_execution_result.name}: "
            f"success={tool_execution_result.success}, "
            f"content={tool_execution_result.content!r}"
        )
    if result.final_response is not None:
        print("Final LLM response:")
        print(result.final_response.content)
    print_timeline(events)

    try:
        assert_multi_tool_demo_result(result, provider)
    except AssertionError as exc:
        print("\nSafe runtime events:")
        for event in events.events:
            print(event.event_type, event.metadata)
        print("\nSteps:")
        for step in result.steps:
            print(step)
        raise RuntimeError("Agent tool loop multi-tool demo failed") from exc

    print("\nAgent Tool Loop multi-tool demo passed.")


if __name__ == "__main__":
    main()
