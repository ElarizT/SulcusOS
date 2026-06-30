from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.agent_tool_loop import AgentToolLoop, AgentToolLoopConfig, ToolPermissionPolicy
from kernel.events import RuntimeEvent, RuntimeEventLog
from kernel.llm import LLMRequest, LLMResponse, LLMRuntime, LLMToolCall
from kernel.tools import ToolRegistry, ToolRuntime


TIMELINE_EVENT_TYPES = {
    "agent_tool_loop_started",
    "llm_request_started",
    "llm_response_received",
    "llm_followup_request_started",
    "llm_followup_response_received",
    "tool_execution_group_started",
    "tool_execution_group_completed",
    "tool_execution_group_failed",
    "tool_call_requested",
    "tool_call_denied",
    "tool_execution_started",
    "tool_execution_completed",
    "tool_execution_failed",
    "llm_final_request_started",
    "llm_final_response_received",
    "agent_tool_loop_completed",
    "agent_tool_loop_failed",
}


class ScriptedToolPermissionProvider:
    name = "scripted-agent-tool-loop-tool-permission-demo"
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
                ),
            ),
            LLMResponse(
                content="",
                provider=self.name,
                model=self.default_model,
                tool_calls=(
                    LLMToolCall(
                        id="call_multiply",
                        name="multiply_numbers",
                        arguments={"a": 42, "b": 2},
                        provider=self.name,
                        model=self.default_model,
                    ),
                ),
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
        round_index = metadata.get("round_index")
        round_suffix = f" round={round_index}" if isinstance(round_index, int) else ""
        matched_rule = metadata.get("matched_rule")
        rule_suffix = f" matched_rule={matched_rule}" if isinstance(matched_rule, str) else ""
        print(f"{index}. {event.event_type}{tool_suffix}{round_suffix}{rule_suffix}")


def main() -> None:
    events = RuntimeEventLog()
    provider = ScriptedToolPermissionProvider()
    registry = build_tool_registry()
    policy = ToolPermissionPolicy(default_allow=False, allowed_tools={"add_numbers"})
    loop = AgentToolLoop(
        llm_runtime=LLMRuntime(provider=provider, event_sink=events),
        tool_runtime=ToolRuntime(registry=registry, event_sink=events),
        config=AgentToolLoopConfig(tool_permission_policy=policy),
        event_sink=events,
    )

    result = loop.run(
        messages=[
            {
                "role": "user",
                "content": (
                    "First add 20 + 22, then multiply the result by 2. "
                    "Only approved tools may run."
                ),
            }
        ],
        tools=registry.llm_tool_definitions(),
        max_steps=4,
        temperature=0.0,
        tool_choice="auto",
    )

    print("Agent Tool Loop tool permission demo")
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
        "Tool call denied by permission policy: multiply_numbers"
    )
    assert [event.metadata["tool_name"] for event in events.by_type("tool_call_denied")] == [
        "multiply_numbers"
    ]
    assert [
        event.metadata["tool_name"] for event in events.by_type("tool_execution_started")
    ] == ["add_numbers"]

    print("\nAgent Tool Loop tool permission demo passed.")


if __name__ == "__main__":
    main()
