from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentos.runtime import AgentToolLoop
from kernel.events import RuntimeEvent, RuntimeEventLog
from agentos.llm import LLMRequest, LLMResponse, LLMRuntime, LLMToolCall
from agentos.tools import ToolRegistry, ToolRuntime


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
    "tool_execution_started",
    "tool_execution_completed",
    "tool_execution_failed",
    "llm_final_request_started",
    "llm_final_response_received",
    "agent_tool_loop_completed",
    "agent_tool_loop_failed",
}


class ScriptedMultiRoundProvider:
    name = "scripted-agent-tool-loop-multi-round-demo"
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
            LLMResponse(
                content="The final answer is 84.",
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


def print_timeline(events: RuntimeEventLog) -> None:
    print("\nTimeline:")
    timeline_events = [
        event
        for event in events.events
        if isinstance(event, RuntimeEvent) and event.event_type in TIMELINE_EVENT_TYPES
    ]
    for index, event in enumerate(timeline_events, start=1):
        round_index = event.metadata.get("round_index")
        round_suffix = f" round={round_index}" if isinstance(round_index, int) else ""
        final_attempt = event.metadata.get("final_attempt")
        final_suffix = (
            f" final_attempt={final_attempt}" if isinstance(final_attempt, bool) else ""
        )
        execution_mode = event.metadata.get("execution_mode")
        mode_suffix = (
            f" execution_mode={execution_mode}"
            if isinstance(execution_mode, str)
            else ""
        )
        tool_name = event.metadata.get("tool_name")
        tool_suffix = f" {tool_name}" if isinstance(tool_name, str) else ""
        print(
            f"{index}. {event.event_type}{tool_suffix}"
            f"{round_suffix}{mode_suffix}{final_suffix}"
        )


def assert_multi_round_demo_result(
    result,
    provider: ScriptedMultiRoundProvider,
) -> None:
    assert result.completed is True
    assert result.reason == "completed"
    assert result.final_response is not None
    assert "84" in result.final_response.content
    assert [tool_result.name for tool_result in result.tool_results] == [
        "add_numbers",
        "multiply_numbers",
    ]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "84"]
    assert all(tool_result.success for tool_result in result.tool_results)
    assert len(provider.requests) == 3
    assert [message.role for message in provider.requests[2].messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]


def main() -> None:
    events = RuntimeEventLog()
    provider = ScriptedMultiRoundProvider()
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
                    "First add 20 + 22, then multiply that result by 2, "
                    "then answer with the final result."
                ),
            }
        ],
        tools=registry.llm_tool_definitions(),
        max_steps=4,
        temperature=0.0,
        tool_choice="auto",
    )

    print("Agent Tool Loop multi-round demo")
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
        assert_multi_round_demo_result(result, provider)
    except AssertionError as exc:
        print("\nSafe runtime events:")
        for event in events.events:
            print(event.event_type, event.metadata)
        print("\nSteps:")
        for step in result.steps:
            print(step)
        raise RuntimeError("Agent tool loop multi-round demo failed") from exc

    print("\nAgent Tool Loop multi-round demo passed.")


if __name__ == "__main__":
    main()
