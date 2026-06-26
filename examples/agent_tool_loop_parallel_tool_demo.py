from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.agent_tool_loop import AgentToolLoop, AgentToolLoopConfig
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
    "tool_execution_started",
    "tool_execution_completed",
    "tool_execution_failed",
    "llm_final_request_started",
    "llm_final_response_received",
    "agent_tool_loop_completed",
    "agent_tool_loop_failed",
}


class ScriptedParallelToolProvider:
    name = "scripted-agent-tool-loop-parallel-demo"
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
                        id="call_slow_add",
                        name="slow_add_numbers",
                        arguments={"a": 20, "b": 22, "delay_ms": 80},
                        provider=self.name,
                        model=self.default_model,
                    ),
                    LLMToolCall(
                        id="call_fast_multiply",
                        name="slow_multiply_numbers",
                        arguments={"a": 6, "b": 7, "delay_ms": 20},
                        provider=self.name,
                        model=self.default_model,
                    ),
                ),
            ),
            LLMResponse(
                content="The sum is 42 and the product is 42.",
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
            "delay_ms": {"type": "integer", "description": "Tiny demo delay"},
        },
        "required": ["a", "b"],
        "additionalProperties": False,
    }


def build_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    def slow_add_numbers(a: float, b: float, delay_ms: int = 80) -> float:
        time.sleep(delay_ms / 1000)
        return a + b

    def slow_multiply_numbers(a: float, b: float, delay_ms: int = 20) -> float:
        time.sleep(delay_ms / 1000)
        return a * b

    registry.register(
        name="slow_add_numbers",
        description="Add two numbers after a tiny deterministic delay.",
        parameters_schema=number_pair_schema(),
        func=slow_add_numbers,
        parallel_safe=True,
    )
    registry.register(
        name="slow_multiply_numbers",
        description="Multiply two numbers after a tiny deterministic delay.",
        parameters_schema=number_pair_schema(),
        func=slow_multiply_numbers,
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
        round_index = metadata.get("round_index")
        round_suffix = f" round={round_index}" if isinstance(round_index, int) else ""
        requested = metadata.get("requested_execution_mode")
        effective = metadata.get("effective_execution_mode")
        mode_suffix = ""
        if isinstance(requested, str) and isinstance(effective, str):
            mode_suffix = f" requested={requested} effective={effective}"
        print(f"{index}. {event.event_type}{tool_suffix}{round_suffix}{mode_suffix}")


def main() -> None:
    events = RuntimeEventLog()
    provider = ScriptedParallelToolProvider()
    registry = build_tool_registry()
    loop = AgentToolLoop(
        llm_runtime=LLMRuntime(provider=provider, event_sink=events),
        tool_runtime=ToolRuntime(registry=registry, event_sink=events),
        config=AgentToolLoopConfig(tool_execution_mode="parallel"),
        event_sink=events,
    )

    result = loop.run(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use slow_add_numbers for 20 + 22 and "
                    "slow_multiply_numbers for 6 * 7, then answer with both results."
                ),
            }
        ],
        tools=registry.llm_tool_definitions(),
        max_steps=4,
        temperature=0.0,
        tool_choice="auto",
    )

    group_started = events.by_type("tool_execution_group_started")[0]
    print("Agent Tool Loop parallel tool demo")
    print("Completed:", result.completed)
    print("Reason:", result.reason)
    print("Final response:", result.final_response.content if result.final_response else None)
    print("Requested execution mode:", group_started.metadata["requested_execution_mode"])
    print("Effective execution mode:", group_started.metadata["effective_execution_mode"])
    print("Tool results in request order:")
    for index, tool_execution_result in enumerate(result.tool_results, start=1):
        print(
            "  "
            f"{index}. {tool_execution_result.name}: "
            f"success={tool_execution_result.success}, "
            f"content={tool_execution_result.content!r}"
        )
    print_timeline(events)

    assert result.completed is True
    assert result.reason == "completed"
    assert [tool_result.name for tool_result in result.tool_results] == [
        "slow_add_numbers",
        "slow_multiply_numbers",
    ]
    assert [tool_result.content for tool_result in result.tool_results] == ["42", "42"]
    assert group_started.metadata["requested_execution_mode"] == "parallel"
    assert group_started.metadata["effective_execution_mode"] == "parallel"
    assert group_started.metadata["parallel_safe_tool_count"] == 2
    assert group_started.metadata["unsafe_tool_count"] == 0

    print("\nAgent Tool Loop parallel tool demo passed.")


if __name__ == "__main__":
    main()
