import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentos.runtime import AgentToolLoop
from kernel.events import RuntimeEventLog
from agentos.llm import LLMRuntime, OpenAICompatibleProvider
from agentos.tools import ToolRegistry, ToolRuntime


EXPECTED_TOOL_NAME = "add_numbers"
EXPECTED_TOOL_RESULT = "42"


def add_numbers(a: float, b: float) -> float:
    return a + b


def assert_successful_agent_tool_loop_result(result) -> None:
    assert result.completed is True
    assert result.reason == "completed"
    assert len(result.tool_results) == 1

    tool_execution_result = result.tool_results[0]
    assert tool_execution_result.success is True
    assert tool_execution_result.name == EXPECTED_TOOL_NAME
    assert tool_execution_result.content == EXPECTED_TOOL_RESULT

    final_llm_response = result.final_response
    assert final_llm_response is not None
    assert EXPECTED_TOOL_RESULT in final_llm_response.content


def main() -> None:
    api_key = os.environ.get("AGENTOS_LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AGENTOS_LLM_API_KEY")

    provider = OpenAICompatibleProvider(
        provider_name=os.environ.get("AGENTOS_LLM_PROVIDER", "openrouter"),
        api_key=api_key,
        base_url=os.environ.get(
            "AGENTOS_LLM_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        default_model=os.environ.get(
            "AGENTOS_LLM_MODEL",
            "openai/gpt-oss-120b:free",
        ),
        timeout_seconds=60,
    )

    registry = ToolRegistry()
    registry.register(
        name=EXPECTED_TOOL_NAME,
        description="Add two numbers together.",
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
            "additionalProperties": False,
        },
        func=add_numbers,
    )

    events = RuntimeEventLog()
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
                    "Use the add_numbers tool to calculate 15 + 27, then answer "
                    "with the result."
                ),
            }
        ],
        tools=registry.llm_tool_definitions(),
        max_steps=4,
        temperature=0.0,
        tool_choice="auto",
    )

    print("Phase 6 Agent Tool Loop smoke test")
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

    try:
        assert_successful_agent_tool_loop_result(result)
    except AssertionError as exc:
        print("\nSafe runtime events:")
        for event in events.events:
            print(event.event_type, event.metadata)
        print("\nSteps:")
        for step in result.steps:
            print(step)
        raise RuntimeError(
            "Agent tool loop smoke test failed regression assertions"
        ) from exc

    print("\nPhase 6 Agent Tool Loop smoke test passed.")


if __name__ == "__main__":
    main()
