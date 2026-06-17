import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.agent_tool_loop import AgentToolLoop
from kernel.events import RuntimeEventLog
from kernel.llm import LLMRuntime, OpenAICompatibleProvider
from kernel.tools import ToolRegistry, ToolRuntime


def add_numbers(a: float, b: float) -> float:
    return a + b


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
        name="add_numbers",
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

    print("Completed:", result.completed)
    print("Reason:", result.reason)
    print("Tool results:", result.tool_results)
    if result.final_response is not None:
        print("Final response:")
        print(result.final_response.content)

    if not result.completed:
        print("\nSafe runtime events:")
        for event in events.events:
            print(event.event_type, event.metadata)
        print("\nSteps:")
        for step in result.steps:
            print(step)
        raise RuntimeError(
            "Agent tool loop smoke test did not complete. "
            f"reason={result.reason}"
        )

    if result.final_response is None:
        raise RuntimeError("Agent tool loop completed without a final response")
    if not result.tool_results:
        raise RuntimeError("Agent tool loop completed without executing a tool")
    if result.tool_results[0].content != "42":
        raise RuntimeError(
            "Expected add_numbers tool result content to be 42, got "
            f"{result.tool_results[0].content!r}"
        )

    print("\nPhase 6 Agent Tool Loop smoke test passed.")


if __name__ == "__main__":
    main()
