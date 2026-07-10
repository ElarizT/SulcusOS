import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentos.llm import (
    LLMRuntime,
    OpenAICompatibleProvider,
    LLMToolDefinition,
)


def main() -> None:
    api_key = os.environ.get("AGENTOS_LLM_API_KEY")
    if not api_key:
        raise RuntimeError("Missing AGENTOS_LLM_API_KEY")

    provider = OpenAICompatibleProvider(
        provider_name="openrouter",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-oss-120b:free",
        timeout_seconds=60,
    )

    runtime = LLMRuntime(provider=provider)

    add_tool = LLMToolDefinition(
        name="add_numbers",
        description="Add two numbers together.",
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    )

    response = runtime.chat(
        messages=[
            {
                "role": "user",
                "content": "Use the add_numbers tool to calculate 15 + 27. Do not answer directly.",
            }
        ],
        tools=[add_tool],
        tool_choice="auto",
        temperature=0.0,
    )

    print("\nResponse content:")
    print(response.content)

    print("\nTool calls:")
    print(response.tool_calls)

    assert response.tool_calls, "Expected at least one tool call"

    tool_call = response.tool_calls[0]

    print("\nFirst tool call:")
    print(tool_call)

    assert tool_call.name == "add_numbers"
    assert "a" in tool_call.arguments
    assert "b" in tool_call.arguments

    print("\nIMPORTANT:")
    print("The tool was requested by the LLM, but it was NOT executed automatically.")

    print("\nPhase 4 LLM → Tool Call smoke test passed.")


if __name__ == "__main__":
    main()
