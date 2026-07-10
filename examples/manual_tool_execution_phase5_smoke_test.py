import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from agentos.llm import (
    LLMRuntime,
    OpenAICompatibleProvider,
    LLMToolDefinition,
)

from agentos.tools import (
    ToolRegistry,
    ToolRuntime,
)


def add_numbers(a: float, b: float) -> float:
    return a + b


def main() -> None:
    api_key = os.environ.get("AGENTOS_LLM_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing AGENTOS_LLM_API_KEY environment variable."
        )

    provider = OpenAICompatibleProvider(
        provider_name="openrouter",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-oss-120b:free",
        timeout_seconds=60,
    )

    llm_runtime = LLMRuntime(provider=provider)

    registry = ToolRegistry()

    registry.register(
        name="add_numbers",
        description="Add two numbers together.",
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
        func=add_numbers,
    )

    tool_runtime = ToolRuntime(registry=registry)

    add_tool = LLMToolDefinition(
        name="add_numbers",
        description="Add two numbers together.",
        parameters_schema={
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number",
                },
            },
            "required": ["a", "b"],
        },
    )

    print("\n1. ASK LLM TO REQUEST TOOL")

    response = llm_runtime.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    "Use the add_numbers tool to calculate "
                    "15 + 27. Do not answer directly."
                ),
            }
        ],
        tools=[add_tool],
        tool_choice="auto",
        temperature=0.0,
    )

    print("LLM response content:")
    print(response.content)

    print("\nLLM tool calls:")
    print(response.tool_calls)

    assert response.tool_calls, (
        "Expected the LLM to request a tool call."
    )

    tool_call = response.tool_calls[0]

    assert tool_call.name == "add_numbers"
    assert tool_call.arguments["a"] == 15
    assert tool_call.arguments["b"] == 27

    print("\n2. EXECUTE TOOL EXPLICITLY")

    tool_result = tool_runtime.execute(
        tool_call.name,
        tool_call.arguments,
    )

    print("Tool result:")
    print(tool_result)

    assert tool_result.success is True
    assert tool_result.content == 42

    print("\n3. CONVERT TOOL RESULT TO LLM TOOL RESULT")

    llm_tool_result = tool_result.to_llm_tool_result()

    print("LLM tool result:")
    print(llm_tool_result)

    assert llm_tool_result.name == "add_numbers"
    assert llm_tool_result.success is True
    assert llm_tool_result.content == "42"

    print("\n4. FINAL SUMMARY")

    print("✓ LLM requested the tool")
    print("✓ Tool was NOT automatically executed")
    print("✓ ToolRuntime explicitly executed the tool")
    print("✓ Result = 42")
    print("✓ Result successfully converted to LLMToolResult")

    print("\n🎉 Phase 5 Manual Tool Execution smoke test passed.")


if __name__ == "__main__":
    main()
