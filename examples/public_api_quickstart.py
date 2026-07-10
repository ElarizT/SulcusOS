"""Offline quickstart using only documented ``agentos`` public imports."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentos.llm import LLMResponse, LLMRuntime, LLMToolCall, LLMToolDefinition
from agentos.runtime import AgentToolLoop
from agentos.tools import ToolRegistry, ToolRuntime


class ScriptedProvider:
    name = "public-api-demo"
    default_model = "offline"

    def __init__(self) -> None:
        self.responses = [
            LLMResponse(
                content="",
                provider=self.name,
                model=self.default_model,
                tool_calls=(LLMToolCall("add-1", "add_numbers", {"a": 20, "b": 22}),),
            ),
            LLMResponse(content="The answer is 42.", provider=self.name, model=self.default_model),
        ]

    def complete(self, request: object) -> LLMResponse:
        return self.responses.pop(0)


def main() -> int:
    registry = ToolRegistry()
    registry.register(
        name="add_numbers",
        description="Add two numbers.",
        parameters_schema={"type": "object", "required": ["a", "b"]},
        func=lambda a, b: a + b,
    )
    loop = AgentToolLoop(
        llm_runtime=LLMRuntime(provider=ScriptedProvider()),
        tool_runtime=ToolRuntime(registry=registry),
    )
    result = loop.run(
        [{"role": "user", "content": "What is 20 plus 22?"}],
        [LLMToolDefinition("add_numbers", "Add two numbers.", {"type": "object"})],
    )
    assert result.completed and result.final_response is not None
    print(result.final_response.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
