import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kernel.tools import ToolRegistry, ToolRuntime


def add_numbers(a: float, b: float) -> float:
    return a + b


registry = ToolRegistry()

registry.register(
    name="add_numbers",
    description="Add two numbers.",
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

runtime = ToolRuntime(registry=registry)

print("\n1. VALID TOOL CALL")
result = runtime.execute("add_numbers", {"a": 5, "b": 7})
print(result)
assert result.success is True
assert result.content == 12

print("\n2. MISSING REQUIRED ARGUMENT")
bad_result = runtime.execute("add_numbers", {"a": 5})
print(bad_result)
assert bad_result.success is False

print("\n3. UNKNOWN TOOL")
unknown_result = runtime.execute("missing_tool", {})
print(unknown_result)
assert unknown_result.success is False

print("\nPhase 3 Tool Runtime smoke test passed.")