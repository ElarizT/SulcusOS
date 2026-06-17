"""Explicit runtime for executing registered Agent OS tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any

from kernel.events import RuntimeEvent
from kernel.llm.types import LLMToolCall
from kernel.tools.registry import ToolRegistry
from kernel.tools.types import (
    ToolCallRequest,
    ToolDefinition,
    ToolExecutionResult,
    ToolRuntimeError,
)


EventSink = Callable[[RuntimeEvent], None] | Any


class ToolRuntime:
    """Execute explicitly requested registered tools only."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        event_sink: EventSink | None = None,
        default_timeout_seconds: float | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise ToolRuntimeError("registry must be a ToolRegistry")
        if default_timeout_seconds is not None:
            if isinstance(default_timeout_seconds, bool) or not isinstance(
                default_timeout_seconds, (int, float)
            ):
                raise ToolRuntimeError("default_timeout_seconds must be a positive number")
            if default_timeout_seconds <= 0:
                raise ToolRuntimeError("default_timeout_seconds must be positive")
        if not callable(clock):
            raise ToolRuntimeError("clock must be callable")
        self.registry = registry
        self.event_sink = event_sink
        self.default_timeout_seconds = default_timeout_seconds
        self._clock = clock

    def execute(
        self,
        tool_call_or_name: ToolCallRequest | LLMToolCall | str,
        arguments: Mapping[str, Any] | None = None,
    ) -> ToolExecutionResult:
        """Execute a registered tool by request object or explicit name/arguments."""
        request = _coerce_tool_call_request(tool_call_or_name, arguments)
        argument_keys = _argument_keys(request.arguments)
        self._emit_execution_event(
            "tool.execution_requested",
            f"Tool execution requested: {request.name}",
            request.name,
            argument_keys=argument_keys,
            success=False,
        )

        definition = self.registry.get(request.name)
        if definition is None:
            return self._rejected_result(
                request,
                argument_keys,
                error_type="UnknownToolError",
                error_category="unknown_tool",
            )

        validation_error = _validate_arguments(
            request.arguments,
            definition.parameters_schema,
        )
        if validation_error is not None:
            return self._rejected_result(
                request,
                argument_keys,
                error_type="ToolValidationError",
                error_category="validation",
                error=validation_error,
            )

        self._emit_execution_event(
            "tool.execution_started",
            f"Tool execution started: {request.name}",
            request.name,
            argument_keys=argument_keys,
            success=False,
        )
        start = self._clock()
        try:
            content = definition.func(**request.arguments)
        except Exception as exc:
            duration_ms = _duration_ms(start, self._clock())
            result = ToolExecutionResult(
                name=request.name,
                success=False,
                tool_call_id=request.tool_call_id,
                error="Tool execution failed",
                error_type=exc.__class__.__name__,
                error_category="exception",
                duration_ms=duration_ms,
            )
            self._emit_execution_event(
                "tool.execution_failed",
                f"Tool execution failed: {request.name}",
                request.name,
                argument_keys=argument_keys,
                success=False,
                error_type=result.error_type,
                error_category=result.error_category,
                duration_ms=duration_ms,
                error=True,
            )
            return result

        duration_ms = _duration_ms(start, self._clock())
        timeout_seconds = _effective_timeout(definition, self.default_timeout_seconds)
        if timeout_seconds is not None and duration_ms > int(timeout_seconds * 1000):
            result = ToolExecutionResult(
                name=request.name,
                success=False,
                tool_call_id=request.tool_call_id,
                error="Tool execution timed out",
                error_type="ToolTimeoutError",
                error_category="timeout",
                duration_ms=duration_ms,
            )
            self._emit_execution_event(
                "tool.execution_failed",
                f"Tool execution timed out: {request.name}",
                request.name,
                argument_keys=argument_keys,
                success=False,
                error_type=result.error_type,
                error_category=result.error_category,
                duration_ms=duration_ms,
                error=True,
            )
            return result

        result = ToolExecutionResult(
            name=request.name,
            content=content,
            success=True,
            tool_call_id=request.tool_call_id,
            duration_ms=duration_ms,
        )
        self._emit_execution_event(
            "tool.execution_completed",
            f"Tool execution completed: {request.name}",
            request.name,
            argument_keys=argument_keys,
            success=True,
            duration_ms=duration_ms,
        )
        return result

    def _rejected_result(
        self,
        request: ToolCallRequest,
        argument_keys: tuple[str, ...],
        *,
        error_type: str,
        error_category: str,
        error: str = "Tool execution rejected",
    ) -> ToolExecutionResult:
        result = ToolExecutionResult(
            name=request.name,
            success=False,
            tool_call_id=request.tool_call_id,
            error=error,
            error_type=error_type,
            error_category=error_category,
            duration_ms=0,
        )
        self._emit_execution_event(
            "tool.execution_rejected",
            f"Tool execution rejected: {request.name}",
            request.name,
            argument_keys=argument_keys,
            success=False,
            error_type=error_type,
            error_category=error_category,
            duration_ms=0,
            error=True,
        )
        return result

    def _emit_execution_event(
        self,
        event_type: str,
        message: str,
        tool_name: str,
        *,
        argument_keys: tuple[str, ...],
        success: bool,
        error_type: str | None = None,
        error_category: str | None = None,
        duration_ms: int | None = None,
        error: bool = False,
    ) -> None:
        metadata: dict[str, Any] = {
            "tool_name": tool_name,
            "success": success,
            "argument_keys": argument_keys,
        }
        if error_type is not None:
            metadata["error_type"] = error_type
        if error_category is not None:
            metadata["error_category"] = error_category
        if duration_ms is not None:
            metadata["duration_ms"] = duration_ms
        factory = RuntimeEvent.error if error else RuntimeEvent.info
        self._emit(factory("ToolRuntime", event_type, message, metadata))

    def _emit(self, event: RuntimeEvent) -> None:
        if self.event_sink is None:
            return
        try:
            append = getattr(self.event_sink, "append", None)
            if callable(append):
                append(event)
            elif callable(self.event_sink):
                self.event_sink(event)
        except Exception:
            return


def _coerce_tool_call_request(
    tool_call_or_name: ToolCallRequest | LLMToolCall | str,
    arguments: Mapping[str, Any] | None,
) -> ToolCallRequest:
    if isinstance(tool_call_or_name, ToolCallRequest):
        if arguments is not None:
            raise ToolRuntimeError("arguments cannot be provided with ToolCallRequest")
        return tool_call_or_name
    if isinstance(tool_call_or_name, LLMToolCall):
        if arguments is not None:
            raise ToolRuntimeError("arguments cannot be provided with LLMToolCall")
        return ToolCallRequest.from_llm_tool_call(tool_call_or_name)
    if isinstance(tool_call_or_name, str):
        if arguments is not None and not isinstance(arguments, Mapping):
            raise ToolRuntimeError("arguments must be a mapping")
        return ToolCallRequest(
            name=tool_call_or_name,
            arguments=dict(arguments or {}),
        )
    raise ToolRuntimeError("execute requires a tool name, ToolCallRequest, or LLMToolCall")


def _validate_arguments(
    arguments: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> str | None:
    if _schema_type(schema) == "object" and not isinstance(arguments, Mapping):
        return "arguments must be an object"

    properties = schema.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, Mapping):
        return "parameters_schema properties must be an object"

    required = schema.get("required")
    if required is None:
        required = []
    if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
        return "parameters_schema required must be a list of strings"
    for name in required:
        if name not in arguments:
            return f"missing required argument: {name}"

    additional_properties = schema.get("additionalProperties", True)
    if additional_properties is False:
        unknown = sorted(str(name) for name in arguments if name not in properties)
        if unknown:
            return f"unknown argument: {unknown[0]}"

    for name in sorted(properties):
        if name not in arguments:
            continue
        child_schema = properties[name]
        if not isinstance(child_schema, Mapping):
            return f"schema for argument is invalid: {name}"
        error = _validate_value(name, arguments[name], child_schema)
        if error is not None:
            return error
    return None


def _validate_value(name: str, value: Any, schema: Mapping[str, Any]) -> str | None:
    expected_type = _schema_type(schema)
    if expected_type is None:
        return None
    if expected_type == "string":
        if not isinstance(value, str):
            return f"invalid type for argument: {name}"
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"invalid type for argument: {name}"
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"invalid type for argument: {name}"
    elif expected_type == "boolean":
        if not isinstance(value, bool):
            return f"invalid type for argument: {name}"
    elif expected_type == "object":
        if not isinstance(value, Mapping):
            return f"invalid type for argument: {name}"
        nested_error = _validate_arguments(value, schema)
        if nested_error is not None:
            return f"{name}.{nested_error}"
    elif expected_type == "array":
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            return f"invalid type for argument: {name}"
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                item_error = _validate_value(f"{name}[{index}]", item, items)
                if item_error is not None:
                    return item_error
    return None


def _schema_type(schema: Mapping[str, Any]) -> str | None:
    expected = schema.get("type")
    return expected if isinstance(expected, str) else None


def _argument_keys(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in arguments))


def _duration_ms(start: float, end: float) -> int:
    return max(0, int(round((end - start) * 1000)))


def _effective_timeout(
    definition: ToolDefinition,
    default_timeout_seconds: float | None,
) -> float | None:
    return definition.timeout_seconds or default_timeout_seconds
