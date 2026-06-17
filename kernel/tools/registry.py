"""Deterministic registry for approved Agent OS tools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from kernel.events import RuntimeEvent
from kernel.llm.types import LLMToolDefinition
from kernel.tools.types import ToolDefinition, ToolRegistrationError


EventSink = Callable[[RuntimeEvent], None] | Any


class ToolRegistry:
    """Approved tool registry; never imports or resolves callables by name."""

    def __init__(self, event_sink: EventSink | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self.event_sink = event_sink

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters_schema: Mapping[str, Any],
        func: Callable[..., Any],
        timeout_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ToolDefinition:
        """Register one approved callable tool."""
        try:
            definition = ToolDefinition(
                name=name,
                description=description,
                parameters_schema=parameters_schema,
                func=func,
                timeout_seconds=timeout_seconds,
                metadata=dict(metadata or {}),
            )
        except ValueError as exc:
            raise ToolRegistrationError(str(exc)) from None

        if definition.name in self._tools and not overwrite:
            raise ToolRegistrationError(f"tool '{definition.name}' is already registered")
        self._tools[definition.name] = definition
        self._emit_registered(definition)
        return definition

    def register_definition(
        self,
        definition: ToolDefinition,
        *,
        overwrite: bool = False,
    ) -> ToolDefinition:
        """Register an already constructed tool definition."""
        if not isinstance(definition, ToolDefinition):
            raise ToolRegistrationError("definition must be a ToolDefinition")
        if definition.name in self._tools and not overwrite:
            raise ToolRegistrationError(f"tool '{definition.name}' is already registered")
        self._tools[definition.name] = definition
        self._emit_registered(definition)
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        """Return a registered tool by exact name."""
        return self._tools.get(name)

    def require(self, name: str) -> ToolDefinition:
        """Return a registered tool or raise a clean registry error."""
        definition = self.get(name)
        if definition is None:
            raise ToolRegistrationError(f"unknown tool '{name}'")
        return definition

    def names(self) -> tuple[str, ...]:
        """Return registered tool names in deterministic order."""
        return tuple(sorted(self._tools))

    def list(self) -> tuple[ToolDefinition, ...]:
        """Return registered tool definitions in deterministic name order."""
        return tuple(self._tools[name] for name in self.names())

    def snapshot(self) -> tuple[dict[str, Any], ...]:
        """Return a safe deterministic registry snapshot without callables."""
        return tuple(
            {
                "name": definition.name,
                "description": definition.description,
                "parameters_schema": dict(definition.parameters_schema),
                "timeout_seconds": definition.timeout_seconds,
                "metadata": dict(definition.metadata),
            }
            for definition in self.list()
        )

    def llm_tool_definitions(self) -> tuple[LLMToolDefinition, ...]:
        """Return Step 33 LLM tool definitions for registered tools."""
        return tuple(definition.to_llm_tool_definition() for definition in self.list())

    def _emit_registered(self, definition: ToolDefinition) -> None:
        self._emit(
            RuntimeEvent.info(
                "ToolRegistry",
                "tool.registered",
                f"Tool registered: {definition.name}",
                {
                    "tool_name": definition.name,
                    "success": True,
                    "property_count": _property_count(definition.parameters_schema),
                    "required_count": _required_count(definition.parameters_schema),
                },
            )
        )

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


def _property_count(schema: Mapping[str, Any]) -> int:
    properties = schema.get("properties")
    return len(properties) if isinstance(properties, Mapping) else 0


def _required_count(schema: Mapping[str, Any]) -> int:
    required = schema.get("required")
    return len(required) if isinstance(required, list) else 0
