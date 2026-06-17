"""Safe explicit Agent OS loop for LLM tool orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from kernel.events import RuntimeEvent
from kernel.llm.providers import classify_llm_error
from kernel.llm.runtime import LLMRuntime
from kernel.llm.types import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    LLMToolResult,
)
from kernel.tools.runtime import ToolRuntime
from kernel.tools.types import ToolDefinition, ToolExecutionResult


EventSink = Callable[[RuntimeEvent], None] | Any
MessageInput = LLMMessage | Mapping[str, Any]
ToolInput = LLMToolDefinition | ToolDefinition | Mapping[str, Any]


@dataclass(frozen=True)
class AgentToolLoopConfig:
    """Conservative controls for bounded LLM-tool orchestration."""

    max_steps: int = 4
    require_tool_approval: bool = False
    stop_on_tool_error: bool = True
    allow_parallel_tool_calls: bool = False
    include_intermediate_steps: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
        for field_name in (
            "require_tool_approval",
            "stop_on_tool_error",
            "allow_parallel_tool_calls",
            "include_intermediate_steps",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True)
class AgentToolLoopStep:
    """One deterministic step recorded by an agent tool loop run."""

    index: int
    kind: str
    response: LLMResponse | None = None
    tool_calls: tuple[LLMToolCall, ...] = ()
    tool_results: tuple[LLMToolResult, ...] = ()
    success: bool = True
    error_type: str | None = None
    error_category: str | None = None
    provider: str | None = None
    model: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("step index must be a nonnegative integer")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("step kind must not be empty")
        object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        object.__setattr__(self, "tool_results", tuple(self.tool_results))
        if not isinstance(self.success, bool):
            raise ValueError("step success must be a boolean")
        for field_name in ("error_type", "error_category", "provider", "model"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")


@dataclass(frozen=True)
class AgentToolLoopResult:
    """Structured outcome of a bounded AgentToolLoop run."""

    completed: bool
    reason: str
    final_response: LLMResponse | None = None
    steps: tuple[AgentToolLoopStep, ...] = ()
    pending_tool_calls: tuple[LLMToolCall, ...] = ()
    tool_results: tuple[LLMToolResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be a boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("result reason must not be empty")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "pending_tool_calls", tuple(self.pending_tool_calls))
        object.__setattr__(self, "tool_results", tuple(self.tool_results))


@dataclass(frozen=True)
class _ToolOutcome:
    llm_result: LLMToolResult
    success: bool
    error_type: str | None = None
    error_category: str | None = None


class AgentToolLoop:
    """Explicitly orchestrate LLM tool calls through ToolRuntime."""

    def __init__(
        self,
        *,
        llm_runtime: LLMRuntime,
        tool_runtime: ToolRuntime,
        config: AgentToolLoopConfig | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        if not isinstance(llm_runtime, LLMRuntime):
            raise TypeError("llm_runtime must be an LLMRuntime")
        if not isinstance(tool_runtime, ToolRuntime):
            raise TypeError("tool_runtime must be a ToolRuntime")
        self.llm_runtime = llm_runtime
        self.tool_runtime = tool_runtime
        self.config = config or AgentToolLoopConfig()
        if not isinstance(self.config, AgentToolLoopConfig):
            raise TypeError("config must be an AgentToolLoopConfig")
        self.event_sink = (
            event_sink if event_sink is not None else getattr(llm_runtime, "event_sink", None)
        )

    def run(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        provider: str | None = None,
        timeout_seconds: float | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        max_steps: int | None = None,
        require_tool_approval: bool | None = None,
        stop_on_tool_error: bool | None = None,
        allow_parallel_tool_calls: bool | None = None,
        include_intermediate_steps: bool | None = None,
    ) -> AgentToolLoopResult:
        """Run a bounded explicit LLM -> tool -> LLM loop."""
        config = _resolve_config(
            self.config,
            max_steps=max_steps,
            require_tool_approval=require_tool_approval,
            stop_on_tool_error=stop_on_tool_error,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            include_intermediate_steps=include_intermediate_steps,
        )
        if config.allow_parallel_tool_calls:
            raise ValueError("parallel tool calls are not supported yet")

        history = tuple(_coerce_message(message) for message in messages)
        tool_definitions = self._coerce_tool_definitions(tools)
        allowed_tool_names = frozenset(tool.name for tool in tool_definitions)
        route_metadata = _runtime_route_metadata(self.llm_runtime, provider, model)
        steps: list[AgentToolLoopStep] = []
        all_tool_results: list[LLMToolResult] = []

        self._emit(
            "agent_tool_loop.started",
            "Agent tool loop started",
            {
                "max_steps": config.max_steps,
                "tool_count": len(tool_definitions),
                **route_metadata,
            },
        )

        for step_index in range(config.max_steps):
            self._emit(
                "agent_tool_loop.llm_step_started",
                "Agent tool loop LLM step started",
                {
                    "step_index": step_index,
                    "max_steps": config.max_steps,
                    "tool_count": len(tool_definitions),
                    **route_metadata,
                },
            )
            try:
                response = self.llm_runtime.chat(
                    history,
                    model=model,
                    temperature=temperature,
                    metadata=metadata,
                    provider=provider,
                    timeout_seconds=timeout_seconds,
                    tools=tool_definitions,
                    tool_choice=tool_choice,
                )
            except Exception as exc:
                error_category = classify_llm_error(exc)
                failed_step = AgentToolLoopStep(
                    index=step_index,
                    kind="llm_error",
                    success=False,
                    error_type=exc.__class__.__name__,
                    error_category=error_category,
                    provider=route_metadata.get("provider"),
                    model=route_metadata.get("model"),
                )
                steps.append(failed_step)
                self._emit_failed(
                    step_index=step_index,
                    max_steps=config.max_steps,
                    error_type=exc.__class__.__name__,
                    error_category=error_category,
                    provider=route_metadata.get("provider"),
                    model=route_metadata.get("model"),
                )
                return self._result(
                    completed=False,
                    reason="llm_error",
                    steps=steps,
                    config=config,
                    tool_results=all_tool_results,
                )

            llm_step = AgentToolLoopStep(
                index=step_index,
                kind="llm_response",
                response=response,
                tool_calls=response.tool_calls,
                provider=response.provider,
                model=response.model,
            )
            steps.append(llm_step)

            if not response.tool_calls:
                self._emit(
                    "agent_tool_loop.completed",
                    "Agent tool loop completed",
                    {
                        "step_index": step_index,
                        "max_steps": config.max_steps,
                        "success": True,
                        "provider": response.provider,
                        "model": response.model,
                    },
                )
                return self._result(
                    completed=True,
                    reason="completed",
                    final_response=response,
                    steps=steps,
                    config=config,
                    tool_results=all_tool_results,
                )

            tool_names = _tool_names(response.tool_calls)
            self._emit(
                "agent_tool_loop.tool_calls_received",
                "Agent tool loop received tool calls",
                {
                    "step_index": step_index,
                    "max_steps": config.max_steps,
                    "tool_count": len(response.tool_calls),
                    "tool_names": tool_names,
                    "provider": response.provider,
                    "model": response.model,
                },
            )

            if config.require_tool_approval:
                steps.append(
                    AgentToolLoopStep(
                        index=step_index,
                        kind="approval_required",
                        tool_calls=response.tool_calls,
                        success=False,
                        provider=response.provider,
                        model=response.model,
                    )
                )
                self._emit(
                    "agent_tool_loop.approval_required",
                    "Agent tool loop approval required",
                    {
                        "step_index": step_index,
                        "max_steps": config.max_steps,
                        "tool_count": len(response.tool_calls),
                        "tool_names": tool_names,
                        "success": False,
                        "provider": response.provider,
                        "model": response.model,
                    },
                    warning=True,
                )
                return self._result(
                    completed=False,
                    reason="approval_required",
                    steps=steps,
                    config=config,
                    pending_tool_calls=response.tool_calls,
                    tool_results=all_tool_results,
                )

            if step_index + 1 >= config.max_steps:
                self._emit_max_steps_exceeded(
                    step_index=step_index,
                    max_steps=config.max_steps,
                    provider=response.provider,
                    model=response.model,
                )
                return self._result(
                    completed=False,
                    reason="max_steps_exceeded",
                    steps=steps,
                    config=config,
                    pending_tool_calls=response.tool_calls,
                    tool_results=all_tool_results,
                )

            history = (
                *history,
                assistant_tool_call_message(response),
            )
            outcomes: list[_ToolOutcome] = []
            executed_tool_calls: list[LLMToolCall] = []
            for tool_call in response.tool_calls:
                outcome = self._execute_tool_call(
                    tool_call,
                    allowed_tool_names=allowed_tool_names,
                    step_index=step_index,
                    max_steps=config.max_steps,
                    provider=response.provider,
                    model=response.model,
                )
                outcomes.append(outcome)
                executed_tool_calls.append(tool_call)
                all_tool_results.append(outcome.llm_result)
                if not outcome.success and config.stop_on_tool_error:
                    break

            tool_results = tuple(outcome.llm_result for outcome in outcomes)
            failed_outcome = next(
                (outcome for outcome in outcomes if not outcome.success),
                None,
            )
            steps.append(
                AgentToolLoopStep(
                    index=step_index,
                    kind="tool_execution",
                    tool_calls=tuple(executed_tool_calls),
                    tool_results=tool_results,
                    success=failed_outcome is None,
                    error_type=None if failed_outcome is None else failed_outcome.error_type,
                    error_category=(
                        None if failed_outcome is None else failed_outcome.error_category
                    ),
                    provider=response.provider,
                    model=response.model,
                )
            )

            if failed_outcome is not None and config.stop_on_tool_error:
                self._emit_failed(
                    step_index=step_index,
                    max_steps=config.max_steps,
                    error_type=failed_outcome.error_type or "ToolExecutionError",
                    error_category=failed_outcome.error_category or "tool_error",
                    provider=response.provider,
                    model=response.model,
                )
                return self._result(
                    completed=False,
                    reason="tool_error",
                    steps=steps,
                    config=config,
                    tool_results=all_tool_results,
                )

            history = (
                *history,
                *tuple(tool_result_message(tool_result) for tool_result in tool_results),
            )

        self._emit_max_steps_exceeded(
            step_index=config.max_steps,
            max_steps=config.max_steps,
            provider=route_metadata.get("provider"),
            model=route_metadata.get("model"),
        )
        return self._result(
            completed=False,
            reason="max_steps_exceeded",
            steps=steps,
            config=config,
            tool_results=all_tool_results,
        )

    def _execute_tool_call(
        self,
        tool_call: LLMToolCall,
        *,
        allowed_tool_names: frozenset[str],
        step_index: int,
        max_steps: int,
        provider: str,
        model: str,
    ) -> _ToolOutcome:
        self._emit(
            "agent_tool_loop.tool_execution_started",
            "Agent tool loop tool execution started",
            {
                "step_index": step_index,
                "max_steps": max_steps,
                "tool_name": tool_call.name,
                "success": False,
                "provider": provider,
                "model": model,
            },
        )

        if tool_call.name not in allowed_tool_names and self.tool_runtime.registry.get(
            tool_call.name
        ) is not None:
            outcome = _failed_tool_outcome(
                tool_call,
                error_type="ToolNotAllowedError",
                error_category="not_allowed",
            )
        else:
            try:
                execution_result = self.tool_runtime.execute(tool_call)
            except Exception as exc:
                outcome = _failed_tool_outcome(
                    tool_call,
                    error_type=exc.__class__.__name__,
                    error_category="execution_error",
                )
            else:
                outcome = _outcome_from_execution_result(execution_result)

        event_type = (
            "agent_tool_loop.tool_execution_completed"
            if outcome.success
            else "agent_tool_loop.tool_execution_failed"
        )
        self._emit(
            event_type,
            (
                "Agent tool loop tool execution completed"
                if outcome.success
                else "Agent tool loop tool execution failed"
            ),
            {
                "step_index": step_index,
                "max_steps": max_steps,
                "tool_name": tool_call.name,
                "success": outcome.success,
                "provider": provider,
                "model": model,
                **_safe_error_metadata(outcome.error_type, outcome.error_category),
            },
            error=not outcome.success,
        )
        return outcome

    def _coerce_tool_definitions(
        self,
        tools: Sequence[ToolInput] | None,
    ) -> tuple[LLMToolDefinition, ...]:
        if tools is None:
            return self.tool_runtime.registry.llm_tool_definitions()
        return tuple(_coerce_tool_definition(tool) for tool in tools)

    def _result(
        self,
        *,
        completed: bool,
        reason: str,
        config: AgentToolLoopConfig,
        final_response: LLMResponse | None = None,
        steps: Sequence[AgentToolLoopStep] = (),
        pending_tool_calls: Sequence[LLMToolCall] = (),
        tool_results: Sequence[LLMToolResult] = (),
    ) -> AgentToolLoopResult:
        return AgentToolLoopResult(
            completed=completed,
            reason=reason,
            final_response=final_response,
            steps=tuple(steps) if config.include_intermediate_steps else (),
            pending_tool_calls=tuple(pending_tool_calls),
            tool_results=tuple(tool_results),
        )

    def _emit_max_steps_exceeded(
        self,
        *,
        step_index: int,
        max_steps: int,
        provider: str | None,
        model: str | None,
    ) -> None:
        self._emit(
            "agent_tool_loop.max_steps_exceeded",
            "Agent tool loop max steps exceeded",
            {
                "step_index": step_index,
                "max_steps": max_steps,
                "success": False,
                **_safe_request_route_metadata(provider, model),
            },
            warning=True,
        )
        self._emit_failed(
            step_index=step_index,
            max_steps=max_steps,
            error_type="AgentToolLoopMaxStepsExceeded",
            error_category="max_steps_exceeded",
            provider=provider,
            model=model,
        )

    def _emit_failed(
        self,
        *,
        step_index: int,
        max_steps: int,
        error_type: str,
        error_category: str,
        provider: str | None,
        model: str | None,
    ) -> None:
        self._emit(
            "agent_tool_loop.failed",
            "Agent tool loop failed",
            {
                "step_index": step_index,
                "max_steps": max_steps,
                "success": False,
                "error_type": error_type,
                "error_category": error_category,
                **_safe_request_route_metadata(provider, model),
            },
            error=True,
        )

    def _emit(
        self,
        event_type: str,
        message: str,
        metadata: dict[str, Any],
        *,
        error: bool = False,
        warning: bool = False,
    ) -> None:
        if self.event_sink is None:
            return
        if error:
            factory = RuntimeEvent.error
        elif warning:
            factory = RuntimeEvent.warning
        else:
            factory = RuntimeEvent.info
        event = factory("AgentToolLoop", event_type, message, metadata)
        try:
            append = getattr(self.event_sink, "append", None)
            if callable(append):
                append(event)
            elif callable(self.event_sink):
                self.event_sink(event)
        except Exception:
            return


def assistant_tool_call_message(response: LLMResponse) -> LLMMessage:
    """Return the provider-neutral assistant message for returned tool calls."""
    return LLMMessage(
        "assistant",
        response.content,
        {
            "tool_calls": response.tool_calls,
            "provider": response.provider,
            "model": response.model,
        },
    )


def tool_result_message(tool_result: LLMToolResult) -> LLMMessage:
    """Return the provider-neutral tool result feedback message."""
    content = tool_result.content
    if not tool_result.success:
        content = tool_result.error or "Tool execution failed"
    return LLMMessage(
        "tool",
        content,
        {
            "tool_call_id": tool_result.tool_call_id,
            "name": tool_result.name,
            "success": tool_result.success,
        },
    )


def _resolve_config(
    config: AgentToolLoopConfig,
    *,
    max_steps: int | None,
    require_tool_approval: bool | None,
    stop_on_tool_error: bool | None,
    allow_parallel_tool_calls: bool | None,
    include_intermediate_steps: bool | None,
) -> AgentToolLoopConfig:
    replacements: dict[str, Any] = {}
    if max_steps is not None:
        replacements["max_steps"] = max_steps
    if require_tool_approval is not None:
        replacements["require_tool_approval"] = require_tool_approval
    if stop_on_tool_error is not None:
        replacements["stop_on_tool_error"] = stop_on_tool_error
    if allow_parallel_tool_calls is not None:
        replacements["allow_parallel_tool_calls"] = allow_parallel_tool_calls
    if include_intermediate_steps is not None:
        replacements["include_intermediate_steps"] = include_intermediate_steps
    return replace(config, **replacements) if replacements else config


def _coerce_message(message: MessageInput) -> LLMMessage:
    if isinstance(message, LLMMessage):
        return message
    if isinstance(message, Mapping):
        metadata = message.get("metadata", {})
        return LLMMessage(
            role=str(message.get("role", "")),
            content=str(message.get("content", "")),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
        )
    raise TypeError("messages must be LLMMessage objects or mappings")


def _coerce_tool_definition(tool: ToolInput) -> LLMToolDefinition:
    if isinstance(tool, LLMToolDefinition):
        return tool
    if isinstance(tool, ToolDefinition):
        return tool.to_llm_tool_definition()
    if isinstance(tool, Mapping):
        parameters_schema = tool.get("parameters_schema", tool.get("parameters", {}))
        if not isinstance(parameters_schema, Mapping):
            raise TypeError("tool parameters_schema must be a mapping")
        return LLMToolDefinition(
            name=str(tool.get("name", "")),
            description=str(tool.get("description", "")),
            parameters_schema=dict(parameters_schema),
        )
    raise TypeError("tools must be LLMToolDefinition, ToolDefinition, or mappings")


def _outcome_from_execution_result(result: ToolExecutionResult) -> _ToolOutcome:
    return _ToolOutcome(
        llm_result=result.to_llm_tool_result(),
        success=result.success,
        error_type=result.error_type,
        error_category=result.error_category,
    )


def _failed_tool_outcome(
    tool_call: LLMToolCall,
    *,
    error_type: str,
    error_category: str,
) -> _ToolOutcome:
    return _ToolOutcome(
        llm_result=LLMToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content="",
            success=False,
            error="Tool execution rejected",
        ),
        success=False,
        error_type=error_type,
        error_category=error_category,
    )


def _tool_names(tool_calls: Sequence[LLMToolCall]) -> tuple[str, ...]:
    return tuple(tool_call.name for tool_call in tool_calls)


def _safe_request_route_metadata(
    provider: str | None,
    model: str | None,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if isinstance(provider, str) and provider.strip():
        metadata["provider"] = provider
    if isinstance(model, str) and model.strip():
        metadata["model"] = model
    return metadata


def _runtime_route_metadata(
    runtime: LLMRuntime,
    provider: str | None,
    model: str | None,
) -> dict[str, str]:
    selected_provider = provider
    provider_object = None
    if selected_provider is None:
        default_provider = getattr(runtime, "default_provider", None)
        if isinstance(default_provider, str) and default_provider.strip():
            selected_provider = default_provider
            providers = getattr(runtime, "providers", {})
            if isinstance(providers, Mapping):
                provider_object = providers.get(selected_provider)
        else:
            provider_object = getattr(runtime, "provider", None)
            provider_name = getattr(provider_object, "name", None)
            if isinstance(provider_name, str) and provider_name.strip():
                selected_provider = provider_name

    selected_model = model
    if selected_model is None and provider_object is None and selected_provider is not None:
        providers = getattr(runtime, "providers", {})
        if isinstance(providers, Mapping):
            provider_object = providers.get(selected_provider)
    if selected_model is None and provider_object is None:
        provider_object = getattr(runtime, "provider", None)
    if selected_model is None:
        default_model = getattr(provider_object, "default_model", None)
        if isinstance(default_model, str) and default_model.strip():
            selected_model = default_model

    return _safe_request_route_metadata(selected_provider, selected_model)


def _safe_error_metadata(
    error_type: str | None,
    error_category: str | None,
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if error_type is not None:
        metadata["error_type"] = error_type
    if error_category is not None:
        metadata["error_category"] = error_category
    return metadata
