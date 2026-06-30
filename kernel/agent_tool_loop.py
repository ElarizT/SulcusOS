"""Safe explicit Agent OS loop for LLM tool orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal

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
ToolExecutionMode = Literal["sequential", "parallel"]
SUPPORTED_TOOL_EXECUTION_MODES: tuple[str, ...] = ("sequential", "parallel")


@dataclass(frozen=True)
class ToolPermissionPolicy:
    """Allow or deny tool calls before execution.

    Deny rules always win. With the default permissive policy, all tools are
    allowed unless explicitly denied.
    """

    allowed_tools: frozenset[str] | None = None
    denied_tools: frozenset[str] | None = None
    default_allow: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.default_allow, bool):
            raise ValueError("default_allow must be a boolean")
        object.__setattr__(
            self,
            "allowed_tools",
            _coerce_policy_tool_names(self.allowed_tools, "allowed_tools"),
        )
        object.__setattr__(
            self,
            "denied_tools",
            _coerce_policy_tool_names(self.denied_tools, "denied_tools"),
        )

    def check(self, tool_name: str) -> "ToolPermissionDecision":
        """Return whether a tool name is allowed and which rule matched."""
        if tool_name in self.denied_tools:
            return ToolPermissionDecision(False, "denied_tools", self.default_allow)
        if self.default_allow:
            return ToolPermissionDecision(True, None, self.default_allow)
        if tool_name in self.allowed_tools:
            return ToolPermissionDecision(True, None, self.default_allow)
        return ToolPermissionDecision(False, "not_in_allowed_tools", self.default_allow)


@dataclass(frozen=True)
class ToolPermissionDecision:
    """Result of checking one tool call against a permission policy."""

    allowed: bool
    matched_rule: str | None = None
    policy_default_allow: bool = True


@dataclass(frozen=True)
class AgentToolLoopConfig:
    """Conservative controls for bounded LLM-tool orchestration.

    One LLM response counts as one step/round. Tool calls returned by that
    response are executed sequentially before the next LLM round.
    """

    max_steps: int = 4
    require_tool_approval: bool = False
    stop_on_tool_error: bool = True
    allow_parallel_tool_calls: bool = False
    tool_execution_mode: ToolExecutionMode = "sequential"
    include_intermediate_steps: bool = True
    tool_permission_policy: ToolPermissionPolicy | None = None

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
        _validate_tool_execution_mode(self.tool_execution_mode)
        if self.tool_permission_policy is not None and not isinstance(
            self.tool_permission_policy,
            ToolPermissionPolicy,
        ):
            raise ValueError("tool_permission_policy must be a ToolPermissionPolicy")


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
    denied: bool = False


@dataclass(frozen=True)
class _ResolvedToolExecutionMode:
    effective_execution_mode: ToolExecutionMode
    fallback_reason: str | None
    parallel_safe_tool_count: int
    unsafe_tool_count: int


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
        tool_execution_mode: ToolExecutionMode | str | None = None,
        include_intermediate_steps: bool | None = None,
        tool_permission_policy: ToolPermissionPolicy | None = None,
    ) -> AgentToolLoopResult:
        """Run a bounded explicit LLM -> tool -> LLM loop."""
        config = _resolve_config(
            self.config,
            max_steps=max_steps,
            require_tool_approval=require_tool_approval,
            stop_on_tool_error=stop_on_tool_error,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            tool_execution_mode=tool_execution_mode,
            include_intermediate_steps=include_intermediate_steps,
            tool_permission_policy=tool_permission_policy,
        )
        if config.allow_parallel_tool_calls:
            raise ValueError("parallel tool calls are not supported yet")

        history = tuple(_coerce_message(message) for message in messages)
        tool_definitions = self._coerce_tool_definitions(tools)
        allowed_tool_names = frozenset(tool.name for tool in tool_definitions)
        permission_policy = config.tool_permission_policy or ToolPermissionPolicy()
        policy_summary = _tool_permission_policy_summary(config.tool_permission_policy)
        route_metadata = _runtime_route_metadata(self.llm_runtime, provider, model)
        steps: list[AgentToolLoopStep] = []
        all_tool_results: list[LLMToolResult] = []

        self._emit(
            "agent_tool_loop_started",
            "Agent tool loop started",
            {
                "max_steps": config.max_steps,
                "require_tool_approval": config.require_tool_approval,
                "stop_on_tool_error": config.stop_on_tool_error,
                "allow_parallel_tool_calls": config.allow_parallel_tool_calls,
                "execution_mode": config.tool_execution_mode,
                "tool_count": len(tool_definitions),
                "tool_names": tuple(tool.name for tool in tool_definitions),
                **policy_summary,
                **route_metadata,
                **_agent_metadata(metadata),
            },
        )
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
            history_tool_results = tuple(
                message for message in history if message.role == "tool"
            )
            if history_tool_results:
                successful_tool_results = sum(
                    1
                    for message in history_tool_results
                    if message.metadata.get("success") is True
                )
                followup_metadata = {
                    "round_index": step_index,
                    "step_index": step_index,
                    "provider": route_metadata.get("provider"),
                    "model": route_metadata.get("model"),
                    "tool_result_count": len(history_tool_results),
                    "successful_tool_result_count": successful_tool_results,
                    "failed_tool_result_count": (
                        len(history_tool_results) - successful_tool_results
                    ),
                }
                self._emit(
                    "llm_followup_request_started",
                    "Agent tool loop follow-up LLM request started",
                    followup_metadata,
                )
                self._emit(
                    "llm_final_request_started",
                    "Agent tool loop final LLM request started",
                    {
                        **followup_metadata,
                        "final_attempt": False,
                    },
                )
            else:
                self._emit(
                    "llm_request_started",
                    "Agent tool loop LLM request started",
                    {
                        "round_index": step_index,
                        "step_index": step_index,
                        "provider": route_metadata.get("provider"),
                        "model": route_metadata.get("model"),
                        "message_count": len(history),
                        "tool_names": tuple(tool.name for tool in tool_definitions),
                    },
                )
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
                self._emit(
                    "agent_tool_loop_failed",
                    "Agent tool loop failed",
                    {
                        "completed": False,
                        "reason": "llm_error",
                        "round_index": step_index,
                        "step_index": step_index,
                        "error_type": exc.__class__.__name__,
                        "error_category": error_category,
                        **_safe_request_route_metadata(
                            route_metadata.get("provider"),
                            route_metadata.get("model"),
                        ),
                    },
                    error=True,
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

            final_response_received = bool(history_tool_results) and not response.tool_calls
            self._emit(
                (
                    "llm_followup_response_received"
                    if history_tool_results
                    else "llm_response_received"
                ),
                (
                    "Agent tool loop follow-up LLM response received"
                    if history_tool_results
                    else "Agent tool loop LLM response received"
                ),
                {
                    "round_index": step_index,
                    "step_index": step_index,
                    "provider": response.provider,
                    "model": response.model,
                    "has_tool_calls": bool(response.tool_calls),
                    "tool_call_count": len(response.tool_calls),
                    "tool_result_count": len(all_tool_results),
                    "final_response_exists": not response.tool_calls,
                    **_finish_reason_metadata(response),
                    **_usage_metadata(response),
                },
            )
            if final_response_received:
                self._emit(
                    "llm_final_response_received",
                    "Agent tool loop final LLM response received",
                    {
                        "round_index": step_index,
                        "step_index": step_index,
                        "provider": response.provider,
                        "model": response.model,
                        "has_tool_calls": False,
                        "tool_call_count": 0,
                        "tool_result_count": len(all_tool_results),
                        "final_response_exists": True,
                        **_finish_reason_metadata(response),
                        **_usage_metadata(response),
                    },
                )

            if not response.tool_calls:
                self._emit(
                    "agent_tool_loop_completed",
                    "Agent tool loop completed",
                    {
                        "completed": True,
                        "reason": "completed",
                        "round_index": step_index,
                        "tool_result_count": len(all_tool_results),
                        "step_count": len(steps),
                        "provider": response.provider,
                        "model": response.model,
                    },
                )
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
                    "round_index": step_index,
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
                self._emit(
                    "agent_tool_loop_failed",
                    "Agent tool loop approval required",
                    {
                        "completed": False,
                        "reason": "approval_required",
                        "round_index": step_index,
                        "step_index": step_index,
                        "tool_call_count": len(response.tool_calls),
                        "tool_names": tool_names,
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
            permission_decisions = tuple(
                permission_policy.check(tool_call.name)
                for tool_call in response.tool_calls
            )
            permitted_tool_calls = tuple(
                tool_call
                for tool_call, decision in zip(response.tool_calls, permission_decisions)
                if decision.allowed
            )
            denied_tool_count = len(response.tool_calls) - len(permitted_tool_calls)
            resolved_mode = _resolve_tool_execution_group_mode(
                requested_execution_mode=config.tool_execution_mode,
                tool_calls=permitted_tool_calls,
                registry=self.tool_runtime.registry,
            )
            group_metadata = _tool_execution_group_metadata(
                requested_execution_mode=config.tool_execution_mode,
                effective_execution_mode=resolved_mode.effective_execution_mode,
                fallback_reason=resolved_mode.fallback_reason,
                parallel_safe_tool_count=resolved_mode.parallel_safe_tool_count,
                unsafe_tool_count=resolved_mode.unsafe_tool_count,
                step_index=step_index,
                tool_calls=response.tool_calls,
                provider=response.provider,
                model=response.model,
                successful_tool_count=0,
                failed_tool_count=0,
                denied_tool_count=denied_tool_count,
            )
            self._emit(
                "tool_execution_group_started",
                "Agent tool loop tool execution group started",
                group_metadata,
            )
            outcomes = self._execute_tool_group(
                response.tool_calls,
                allowed_tool_names=allowed_tool_names,
                permission_decisions=permission_decisions,
                step_index=step_index,
                max_steps=config.max_steps,
                requested_execution_mode=config.tool_execution_mode,
                effective_execution_mode=resolved_mode.effective_execution_mode,
                provider=response.provider,
                model=response.model,
                stop_on_tool_error=config.stop_on_tool_error,
            )
            executed_tool_calls = tuple(
                tool_call
                for tool_call, outcome in zip(response.tool_calls, outcomes)
                if outcome is not None
            )
            outcomes = [outcome for outcome in outcomes if outcome is not None]
            all_tool_results.extend(outcome.llm_result for outcome in outcomes)

            tool_results = tuple(outcome.llm_result for outcome in outcomes)
            failed_outcome = next(
                (outcome for outcome in outcomes if not outcome.success),
                None,
            )
            successful_tool_count = sum(1 for outcome in outcomes if outcome.success)
            failed_tool_count = len(outcomes) - successful_tool_count
            denied_tool_count = sum(1 for outcome in outcomes if outcome.denied)
            completed_group_metadata = _tool_execution_group_metadata(
                requested_execution_mode=config.tool_execution_mode,
                effective_execution_mode=resolved_mode.effective_execution_mode,
                fallback_reason=resolved_mode.fallback_reason,
                parallel_safe_tool_count=resolved_mode.parallel_safe_tool_count,
                unsafe_tool_count=resolved_mode.unsafe_tool_count,
                step_index=step_index,
                tool_calls=response.tool_calls,
                provider=response.provider,
                model=response.model,
                successful_tool_count=successful_tool_count,
                failed_tool_count=failed_tool_count,
                denied_tool_count=denied_tool_count,
            )
            if failed_outcome is None:
                self._emit(
                    "tool_execution_group_completed",
                    "Agent tool loop tool execution group completed",
                    completed_group_metadata,
                )
            else:
                self._emit(
                    "tool_execution_group_failed",
                    "Agent tool loop tool execution group failed",
                    {
                        **completed_group_metadata,
                        **_safe_error_metadata(
                            failed_outcome.error_type,
                            failed_outcome.error_category,
                        ),
                    },
                    error=True,
                )
            steps.append(
                AgentToolLoopStep(
                    index=step_index,
                    kind="tool_execution",
                    tool_calls=executed_tool_calls,
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
                self._emit(
                    "agent_tool_loop_failed",
                    "Agent tool loop failed",
                    {
                        "completed": False,
                        "reason": "tool_error",
                        "round_index": step_index,
                        "step_index": step_index,
                        "tool_result_count": len(all_tool_results),
                        "error_type": failed_outcome.error_type or "ToolExecutionError",
                        "error_category": failed_outcome.error_category or "tool_error",
                        "provider": response.provider,
                        "model": response.model,
                    },
                    error=True,
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

    def _execute_tool_group(
        self,
        tool_calls: Sequence[LLMToolCall],
        *,
        allowed_tool_names: frozenset[str],
        permission_decisions: Sequence[ToolPermissionDecision],
        step_index: int,
        max_steps: int,
        requested_execution_mode: ToolExecutionMode,
        effective_execution_mode: ToolExecutionMode,
        provider: str,
        model: str,
        stop_on_tool_error: bool,
    ) -> list[_ToolOutcome | None]:
        if effective_execution_mode == "parallel":
            return self._execute_tool_group_parallel(
                tool_calls,
                allowed_tool_names=allowed_tool_names,
                permission_decisions=permission_decisions,
                step_index=step_index,
                max_steps=max_steps,
                requested_execution_mode=requested_execution_mode,
                effective_execution_mode=effective_execution_mode,
                provider=provider,
                model=model,
            )

        outcomes: list[_ToolOutcome | None] = []
        for tool_call, permission_decision in zip(tool_calls, permission_decisions):
            self._emit(
                "tool_call_requested",
                "Agent tool loop tool call requested",
                {
                    "round_index": step_index,
                    "step_index": step_index,
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "argument_keys": _argument_keys(tool_call.arguments),
                    "execution_mode": effective_execution_mode,
                    "requested_execution_mode": requested_execution_mode,
                    "effective_execution_mode": effective_execution_mode,
                    "provider": provider,
                    "model": model,
                },
            )
            if permission_decision.allowed:
                outcome = self._execute_tool_call(
                    tool_call,
                    allowed_tool_names=allowed_tool_names,
                    step_index=step_index,
                    max_steps=max_steps,
                    requested_execution_mode=requested_execution_mode,
                    effective_execution_mode=effective_execution_mode,
                    provider=provider,
                    model=model,
                )
            else:
                outcome = self._deny_tool_call(
                    tool_call,
                    permission_decision,
                    step_index=step_index,
                    max_steps=max_steps,
                    requested_execution_mode=requested_execution_mode,
                    effective_execution_mode=effective_execution_mode,
                    provider=provider,
                    model=model,
                )
            outcomes.append(outcome)
            if not outcome.success and stop_on_tool_error:
                break
        return outcomes

    def _execute_tool_group_parallel(
        self,
        tool_calls: Sequence[LLMToolCall],
        *,
        allowed_tool_names: frozenset[str],
        permission_decisions: Sequence[ToolPermissionDecision],
        step_index: int,
        max_steps: int,
        requested_execution_mode: ToolExecutionMode,
        effective_execution_mode: ToolExecutionMode,
        provider: str,
        model: str,
    ) -> list[_ToolOutcome | None]:
        outcomes: list[_ToolOutcome | None] = [None] * len(tool_calls)
        for tool_call in tool_calls:
            self._emit(
                "tool_call_requested",
                "Agent tool loop tool call requested",
                {
                    "round_index": step_index,
                    "step_index": step_index,
                    "tool_call_id": tool_call.id,
                    "tool_name": tool_call.name,
                    "argument_keys": _argument_keys(tool_call.arguments),
                    "execution_mode": effective_execution_mode,
                    "requested_execution_mode": requested_execution_mode,
                    "effective_execution_mode": effective_execution_mode,
                    "provider": provider,
                    "model": model,
                },
            )

        permitted_calls: list[tuple[int, LLMToolCall]] = []
        for index, (tool_call, permission_decision) in enumerate(
            zip(tool_calls, permission_decisions)
        ):
            if permission_decision.allowed:
                permitted_calls.append((index, tool_call))
            else:
                outcomes[index] = self._deny_tool_call(
                    tool_call,
                    permission_decision,
                    step_index=step_index,
                    max_steps=max_steps,
                    requested_execution_mode=requested_execution_mode,
                    effective_execution_mode=effective_execution_mode,
                    provider=provider,
                    model=model,
                )

        if not permitted_calls:
            return outcomes

        with ThreadPoolExecutor(max_workers=len(permitted_calls)) as executor:
            futures = [
                (
                    index,
                    executor.submit(
                        self._execute_tool_call,
                        tool_call,
                        allowed_tool_names=allowed_tool_names,
                        step_index=step_index,
                        max_steps=max_steps,
                        requested_execution_mode=requested_execution_mode,
                        effective_execution_mode=effective_execution_mode,
                        provider=provider,
                        model=model,
                    ),
                )
                for index, tool_call in permitted_calls
            ]
            for index, future in futures:
                try:
                    outcomes[index] = future.result()
                except Exception as exc:
                    tool_call = tool_calls[index]
                    outcome = _failed_tool_outcome(
                        tool_call,
                        error_type=exc.__class__.__name__,
                        error_category="execution_error",
                        error="Tool execution failed",
                    )
                    self._emit_tool_outcome(
                        tool_call,
                        outcome,
                        step_index=step_index,
                        max_steps=max_steps,
                        requested_execution_mode=requested_execution_mode,
                        effective_execution_mode=effective_execution_mode,
                        provider=provider,
                        model=model,
                    )
                    outcomes[index] = outcome
        return outcomes

    def _deny_tool_call(
        self,
        tool_call: LLMToolCall,
        permission_decision: ToolPermissionDecision,
        *,
        step_index: int,
        max_steps: int,
        requested_execution_mode: ToolExecutionMode,
        effective_execution_mode: ToolExecutionMode,
        provider: str,
        model: str,
    ) -> _ToolOutcome:
        matched_rule = permission_decision.matched_rule or "permission_policy"
        outcome = _permission_denied_tool_outcome(tool_call)
        self._emit(
            "tool_call_denied",
            "Agent tool loop tool call denied",
            {
                "round_index": step_index,
                "step_index": step_index,
                "max_steps": max_steps,
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "reason": "permission_policy",
                "requested_execution_mode": requested_execution_mode,
                "effective_execution_mode": effective_execution_mode,
                "execution_mode": effective_execution_mode,
                "policy_default_allow": permission_decision.policy_default_allow,
                "matched_rule": matched_rule,
                "success": False,
                "provider": provider,
                "model": model,
                **_tool_result_preview_metadata(outcome.llm_result),
                **_safe_error_metadata(outcome.error_type, outcome.error_category),
            },
            error=True,
        )
        return outcome

    def _emit_tool_outcome(
        self,
        tool_call: LLMToolCall,
        outcome: _ToolOutcome,
        *,
        step_index: int,
        max_steps: int,
        requested_execution_mode: ToolExecutionMode,
        effective_execution_mode: ToolExecutionMode,
        provider: str,
        model: str,
    ) -> None:
        event_type = (
            "agent_tool_loop.tool_execution_completed"
            if outcome.success
            else "agent_tool_loop.tool_execution_failed"
        )
        metadata = {
            "round_index": step_index,
            "step_index": step_index,
            "max_steps": max_steps,
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "execution_mode": effective_execution_mode,
            "requested_execution_mode": requested_execution_mode,
            "effective_execution_mode": effective_execution_mode,
            "success": outcome.success,
            "provider": provider,
            "model": model,
            **_tool_result_preview_metadata(outcome.llm_result),
            **_safe_error_metadata(outcome.error_type, outcome.error_category),
        }
        self._emit(
            "tool_execution_completed" if outcome.success else "tool_execution_failed",
            (
                "Agent tool loop tool execution completed"
                if outcome.success
                else "Agent tool loop tool execution failed"
            ),
            metadata,
            error=not outcome.success,
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
                "execution_mode": effective_execution_mode,
                "requested_execution_mode": requested_execution_mode,
                "effective_execution_mode": effective_execution_mode,
                "success": outcome.success,
                "provider": provider,
                "model": model,
                **_safe_error_metadata(outcome.error_type, outcome.error_category),
            },
            error=not outcome.success,
        )

    def _execute_tool_call(
        self,
        tool_call: LLMToolCall,
        *,
        allowed_tool_names: frozenset[str],
        step_index: int,
        max_steps: int,
        requested_execution_mode: ToolExecutionMode,
        effective_execution_mode: ToolExecutionMode,
        provider: str,
        model: str,
    ) -> _ToolOutcome:
        self._emit(
            "tool_execution_started",
            "Agent tool loop tool execution started",
            {
                "round_index": step_index,
                "step_index": step_index,
                "max_steps": max_steps,
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "execution_mode": effective_execution_mode,
                "requested_execution_mode": requested_execution_mode,
                "effective_execution_mode": effective_execution_mode,
                "success": False,
                "provider": provider,
                "model": model,
            },
        )
        self._emit(
            "agent_tool_loop.tool_execution_started",
            "Agent tool loop tool execution started",
            {
                "step_index": step_index,
                "max_steps": max_steps,
                "tool_name": tool_call.name,
                "execution_mode": effective_execution_mode,
                "requested_execution_mode": requested_execution_mode,
                "effective_execution_mode": effective_execution_mode,
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

        self._emit_tool_outcome(
            tool_call,
            outcome,
            step_index=step_index,
            max_steps=max_steps,
            requested_execution_mode=requested_execution_mode,
            effective_execution_mode=effective_execution_mode,
            provider=provider,
            model=model,
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
        self._emit(
            "agent_tool_loop_failed",
            "Agent tool loop failed",
            {
                "completed": False,
                "reason": "max_steps_exceeded",
                "round_index": step_index,
                "step_index": step_index,
                "error_type": "AgentToolLoopMaxStepsExceeded",
                "error_category": "max_steps_exceeded",
                **_safe_request_route_metadata(provider, model),
            },
            error=True,
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
    tool_execution_mode: ToolExecutionMode | str | None,
    include_intermediate_steps: bool | None,
    tool_permission_policy: ToolPermissionPolicy | None,
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
    if tool_execution_mode is not None:
        replacements["tool_execution_mode"] = tool_execution_mode
    if include_intermediate_steps is not None:
        replacements["include_intermediate_steps"] = include_intermediate_steps
    if tool_permission_policy is not None:
        replacements["tool_permission_policy"] = tool_permission_policy
    return replace(config, **replacements) if replacements else config


def _validate_tool_execution_mode(mode: str) -> None:
    if mode not in SUPPORTED_TOOL_EXECUTION_MODES:
        supported = ", ".join(SUPPORTED_TOOL_EXECUTION_MODES)
        raise ValueError(
            f"Unsupported tool_execution_mode: {mode}. Supported modes: {supported}."
        )


def _coerce_policy_tool_names(
    tool_names: Iterable[str] | None,
    field_name: str,
) -> frozenset[str]:
    if tool_names is None:
        return frozenset()
    if isinstance(tool_names, (str, bytes)):
        raise ValueError(f"{field_name} must be an iterable of tool names")
    values: list[str] = []
    for tool_name in tool_names:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError(f"{field_name} must contain nonempty strings")
        normalized = tool_name.strip()
        if normalized != tool_name:
            raise ValueError(f"{field_name} tool names must not contain whitespace")
        values.append(tool_name)
    return frozenset(values)


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
    error: str = "Tool execution rejected",
) -> _ToolOutcome:
    return _ToolOutcome(
        llm_result=LLMToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content="",
            success=False,
            error=error,
        ),
        success=False,
        error_type=error_type,
        error_category=error_category,
    )


def _permission_denied_tool_outcome(tool_call: LLMToolCall) -> _ToolOutcome:
    return _ToolOutcome(
        llm_result=LLMToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content="",
            success=False,
            error=f"Tool call denied by permission policy: {tool_call.name}",
        ),
        success=False,
        error_type="ToolPermissionDeniedError",
        error_category="permission_policy",
        denied=True,
    )


def _tool_names(tool_calls: Sequence[LLMToolCall]) -> tuple[str, ...]:
    return tuple(tool_call.name for tool_call in tool_calls)


def _resolve_tool_execution_group_mode(
    *,
    requested_execution_mode: ToolExecutionMode,
    tool_calls: Sequence[LLMToolCall],
    registry: Any,
) -> _ResolvedToolExecutionMode:
    parallel_safe_tool_count = 0
    unsafe_tool_count = 0
    for tool_call in tool_calls:
        definition = registry.get(tool_call.name)
        if definition is not None and definition.parallel_safe:
            parallel_safe_tool_count += 1
        else:
            unsafe_tool_count += 1

    if requested_execution_mode == "parallel" and unsafe_tool_count == 0:
        return _ResolvedToolExecutionMode(
            effective_execution_mode="parallel",
            fallback_reason=None,
            parallel_safe_tool_count=parallel_safe_tool_count,
            unsafe_tool_count=unsafe_tool_count,
        )

    return _ResolvedToolExecutionMode(
        effective_execution_mode="sequential",
        fallback_reason=(
            "not_all_tools_parallel_safe"
            if requested_execution_mode == "parallel" and unsafe_tool_count > 0
            else None
        ),
        parallel_safe_tool_count=parallel_safe_tool_count,
        unsafe_tool_count=unsafe_tool_count,
    )


def _tool_execution_group_metadata(
    *,
    requested_execution_mode: ToolExecutionMode,
    effective_execution_mode: ToolExecutionMode,
    fallback_reason: str | None,
    parallel_safe_tool_count: int,
    unsafe_tool_count: int,
    step_index: int,
    tool_calls: Sequence[LLMToolCall],
    provider: str,
    model: str,
    successful_tool_count: int,
    failed_tool_count: int,
    denied_tool_count: int = 0,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "execution_mode": effective_execution_mode,
        "requested_execution_mode": requested_execution_mode,
        "effective_execution_mode": effective_execution_mode,
        "round_index": step_index,
        "step_index": step_index,
        "tool_call_count": len(tool_calls),
        "tool_names": _tool_names(tool_calls),
        "parallel_safe_tool_count": parallel_safe_tool_count,
        "unsafe_tool_count": unsafe_tool_count,
        "successful_tool_count": successful_tool_count,
        "failed_tool_count": failed_tool_count,
        "denied_tool_count": denied_tool_count,
        "provider": provider,
        "model": model,
    }
    if fallback_reason is not None:
        metadata["fallback_reason"] = fallback_reason
    return metadata


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


def _tool_permission_policy_summary(
    policy: ToolPermissionPolicy | None,
) -> dict[str, int | bool]:
    effective_policy = policy or ToolPermissionPolicy()
    return {
        "tool_policy_enabled": policy is not None,
        "policy_default_allow": effective_policy.default_allow,
        "allowed_tool_count": len(effective_policy.allowed_tools),
        "denied_tool_count": len(effective_policy.denied_tools),
    }


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


def _agent_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    if metadata is None:
        return {}
    values: dict[str, str] = {}
    for source_key, target_key in (
        ("agent", "agent"),
        ("agent_id", "agent_id"),
        ("agent_name", "agent_name"),
    ):
        value = metadata.get(source_key)
        if _is_short_scalar(value):
            values[target_key] = str(value)
    return values


def _finish_reason_metadata(response: LLMResponse) -> dict[str, str]:
    finish_reason = response.metadata.get("finish_reason")
    if _is_short_scalar(finish_reason):
        return {"finish_reason": str(finish_reason)}
    return {}


def _usage_metadata(response: LLMResponse) -> dict[str, int]:
    usage = response.usage
    if usage is None:
        return {}
    metadata: dict[str, int] = {}
    if usage.prompt_tokens is not None:
        metadata["prompt_tokens"] = usage.prompt_tokens
    if usage.completion_tokens is not None:
        metadata["completion_tokens"] = usage.completion_tokens
    if usage.total_tokens is not None:
        metadata["total_tokens"] = usage.total_tokens
    return metadata


def _argument_keys(arguments: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(str(key) for key in arguments))


def _tool_result_preview_metadata(tool_result: LLMToolResult) -> dict[str, str]:
    preview_source = tool_result.content if tool_result.success else tool_result.error
    if not isinstance(preview_source, str) or not preview_source:
        return {}
    preview = preview_source.replace("\n", " ")
    if tool_result.success and preview.strip().startswith(("{", "[")):
        return {}
    if len(preview) > 80:
        preview = f"{preview[:77]}..."
    return {"output_preview" if tool_result.success else "error_preview": preview}


def _is_short_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) and len(str(value)) <= 120
