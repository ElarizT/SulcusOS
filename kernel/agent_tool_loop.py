"""Safe explicit Agent OS loop for LLM tool orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from typing import Any, Literal
from uuid import uuid4

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
class ToolResourceLimits:
    """Optional per-run safety limits for agent-loop tool usage.

    Requested calls count toward call limits, including permission-denied and
    resource-denied calls. Timeouts apply only after actual execution starts.
    """

    max_tool_calls_per_loop: int | None = None
    max_tool_calls_per_round: int | None = None
    max_calls_per_tool: Mapping[str, int] | None = None
    tool_timeout_ms: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "max_tool_calls_per_loop",
            "max_tool_calls_per_round",
            "tool_timeout_ms",
        ):
            _validate_optional_nonnegative_int(getattr(self, field_name), field_name)
        if self.max_calls_per_tool is None:
            return
        if not isinstance(self.max_calls_per_tool, Mapping):
            raise ValueError("max_calls_per_tool must be a mapping of tool names to limits")
        normalized: dict[str, int] = {}
        for tool_name, limit in self.max_calls_per_tool.items():
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("max_calls_per_tool keys must be nonempty strings")
            if tool_name.strip() != tool_name:
                raise ValueError("max_calls_per_tool tool names must not contain whitespace")
            _validate_optional_nonnegative_int(limit, "max_calls_per_tool values")
            assert limit is not None
            normalized[tool_name] = limit
        object.__setattr__(self, "max_calls_per_tool", normalized)


@dataclass(frozen=True)
class ToolApprovalDecision:
    """Caller decision for one pending tool call.

    ``reason`` is deliberately not copied into runtime events or tool feedback.
    It is useful to a caller's own audit trail only.
    """

    tool_call_id: str
    approved: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
            raise ValueError("tool_call_id must be a nonempty string")
        if not isinstance(self.approved, bool):
            raise ValueError("approved must be a boolean")
        if self.reason is not None and not isinstance(self.reason, str):
            raise ValueError("reason must be a string or None")


@dataclass(frozen=True)
class PendingToolApproval:
    """Safe, inspectable metadata for a tool call awaiting caller approval."""

    tool_call_id: str
    tool_name: str
    round_index: int
    call_index: int
    requested_execution_mode: ToolExecutionMode
    effective_execution_mode: ToolExecutionMode


@dataclass(frozen=True)
class AgentToolLoopCheckpoint:
    """In-memory resume state for a paused approval round.

    This intentionally contains tool definitions and call data, but never a
    registered Python callable. Resume validates those definitions against the
    live ToolRuntime registry owned by the same loop instance.
    """

    checkpoint_version: int
    checkpoint_id: str
    loop_id: str
    round_index: int
    history: tuple[LLMMessage, ...]
    response: LLMResponse
    tool_definitions: tuple[LLMToolDefinition, ...]
    allowed_tool_names: frozenset[str]
    config: "AgentToolLoopConfig"
    steps: tuple["AgentToolLoopStep", ...]
    tool_results: tuple[LLMToolResult, ...]
    pending_approvals: tuple[PendingToolApproval, ...]
    preflight_outcomes: tuple["_ToolOutcome | None", ...]
    requested_execution_mode: ToolExecutionMode
    effective_execution_mode: ToolExecutionMode
    fallback_reason: str | None
    parallel_safe_tool_count: int
    unsafe_tool_count: int
    total_requested: int
    round_requested: tuple[tuple[int, int], ...]
    tool_requested: tuple[tuple[str, int], ...]
    provider: str
    model: str


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
    tool_resource_limits: ToolResourceLimits | None = None

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
        if self.tool_resource_limits is not None and not isinstance(
            self.tool_resource_limits,
            ToolResourceLimits,
        ):
            raise ValueError("tool_resource_limits must be a ToolResourceLimits")


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
    pending_approvals: tuple[PendingToolApproval, ...] = ()
    checkpoint: AgentToolLoopCheckpoint | None = None
    current_round_index: int | None = None
    current_step_index: int | None = None
    tool_results: tuple[LLMToolResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.completed, bool):
            raise ValueError("completed must be a boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("result reason must not be empty")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "pending_tool_calls", tuple(self.pending_tool_calls))
        object.__setattr__(self, "pending_approvals", tuple(self.pending_approvals))
        object.__setattr__(self, "tool_results", tuple(self.tool_results))


@dataclass(frozen=True)
class _ToolOutcome:
    llm_result: LLMToolResult
    success: bool
    error_type: str | None = None
    error_category: str | None = None
    denied: bool = False
    resource_denied: bool = False
    timed_out: bool = False
    limit_name: str | None = None
    limit_value: int | None = None


@dataclass(frozen=True)
class _ResolvedToolExecutionMode:
    effective_execution_mode: ToolExecutionMode
    fallback_reason: str | None
    parallel_safe_tool_count: int
    unsafe_tool_count: int


@dataclass(frozen=True)
class _ToolResourceDecision:
    allowed: bool
    limit_name: str | None = None
    limit_value: int | None = None
    current_count: int | None = None


class _ToolResourceLimitState:
    """Mutable per-run requested-call counters for tool resource limits."""

    def __init__(self, limits: ToolResourceLimits | None) -> None:
        self.limits = limits
        self.total_requested = 0
        self.round_requested: dict[int, int] = {}
        self.tool_requested: dict[str, int] = {}

    def record_requested(
        self,
        tool_call: LLMToolCall,
        *,
        round_index: int,
    ) -> _ToolResourceDecision:
        self.total_requested += 1
        round_count = self.round_requested.get(round_index, 0) + 1
        self.round_requested[round_index] = round_count
        tool_count = self.tool_requested.get(tool_call.name, 0) + 1
        self.tool_requested[tool_call.name] = tool_count

        limits = self.limits
        if limits is None:
            return _ToolResourceDecision(True)
        if (
            limits.max_tool_calls_per_loop is not None
            and self.total_requested > limits.max_tool_calls_per_loop
        ):
            return _ToolResourceDecision(
                False,
                "max_tool_calls_per_loop",
                limits.max_tool_calls_per_loop,
                self.total_requested,
            )
        if (
            limits.max_tool_calls_per_round is not None
            and round_count > limits.max_tool_calls_per_round
        ):
            return _ToolResourceDecision(
                False,
                "max_tool_calls_per_round",
                limits.max_tool_calls_per_round,
                round_count,
            )
        per_tool_limits = limits.max_calls_per_tool or {}
        per_tool_limit = per_tool_limits.get(tool_call.name)
        if per_tool_limit is not None and tool_count > per_tool_limit:
            return _ToolResourceDecision(
                False,
                "max_calls_per_tool",
                per_tool_limit,
                tool_count,
            )
        return _ToolResourceDecision(True)

    def snapshot(self) -> tuple[int, tuple[tuple[int, int], ...], tuple[tuple[str, int], ...]]:
        return (self.total_requested, tuple(sorted(self.round_requested.items())), tuple(sorted(self.tool_requested.items())))

    @classmethod
    def from_snapshot(
        cls,
        limits: ToolResourceLimits | None,
        total_requested: int,
        round_requested: Sequence[tuple[int, int]],
        tool_requested: Sequence[tuple[str, int]],
    ) -> "_ToolResourceLimitState":
        state = cls(limits)
        state.total_requested = total_requested
        state.round_requested = dict(round_requested)
        state.tool_requested = dict(tool_requested)
        return state


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
        self._loop_id = uuid4().hex
        self._consumed_checkpoints: set[str] = set()

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
        tool_resource_limits: ToolResourceLimits | None = None,
        _checkpoint: AgentToolLoopCheckpoint | None = None,
        _approval_decisions: Mapping[str, ToolApprovalDecision] | None = None,
    ) -> AgentToolLoopResult:
        """Run a bounded explicit LLM -> tool -> LLM loop."""
        config = (_checkpoint.config if _checkpoint is not None else _resolve_config(
            self.config,
            max_steps=max_steps,
            require_tool_approval=require_tool_approval,
            stop_on_tool_error=stop_on_tool_error,
            allow_parallel_tool_calls=allow_parallel_tool_calls,
            tool_execution_mode=tool_execution_mode,
            include_intermediate_steps=include_intermediate_steps,
            tool_permission_policy=tool_permission_policy,
            tool_resource_limits=tool_resource_limits,
        ))
        if config.allow_parallel_tool_calls:
            raise ValueError("parallel tool calls are not supported yet")

        history = (_checkpoint.history if _checkpoint is not None else tuple(_coerce_message(message) for message in messages))
        tool_definitions = (_checkpoint.tool_definitions if _checkpoint is not None else self._coerce_tool_definitions(tools))
        allowed_tool_names = (_checkpoint.allowed_tool_names if _checkpoint is not None else frozenset(tool.name for tool in tool_definitions))
        permission_policy = config.tool_permission_policy or ToolPermissionPolicy()
        policy_summary = _tool_permission_policy_summary(config.tool_permission_policy)
        resource_limit_state = (_ToolResourceLimitState.from_snapshot(
            config.tool_resource_limits,
            _checkpoint.total_requested,
            _checkpoint.round_requested,
            _checkpoint.tool_requested,
        ) if _checkpoint is not None else _ToolResourceLimitState(config.tool_resource_limits))
        resource_limit_summary = _tool_resource_limits_summary(config.tool_resource_limits)
        route_metadata = _runtime_route_metadata(self.llm_runtime, provider, model)
        steps: list[AgentToolLoopStep] = list(_checkpoint.steps) if _checkpoint is not None else []
        all_tool_results: list[LLMToolResult] = list(_checkpoint.tool_results) if _checkpoint is not None else []

        if _checkpoint is None:
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
                **resource_limit_summary,
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

        for step_index in range(_checkpoint.round_index if _checkpoint is not None else 0, config.max_steps):
            resuming_round = _checkpoint is not None and step_index == _checkpoint.round_index
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
                response = (
                    _checkpoint.response
                    if resuming_round
                    else self.llm_runtime.chat(
                        history,
                        model=model,
                        temperature=temperature,
                        metadata=metadata,
                        provider=provider,
                        timeout_seconds=timeout_seconds,
                        tools=tool_definitions,
                        tool_choice=tool_choice,
                    )
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
            if not resuming_round:
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

            if config.require_tool_approval and not resuming_round:
                history = (*history, assistant_tool_call_message(response))
                permission_decisions = tuple(
                    permission_policy.check(tool_call.name) for tool_call in response.tool_calls
                )
                preflight: list[_ToolOutcome | None] = []
                approvable: list[tuple[int, LLMToolCall]] = []
                for call_index, (tool_call, permission_decision) in enumerate(
                    zip(response.tool_calls, permission_decisions)
                ):
                    self._emit_tool_call_requested(
                        tool_call, step_index, config.tool_execution_mode,
                        response.provider, response.model,
                    )
                    resource_decision = resource_limit_state.record_requested(
                        tool_call, round_index=step_index
                    )
                    if not permission_decision.allowed:
                        preflight.append(self._deny_tool_call(
                            tool_call, permission_decision, step_index=step_index,
                            max_steps=config.max_steps,
                            requested_execution_mode=config.tool_execution_mode,
                            effective_execution_mode="sequential", provider=response.provider,
                            model=response.model,
                        ))
                    elif not resource_decision.allowed:
                        preflight.append(self._resource_deny_tool_call(
                            tool_call, resource_decision, step_index=step_index,
                            max_steps=config.max_steps,
                            requested_execution_mode=config.tool_execution_mode,
                            effective_execution_mode="sequential", provider=response.provider,
                            model=response.model,
                        ))
                    else:
                        preflight.append(None)
                        approvable.append((call_index, tool_call))
                resolved_mode = _resolve_tool_execution_group_mode(
                    requested_execution_mode=config.tool_execution_mode,
                    tool_calls=tuple(call for _, call in approvable), registry=self.tool_runtime.registry,
                )
                pending = tuple(
                    PendingToolApproval(
                        tool_call_id=call.id, tool_name=call.name, round_index=step_index,
                        call_index=index, requested_execution_mode=config.tool_execution_mode,
                        effective_execution_mode=resolved_mode.effective_execution_mode,
                    ) for index, call in approvable
                )
                if pending:
                    total_requested, round_requested, tool_requested = resource_limit_state.snapshot()
                    paused_step = AgentToolLoopStep(
                        index=step_index, kind="approval_required", tool_calls=response.tool_calls,
                        success=False, provider=response.provider, model=response.model,
                    )
                    checkpoint = AgentToolLoopCheckpoint(
                        checkpoint_version=1, checkpoint_id=uuid4().hex, loop_id=self._loop_id,
                        round_index=step_index, history=history, response=response,
                        tool_definitions=tool_definitions, allowed_tool_names=allowed_tool_names,
                        config=config, steps=(*steps, paused_step), tool_results=tuple(all_tool_results),
                        pending_approvals=pending, preflight_outcomes=tuple(preflight),
                        requested_execution_mode=config.tool_execution_mode,
                        effective_execution_mode=resolved_mode.effective_execution_mode,
                        fallback_reason=resolved_mode.fallback_reason,
                        parallel_safe_tool_count=resolved_mode.parallel_safe_tool_count,
                        unsafe_tool_count=resolved_mode.unsafe_tool_count,
                        total_requested=total_requested, round_requested=round_requested,
                        tool_requested=tool_requested, provider=response.provider, model=response.model,
                    )
                    steps.append(paused_step)
                    for approval in pending:
                        self._emit("tool_approval_requested", "Tool approval requested", {
                            "round_index": step_index, "step_index": step_index,
                            "tool_call_id": approval.tool_call_id, "tool_name": approval.tool_name,
                            "call_index": approval.call_index, "pending_approval_count": len(pending),
                            "requested_execution_mode": approval.requested_execution_mode,
                            "effective_execution_mode": approval.effective_execution_mode,
                        }, warning=True)
                    self._emit("agent_tool_loop_paused", "Agent tool loop paused for approval", {
                        "round_index": step_index, "step_index": step_index,
                        "pending_approval_count": len(pending),
                    }, warning=True)
                    self._emit("agent_tool_loop.approval_required", "Agent tool loop approval required", {
                        "step_index": step_index, "max_steps": config.max_steps,
                        "tool_count": len(response.tool_calls), "tool_names": tool_names,
                        "success": False, "provider": response.provider, "model": response.model,
                    }, warning=True)
                    return self._result(
                        completed=False, reason="approval_required", steps=steps, config=config,
                        pending_tool_calls=tuple(response.tool_calls[index] for index, _ in approvable),
                        pending_approvals=pending, checkpoint=checkpoint,
                        current_round_index=step_index, current_step_index=step_index,
                        tool_results=all_tool_results,
                    )
                # Nothing passed preflight, so approval is irrelevant. Feed the
                # already-accounted denial results back to the LLM without
                # charging this attempted group a second time.
                outcomes = [outcome for outcome in preflight if outcome is not None]
                all_tool_results.extend(outcome.llm_result for outcome in outcomes)
                failed_outcome = next((outcome for outcome in outcomes if not outcome.success), None)
                steps.append(AgentToolLoopStep(
                    index=step_index, kind="tool_execution", tool_calls=response.tool_calls,
                    tool_results=tuple(outcome.llm_result for outcome in outcomes),
                    success=failed_outcome is None,
                    error_type=None if failed_outcome is None else failed_outcome.error_type,
                    error_category=None if failed_outcome is None else failed_outcome.error_category,
                    provider=response.provider, model=response.model,
                ))
                if failed_outcome is not None and config.stop_on_tool_error:
                    return self._result(
                        completed=False, reason="tool_error", steps=steps, config=config,
                        tool_results=all_tool_results,
                    )
                history = (*history, *(tool_result_message(outcome.llm_result) for outcome in outcomes))
                continue

            if step_index + 1 >= config.max_steps and not resuming_round:
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

            if not resuming_round:
                history = (*history, assistant_tool_call_message(response))
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
            resolved_mode = (_ResolvedToolExecutionMode(
                _checkpoint.effective_execution_mode,
                _checkpoint.fallback_reason,
                _checkpoint.parallel_safe_tool_count,
                _checkpoint.unsafe_tool_count,
            ) if resuming_round else _resolve_tool_execution_group_mode(
                requested_execution_mode=config.tool_execution_mode,
                tool_calls=permitted_tool_calls,
                registry=self.tool_runtime.registry,
            ))
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
                resource_denied_tool_count=0,
                timed_out_tool_count=0,
                tool_resource_limits=config.tool_resource_limits,
            )
            self._emit(
                "tool_execution_group_started",
                "Agent tool loop tool execution group started",
                group_metadata,
            )
            outcomes = (
                self._execute_approved_tool_group(
                    response.tool_calls,
                    preflight_outcomes=_checkpoint.preflight_outcomes,
                    decisions=_approval_decisions or {},
                    allowed_tool_names=allowed_tool_names,
                    step_index=step_index,
                    max_steps=config.max_steps,
                    requested_execution_mode=_checkpoint.requested_execution_mode,
                    effective_execution_mode=_checkpoint.effective_execution_mode,
                    tool_resource_limits=config.tool_resource_limits,
                    provider=response.provider,
                    model=response.model,
                    stop_on_tool_error=config.stop_on_tool_error,
                ) if resuming_round else self._execute_tool_group(
                    response.tool_calls,
                    allowed_tool_names=allowed_tool_names,
                    permission_decisions=permission_decisions,
                    step_index=step_index,
                    max_steps=config.max_steps,
                    requested_execution_mode=config.tool_execution_mode,
                    effective_execution_mode=resolved_mode.effective_execution_mode,
                    resource_limit_state=resource_limit_state,
                    tool_resource_limits=config.tool_resource_limits,
                    provider=response.provider,
                    model=response.model,
                    stop_on_tool_error=config.stop_on_tool_error,
                )
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
            resource_denied_tool_count = sum(
                1 for outcome in outcomes if outcome.resource_denied
            )
            timed_out_tool_count = sum(1 for outcome in outcomes if outcome.timed_out)
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
                resource_denied_tool_count=resource_denied_tool_count,
                timed_out_tool_count=timed_out_tool_count,
                tool_resource_limits=config.tool_resource_limits,
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

    def resume(
        self,
        *,
        checkpoint: AgentToolLoopCheckpoint,
        approval_decisions: Sequence[ToolApprovalDecision],
    ) -> AgentToolLoopResult:
        """Resume a paused approval round without repeating its LLM request."""
        if not isinstance(checkpoint, AgentToolLoopCheckpoint):
            raise TypeError("checkpoint must be an AgentToolLoopCheckpoint")
        if checkpoint.checkpoint_version != 1:
            raise ValueError("unsupported checkpoint_version")
        if checkpoint.loop_id != self._loop_id:
            raise ValueError("checkpoint belongs to a different AgentToolLoop instance")
        if checkpoint.checkpoint_id in self._consumed_checkpoints:
            raise ValueError("checkpoint has already been consumed")
        pending_ids = {approval.tool_call_id for approval in checkpoint.pending_approvals}
        decisions: dict[str, ToolApprovalDecision] = {}
        for decision in approval_decisions:
            if not isinstance(decision, ToolApprovalDecision):
                raise TypeError("approval_decisions must contain ToolApprovalDecision values")
            if decision.tool_call_id not in pending_ids:
                raise ValueError("approval decision references an unknown or resolved tool call")
            if decision.tool_call_id in decisions:
                raise ValueError("duplicate approval decision for tool call")
            decisions[decision.tool_call_id] = decision
        if set(decisions) != pending_ids:
            # Partial submissions are intentionally non-consuming and leave the
            # same checkpoint inspectable until every pending call is decided.
            return self._result(
                completed=False, reason="approval_required", config=checkpoint.config,
                steps=checkpoint.steps,
                pending_tool_calls=tuple(
                    call for call in checkpoint.response.tool_calls if call.id in pending_ids
                ),
                pending_approvals=checkpoint.pending_approvals, checkpoint=checkpoint,
                current_round_index=checkpoint.round_index,
                current_step_index=checkpoint.round_index,
                tool_results=checkpoint.tool_results,
            )
        for definition in checkpoint.tool_definitions:
            registered = self.tool_runtime.registry.get(definition.name)
            if registered is None:
                raise ValueError("checkpoint tool is no longer registered")
            if registered.to_llm_tool_definition().parameters_schema != definition.parameters_schema:
                raise ValueError("checkpoint tool definitions are incompatible with the registry")
        self._consumed_checkpoints.add(checkpoint.checkpoint_id)
        self._emit("agent_tool_loop_resumed", "Agent tool loop resumed", {
            "round_index": checkpoint.round_index, "step_index": checkpoint.round_index,
            "pending_approval_count": len(checkpoint.pending_approvals),
            "requested_execution_mode": checkpoint.requested_execution_mode,
            "effective_execution_mode": checkpoint.effective_execution_mode,
        })
        return self.run(
            (), (), _checkpoint=checkpoint, _approval_decisions=decisions,
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
        resource_limit_state: _ToolResourceLimitState,
        tool_resource_limits: ToolResourceLimits | None,
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
                resource_limit_state=resource_limit_state,
                tool_resource_limits=tool_resource_limits,
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
            resource_decision = resource_limit_state.record_requested(
                tool_call,
                round_index=step_index,
            )
            if permission_decision.allowed:
                if resource_decision.allowed:
                    outcome = self._execute_tool_call(
                        tool_call,
                        allowed_tool_names=allowed_tool_names,
                        step_index=step_index,
                        max_steps=max_steps,
                        requested_execution_mode=requested_execution_mode,
                        effective_execution_mode=effective_execution_mode,
                        tool_resource_limits=tool_resource_limits,
                        provider=provider,
                        model=model,
                    )
                else:
                    outcome = self._resource_deny_tool_call(
                        tool_call,
                        resource_decision,
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

    def _emit_tool_call_requested(
        self,
        tool_call: LLMToolCall,
        step_index: int,
        requested_execution_mode: ToolExecutionMode,
        provider: str,
        model: str,
    ) -> None:
        """Emit the safe requested event used before approval preflight."""
        self._emit("tool_call_requested", "Agent tool loop tool call requested", {
            "round_index": step_index, "step_index": step_index,
            "tool_call_id": tool_call.id, "tool_name": tool_call.name,
            "argument_keys": _argument_keys(tool_call.arguments),
            "requested_execution_mode": requested_execution_mode,
            "provider": provider, "model": model,
        })

    def _execute_approved_tool_group(
        self,
        tool_calls: Sequence[LLMToolCall],
        *,
        preflight_outcomes: Sequence[_ToolOutcome | None],
        decisions: Mapping[str, ToolApprovalDecision],
        allowed_tool_names: frozenset[str],
        step_index: int,
        max_steps: int,
        requested_execution_mode: ToolExecutionMode,
        effective_execution_mode: ToolExecutionMode,
        tool_resource_limits: ToolResourceLimits | None,
        provider: str,
        model: str,
        stop_on_tool_error: bool,
    ) -> list[_ToolOutcome | None]:
        """Finish a preflighted group without charging resource limits again."""
        outcomes: list[_ToolOutcome | None] = list(preflight_outcomes)
        executable: list[tuple[int, LLMToolCall]] = []
        for index, tool_call in enumerate(tool_calls):
            if outcomes[index] is not None:
                continue
            decision = decisions[tool_call.id]
            if decision.approved:
                self._emit("tool_approval_granted", "Tool approval granted", {
                    "round_index": step_index, "step_index": step_index,
                    "tool_call_id": tool_call.id, "tool_name": tool_call.name,
                    "call_index": index, "requested_execution_mode": requested_execution_mode,
                    "effective_execution_mode": effective_execution_mode,
                })
                executable.append((index, tool_call))
            else:
                outcome = _approval_denied_tool_outcome(tool_call)
                self._emit("tool_approval_denied", "Tool approval denied", {
                    "round_index": step_index, "step_index": step_index,
                    "tool_call_id": tool_call.id, "tool_name": tool_call.name,
                    "call_index": index, "requested_execution_mode": requested_execution_mode,
                    "effective_execution_mode": effective_execution_mode,
                    "success": False, "error_category": "approval_denied",
                }, warning=True)
                outcomes[index] = outcome

        if effective_execution_mode == "parallel":
            with ThreadPoolExecutor(max_workers=max(1, len(executable))) as executor:
                futures = [(index, executor.submit(
                    self._execute_tool_call, call, allowed_tool_names=allowed_tool_names,
                    step_index=step_index, max_steps=max_steps,
                    requested_execution_mode=requested_execution_mode,
                    effective_execution_mode=effective_execution_mode,
                    tool_resource_limits=tool_resource_limits, provider=provider, model=model,
                )) for index, call in executable]
                for index, future in futures:
                    outcomes[index] = future.result()
            return outcomes

        for index, tool_call in executable:
            outcome = self._execute_tool_call(
                tool_call, allowed_tool_names=allowed_tool_names, step_index=step_index,
                max_steps=max_steps, requested_execution_mode=requested_execution_mode,
                effective_execution_mode=effective_execution_mode,
                tool_resource_limits=tool_resource_limits, provider=provider, model=model,
            )
            outcomes[index] = outcome
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
        resource_limit_state: _ToolResourceLimitState,
        tool_resource_limits: ToolResourceLimits | None,
        provider: str,
        model: str,
    ) -> list[_ToolOutcome | None]:
        outcomes: list[_ToolOutcome | None] = [None] * len(tool_calls)
        permitted_calls: list[tuple[int, LLMToolCall]] = []
        for index, (tool_call, permission_decision) in enumerate(
            zip(tool_calls, permission_decisions)
        ):
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
            resource_decision = resource_limit_state.record_requested(
                tool_call,
                round_index=step_index,
            )
            if permission_decision.allowed:
                if resource_decision.allowed:
                    permitted_calls.append((index, tool_call))
                else:
                    outcomes[index] = self._resource_deny_tool_call(
                        tool_call,
                        resource_decision,
                        step_index=step_index,
                        max_steps=max_steps,
                        requested_execution_mode=requested_execution_mode,
                        effective_execution_mode=effective_execution_mode,
                        provider=provider,
                        model=model,
                    )
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
                        tool_resource_limits=tool_resource_limits,
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

    def _resource_deny_tool_call(
        self,
        tool_call: LLMToolCall,
        resource_decision: _ToolResourceDecision,
        *,
        step_index: int,
        max_steps: int,
        requested_execution_mode: ToolExecutionMode,
        effective_execution_mode: ToolExecutionMode,
        provider: str,
        model: str,
    ) -> _ToolOutcome:
        limit_name = resource_decision.limit_name or "resource_limits"
        outcome = _resource_denied_tool_outcome(
            tool_call,
            limit_name=limit_name,
        )
        self._emit(
            "tool_call_resource_denied",
            "Agent tool loop tool call denied by resource limits",
            {
                "round_index": step_index,
                "step_index": step_index,
                "max_steps": max_steps,
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
                "reason": "resource_limits",
                "limit_name": limit_name,
                "limit_value": resource_decision.limit_value,
                "current_count": resource_decision.current_count,
                "requested_execution_mode": requested_execution_mode,
                "effective_execution_mode": effective_execution_mode,
                "execution_mode": effective_execution_mode,
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
        if outcome.limit_name is not None:
            metadata["limit_name"] = outcome.limit_name
        if outcome.limit_value is not None:
            metadata["limit_value"] = outcome.limit_value
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
                **_limit_metadata(outcome),
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
        tool_resource_limits: ToolResourceLimits | None,
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
                execution_result = self._execute_with_optional_timeout(
                    tool_call,
                    tool_resource_limits=tool_resource_limits,
                )
            except Exception as exc:
                outcome = _failed_tool_outcome(
                    tool_call,
                    error_type=exc.__class__.__name__,
                    error_category="execution_error",
                )
            else:
                outcome = _outcome_from_execution_result(execution_result)
                if (
                    execution_result.error_type == "ToolTimeoutError"
                    and tool_resource_limits is not None
                    and tool_resource_limits.tool_timeout_ms is not None
                ):
                    outcome = replace(
                        outcome,
                        timed_out=True,
                        limit_name="tool_timeout_ms",
                        limit_value=tool_resource_limits.tool_timeout_ms,
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
        return outcome

    def _execute_with_optional_timeout(
        self,
        tool_call: LLMToolCall,
        *,
        tool_resource_limits: ToolResourceLimits | None,
    ) -> ToolExecutionResult:
        timeout_ms = (
            None if tool_resource_limits is None else tool_resource_limits.tool_timeout_ms
        )
        if timeout_ms is None:
            return self.tool_runtime.execute(tool_call)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self.tool_runtime.execute, tool_call)
        try:
            return future.result(timeout=timeout_ms / 1000)
        except FutureTimeoutError:
            future.cancel()
            return ToolExecutionResult(
                name=tool_call.name,
                success=False,
                tool_call_id=tool_call.id,
                error=f"Tool execution timed out after {timeout_ms}ms",
                error_type="ToolTimeoutError",
                error_category="timeout",
                duration_ms=timeout_ms,
            )
        finally:
            executor.shutdown(wait=future.done(), cancel_futures=True)

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
        pending_approvals: Sequence[PendingToolApproval] = (),
        checkpoint: AgentToolLoopCheckpoint | None = None,
        current_round_index: int | None = None,
        current_step_index: int | None = None,
        tool_results: Sequence[LLMToolResult] = (),
    ) -> AgentToolLoopResult:
        return AgentToolLoopResult(
            completed=completed,
            reason=reason,
            final_response=final_response,
            steps=tuple(steps) if config.include_intermediate_steps else (),
            pending_tool_calls=tuple(pending_tool_calls),
            pending_approvals=tuple(pending_approvals),
            checkpoint=checkpoint,
            current_round_index=current_round_index,
            current_step_index=current_step_index,
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
    tool_resource_limits: ToolResourceLimits | None,
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
    if tool_resource_limits is not None:
        replacements["tool_resource_limits"] = tool_resource_limits
    return replace(config, **replacements) if replacements else config


def _validate_optional_nonnegative_int(value: Any, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be a nonnegative integer or None")
    if value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer or None")


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


def _resource_denied_tool_outcome(
    tool_call: LLMToolCall,
    *,
    limit_name: str,
) -> _ToolOutcome:
    error = f"Tool call denied by resource limits: {limit_name} exceeded"
    if limit_name == "max_calls_per_tool":
        error = (
            "Tool call denied by resource limits: "
            f"max_calls_per_tool exceeded for {tool_call.name}"
        )
    return _ToolOutcome(
        llm_result=LLMToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content="",
            success=False,
            error=error,
        ),
        success=False,
        error_type="ToolResourceLimitExceededError",
        error_category="resource_limits",
        denied=True,
        resource_denied=True,
        limit_name=limit_name,
    )


def _approval_denied_tool_outcome(tool_call: LLMToolCall) -> _ToolOutcome:
    return _ToolOutcome(
        llm_result=LLMToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            content="",
            success=False,
            error=f"Tool call denied by approval: {tool_call.name}",
        ),
        success=False,
        error_type="ToolApprovalDeniedError",
        error_category="approval_denied",
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
    resource_denied_tool_count: int = 0,
    timed_out_tool_count: int = 0,
    tool_resource_limits: ToolResourceLimits | None = None,
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
        "resource_denied_tool_count": resource_denied_tool_count,
        "timed_out_tool_count": timed_out_tool_count,
        "provider": provider,
        "model": model,
        **_tool_execution_group_resource_limit_metadata(tool_resource_limits),
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


def _tool_resource_limits_summary(
    limits: ToolResourceLimits | None,
) -> dict[str, int | bool | None]:
    return {
        "tool_resource_limits_enabled": limits is not None,
        "max_tool_calls_per_loop": (
            None if limits is None else limits.max_tool_calls_per_loop
        ),
        "max_tool_calls_per_round": (
            None if limits is None else limits.max_tool_calls_per_round
        ),
        "max_calls_per_tool_count": (
            0
            if limits is None or limits.max_calls_per_tool is None
            else len(limits.max_calls_per_tool)
        ),
        "tool_timeout_ms": None if limits is None else limits.tool_timeout_ms,
    }


def _tool_execution_group_resource_limit_metadata(
    limits: ToolResourceLimits | None,
) -> dict[str, int | bool | None]:
    return {
        "resource_limits_enabled": limits is not None,
        "max_tool_calls_per_loop": (
            None if limits is None else limits.max_tool_calls_per_loop
        ),
        "max_tool_calls_per_round": (
            None if limits is None else limits.max_tool_calls_per_round
        ),
        "tool_timeout_ms": None if limits is None else limits.tool_timeout_ms,
    }


def _limit_metadata(outcome: _ToolOutcome) -> dict[str, int | str]:
    metadata: dict[str, int | str] = {}
    if outcome.limit_name is not None:
        metadata["limit_name"] = outcome.limit_name
    if outcome.limit_value is not None:
        metadata["limit_value"] = outcome.limit_value
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
