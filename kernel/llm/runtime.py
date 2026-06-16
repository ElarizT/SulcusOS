"""Provider-neutral LLM runtime facade with structured observability."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from time import sleep
from typing import Any

from kernel.events import RuntimeEvent
from kernel.llm.cache import (
    LLMCacheStats,
    LLMResponseCache,
    build_llm_cache_key,
    copy_llm_response,
)
from kernel.llm.cost import (
    LLMCostLedger,
    LLMCostTable,
    append_cost_record,
    calculate_cost_record,
    format_decimal_cost,
)
from kernel.llm.providers import (
    LLMBudgetExceededError,
    LLMProvider,
    LLMProviderError,
    LLMRuntimeError,
    LLMStreamingUnsupportedError,
    classify_llm_error,
)
from kernel.llm.types import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRetryPolicy,
    LLMStreamChunk,
    LLMToolDefinition,
    LLMTokenBudget,
    LLMUsage,
    LLMUsageLedger,
    apply_usage_to_ledger,
    check_token_budget,
)


EventSink = Callable[[RuntimeEvent], None] | Any
MessageInput = LLMMessage | Mapping[str, Any]
ToolInput = LLMToolDefinition | Mapping[str, Any]


class LLMRuntime:
    """Stable Agent OS interface for synchronous LLM provider calls."""

    def __init__(
        self,
        provider: LLMProvider | None = None,
        event_sink: EventSink | None = None,
        *,
        providers: Mapping[str, LLMProvider] | None = None,
        default_provider: str | None = None,
        fallback_providers: Sequence[str] | None = None,
        retry_policy: LLMRetryPolicy | None = None,
        timeout_seconds: float | None = None,
        sleeper: Callable[[float], None] = sleep,
        token_budget: LLMTokenBudget | None = None,
        cache: LLMResponseCache | None = None,
        cost_table: LLMCostTable | None = None,
    ) -> None:
        if provider is not None and providers is not None:
            raise LLMRuntimeError("configure either provider or providers, not both")
        if provider is None and providers is None:
            raise LLMRuntimeError("LLMRuntime requires at least one provider")

        self.providers = _provider_registry(providers) if providers is not None else {}
        self.default_provider = _optional_provider_name(default_provider, "default_provider")
        self.fallback_providers = _provider_names(fallback_providers or ())
        self.retry_policy = retry_policy or LLMRetryPolicy()
        if not isinstance(self.retry_policy, LLMRetryPolicy):
            raise LLMRuntimeError("retry_policy must be an LLMRetryPolicy")
        if timeout_seconds is not None:
            if isinstance(timeout_seconds, bool) or not isinstance(
                timeout_seconds, (int, float)
            ):
                raise LLMRuntimeError("timeout_seconds must be a positive number")
            if timeout_seconds <= 0:
                raise LLMRuntimeError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds
        self._sleeper = sleeper
        if token_budget is not None and not isinstance(token_budget, LLMTokenBudget):
            raise LLMRuntimeError("token_budget must be an LLMTokenBudget")
        self.token_budget = token_budget
        self._usage_ledger = LLMUsageLedger()
        if cache is not None and not isinstance(cache, LLMResponseCache):
            raise LLMRuntimeError("cache must be an LLMResponseCache")
        self.cache = cache
        if cost_table is not None and not isinstance(cost_table, LLMCostTable):
            raise LLMRuntimeError("cost_table must be an LLMCostTable")
        self.cost_table = cost_table
        self._cost_ledger = LLMCostLedger(
            currency=cost_table.currency if cost_table is not None else "USD"
        )
        self._uses_registry = providers is not None

        if self._uses_registry:
            _validate_configured_provider(self.default_provider, self.providers, "default")
            for fallback_provider in self.fallback_providers:
                _validate_configured_provider(fallback_provider, self.providers, "fallback")
            self.provider = (
                self.providers[self.default_provider]
                if self.default_provider is not None
                else None
            )
        else:
            if default_provider is not None or fallback_providers:
                raise LLMRuntimeError(
                    "default_provider and fallback_providers require a providers registry"
                )
            self.provider = provider
        self.event_sink = event_sink

    def usage_snapshot(self) -> LLMUsageLedger:
        """Return an immutable snapshot of cumulative provider-reported usage."""
        return self._usage_ledger

    def cache_snapshot(self) -> LLMCacheStats:
        """Return immutable cache statistics, or zeros when no cache exists."""
        return self.cache.stats() if self.cache is not None else LLMCacheStats()

    def cost_snapshot(self) -> LLMCostLedger:
        """Return an immutable snapshot of configured provider-reported costs."""
        return self._cost_ledger

    def clear_cache(self) -> None:
        """Clear cached responses and emit a safe event when a cache exists."""
        if self.cache is None:
            return
        self.cache.clear()
        self._emit_cache_event(
            "llm.cache_cleared",
            "LLM response cache cleared",
            size=self.cache.stats().size,
        )

    def chat(
        self,
        messages: Sequence[MessageInput],
        model: str | None = None,
        temperature: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        provider: str | None = None,
        timeout_seconds: float | None = None,
        tools: Sequence[ToolInput] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        if self._uses_registry:
            return self._chat_with_routing(
                messages,
                model=model,
                temperature=temperature,
                metadata=metadata,
                provider=provider,
                timeout_seconds=timeout_seconds,
                tools=tools,
                tool_choice=tool_choice,
            )
        if provider is not None:
            raise LLMRuntimeError(
                "explicit provider selection requires a providers registry"
            )
        return self._chat_single_provider(
            messages,
            model=model,
            temperature=temperature,
            metadata=metadata,
            timeout_seconds=timeout_seconds,
            tools=tools,
            tool_choice=tool_choice,
        )

    def stream_chat(
        self,
        messages: Sequence[MessageInput],
        model: str | None = None,
        temperature: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        provider: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Stream a chat response, retrying or falling back only before output."""
        if self._uses_registry:
            selected_name = (
                _optional_provider_name(provider, "provider") or self.default_provider
            )
            if selected_name is None:
                raise LLMRuntimeError(
                    "provider must be specified when no default_provider is configured"
                )
            _validate_configured_provider(selected_name, self.providers, "requested")
            attempts = _ordered_unique((selected_name, *self.fallback_providers))
            selected_provider = self.providers[selected_name]
        else:
            if provider is not None:
                raise LLMRuntimeError(
                    "explicit provider selection requires a providers registry"
                )
            selected_provider = self.provider
            if selected_provider is None:
                raise LLMRuntimeError("LLMRuntime has no configured provider")
            selected_name = _provider_name(selected_provider)
            attempts = (selected_name,)

        request = LLMRequest(
            messages=tuple(_coerce_message(message) for message in messages),
            model=model or _default_model(selected_provider),
            temperature=temperature,
            metadata=dict(metadata or {}),
            timeout_seconds=_resolve_timeout(timeout_seconds, self.timeout_seconds),
        )
        self._check_request_budget(request)
        if self._uses_registry:
            self._emit_routing_event(
                "llm.provider_selected",
                f"LLM provider selected: {selected_name}",
                {
                    "provider": selected_name,
                    "model": request.model,
                    "attempt": 1,
                    "success": True,
                    "streaming": True,
                },
            )
        self._emit_stream_event(
            "llm.stream_requested",
            f"LLM stream requested from {selected_name}",
            {"provider": selected_name, "model": request.model},
        )
        if self.cache is not None and self.cache.enabled:
            self._emit_cache_event(
                "llm.cache_bypassed",
                "LLM response cache bypassed for streaming",
                provider=selected_name,
                model=request.model,
                reason="streaming",
            )
        return self._stream_with_routing(request, selected_name, attempts)

    def _stream_with_routing(
        self,
        request: LLMRequest,
        selected_name: str,
        attempts: tuple[str, ...],
    ) -> Iterator[LLMStreamChunk]:
        for route_attempt, provider_name in enumerate(attempts, start=1):
            if route_attempt > 1:
                self._emit_routing_event(
                    "llm.fallback_started",
                    f"LLM fallback started: {provider_name}",
                    {
                        "provider": selected_name,
                        "fallback_provider": provider_name,
                        "model": request.model,
                        "attempt": route_attempt,
                        "success": False,
                        "streaming": True,
                    },
                )
            active_provider = (
                self.providers[provider_name] if self._uses_registry else self.provider
            )
            if active_provider is None:
                raise LLMRuntimeError("LLMRuntime has no configured provider")

            try:
                chunks, usage = yield from self._stream_provider_with_retries(
                    active_provider,
                    provider_name=provider_name,
                    request=request,
                )
            except Exception as exc:
                if isinstance(exc, LLMBudgetExceededError):
                    raise
                category = classify_llm_error(exc)
                if self._uses_registry:
                    self._emit_routing_event(
                        "llm.provider_failed",
                        f"LLM streaming provider failed: {provider_name}",
                        {
                            "provider": provider_name,
                            "model": request.model,
                            "attempt": route_attempt,
                            "error_category": category,
                            "error_type": exc.__class__.__name__,
                            "success": False,
                            "streaming": True,
                        },
                        error=True,
                    )
                if category != "stream_interrupted" and route_attempt < len(attempts) and (
                    isinstance(exc, LLMProviderError)
                    or (category != "configuration" and category in self.retry_policy.retry_on)
                ):
                    continue
                if (
                    category != "stream_interrupted"
                    and self._uses_registry
                    and len(attempts) > 1
                    and route_attempt == len(attempts)
                ):
                    self._emit_routing_event(
                        "llm.fallback_exhausted",
                        "LLM streaming provider fallbacks exhausted",
                        {
                            "provider": selected_name,
                            "model": request.model,
                            "attempt": route_attempt,
                            "success": False,
                            "streaming": True,
                        },
                        error=True,
                    )
                self._emit_stream_failed(provider_name, request.model, exc)
                if isinstance(exc, LLMStreamingUnsupportedError) or (
                    isinstance(exc, LLMProviderError)
                    and category == "stream_interrupted"
                ):
                    raise
                raise LLMProviderError(
                    f"LLM stream failed for provider '{provider_name}'",
                    category=category,
                ) from None

            if route_attempt > 1:
                self._emit_routing_event(
                    "llm.fallback_succeeded",
                    f"LLM fallback succeeded: {provider_name}",
                    {
                        "provider": selected_name,
                        "fallback_provider": provider_name,
                        "model": request.model,
                        "attempt": route_attempt,
                        "success": True,
                        "streaming": True,
                        **_usage_metadata(usage),
                    },
                )
            self._emit_stream_event(
                "llm.stream_completed",
                f"LLM stream completed with {provider_name}",
                {
                    "provider": provider_name,
                    "model": request.model,
                    "chunk_count": chunks,
                    **_usage_metadata(usage),
                },
            )
            self._record_cost(_provider_name(active_provider), request.model, usage)
            self._apply_stream_usage(usage)
            return

        raise AssertionError("stream routing loop completed without a result")

    def _stream_provider_with_retries(
        self,
        provider: LLMProvider,
        *,
        provider_name: str,
        request: LLMRequest,
    ) -> Iterator[LLMStreamChunk]:
        stream = getattr(provider, "stream", None)
        if not callable(stream) or getattr(provider, "supports_streaming", True) is False:
            raise LLMStreamingUnsupportedError(
                f"LLM provider '{provider_name}' does not support streaming"
            )

        policy = self.retry_policy
        last_category: str | None = None
        last_error_type: str | None = None
        for attempt in range(1, policy.max_attempts + 1):
            yielded = 0
            final_usage: LLMUsage | None = None
            started = False
            try:
                stream_iterator = iter(stream(request))
                for raw_chunk in stream_iterator:
                    if not isinstance(raw_chunk, LLMStreamChunk):
                        raise TypeError("provider returned an invalid LLM stream chunk")
                    chunk = replace(
                        raw_chunk,
                        provider=raw_chunk.provider or provider_name,
                        model=raw_chunk.model or request.model,
                    )
                    if not started:
                        self._emit_stream_event(
                            "llm.stream_started",
                            f"LLM stream started with {provider_name}",
                            {"provider": provider_name, "model": request.model},
                        )
                        started = True
                    self._emit_stream_event(
                        "llm.stream_chunk",
                        f"LLM stream chunk received from {provider_name}",
                        {
                            "provider": provider_name,
                            "model": request.model,
                            "chunk_index": chunk.index,
                            "delta_chars": len(chunk.delta),
                            "done": chunk.done,
                            **_usage_metadata(chunk.usage if chunk.done else None),
                        },
                    )
                    yielded += 1
                    if chunk.done and chunk.usage is not None:
                        final_usage = chunk.usage
                    yield chunk
                if not started:
                    self._emit_stream_event(
                        "llm.stream_started",
                        f"LLM stream started with {provider_name}",
                        {"provider": provider_name, "model": request.model},
                    )
            except Exception as exc:
                category = classify_llm_error(exc)
                if yielded:
                    raise LLMProviderError(
                        f"LLM stream failed after output started for provider '{provider_name}'",
                        category="stream_interrupted",
                    ) from None
                retryable = category != "configuration" and category in policy.retry_on
                if not retryable or attempt >= policy.max_attempts:
                    if retryable and policy.max_attempts > 1:
                        self._emit_retry_event(
                            "llm.retry_exhausted",
                            f"LLM streaming retries exhausted for {provider_name}",
                            provider_name,
                            request.model,
                            attempt,
                            category,
                            exc.__class__.__name__,
                            error=True,
                        )
                    raise
                backoff_seconds = _retry_backoff(policy, attempt)
                self._emit_retry_event(
                    "llm.retry_scheduled",
                    f"LLM streaming retry scheduled for {provider_name}",
                    provider_name,
                    request.model,
                    attempt + 1,
                    category,
                    exc.__class__.__name__,
                    backoff_seconds=backoff_seconds,
                )
                if backoff_seconds > 0:
                    self._sleeper(backoff_seconds)
                self._emit_retry_event(
                    "llm.retry_started",
                    f"LLM streaming retry started for {provider_name}",
                    provider_name,
                    request.model,
                    attempt + 1,
                    category,
                    exc.__class__.__name__,
                    backoff_seconds=backoff_seconds,
                )
                last_category = category
                last_error_type = exc.__class__.__name__
                continue

            if attempt > 1:
                self._emit_retry_event(
                    "llm.retry_succeeded",
                    f"LLM streaming retry succeeded for {provider_name}",
                    provider_name,
                    request.model,
                    attempt,
                    last_category,
                    last_error_type,
                )
            return yielded, final_usage

        raise AssertionError("stream retry loop completed without a result or error")

    def _apply_stream_usage(self, usage: LLMUsage | None) -> None:
        if usage is None:
            return
        self._usage_ledger = apply_usage_to_ledger(self._usage_ledger, usage)
        budget = self.token_budget
        if budget is None:
            return
        self._emit_budget_event(
            "llm.budget_updated",
            "LLM token budget usage updated",
            (),
        )
        self._handle_budget_exceeded(check_token_budget(budget, self._usage_ledger))

    def _emit_stream_failed(
        self,
        provider: str,
        model: str,
        error: Exception,
    ) -> None:
        self._emit_stream_event(
            "llm.stream_failed",
            f"LLM stream failed for {provider}",
            {
                "provider": provider,
                "model": model,
                "error_category": classify_llm_error(error),
                "error_type": error.__class__.__name__,
            },
            error=True,
        )

    def _emit_stream_event(
        self,
        event_type: str,
        message: str,
        metadata: dict[str, Any],
        *,
        error: bool = False,
    ) -> None:
        factory = RuntimeEvent.error if error else RuntimeEvent.info
        self._emit(factory("LLMRuntime", event_type, message, metadata))

    def _chat_single_provider(
        self,
        messages: Sequence[MessageInput],
        *,
        model: str | None,
        temperature: float,
        metadata: Mapping[str, Any] | None,
        timeout_seconds: float | None,
        tools: Sequence[ToolInput] | None,
        tool_choice: str | Mapping[str, Any] | None,
    ) -> LLMResponse:
        active_provider = self.provider
        if active_provider is None:
            raise LLMRuntimeError("LLMRuntime has no configured provider")
        resolved_model = model or _default_model(active_provider)
        request = LLMRequest(
            messages=tuple(_coerce_message(message) for message in messages),
            model=resolved_model,
            temperature=temperature,
            metadata=dict(metadata or {}),
            timeout_seconds=_resolve_timeout(timeout_seconds, self.timeout_seconds),
            tools=tuple(_coerce_tool_definition(tool) for tool in tools or ()),
            tool_choice=tool_choice,
        )
        provider_name = _provider_name(active_provider)
        self._check_request_budget(request)
        cached_response = self._get_cached_response(request, provider_name)
        if cached_response is not None:
            self._emit_tool_call_events(cached_response)
            self._emit_completed(cached_response)
            return cached_response
        event_metadata = {"provider": provider_name, "model": request.model}
        self._emit_tools_available(provider_name, request)
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.requested",
                f"LLM request sent to {provider_name}",
                event_metadata,
            )
        )

        try:
            response = self._complete_with_retries(
                active_provider,
                provider_name=provider_name,
                request=request,
            )
        except Exception as exc:
            self._emit(
                RuntimeEvent.error(
                    "LLMRuntime",
                    "llm.failed",
                    f"LLM request failed for {provider_name}",
                    {
                        **event_metadata,
                        "error": True,
                        "error_type": exc.__class__.__name__,
                    },
                )
            )
            if isinstance(exc, LLMProviderError):
                raise
            raise LLMProviderError(f"LLM provider '{provider_name}' failed") from exc

        self._record_cost(response.provider, response.model, response.usage)
        self._apply_response_usage(response)
        self._store_cached_response(request, provider_name, response)
        self._emit_tool_call_events(response)
        self._emit_completed(response)
        return response

    def _chat_with_routing(
        self,
        messages: Sequence[MessageInput],
        *,
        model: str | None,
        temperature: float,
        metadata: Mapping[str, Any] | None,
        provider: str | None,
        timeout_seconds: float | None,
        tools: Sequence[ToolInput] | None,
        tool_choice: str | Mapping[str, Any] | None,
    ) -> LLMResponse:
        selected_name = _optional_provider_name(provider, "provider") or self.default_provider
        if selected_name is None:
            raise LLMRuntimeError(
                "provider must be specified when no default_provider is configured"
            )
        _validate_configured_provider(selected_name, self.providers, "requested")

        attempts = _ordered_unique((selected_name, *self.fallback_providers))
        selected_provider = self.providers[selected_name]
        request = LLMRequest(
            messages=tuple(_coerce_message(message) for message in messages),
            model=model or _default_model(selected_provider),
            temperature=temperature,
            metadata=dict(metadata or {}),
            timeout_seconds=_resolve_timeout(timeout_seconds, self.timeout_seconds),
            tools=tuple(_coerce_tool_definition(tool) for tool in tools or ()),
            tool_choice=tool_choice,
        )
        self._check_request_budget(request)
        self._emit_routing_event(
            "llm.provider_selected",
            f"LLM provider selected: {selected_name}",
            {
                "provider": selected_name,
                "model": request.model,
                "attempt": 1,
                "success": True,
            },
        )
        for attempt, provider_name in enumerate(attempts, start=1):
            if attempt > 1:
                self._emit_routing_event(
                    "llm.fallback_started",
                    f"LLM fallback started: {provider_name}",
                    {
                        "provider": selected_name,
                        "fallback_provider": provider_name,
                        "model": request.model,
                        "attempt": attempt,
                        "success": False,
                    },
                )
            cached_response = self._get_cached_response(request, provider_name)
            if cached_response is not None:
                if attempt > 1:
                    self._emit_routing_event(
                        "llm.fallback_succeeded",
                        f"LLM fallback succeeded: {provider_name}",
                        {
                            "provider": selected_name,
                            "fallback_provider": provider_name,
                            "model": cached_response.model,
                            "attempt": attempt,
                            "success": True,
                            "cached": True,
                        },
                    )
                self._emit_tool_call_events(cached_response)
                self._emit_completed(cached_response)
                return cached_response
            if attempt == 1:
                self._emit_tools_available(selected_name, request)
                self._emit(
                    RuntimeEvent.info(
                        "LLMRuntime",
                        "llm.requested",
                        f"LLM request sent to {selected_name}",
                        {"provider": selected_name, "model": request.model},
                    )
                )
            try:
                response = self._complete_with_retries(
                    self.providers[provider_name],
                    provider_name=provider_name,
                    request=request,
                )
            except LLMProviderError as exc:
                self._emit_routing_event(
                    "llm.provider_failed",
                    f"LLM provider failed: {provider_name}",
                    {
                        "provider": provider_name,
                        "model": request.model,
                        "attempt": attempt,
                        "error_type": exc.__class__.__name__,
                        "success": False,
                    },
                    error=True,
                )
                continue
            except Exception as exc:
                self._emit_routing_event(
                    "llm.provider_failed",
                    f"LLM provider failed: {provider_name}",
                    {
                        "provider": provider_name,
                        "model": request.model,
                        "attempt": attempt,
                        "error_type": exc.__class__.__name__,
                        "success": False,
                    },
                    error=True,
                )
                category = classify_llm_error(exc)
                if category != "configuration" and category in self.retry_policy.retry_on:
                    continue
                self._emit(
                    RuntimeEvent.error(
                        "LLMRuntime",
                        "llm.failed",
                        f"LLM request failed for {provider_name}",
                        {
                            "provider": provider_name,
                            "model": request.model,
                            "error": True,
                            "error_type": exc.__class__.__name__,
                        },
                    )
                )
                raise LLMProviderError(
                    f"LLM provider '{provider_name}' failed"
                ) from None

            self._record_cost(response.provider, response.model, response.usage)
            self._apply_response_usage(response)
            self._store_cached_response(request, provider_name, response)
            if attempt > 1:
                self._emit_routing_event(
                    "llm.fallback_succeeded",
                    f"LLM fallback succeeded: {provider_name}",
                    {
                        "provider": selected_name,
                        "fallback_provider": provider_name,
                        "model": response.model,
                        "attempt": attempt,
                        "success": True,
                        **_usage_metadata(response.usage),
                    },
                )
            self._emit_tool_call_events(response)
            self._emit_completed(response)
            return response

        self._emit_routing_event(
            "llm.fallback_exhausted",
            "LLM provider fallbacks exhausted",
            {
                "provider": selected_name,
                "model": request.model,
                "attempt": len(attempts),
                "success": False,
            },
            error=True,
        )
        self._emit(
            RuntimeEvent.error(
                "LLMRuntime",
                "llm.failed",
                f"LLM request failed after {len(attempts)} provider attempts",
                {
                    "provider": selected_name,
                    "model": request.model,
                    "error": True,
                    "error_type": "LLMProviderError",
                },
            )
        )
        provider_summary = ", ".join(attempts)
        raise LLMProviderError(
            f"All LLM providers failed after {len(attempts)} attempts: {provider_summary}"
        ) from None

    def _check_request_budget(self, request: LLMRequest) -> None:
        budget = self.token_budget
        if budget is None:
            return

        max_tokens = _request_max_tokens(request.metadata)
        exceeded = list(check_token_budget(budget, self._usage_ledger))
        if max_tokens is not None:
            if (
                budget.max_completion_tokens is not None
                and self._usage_ledger.completion_tokens + max_tokens
                > budget.max_completion_tokens
                and "completion" not in exceeded
            ):
                exceeded.append("completion")
            if (
                budget.max_total_tokens is not None
                and self._usage_ledger.total_tokens + max_tokens
                > budget.max_total_tokens
                and "total" not in exceeded
            ):
                exceeded.append("total")

        self._emit_budget_event(
            "llm.budget_checked",
            "LLM token budget checked",
            tuple(exceeded),
        )
        self._handle_budget_exceeded(tuple(exceeded))

    def _apply_response_usage(self, response: LLMResponse) -> None:
        self._usage_ledger = apply_usage_to_ledger(self._usage_ledger, response.usage)
        budget = self.token_budget
        if budget is None:
            return

        self._emit_budget_event(
            "llm.budget_updated",
            "LLM token budget usage updated",
            (),
        )
        exceeded = check_token_budget(budget, self._usage_ledger)
        self._handle_budget_exceeded(exceeded)

    def _record_cost(
        self,
        provider: str,
        model: str,
        usage: LLMUsage | None,
    ) -> None:
        table = self.cost_table
        if table is None:
            return
        if usage is None:
            self._emit_cost_skipped(provider, model, usage, "missing_usage")
            return
        rate = table.match(provider, model)
        if rate is None:
            self._emit_cost_skipped(provider, model, usage, "rate_not_found")
            return
        record = calculate_cost_record(rate, provider, model, usage)
        if record is None:
            self._emit_cost_skipped(provider, model, usage, "incomplete_usage")
            return
        self._cost_ledger = append_cost_record(self._cost_ledger, record)
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.cost_recorded",
                f"LLM cost recorded for {provider}",
                {
                    "provider": provider,
                    "model": model,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "total_tokens": record.total_tokens,
                    "total_cost": format_decimal_cost(record.total_cost),
                    "currency": record.currency,
                },
            )
        )

    def _emit_cost_skipped(
        self,
        provider: str,
        model: str,
        usage: LLMUsage | None,
        reason: str,
    ) -> None:
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.cost_skipped",
                f"LLM cost skipped for {provider}",
                {
                    "provider": provider,
                    "model": model,
                    **_usage_metadata(usage),
                    "total_cost": "0",
                    "currency": self._cost_ledger.currency,
                    "reason": reason,
                },
            )
        )

    def _get_cached_response(
        self,
        request: LLMRequest,
        provider_name: str,
    ) -> LLMResponse | None:
        if not self._cache_enabled_for_request(request):
            return None
        cache = self.cache
        if cache is None:
            return None
        key = build_llm_cache_key(request, provider_name)
        self._emit_cache_event(
            "llm.cache_checked",
            f"LLM response cache checked for {provider_name}",
            provider=provider_name,
            model=request.model,
            cache_key=key.short_hash,
        )
        response = cache.get(key)
        if response is None:
            self._emit_cache_event(
                "llm.cache_miss",
                f"LLM response cache miss for {provider_name}",
                provider=provider_name,
                model=request.model,
                cache_key=key.short_hash,
                hit=False,
                size=cache.stats().size,
            )
            return None
        metadata = dict(response.metadata)
        metadata.update({"cached": True, "cache_key": key.short_hash})
        cached_response = copy_llm_response(response, metadata=metadata)
        self._emit_cache_event(
            "llm.cache_hit",
            f"LLM response cache hit for {provider_name}",
            provider=provider_name,
            model=request.model,
            cache_key=key.short_hash,
            hit=True,
            size=cache.stats().size,
        )
        return cached_response

    def _store_cached_response(
        self,
        request: LLMRequest,
        provider_name: str,
        response: LLMResponse,
    ) -> None:
        if not self._cache_enabled_for_request(request):
            return
        cache = self.cache
        if cache is None:
            return
        key = build_llm_cache_key(request, provider_name)
        cache.set(key, response)
        self._emit_cache_event(
            "llm.cache_stored",
            f"LLM response cached for {provider_name}",
            provider=provider_name,
            model=request.model,
            cache_key=key.short_hash,
            size=cache.stats().size,
        )

    def _cache_enabled_for_request(self, request: LLMRequest) -> bool:
        cache = self.cache
        if cache is None or not cache.enabled:
            return False
        return _request_cache_enabled(request.metadata)

    def _emit_completed(self, response: LLMResponse) -> None:
        metadata: dict[str, Any] = {
            "provider": response.provider,
            "model": response.model,
        }
        if response.metadata.get("cached") is True:
            metadata["cached"] = True
        else:
            metadata.update(_usage_metadata(response.usage))
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.completed",
                f"LLM request completed with {response.provider}",
                metadata,
            )
        )

    def _emit_tools_available(self, provider: str, request: LLMRequest) -> None:
        if not request.tools:
            return
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.tools_available",
                f"LLM tools available for {provider}",
                {
                    "provider": provider,
                    "model": request.model,
                    "tool_count": len(request.tools),
                },
            )
        )

    def _emit_tool_call_events(self, response: LLMResponse) -> None:
        if not response.tool_calls:
            return
        tool_call_count = len(response.tool_calls)
        for tool_call in response.tool_calls:
            self._emit(
                RuntimeEvent.info(
                    "LLMRuntime",
                    "llm.tool_call_requested",
                    f"LLM requested tool call: {tool_call.name}",
                    {
                        "provider": response.provider,
                        "model": response.model,
                        "tool_name": tool_call.name,
                        "tool_call_count": tool_call_count,
                    },
                )
            )

    def _handle_budget_exceeded(self, exceeded: tuple[str, ...]) -> None:
        if not exceeded:
            return
        self._emit_budget_event(
            "llm.budget_exceeded",
            "LLM token budget exceeded",
            exceeded,
            error=True,
        )
        budget = self.token_budget
        if budget is not None and budget.strict:
            categories = ", ".join(exceeded)
            raise LLMBudgetExceededError(
                f"LLM token budget '{_budget_name(budget)}' exceeded: {categories}"
            ) from None

    def _emit_budget_event(
        self,
        event_type: str,
        message: str,
        exceeded: tuple[str, ...],
        *,
        error: bool = False,
    ) -> None:
        budget = self.token_budget
        if budget is None:
            return
        metadata: dict[str, Any] = {
            "budget": _budget_name(budget),
            "prompt_tokens_used": self._usage_ledger.prompt_tokens,
            "completion_tokens_used": self._usage_ledger.completion_tokens,
            "total_tokens_used": self._usage_ledger.total_tokens,
            "exceeded": ", ".join(exceeded) if exceeded else "none",
        }
        for field_name in (
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_total_tokens",
        ):
            value = getattr(budget, field_name)
            if value is not None:
                metadata[field_name] = value
        factory = RuntimeEvent.error if error else RuntimeEvent.info
        self._emit(factory("LLMRuntime", event_type, message, metadata))

    def _complete_with_retries(
        self,
        provider: LLMProvider,
        *,
        provider_name: str,
        request: LLMRequest,
    ) -> LLMResponse:
        policy = self.retry_policy
        last_category: str | None = None
        last_error_type: str | None = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                response = provider.complete(request)
                if not isinstance(response, LLMResponse):
                    raise TypeError("provider returned an invalid LLM response")
            except Exception as exc:
                category = classify_llm_error(exc)
                retryable = category != "configuration" and category in policy.retry_on
                if not retryable or attempt >= policy.max_attempts:
                    if retryable and policy.max_attempts > 1:
                        self._emit_retry_event(
                            "llm.retry_exhausted",
                            f"LLM retries exhausted for {provider_name}",
                            provider_name,
                            request.model,
                            attempt,
                            category,
                            exc.__class__.__name__,
                            error=True,
                        )
                    raise

                backoff_seconds = _retry_backoff(policy, attempt)
                self._emit_retry_event(
                    "llm.retry_scheduled",
                    f"LLM retry scheduled for {provider_name}",
                    provider_name,
                    request.model,
                    attempt + 1,
                    category,
                    exc.__class__.__name__,
                    backoff_seconds=backoff_seconds,
                )
                if backoff_seconds > 0:
                    self._sleeper(backoff_seconds)
                self._emit_retry_event(
                    "llm.retry_started",
                    f"LLM retry started for {provider_name}",
                    provider_name,
                    request.model,
                    attempt + 1,
                    category,
                    exc.__class__.__name__,
                    backoff_seconds=backoff_seconds,
                )
                last_category = category
                last_error_type = exc.__class__.__name__
                continue

            if attempt > 1:
                self._emit_retry_event(
                    "llm.retry_succeeded",
                    f"LLM retry succeeded for {provider_name}",
                    provider_name,
                    response.model,
                    attempt,
                    last_category,
                    last_error_type,
                )
            return response

        raise AssertionError("retry loop completed without a response or error")

    def _emit_retry_event(
        self,
        event_type: str,
        message: str,
        provider: str,
        model: str,
        attempt: int,
        category: str | None,
        error_type: str | None,
        *,
        backoff_seconds: float | None = None,
        error: bool = False,
    ) -> None:
        metadata: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "attempt": attempt,
            "max_attempts": self.retry_policy.max_attempts,
        }
        if category is not None:
            metadata["error_category"] = category
        if error_type is not None:
            metadata["error_type"] = error_type
        if backoff_seconds is not None:
            metadata["backoff_seconds"] = backoff_seconds
        factory = RuntimeEvent.error if error else RuntimeEvent.info
        self._emit(factory("LLMRuntime", event_type, message, metadata))

    def _emit_routing_event(
        self,
        event_type: str,
        message: str,
        metadata: dict[str, Any],
        *,
        error: bool = False,
    ) -> None:
        factory = RuntimeEvent.error if error else RuntimeEvent.info
        self._emit(factory("LLMRuntime", event_type, message, metadata))

    def _emit_cache_event(
        self,
        event_type: str,
        message: str,
        **metadata: Any,
    ) -> None:
        self._emit(RuntimeEvent.info("LLMRuntime", event_type, message, metadata))

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
            # Observability must not change LLM call behavior.
            return


def _coerce_message(message: MessageInput) -> LLMMessage:
    if isinstance(message, LLMMessage):
        return message
    if isinstance(message, Mapping):
        return LLMMessage(
            role=str(message.get("role", "")),
            content=str(message.get("content", "")),
            metadata=dict(message.get("metadata", {})),
        )
    raise TypeError("messages must be LLMMessage objects or mappings")


def _coerce_tool_definition(tool: ToolInput) -> LLMToolDefinition:
    if isinstance(tool, LLMToolDefinition):
        return tool
    if isinstance(tool, Mapping):
        parameters_schema = tool.get("parameters_schema", tool.get("parameters", {}))
        if not isinstance(parameters_schema, Mapping):
            raise TypeError("tool parameters_schema must be a mapping")
        return LLMToolDefinition(
            name=str(tool.get("name", "")),
            description=str(tool.get("description", "")),
            parameters_schema=dict(parameters_schema),
        )
    raise TypeError("tools must be LLMToolDefinition objects or mappings")


def _provider_name(provider: LLMProvider) -> str:
    name = str(getattr(provider, "name", "")).strip()
    return name or provider.__class__.__name__


def _provider_registry(
    providers: Mapping[str, LLMProvider],
) -> dict[str, LLMProvider]:
    registry: dict[str, LLMProvider] = {}
    for raw_name, provider in providers.items():
        name = str(raw_name).strip()
        if not name:
            raise LLMRuntimeError("provider registry names must not be empty")
        if provider is None:
            raise LLMRuntimeError(f"provider registry entry '{name}' must not be empty")
        registry[name] = provider
    if not registry:
        raise LLMRuntimeError("providers registry must not be empty")
    return registry


def _provider_names(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(_optional_provider_name(name, "fallback provider") or "" for name in names)


def _optional_provider_name(name: str | None, label: str) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip()
    if not normalized:
        raise LLMRuntimeError(f"{label} must not be empty")
    return normalized


def _validate_configured_provider(
    name: str | None,
    providers: Mapping[str, LLMProvider],
    label: str,
) -> None:
    if name is not None and name not in providers:
        raise LLMRuntimeError(f"unknown {label} LLM provider '{name}'")


def _ordered_unique(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(names))


def _default_model(provider: LLMProvider) -> str:
    return str(getattr(provider, "default_model", "") or "")


def _usage_metadata(usage: LLMUsage | None) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        name: value
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if (value := getattr(usage, name)) is not None
    }


def _resolve_timeout(request_timeout: float | None, runtime_timeout: float | None) -> float | None:
    return request_timeout if request_timeout is not None else runtime_timeout


def _retry_backoff(policy: LLMRetryPolicy, failed_attempt: int) -> float:
    backoff = policy.backoff_seconds * (2 ** (failed_attempt - 1))
    if policy.max_backoff_seconds is not None:
        backoff = min(backoff, policy.max_backoff_seconds)
    return backoff


def _request_max_tokens(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("max_tokens")
    options = metadata.get("options")
    if value is None and isinstance(options, Mapping):
        value = options.get("max_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _request_cache_enabled(metadata: Mapping[str, Any]) -> bool:
    value = metadata.get("cache")
    options = metadata.get("options")
    if value is None and isinstance(options, Mapping):
        value = options.get("cache")
    return value is not False


def _budget_name(budget: LLMTokenBudget) -> str:
    return budget.name or "default"
