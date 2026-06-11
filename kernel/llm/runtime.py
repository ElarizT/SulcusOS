"""Provider-neutral LLM runtime facade with structured observability."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from kernel.events import RuntimeEvent
from kernel.llm.providers import LLMProvider, LLMProviderError, LLMRuntimeError
from kernel.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMUsage


EventSink = Callable[[RuntimeEvent], None] | Any
MessageInput = LLMMessage | Mapping[str, Any]


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
    ) -> None:
        if provider is not None and providers is not None:
            raise LLMRuntimeError("configure either provider or providers, not both")
        if provider is None and providers is None:
            raise LLMRuntimeError("LLMRuntime requires at least one provider")

        self.providers = _provider_registry(providers) if providers is not None else {}
        self.default_provider = _optional_provider_name(default_provider, "default_provider")
        self.fallback_providers = _provider_names(fallback_providers or ())
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

    def chat(
        self,
        messages: Sequence[MessageInput],
        model: str | None = None,
        temperature: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
        provider: str | None = None,
    ) -> LLMResponse:
        if self._uses_registry:
            return self._chat_with_routing(
                messages,
                model=model,
                temperature=temperature,
                metadata=metadata,
                provider=provider,
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
        )

    def _chat_single_provider(
        self,
        messages: Sequence[MessageInput],
        *,
        model: str | None,
        temperature: float,
        metadata: Mapping[str, Any] | None,
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
        )
        provider_name = _provider_name(active_provider)
        event_metadata = {"provider": provider_name, "model": request.model}
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.requested",
                f"LLM request sent to {provider_name}",
                event_metadata,
            )
        )

        try:
            response = active_provider.complete(request)
            if not isinstance(response, LLMResponse):
                raise TypeError("provider returned an invalid LLM response")
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

        completed_metadata = {
            "provider": response.provider,
            "model": response.model,
            **_usage_metadata(response.usage),
        }
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.completed",
                f"LLM request completed with {response.provider}",
                completed_metadata,
            )
        )
        return response

    def _chat_with_routing(
        self,
        messages: Sequence[MessageInput],
        *,
        model: str | None,
        temperature: float,
        metadata: Mapping[str, Any] | None,
        provider: str | None,
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
        )
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
        self._emit(
            RuntimeEvent.info(
                "LLMRuntime",
                "llm.requested",
                f"LLM request sent to {selected_name}",
                {"provider": selected_name, "model": request.model},
            )
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
            try:
                response = self.providers[provider_name].complete(request)
                if not isinstance(response, LLMResponse):
                    raise TypeError("provider returned an invalid LLM response")
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
            self._emit(
                RuntimeEvent.info(
                    "LLMRuntime",
                    "llm.completed",
                    f"LLM request completed with {response.provider}",
                    {
                        "provider": response.provider,
                        "model": response.model,
                        **_usage_metadata(response.usage),
                    },
                )
            )
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
