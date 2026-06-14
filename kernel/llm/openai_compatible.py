"""Optional synchronous provider for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from kernel.llm.providers import LLMProviderError, classify_llm_error
from kernel.llm.types import LLMRequest, LLMResponse, LLMUsage


class OpenAICompatibleProvider:
    """Small, lazily initialized adapter for OpenAI-compatible APIs."""

    supports_streaming = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
        provider_name: str | None = None,
        timeout_seconds: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise ValueError("timeout_seconds must be a positive number")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.api_key = _configured_value(api_key, "AGENTOS_LLM_API_KEY")
        self.base_url = _configured_value(base_url, "AGENTOS_LLM_BASE_URL")
        self.default_model = _configured_value(default_model, "AGENTOS_LLM_MODEL")
        self.name = (
            _configured_value(provider_name, "AGENTOS_LLM_PROVIDER")
            or "openai-compatible"
        )
        self.timeout_seconds = timeout_seconds
        self._client = client

    def complete(self, request: LLMRequest) -> LLMResponse:
        if not self.api_key:
            raise LLMProviderError(
                "OpenAI-compatible provider requires an API key; set "
                "AGENTOS_LLM_API_KEY or pass api_key",
                category="configuration",
            )

        client = self._get_client()
        payload = _request_payload(request)
        try:
            completion = client.chat.completions.create(**payload)
        except Exception as exc:
            raise LLMProviderError(
                "OpenAI-compatible provider request failed",
                category=classify_llm_error(exc),
            ) from None

        return _response_from_completion(completion, provider=self.name, request=request)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except (ImportError, ModuleNotFoundError):
            raise LLMProviderError(
                "OpenAI-compatible provider requires the optional 'openai' package",
                category="configuration",
            ) from None

        client_options: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.timeout_seconds,
        }
        if self.base_url:
            client_options["base_url"] = self.base_url
        try:
            self._client = OpenAI(**client_options)
        except Exception:
            raise LLMProviderError(
                "OpenAI-compatible provider client initialization failed",
                category="configuration",
            ) from None
        return self._client


def _configured_value(explicit: str | None, environment_name: str) -> str | None:
    value = explicit if explicit is not None else os.getenv(environment_name)
    if value is None:
        return None
    return value.strip() or None


def _request_payload(request: LLMRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in request.messages
        ],
        "temperature": request.temperature,
    }
    max_tokens = _safe_max_tokens(request.metadata)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if request.timeout_seconds is not None:
        payload["timeout"] = request.timeout_seconds
    return payload


def _safe_max_tokens(metadata: Mapping[str, Any]) -> int | None:
    value = metadata.get("max_tokens")
    options = metadata.get("options")
    if value is None and isinstance(options, Mapping):
        value = options.get("max_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LLMProviderError("OpenAI-compatible max_tokens must be a positive integer")
    return value


def _response_from_completion(
    completion: Any,
    *,
    provider: str,
    request: LLMRequest,
) -> LLMResponse:
    try:
        choices = completion.choices
        if not choices:
            raise ValueError
        content = choices[0].message.content
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise TypeError

        model = getattr(completion, "model", None) or request.model
        if not isinstance(model, str) or not model.strip():
            raise TypeError

        usage = _map_usage(getattr(completion, "usage", None))
        metadata = _safe_response_metadata(completion)
        return LLMResponse(
            content=content,
            model=model,
            provider=provider,
            usage=usage,
            metadata=metadata,
        )
    except LLMProviderError:
        raise
    except Exception:
        raise LLMProviderError(
            "OpenAI-compatible provider returned a malformed response"
        ) from None


def _map_usage(usage: Any) -> LLMUsage | None:
    if usage is None:
        return None
    return LLMUsage(
        prompt_tokens=_optional_nonnegative_int(usage, "prompt_tokens"),
        completion_tokens=_optional_nonnegative_int(usage, "completion_tokens"),
        total_tokens=_optional_nonnegative_int(usage, "total_tokens"),
    )


def _optional_nonnegative_int(value: Any, attribute: str) -> int | None:
    token_count = getattr(value, attribute, None)
    if token_count is None:
        return None
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise LLMProviderError(
            "OpenAI-compatible provider returned malformed usage data"
        )
    return token_count


def _safe_response_metadata(completion: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    response_id = getattr(completion, "id", None)
    if isinstance(response_id, str) and response_id:
        metadata["response_id"] = response_id
    created = getattr(completion, "created", None)
    if isinstance(created, int) and not isinstance(created, bool):
        metadata["created"] = created
    system_fingerprint = getattr(completion, "system_fingerprint", None)
    if isinstance(system_fingerprint, str) and system_fingerprint:
        metadata["system_fingerprint"] = system_fingerprint
    return metadata
