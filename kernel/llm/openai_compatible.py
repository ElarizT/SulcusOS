"""Synchronous provider for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import os
import json
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from kernel.llm.providers import LLMProviderError, classify_llm_error
from kernel.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMToolCall, LLMUsage


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
            self._client = _HTTPCompatibleClient(
                api_key=self.api_key or "",
                base_url=self.base_url or "https://api.openai.com/v1",
                timeout_seconds=self.timeout_seconds,
            )
            return self._client

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


class _HTTPCompatibleClient:
    """Tiny OpenAI-compatible chat client used when the official SDK is unavailable."""

    def __init__(self, *, api_key: str, base_url: str, timeout_seconds: float) -> None:
        try:
            import httpx
        except (ImportError, ModuleNotFoundError):
            raise LLMProviderError(
                "OpenAI-compatible provider requires either 'openai' or 'httpx'",
                category="configuration",
            ) from None

        self.chat = _HTTPChat(
            httpx.Client(
                base_url=base_url.rstrip("/") + "/",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout_seconds,
            )
        )


class _HTTPChat:
    def __init__(self, client: Any) -> None:
        self.completions = _HTTPCompletions(client)


class _HTTPCompletions:
    def __init__(self, client: Any) -> None:
        self._client = client

    def create(self, **payload: Any) -> dict[str, Any]:
        request_payload = dict(payload)
        timeout = request_payload.pop("timeout", None)
        response = self._client.post(
            "chat/completions",
            json=request_payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()


def _configured_value(explicit: str | None, environment_name: str) -> str | None:
    value = explicit if explicit is not None else os.getenv(environment_name)
    if value is None:
        return None
    return value.strip() or None


def _request_payload(request: LLMRequest) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [_message_payload(message) for message in request.messages],
        "temperature": request.temperature,
    }
    max_tokens = _safe_max_tokens(request.metadata)
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if request.timeout_seconds is not None:
        payload["timeout"] = request.timeout_seconds
    if request.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": deepcopy(tool.parameters_schema),
                },
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        payload["tool_choice"] = _map_tool_choice(request.tool_choice)
    return payload


def _message_payload(message: LLMMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    tool_calls = message.metadata.get("tool_calls")
    if tool_calls is not None:
        payload["tool_calls"] = _map_message_tool_calls(tool_calls)
    if message.role == "tool":
        tool_call_id = message.metadata.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise LLMProviderError("OpenAI-compatible tool messages require tool_call_id")
        payload["tool_call_id"] = tool_call_id
    return payload


def _map_message_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if isinstance(tool_calls, (str, bytes)):
        raise LLMProviderError("OpenAI-compatible message tool_calls must be a sequence")
    try:
        iterable = list(tool_calls)
    except TypeError:
        raise LLMProviderError(
            "OpenAI-compatible message tool_calls must be a sequence"
        ) from None

    mapped: list[dict[str, Any]] = []
    for tool_call in iterable:
        if isinstance(tool_call, LLMToolCall):
            tool_call_id = tool_call.id
            name = tool_call.name
            arguments = tool_call.arguments
        elif isinstance(tool_call, Mapping):
            tool_call_id = tool_call.get("id")
            name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
        else:
            raise LLMProviderError(
                "OpenAI-compatible message tool_calls must contain LLMToolCall objects"
            )
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise LLMProviderError("OpenAI-compatible message tool_call id is invalid")
        if not isinstance(name, str) or not name.strip():
            raise LLMProviderError("OpenAI-compatible message tool_call name is invalid")
        if not isinstance(arguments, Mapping):
            raise LLMProviderError(
                "OpenAI-compatible message tool_call arguments must be a mapping"
            )
        mapped.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        deepcopy(dict(arguments)),
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                },
            }
        )
    return mapped


def _map_tool_choice(tool_choice: str | Mapping[str, Any]) -> str | dict[str, Any]:
    if isinstance(tool_choice, str):
        return tool_choice
    return deepcopy(dict(tool_choice))


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
        choices = _field(completion, "choices")
        if not choices:
            raise ValueError
        message = _field(choices[0], "message")
        content = _field(message, "content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise TypeError

        model = _field(completion, "model") or request.model
        if not isinstance(model, str) or not model.strip():
            raise TypeError

        usage = _map_usage(_field(completion, "usage"))
        metadata = _safe_response_metadata(completion)
        tool_calls = _map_tool_calls(
            _field(message, "tool_calls"),
            provider=provider,
            model=model,
        )
        return LLMResponse(
            content=content,
            model=model,
            provider=provider,
            usage=usage,
            metadata=metadata,
            tool_calls=tool_calls,
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
    token_count = _field(value, attribute)
    if token_count is None:
        return None
    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 0:
        raise LLMProviderError(
            "OpenAI-compatible provider returned malformed usage data"
        )
    return token_count


def _map_tool_calls(
    raw_tool_calls: Any,
    *,
    provider: str,
    model: str,
) -> tuple[LLMToolCall, ...]:
    if raw_tool_calls is None:
        return ()
    if isinstance(raw_tool_calls, (str, bytes)):
        raise LLMProviderError(
            "OpenAI-compatible provider returned malformed tool call data"
        )
    mapped: list[LLMToolCall] = []
    for index, raw_tool_call in enumerate(raw_tool_calls):
        tool_call_id = _field(raw_tool_call, "id")
        function = _field(raw_tool_call, "function")
        name = _field(function, "name") if function is not None else None
        arguments = _field(function, "arguments") if function is not None else None
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            raise LLMProviderError(
                "OpenAI-compatible provider returned malformed tool call data"
            )
        if not isinstance(name, str) or not name.strip():
            raise LLMProviderError(
                "OpenAI-compatible provider returned malformed tool call data"
            )
        mapped.append(
            LLMToolCall(
                id=tool_call_id,
                name=name,
                arguments=_parse_tool_arguments(arguments),
                provider=provider,
                model=model,
                metadata={"index": index},
            )
        )
    return tuple(mapped)


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, Mapping):
        return deepcopy(dict(arguments))
    if not isinstance(arguments, str):
        raise LLMProviderError(
            "OpenAI-compatible provider returned malformed tool call arguments"
        )
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        raise LLMProviderError(
            "OpenAI-compatible provider returned malformed tool call arguments"
        ) from None
    if not isinstance(parsed, Mapping):
        raise LLMProviderError(
            "OpenAI-compatible provider returned malformed tool call arguments"
        )
    return deepcopy(dict(parsed))


def _safe_response_metadata(completion: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    response_id = _field(completion, "id")
    if isinstance(response_id, str) and response_id:
        metadata["response_id"] = response_id
    created = _field(completion, "created")
    if isinstance(created, int) and not isinstance(created, bool):
        metadata["created"] = created
    system_fingerprint = _field(completion, "system_fingerprint")
    if isinstance(system_fingerprint, str) and system_fingerprint:
        metadata["system_fingerprint"] = system_fingerprint
    return metadata
