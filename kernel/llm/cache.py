"""Deterministic in-memory caching for provider-neutral LLM responses."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any

from kernel.llm.types import LLMRequest, LLMResponse


@dataclass(frozen=True)
class LLMCacheKey:
    """Opaque deterministic cache key that never retains raw request content."""

    digest: str

    @property
    def short_hash(self) -> str:
        return self.digest[:12]


@dataclass(frozen=True)
class LLMCacheEntry:
    """One immutable cache entry."""

    key: LLMCacheKey
    response: LLMResponse


@dataclass(frozen=True)
class LLMCacheStats:
    """Deterministic cache counters and current entry count."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    size: int = 0


class LLMResponseCache:
    """Small optional in-memory response cache."""

    def __init__(self, *, enabled: bool = False) -> None:
        if not isinstance(enabled, bool):
            raise ValueError("cache enabled must be a boolean")
        self.enabled = enabled
        self._entries: dict[LLMCacheKey, LLMCacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._lock = Lock()

    def get(self, key: LLMCacheKey) -> LLMResponse | None:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            self._hits += 1
            return copy_llm_response(entry.response)

    def set(self, key: LLMCacheKey, response: LLMResponse) -> None:
        if not self.enabled:
            return
        if not isinstance(response, LLMResponse):
            raise TypeError("cache response must be an LLMResponse")
        with self._lock:
            self._entries[key] = LLMCacheEntry(key, copy_llm_response(response))
            self._stores += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> LLMCacheStats:
        with self._lock:
            return LLMCacheStats(
                hits=self._hits,
                misses=self._misses,
                stores=self._stores,
                size=len(self._entries),
            )


def build_llm_cache_key(request: LLMRequest, provider_name: str) -> LLMCacheKey:
    """Build a stable opaque key from provider-safe request fields."""
    normalized_provider = str(provider_name).strip()
    if not normalized_provider:
        raise ValueError("cache provider name must not be empty")

    payload = {
        "provider": normalized_provider,
        "model": request.model,
        "messages": [
            {
                "role": message.role,
                "content": message.content,
                "protocol": _message_protocol_metadata(message),
            }
            for message in request.messages
        ],
        "temperature": _canonical_value(request.temperature),
        "options": {
            "max_tokens": _canonical_value(_request_option(request.metadata, "max_tokens"))
        },
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters_schema": _canonical_value(tool.parameters_schema),
            }
            for tool in request.tools
        ],
        "tool_choice": _canonical_value(request.tool_choice),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return LLMCacheKey(hashlib.sha256(serialized).hexdigest())


def copy_llm_response(
    response: LLMResponse,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> LLMResponse:
    """Return a response copy with independent metadata."""
    return LLMResponse(
        content=response.content,
        model=response.model,
        provider=response.provider,
        usage=response.usage,
        metadata=deepcopy(dict(response.metadata if metadata is None else metadata)),
        tool_calls=response.tool_calls,
    )


def _message_protocol_metadata(message: Any) -> dict[str, Any]:
    metadata = getattr(message, "metadata", {})
    if not isinstance(metadata, Mapping):
        return {}

    protocol: dict[str, Any] = {}
    tool_call_id = metadata.get("tool_call_id")
    if isinstance(tool_call_id, str) and tool_call_id:
        protocol["tool_call_id"] = tool_call_id

    tool_calls = metadata.get("tool_calls")
    if tool_calls is not None and not isinstance(tool_calls, (str, bytes)):
        try:
            protocol["tool_calls"] = [
                _tool_call_protocol_metadata(tool_call) for tool_call in tool_calls
            ]
        except TypeError:
            pass
    return protocol


def _tool_call_protocol_metadata(tool_call: Any) -> dict[str, Any]:
    if isinstance(tool_call, Mapping):
        tool_call_id = tool_call.get("id")
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
    else:
        tool_call_id = getattr(tool_call, "id", None)
        name = getattr(tool_call, "name", None)
        arguments = getattr(tool_call, "arguments", None)
    protocol: dict[str, Any] = {}
    if isinstance(tool_call_id, str) and tool_call_id:
        protocol["id"] = tool_call_id
    if isinstance(name, str) and name:
        protocol["name"] = name
    if isinstance(arguments, Mapping):
        protocol["arguments"] = _canonical_value(arguments)
    return protocol


def _request_option(metadata: Mapping[str, Any], name: str) -> Any:
    value = metadata.get(name)
    options = metadata.get("options")
    if value is None and isinstance(options, Mapping):
        value = options.get(name)
    return value


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"type": "float", "value": str(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(child) for child in value]
    return {
        "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
    }
