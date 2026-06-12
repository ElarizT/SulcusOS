"""Stable provider-neutral LLM Runtime Layer for Agent OS."""

from importlib import import_module
from typing import Any

from kernel.llm.providers import (
    DeterministicLLMProvider,
    EchoLLMProvider,
    LLMProvider,
    LLMProviderError,
    LLMRuntimeError,
    classify_llm_error,
)
from kernel.llm.openai_compatible import OpenAICompatibleProvider
from kernel.llm.runtime import LLMRuntime
from kernel.llm.types import LLMMessage, LLMRequest, LLMResponse, LLMRetryPolicy, LLMUsage


_LEGACY_EXPORTS = {
    "AsyncLLMManager",
    "LLMConfig",
    "LLMError",
    "LegacyLLMResponse",
    "extract_python_code_blocks",
    "normalize_code_block",
}


def __getattr__(name: str) -> Any:
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module("kernel.llm.legacy"), name)


__all__ = [
    "AsyncLLMManager",
    "DeterministicLLMProvider",
    "EchoLLMProvider",
    "LLMConfig",
    "LLMError",
    "LLMMessage",
    "LLMProvider",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMRetryPolicy",
    "LLMRuntime",
    "LLMRuntimeError",
    "LLMUsage",
    "LegacyLLMResponse",
    "OpenAICompatibleProvider",
    "classify_llm_error",
    "extract_python_code_blocks",
    "normalize_code_block",
]
