"""Stable provider-neutral LLM Runtime Layer for Agent OS."""

from importlib import import_module
from typing import Any

from kernel.llm.cache import (
    LLMCacheEntry,
    LLMCacheKey,
    LLMCacheStats,
    LLMResponseCache,
    build_llm_cache_key,
)
from kernel.llm.cost import (
    LLMCostLedger,
    LLMCostRate,
    LLMCostRecord,
    LLMCostTable,
    format_cost_ledger,
    format_cost_record,
)
from kernel.llm.providers import (
    DeterministicLLMProvider,
    DeterministicStreamingLLMProvider,
    EchoLLMProvider,
    LLMBudgetExceededError,
    LLMProvider,
    LLMProviderError,
    LLMRuntimeError,
    LLMStreamingUnsupportedError,
    StreamingLLMProvider,
    classify_llm_error,
)
from kernel.llm.openai_compatible import OpenAICompatibleProvider
from kernel.llm.runtime import LLMRuntime
from kernel.llm.types import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRetryPolicy,
    LLMStreamChunk,
    LLMStreamResult,
    LLMTokenBudget,
    LLMUsage,
    LLMUsageLedger,
    apply_usage_to_ledger,
    check_token_budget,
    format_usage_ledger,
)


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
    "DeterministicStreamingLLMProvider",
    "EchoLLMProvider",
    "LLMConfig",
    "LLMCostLedger",
    "LLMCostRate",
    "LLMCostRecord",
    "LLMCostTable",
    "LLMCacheEntry",
    "LLMCacheKey",
    "LLMCacheStats",
    "LLMError",
    "LLMMessage",
    "LLMBudgetExceededError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseCache",
    "LLMRetryPolicy",
    "LLMTokenBudget",
    "LLMRuntime",
    "LLMRuntimeError",
    "LLMStreamChunk",
    "LLMStreamResult",
    "LLMStreamingUnsupportedError",
    "LLMUsage",
    "LLMUsageLedger",
    "LegacyLLMResponse",
    "OpenAICompatibleProvider",
    "StreamingLLMProvider",
    "classify_llm_error",
    "apply_usage_to_ledger",
    "build_llm_cache_key",
    "check_token_budget",
    "extract_python_code_blocks",
    "format_usage_ledger",
    "format_cost_ledger",
    "format_cost_record",
    "normalize_code_block",
]
