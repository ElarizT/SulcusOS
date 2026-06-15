from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import pytest

from kernel.events import RuntimeEventLog
from kernel.llm import (
    LLMCostLedger,
    LLMCostRate,
    LLMCostTable,
    LLMMessage,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMResponseCache,
    LLMRuntime,
    LLMStreamChunk,
    LLMUsage,
    format_cost_ledger,
    format_cost_record,
)


class CostProvider:
    supports_streaming = True

    def __init__(
        self,
        name: str,
        *,
        usage: LLMUsage | None = None,
        fail: bool = False,
    ) -> None:
        self.name = name
        self.usage = usage
        self.fail = fail
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.fail:
            raise LLMProviderError("private failure")
        return LLMResponse("private response", request.model, self.name, self.usage)

    def stream(self, request: LLMRequest) -> Iterator[LLMStreamChunk]:
        self.requests.append(request)
        yield LLMStreamChunk("private streamed text", 0)
        yield LLMStreamChunk("", 1, done=True, usage=self.usage)


def rates(*items: LLMCostRate) -> LLMCostTable:
    return LLMCostTable(items)


def test_no_cost_table_preserves_behavior() -> None:
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        CostProvider("openai", usage=LLMUsage(10, 5, 15)),
        events,
    )

    runtime.chat([LLMMessage("user", "private prompt")], model="model")

    assert runtime.cost_snapshot() == LLMCostLedger()
    assert [event.event_type for event in events.events] == [
        "llm.requested",
        "llm.completed",
    ]


def test_successful_usage_records_exact_decimal_cost_and_safe_event() -> None:
    prompt = "private prompt"
    api_key = "private api key"
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        CostProvider("openai", usage=LLMUsage(1_000, 500, 1_500)),
        events,
        cost_table=rates(LLMCostRate("openai", "model", 0.40, 1.60)),
    )

    runtime.chat(
        [LLMMessage("user", prompt)],
        model="model",
        metadata={"api_key": api_key},
    )

    ledger = runtime.cost_snapshot()
    assert ledger.total_cost == Decimal("0.00120")
    assert len(ledger.records) == 1
    assert ledger.records[0].prompt_cost == Decimal("0.00040")
    assert ledger.records[0].completion_cost == Decimal("0.00080")
    event = events.by_type("llm.cost_recorded")[0]
    assert event.metadata == {
        "provider": "openai",
        "model": "model",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "total_cost": "0.0012",
        "currency": "USD",
    }
    assert prompt not in repr(events.events)
    assert api_key not in repr(events.events)
    assert "private response" not in repr(events.events)


@pytest.mark.parametrize(
    ("usage", "reason"),
    [
        (None, "missing_usage"),
        (LLMUsage(total_tokens=3), "incomplete_usage"),
    ],
)
def test_missing_or_incomplete_usage_skips_cost_safely(
    usage: LLMUsage | None,
    reason: str,
) -> None:
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        CostProvider("openai", usage=usage),
        events,
        cost_table=rates(LLMCostRate("openai", "model", 1, 2)),
    )

    runtime.chat([LLMMessage("user", "private")], model="model")

    assert runtime.cost_snapshot().records == ()
    event = events.by_type("llm.cost_skipped")[0]
    assert event.metadata["reason"] == reason
    assert event.metadata["total_cost"] == "0"
    assert event.metadata["currency"] == "USD"


def test_no_matching_rate_skips_and_exact_match_precedes_wildcard() -> None:
    usage = LLMUsage(1_000_000, 1_000_000, 2_000_000)
    table = rates(
        LLMCostRate("openai", "*", 10, 10),
        LLMCostRate("openai", "exact", 1, 2),
    )
    exact = LLMRuntime(CostProvider("openai", usage=usage), cost_table=table)
    missing_events = RuntimeEventLog()
    missing = LLMRuntime(
        CostProvider("other", usage=usage),
        missing_events,
        cost_table=table,
    )

    exact.chat([LLMMessage("user", "private")], model="exact")
    missing.chat([LLMMessage("user", "private")], model="exact")

    assert exact.cost_snapshot().total_cost == Decimal("3")
    assert missing.cost_snapshot().records == ()
    assert missing_events.by_type("llm.cost_skipped")[0].metadata["reason"] == "rate_not_found"


def test_wildcard_rate_matches_when_exact_rate_is_absent() -> None:
    runtime = LLMRuntime(
        CostProvider("openai", usage=LLMUsage(1_000_000, 1_000_000)),
        cost_table=rates(LLMCostRate("openai", "*", 2, 3)),
    )

    runtime.chat([LLMMessage("user", "private")], model="any-model")

    assert runtime.cost_snapshot().total_cost == Decimal("5")


def test_cache_hit_does_not_double_count_cost() -> None:
    provider = CostProvider("openai", usage=LLMUsage(10, 5, 15))
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        provider,
        events,
        cache=LLMResponseCache(enabled=True),
        cost_table=rates(LLMCostRate("openai", "model", 1, 2)),
    )
    messages = [LLMMessage("user", "same")]

    runtime.chat(messages, model="model")
    runtime.chat(messages, model="model")

    assert len(provider.requests) == 1
    assert len(runtime.cost_snapshot().records) == 1
    assert len(events.by_type("llm.cost_recorded")) == 1


def test_fallback_records_only_successful_provider_cost() -> None:
    runtime = LLMRuntime(
        providers={
            "primary": CostProvider("primary-provider", fail=True),
            "fallback": CostProvider("fallback-provider", usage=LLMUsage(10, 5, 15)),
        },
        default_provider="primary",
        fallback_providers=["fallback"],
        cost_table=rates(LLMCostRate("fallback-provider", "model", 1, 2)),
    )

    runtime.chat([LLMMessage("user", "private")], model="model")

    assert [record.provider for record in runtime.cost_snapshot().records] == [
        "fallback-provider"
    ]


def test_stream_final_usage_records_cost_exactly_once() -> None:
    events = RuntimeEventLog()
    runtime = LLMRuntime(
        CostProvider("openai", usage=LLMUsage(10, 5, 15)),
        events,
        cost_table=rates(LLMCostRate("openai", "model", 1, 2)),
    )

    list(runtime.stream_chat([LLMMessage("user", "private prompt")], model="model"))

    assert len(runtime.cost_snapshot().records) == 1
    assert len(events.by_type("llm.cost_recorded")) == 1
    assert "private streamed text" not in repr(events.events)


def test_cost_snapshot_and_formatting_are_deterministic() -> None:
    runtime = LLMRuntime(
        CostProvider("openai", usage=LLMUsage(1_000, 500, 1_500)),
        cost_table=rates(LLMCostRate("openai", "model", 0.40, 1.60)),
    )
    runtime.chat([LLMMessage("user", "private")], model="model")
    record = runtime.cost_snapshot().records[0]

    assert format_cost_record(record) == "openai/model  tokens=1500  cost=$0.0012 USD"
    assert format_cost_ledger(runtime.cost_snapshot()) == (
        "total: $0.0012 USD\n"
        "openai/model  calls=1  tokens=1500  cost=$0.0012 USD"
    )


def test_invalid_cost_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        LLMCostRate("openai", "model", -1, 1)
    with pytest.raises(ValueError, match="unique"):
        LLMCostTable(
            [
                LLMCostRate("openai", "model", 1, 1),
                LLMCostRate("openai", "model", 2, 2),
            ]
        )
    with pytest.raises(ValueError, match="one currency"):
        LLMCostTable(
            [
                LLMCostRate("openai", "one", 1, 1, "USD"),
                LLMCostRate("openai", "two", 1, 1, "EUR"),
            ]
        )
    with pytest.raises(Exception, match="cost_table must be"):
        LLMRuntime(CostProvider("openai"), cost_table=object())  # type: ignore[arg-type]
