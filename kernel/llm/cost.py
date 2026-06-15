"""Optional deterministic cost accounting for provider-reported LLM usage."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from kernel.llm.types import LLMUsage


_ONE_MILLION = Decimal("1000000")


@dataclass(frozen=True)
class LLMCostRate:
    """User-configured token prices for one provider and model."""

    provider: str
    model: str
    prompt_per_1m_tokens: Decimal
    completion_per_1m_tokens: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        for name in ("provider", "model", "currency"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"cost rate {name} must be a nonempty string")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(
            self,
            "prompt_per_1m_tokens",
            _nonnegative_decimal(self.prompt_per_1m_tokens, "prompt_per_1m_tokens"),
        )
        object.__setattr__(
            self,
            "completion_per_1m_tokens",
            _nonnegative_decimal(
                self.completion_per_1m_tokens,
                "completion_per_1m_tokens",
            ),
        )


class LLMCostTable:
    """Deterministic exact-then-wildcard provider/model rate lookup."""

    def __init__(self, rates: Iterable[LLMCostRate] = ()) -> None:
        self.rates = tuple(rates)
        if any(not isinstance(rate, LLMCostRate) for rate in self.rates):
            raise ValueError("cost table rates must be LLMCostRate objects")
        keys = [(rate.provider, rate.model) for rate in self.rates]
        if len(keys) != len(set(keys)):
            raise ValueError("cost table provider/model rates must be unique")
        currencies = {rate.currency for rate in self.rates}
        if len(currencies) > 1:
            raise ValueError("cost table rates must use one currency")
        self.currency = next(iter(currencies), "USD")
        self._rates = {(rate.provider, rate.model): rate for rate in self.rates}

    def match(self, provider: str, model: str) -> LLMCostRate | None:
        """Return an exact match before a provider wildcard match."""
        return self._rates.get((provider, model)) or self._rates.get((provider, "*"))


@dataclass(frozen=True)
class LLMCostRecord:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cost: Decimal
    completion_cost: Decimal
    total_cost: Decimal
    currency: str


@dataclass(frozen=True)
class LLMCostLedger:
    """Immutable snapshot of recorded LLM costs."""

    records: tuple[LLMCostRecord, ...] = ()
    total_cost: Decimal = Decimal("0")
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "total_cost", Decimal(self.total_cost))


def calculate_cost_record(
    rate: LLMCostRate,
    provider: str,
    model: str,
    usage: LLMUsage,
) -> LLMCostRecord | None:
    """Calculate cost when prompt and completion usage are both known."""
    if not _known_token_count(usage.prompt_tokens) or not _known_token_count(
        usage.completion_tokens
    ):
        return None
    if usage.total_tokens is not None and not _known_token_count(usage.total_tokens):
        return None
    prompt_cost = Decimal(usage.prompt_tokens) * rate.prompt_per_1m_tokens / _ONE_MILLION
    completion_cost = (
        Decimal(usage.completion_tokens)
        * rate.completion_per_1m_tokens
        / _ONE_MILLION
    )
    return LLMCostRecord(
        provider=provider,
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens
        if usage.total_tokens is not None
        else usage.prompt_tokens + usage.completion_tokens,
        prompt_cost=prompt_cost,
        completion_cost=completion_cost,
        total_cost=prompt_cost + completion_cost,
        currency=rate.currency,
    )


def append_cost_record(ledger: LLMCostLedger, record: LLMCostRecord) -> LLMCostLedger:
    """Return a new ledger with one additional record."""
    if ledger.currency != record.currency:
        raise ValueError("cost record currency must match ledger currency")
    return LLMCostLedger(
        records=(*ledger.records, record),
        total_cost=ledger.total_cost + record.total_cost,
        currency=ledger.currency,
    )


def format_cost_record(record: LLMCostRecord) -> str:
    """Format one cost record without request or response content."""
    return (
        f"{record.provider}/{record.model}  "
        f"tokens={record.total_tokens}  "
        f"cost={_format_money(record.total_cost, record.currency)}"
    )


def format_cost_ledger(ledger: LLMCostLedger) -> str:
    """Format a compact deterministic provider/model cost summary."""
    lines = [f"total: {_format_money(ledger.total_cost, ledger.currency)}"]
    grouped: dict[tuple[str, str], tuple[int, int, Decimal]] = {}
    for record in ledger.records:
        key = (record.provider, record.model)
        calls, tokens, cost = grouped.get(key, (0, 0, Decimal("0")))
        grouped[key] = (calls + 1, tokens + record.total_tokens, cost + record.total_cost)
    for provider, model in sorted(grouped):
        calls, tokens, cost = grouped[(provider, model)]
        lines.append(
            f"{provider}/{model}  calls={calls}  tokens={tokens}  "
            f"cost={_format_money(cost, ledger.currency)}"
        )
    return "\n".join(lines)


def format_decimal_cost(value: Decimal) -> str:
    """Return a stable plain-decimal representation for safe event metadata."""
    text = format(value, "f").rstrip("0").rstrip(".")
    return text or "0"


def _format_money(value: Decimal, currency: str) -> str:
    prefix = "$" if currency == "USD" else ""
    return f"{prefix}{format_decimal_cost(value)} {currency}"


def _nonnegative_decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a nonnegative number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{label} must be a nonnegative number") from None
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be a nonnegative number")
    return result


def _known_token_count(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0
