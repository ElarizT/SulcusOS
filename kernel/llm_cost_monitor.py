"""Safe event-derived LLM cost dashboard helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from kernel.events import RuntimeEvent
from kernel.llm.cost import LLMCostLedger, LLMCostRecord, format_cost_ledger


@dataclass(frozen=True)
class LLMCostSnapshot:
    ledger: LLMCostLedger = LLMCostLedger()


def build_llm_cost_snapshot(events: Iterable[Any] = ()) -> LLMCostSnapshot:
    """Build cost totals from safe llm.cost_recorded events."""
    records: list[LLMCostRecord] = []
    currency: str | None = None
    for event in events:
        if not isinstance(event, RuntimeEvent) or event.event_type != "llm.cost_recorded":
            continue
        record = _safe_record(event.metadata)
        if record is None or (currency is not None and record.currency != currency):
            continue
        currency = record.currency
        records.append(record)
    return LLMCostSnapshot(
        LLMCostLedger(
            records=tuple(records),
            total_cost=sum((record.total_cost for record in records), Decimal("0")),
            currency=currency or "USD",
        )
    )


def render_llm_cost_snapshot(snapshot: LLMCostSnapshot) -> list[str]:
    """Render stable compact rows with a deterministic empty state."""
    if not snapshot.ledger.records:
        return ["No LLM costs recorded yet."]
    return format_cost_ledger(snapshot.ledger).splitlines()


def _safe_record(metadata: Mapping[str, Any]) -> LLMCostRecord | None:
    provider = _safe_text(metadata, "provider")
    model = _safe_text(metadata, "model")
    currency = _safe_text(metadata, "currency")
    prompt_tokens = _safe_int(metadata, "prompt_tokens")
    completion_tokens = _safe_int(metadata, "completion_tokens")
    total_tokens = _safe_int(metadata, "total_tokens")
    total_cost = _safe_decimal(metadata, "total_cost")
    if None in (
        provider,
        model,
        currency,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        total_cost,
    ):
        return None
    return LLMCostRecord(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        prompt_cost=Decimal("0"),
        completion_cost=total_cost,
        total_cost=total_cost,
        currency=currency,
    )


def _safe_text(metadata: Mapping[str, Any], name: str) -> str | None:
    value = metadata.get(name)
    if not isinstance(value, str):
        return None
    value = " ".join(value.split()).strip()
    return value[:120] if value else None


def _safe_int(metadata: Mapping[str, Any], name: str) -> int | None:
    value = metadata.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _safe_decimal(metadata: Mapping[str, Any], name: str) -> Decimal | None:
    try:
        value = Decimal(str(metadata.get(name)))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() and value >= 0 else None
