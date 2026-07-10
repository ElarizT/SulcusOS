import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentos.llm import (
    LLMRuntime,
    OpenAICompatibleProvider,
    LLMResponseCache,
    LLMTokenBudget,
    LLMCostRate,
    LLMCostTable,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    api_key = os.environ.get("AGENTOS_LLM_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing AGENTOS_LLM_API_KEY. Set it in PowerShell first:\n"
            '$env:AGENTOS_LLM_API_KEY="your_openrouter_key"'
        )

    provider = OpenAICompatibleProvider(
        provider_name="openrouter",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-oss-120b:free",
        timeout_seconds=60,
    )

    runtime = LLMRuntime(
        provider=provider,
        cache=LLMResponseCache(enabled=True),
        token_budget=LLMTokenBudget(
            name="phase-2-smoke-budget",
            max_total_tokens=20_000,
        ),
        cost_table=LLMCostTable(
            [
                LLMCostRate(
                    provider="openrouter",
                    model="openai/gpt-oss-120b:free",
                    prompt_per_1m_tokens=0.0,
                    completion_per_1m_tokens=0.0,
                    currency="USD",
                )
            ]
        ),
    )

    messages = [
        {
            "role": "user",
            "content": "Reply with exactly: Agent OS integration test passed.",
        }
    ]

    print_section("1. FIRST LIVE CALL")

    response_1 = runtime.chat(
        messages=messages,
        temperature=0.0,
    )

    print("Response content:")
    print(response_1.content)

    print("\nProvider:")
    print(response_1.provider)

    print("\nModel:")
    print(response_1.model)

    print("\nUsage snapshot:")
    print(runtime.usage_snapshot())

    print("\nCost snapshot:")
    print(runtime.cost_snapshot())

    print("\nCache snapshot after first call:")
    print(runtime.cache_snapshot())

    print_section("2. SECOND IDENTICAL CALL -- SHOULD HIT CACHE")

    response_2 = runtime.chat(
        messages=messages,
        temperature=0.0,
    )

    print("Response content:")
    print(response_2.content)

    print("\nResponse metadata:")
    print(response_2.metadata)

    print("\nUsage snapshot after cached call:")
    print(runtime.usage_snapshot())

    print("\nCost snapshot after cached call:")
    print(runtime.cost_snapshot())

    print("\nCache snapshot after cached call:")
    print(runtime.cache_snapshot())

    print_section("3. EVENT LOG SAFETY CHECK")

    event_log = getattr(runtime, "event_log", None)

    if event_log is None:
        print("No runtime.event_log attribute found. That may be normal depending on implementation.")
    else:
        events = list(event_log)
        print(f"Total events: {len(events)}")

        for event in events:
            print(event)

    print_section("4. BASIC ASSERTIONS")

    assert response_1.content is not None
    assert "Agent OS integration test passed" in response_1.content
    assert response_1.provider == "openrouter"

    assert response_2.content is not None
    assert response_2.provider == "openrouter"

    print("All basic assertions passed.")


if __name__ == "__main__":
    main()
