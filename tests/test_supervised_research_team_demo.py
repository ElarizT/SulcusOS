from __future__ import annotations

from examples.supervised_research_team.demo import DEFAULT_TOPIC, run_workflow


def test_default_workflow_is_complete_offline_and_denies_publication() -> None:
    result = run_workflow()
    assert result.topic == DEFAULT_TOPIC
    assert "1. Scope" in result.plan
    assert "[supervision.md]" in result.findings
    assert "PASS WITH CAVEATS" in result.critic_review
    assert "# Supervised Research Report" in result.final_report
    assert result.controlled_failure_recovered
    assert not result.publication_approved
    assert not result.published
    assert result.publication_provider_requests == 2


def test_approval_executes_simulated_publication() -> None:
    result = run_workflow(approve_publish=True)
    assert result.publication_approved
    assert result.published
    assert result.publication_provider_requests == 2


def test_parallel_mode_preserves_tool_result_order() -> None:
    sequential = run_workflow(execution_mode="sequential")
    parallel = run_workflow(execution_mode="parallel")
    assert parallel.findings == sequential.findings
    requested = [
        event.metadata.get("tool_name")
        for event in parallel.timeline
        if event.event_type == "tool_call_requested"
    ]
    assert requested[:2] == ["list_sources", "read_source"]
    assert requested[2:5] == ["read_source", "search_sources", "search_sources"]


def test_tight_limits_produces_exactly_one_safe_denial() -> None:
    result = run_workflow(tight_limits=True)
    assert result.resource_denials == 1
    assert "# Supervised Research Report" in result.final_report


def test_timeline_metadata_does_not_copy_raw_argument_values() -> None:
    result = run_workflow(topic="TOPIC_SECRET_47")
    rendered = repr([(event.message, event.metadata) for event in result.timeline])
    assert "TOPIC_SECRET_47" not in rendered
    assert "missing-source.md" not in rendered
    assert "argument_keys" in rendered
