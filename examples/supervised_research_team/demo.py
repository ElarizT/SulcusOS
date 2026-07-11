"""Supervised Research Team: an offline, deterministic Sulcus flagship demo."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from agentos.llm import LLMRequest, LLMResponse, LLMRuntime, LLMToolCall
from agentos.config import load_config, resolve_config
from agentos.runtime import AgentToolLoop, ToolApprovalDecision, ToolResourceLimits
from agentos.tools import ToolRegistry, ToolRuntime
from kernel.events import RuntimeEvent, RuntimeEventLog
from kernel.timeline import render_runtime_timeline


DEFAULT_TOPIC = "How can supervised agent runtimes remain safe and reproducible?"
SOURCE_DIR = Path(__file__).with_name("sources")


class ScriptedProvider:
    """Tiny provider that exercises the real loop while remaining offline."""

    name = "offline-supervised-research"
    default_model = "deterministic-script-v1"

    def __init__(self, responses: Sequence[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("scripted provider received an unexpected request")
        return self.responses.pop(0)


def _response(content: str = "", *calls: LLMToolCall) -> LLMResponse:
    return LLMResponse(
        content=content,
        provider=ScriptedProvider.name,
        model=ScriptedProvider.default_model,
        tool_calls=tuple(calls),
    )


def _call(call_id: str, name: str, **arguments: object) -> LLMToolCall:
    return LLMToolCall(id=call_id, name=name, arguments=arguments)


@dataclass(frozen=True)
class WorkflowResult:
    topic: str
    plan: str
    findings: str
    critic_review: str
    final_report: str
    publication_approved: bool
    published: bool
    controlled_failure_recovered: bool
    resource_denials: int
    publication_provider_requests: int
    timeline: tuple[RuntimeEvent, ...]
    notes: tuple[str, ...]


class ResearchTools:
    """Closed local corpus and in-memory notebook; no ambient filesystem tool."""

    def __init__(self) -> None:
        self._sources = {
            path.name: path.read_text(encoding="utf-8")
            for path in sorted(SOURCE_DIR.glob("*.md"))
        }
        self.notes: list[str] = []
        self.published = False

    def list_sources(self) -> str:
        return "\n".join(sorted(self._sources))

    def read_source(self, source_id: str) -> str:
        if source_id not in self._sources:
            raise KeyError("source is not in the bundled corpus")
        return self._sources[source_id]

    def search_sources(self, query: str) -> str:
        terms = {word.casefold().strip(".,:;?!") for word in query.split() if len(word) > 3}
        matches: list[str] = []
        for source_id, text in sorted(self._sources.items()):
            lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
            if terms.intersection(text.casefold().replace("\n", " ").split()):
                matches.append(f"{source_id}: {lines[0]}")
        return "\n".join(matches) if matches else "No local matches."

    def save_research_note(self, title: str, note: str) -> str:
        self.notes.append(f"{title}: {note}")
        return f"saved note {len(self.notes)}"

    def publish_report(self, report: str) -> str:
        self.published = True
        return "simulated publication completed"


def _object_schema(properties: dict[str, dict[str, object]], required: list[str]) -> dict[str, object]:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def build_registry(tools: ResearchTools, events: RuntimeEventLog) -> ToolRegistry:
    registry = ToolRegistry(event_sink=events)
    string = {"type": "string"}
    registry.register(name="list_sources", description="List bundled research source identifiers.", parameters_schema=_object_schema({}, []), func=tools.list_sources, parallel_safe=True)
    registry.register(name="read_source", description="Read one bundled source by identifier.", parameters_schema=_object_schema({"source_id": string}, ["source_id"]), func=tools.read_source, parallel_safe=True)
    registry.register(name="search_sources", description="Search only the bundled local corpus.", parameters_schema=_object_schema({"query": string}, ["query"]), func=tools.search_sources, parallel_safe=True)
    registry.register(name="save_research_note", description="Save a note in the workflow's in-memory notebook.", parameters_schema=_object_schema({"title": string, "note": string}, ["title", "note"]), func=tools.save_research_note)
    registry.register(name="publish_report", description="Simulate publishing the final report after approval.", parameters_schema=_object_schema({"report": string}, ["report"]), func=tools.publish_report)
    return registry


class Agent:
    role = "Agent"

    def __init__(self, registry: ToolRegistry, runtime: ToolRuntime, events: RuntimeEventLog) -> None:
        self.registry, self.runtime, self.events = registry, runtime, events

    def run_script(self, prompt: str, responses: Sequence[LLMResponse], *, tool_names: Sequence[str] = (), **options: object):
        provider = ScriptedProvider(responses)
        loop = AgentToolLoop(llm_runtime=LLMRuntime(provider=provider, event_sink=self.events), tool_runtime=self.runtime, event_sink=self.events)
        definitions = [self.registry.require(name).to_llm_tool_definition() for name in tool_names]
        result = loop.run([{"role": "user", "content": prompt}], definitions, metadata={"agent_role": self.role}, max_steps=6, **options)
        return result, provider, loop


class PlannerAgent(Agent):
    role = "PlannerAgent"

    def plan(self, topic: str) -> str:
        plan = f"""1. Scope the question: {topic}
2. Inventory the bundled sources.
3. Search for supervision, safety, and determinism evidence.
4. Read primary local passages and record concise notes.
5. Ask the critic to test gaps and claims before synthesis."""
        result, _, _ = self.run_script(f"Create a bounded research plan for: {topic}", [_response(plan)])
        assert result.final_response is not None
        return result.final_response.content


class ResearchAgent(Agent):
    role = "ResearchAgent"

    def research(self, topic: str, mode: str, tight_limits: bool, configured_limits: ToolResourceLimits | None) -> tuple[str, int]:
        first = _response("", _call("sources", "list_sources"), _call("controlled-failure", "read_source", source_id="missing-source.md"))
        calls = [
            _call("read-supervision", "read_source", source_id="supervision.md"),
            _call("search-safety", "search_sources", query="registered tools safety failures"),
            _call("search-order", "search_sources", query="deterministic parallel order"),
        ]
        if tight_limits:
            calls.append(_call("limit-demo", "search_sources", query="bounded resources"))
        second = _response("", *calls)
        third = _response("", _call("note-1", "save_research_note", title="Supervision", note="Keep consequential actions behind supervisor approval."))
        findings = """- Supervision: narrow roles can work autonomously while a supervisor owns consequential actions [supervision.md].
- Recovery: structured tool failures allow a bounded retry against a known source [tool_safety.md].
- Determinism: concurrent outcomes should be returned in requested-call order [determinism.md].
- Privacy: runtime metadata can expose tool names and argument keys without argument values [tool_safety.md]."""
        limits = configured_limits
        if tight_limits:
            limits = ToolResourceLimits(
                max_tool_calls_per_loop=None if limits is None else limits.max_tool_calls_per_loop,
                max_tool_calls_per_round=None if limits is None else limits.max_tool_calls_per_round,
                max_calls_per_tool={"search_sources": 2},
                tool_timeout_ms=None if limits is None else limits.tool_timeout_ms,
            )
        result, _, _ = self.run_script(
            f"Gather local evidence for: {topic}", [first, second, third, _response(findings)],
            tool_names=("list_sources", "read_source", "search_sources", "save_research_note"),
            stop_on_tool_error=False, tool_execution_mode=mode, tool_resource_limits=limits,
        )
        assert result.final_response is not None
        denials = sum(
            1
            for item in result.tool_results
            if item.error and "resource limits" in item.error.casefold()
        )
        return result.final_response.content, denials


class CriticAgent(Agent):
    role = "CriticAgent"

    def review(self, findings: str) -> str:
        review = "PASS WITH CAVEATS: claims are traceable to the closed corpus; the demo proves runtime behavior, not real-world research quality. Keep publication human-controlled and label the source set as illustrative."
        result, _, _ = self.run_script("Review the evidence for gaps and unsupported claims.\n" + findings, [_response(review)])
        assert result.final_response is not None
        return result.final_response.content


class SynthesisAgent(Agent):
    role = "SynthesisAgent"

    def synthesize(self, topic: str, findings: str, review: str) -> str:
        report = f"""# Supervised Research Report

## Topic
{topic}

## Conclusion
Safe supervised runtimes combine narrow registered tools, bounded recovery, deterministic result collection, and an explicit approval checkpoint before side effects.

## Evidence
{findings}

## Critic review
{review}

## Scope
This deterministic report uses only the three bundled illustrative Markdown sources."""
        result, _, _ = self.run_script("Synthesize a concise report from reviewed local evidence.", [_response(report)])
        assert result.final_response is not None
        return result.final_response.content


class ResearchSupervisor(Agent):
    role = "ResearchSupervisor"

    def publish(self, report: str, approved: bool) -> tuple[bool, int]:
        provider = ScriptedProvider([
            _response("", _call("publish-final", "publish_report", report=report)),
            _response("Publication completed." if approved else "Publication was denied; the report remains local."),
        ])
        loop = AgentToolLoop(llm_runtime=LLMRuntime(provider=provider, event_sink=self.events), tool_runtime=self.runtime, event_sink=self.events)
        paused = loop.run(
            [{"role": "user", "content": "Publish the reviewed final report."}],
            [self.registry.require("publish_report").to_llm_tool_definition()],
            metadata={"agent_role": self.role}, require_tool_approval=True, stop_on_tool_error=False,
        )
        assert paused.reason == "approval_required" and paused.checkpoint is not None
        assert len(provider.requests) == 1
        resumed = loop.resume(checkpoint=paused.checkpoint, approval_decisions=[ToolApprovalDecision("publish-final", approved)])
        assert resumed.completed
        return approved and any(item.name == "publish_report" and item.success for item in resumed.tool_results), len(provider.requests)


def run_workflow(topic: str = DEFAULT_TOPIC, execution_mode: str = "sequential", approve_publish: bool = False, tight_limits: bool = False, resource_limits: ToolResourceLimits | None = None) -> WorkflowResult:
    events = RuntimeEventLog()
    local_tools = ResearchTools()
    registry = build_registry(local_tools, events)
    runtime = ToolRuntime(registry=registry, event_sink=events)
    planner = PlannerAgent(registry, runtime, events)
    researcher = ResearchAgent(registry, runtime, events)
    critic = CriticAgent(registry, runtime, events)
    synthesis = SynthesisAgent(registry, runtime, events)
    supervisor = ResearchSupervisor(registry, runtime, events)
    plan = planner.plan(topic)
    findings, resource_denials = researcher.research(topic, execution_mode, tight_limits, resource_limits)
    review = critic.review(findings)
    report = synthesis.synthesize(topic, findings, review)
    published, publication_requests = supervisor.publish(report, approve_publish)
    failures = [e for e in events.events if isinstance(e, RuntimeEvent) and e.event_type == "tool_execution_failed" and e.metadata.get("tool_name") == "read_source"]
    return WorkflowResult(topic, plan, findings, review, report, approve_publish, published, bool(failures), resource_denials, publication_requests, tuple(e for e in events.events if isinstance(e, RuntimeEvent)), tuple(local_tools.notes))


def _print_result(result: WorkflowResult, show_timeline: bool) -> None:
    print("Supervised Research Team")
    print("\nPLAN\n" + result.plan)
    print("\nFINDINGS\n" + result.findings)
    print("\nCRITIC REVIEW\n" + result.critic_review)
    print("\nFINAL REPORT\n" + result.final_report)
    print("\nAPPROVAL CHECKPOINT")
    print("Publication:", "APPROVED (simulated publish succeeded)" if result.published else "DENIED (report kept local)")
    counts = Counter(event.event_type for event in result.timeline)
    print("\nTIMELINE SUMMARY")
    print(f"events={sum(counts.values())} tool_requests={counts['tool_call_requested']} tool_failures={counts['tool_execution_failed']} approvals={counts['tool_approval_requested']} resource_denials={result.resource_denials}")
    print("key_events=controlled_failure->recovery->critic_review->approval_pause->approval_decision")
    if show_timeline:
        print("\nSAFE TIMELINE")
        for row in render_runtime_timeline(result.timeline):
            print(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--execution-mode", choices=("sequential", "parallel"))
    decision = parser.add_mutually_exclusive_group()
    decision.add_argument("--approve-publish", action="store_true")
    decision.add_argument("--deny-publish", action="store_true")
    parser.add_argument("--tight-limits", action="store_true")
    parser.add_argument("--show-timeline", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    effective = resolve_config(load_config(), explicit={"execution_mode": args.execution_mode})
    configured_limits = ToolResourceLimits(
        max_tool_calls_per_loop=effective.limits.max_tool_calls_per_loop,
        max_tool_calls_per_round=effective.limits.max_tool_calls_per_round,
        tool_timeout_ms=effective.limits.tool_timeout_ms,
    )
    if all(value is None for value in (
        configured_limits.max_tool_calls_per_loop,
        configured_limits.max_tool_calls_per_round,
        configured_limits.tool_timeout_ms,
    )):
        configured_limits = None
    result = run_workflow(
        args.topic,
        effective.runtime.execution_mode,
        args.approve_publish and not args.deny_publish,
        args.tight_limits,
        configured_limits,
    )
    _print_result(result, args.show_timeline)
    if not result.controlled_failure_recovered or result.publication_provider_requests != 2:
        raise RuntimeError("workflow safety invariant failed")
    if args.tight_limits and result.resource_denials != 1:
        raise RuntimeError("tight-limit mode did not produce exactly one safe denial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
