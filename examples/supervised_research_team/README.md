# Supervised Research Team

This flagship example shows Sulcus as a supervised multi-agent runtime while
remaining deterministic and completely offline. It needs no API key, network,
Rust toolchain, or `agent_os_core`.

```powershell
python -m examples.supervised_research_team.demo
```

## Roles and flow

`ResearchSupervisor` owns the workflow and the publication boundary.
`PlannerAgent` turns the topic into a structured plan. `ResearchAgent` obtains
all evidence through the registered local tools. `CriticAgent` identifies scope
and evidence limitations. `SynthesisAgent` produces the reviewed report.

The closed tool surface contains `list_sources`, `read_source`,
`search_sources`, `save_research_note`, and `publish_report`. The first four
operate only on bundled Markdown or an in-memory notebook. `publish_report` is
a simulated side effect. There are no network, shell, or general filesystem
tools.

The researcher intentionally requests one missing source. `ToolRuntime` returns
a structured failure, and `AgentToolLoop` continues with `stop_on_tool_error`
disabled. The next round reads a valid source and completes the research. With
`--tight-limits`, a third search exceeds a per-tool call limit and is safely
denied; synthesis still completes from the evidence already gathered.

## Approval and resume

Publication runs in a separate loop with tool approval required. The loop
pauses before `publish_report` and exposes safe pending-call metadata. The
supervisor resumes the existing checkpoint with an approve or deny decision.
Resume processes the saved response, so it does not repeat the original LLM
request. Denial is the default; `--approve-publish` permits only the simulated
publication.

## Modes

```powershell
python -m examples.supervised_research_team.demo --topic "Your topic"
python -m examples.supervised_research_team.demo --execution-mode parallel
python -m examples.supervised_research_team.demo --tight-limits
python -m examples.supervised_research_team.demo --show-timeline
python -m examples.supervised_research_team.demo --approve-publish
python -m examples.supervised_research_team.demo --deny-publish
python -m examples.supervised_research_team.demo --live
```

Sequential and parallel modes use the same requested-call ordering. The
timeline renders only safe runtime metadata such as tool names, counts, modes,
and argument keys—not prompts or raw argument values. `--live` adds a simple
progress-style label but deliberately stays on the same offline scripted
provider. It is intended as a live terminal presentation, not a network-backed
research mode.

