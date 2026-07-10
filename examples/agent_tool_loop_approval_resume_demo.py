"""Offline resumable tool-approval demonstration (no API key required)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agentos.runtime import AgentToolLoop, ToolApprovalDecision
from kernel.events import RuntimeEventLog
from agentos.llm import LLMRequest, LLMResponse, LLMRuntime, LLMToolCall, LLMToolDefinition
from agentos.tools import ToolRegistry, ToolRuntime


class ScriptedProvider:
    name = "offline-demo"
    default_model = "offline-model"

    def __init__(self) -> None:
        self.responses = [
            LLMResponse(
                content="",
                provider=self.name,
                model=self.default_model,
                tool_calls=(
                    LLMToolCall("read-1", "read_document", {"path": "brief.txt"}),
                    LLMToolCall("mail-1", "send_email", {"to": "team@example.test"}),
                ),
            ),
            LLMResponse(
                content="The document was read, but the email was not sent.",
                provider=self.name,
                model=self.default_model,
            ),
        ]

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self.responses.pop(0)


def definition(name: str, description: str) -> LLMToolDefinition:
    return LLMToolDefinition(name=name, description=description, parameters_schema={"type": "object"})


def main() -> None:
    registry = ToolRegistry()
    registry.register(name="read_document", description="Read a document", parameters_schema={"type": "object"}, func=lambda **_: "Document contents")
    registry.register(name="send_email", description="Send email", parameters_schema={"type": "object"}, func=lambda **_: "Email sent")
    events = RuntimeEventLog()
    loop = AgentToolLoop(
        llm_runtime=LLMRuntime(provider=ScriptedProvider(), event_sink=events),
        tool_runtime=ToolRuntime(registry=registry),
        event_sink=events,
    )

    paused = loop.run(
        [{"role": "user", "content": "Read the brief and email the team."}],
        [definition("read_document", "Read a document"), definition("send_email", "Send email")],
        require_tool_approval=True,
        stop_on_tool_error=False,
    )
    print(f"Paused: {paused.reason} (round {paused.current_round_index})")
    print("Pending:", ", ".join(item.tool_name for item in paused.pending_approvals))

    decisions = [
        ToolApprovalDecision("read-1", approved=True),
        ToolApprovalDecision("mail-1", approved=False, reason="Caller chose not to send email"),
    ]
    print("Decisions:", ", ".join(f"{item.tool_call_id}={'approved' if item.approved else 'denied'}" for item in decisions))
    assert paused.checkpoint is not None
    resumed = loop.resume(checkpoint=paused.checkpoint, approval_decisions=decisions)
    print("Results:", [(result.name, result.success) for result in resumed.tool_results])
    print("Final:", resumed.final_response.content if resumed.final_response else resumed.reason)
    print("Timeline:")
    for event in events.events:
        if "approval" in event.event_type or event.event_type == "agent_tool_loop_resumed":
            print(f"- {event.event_type}: {event.metadata}")


if __name__ == "__main__":
    main()
