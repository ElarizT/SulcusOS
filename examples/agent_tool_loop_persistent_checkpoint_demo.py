"""Deterministic offline restart-safe checkpoint demo (no native core or network)."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from agentos.checkpoints import inspect_checkpoint, resume_checkpoint, save_checkpoint
from agentos.llm import LLMRequest, LLMResponse, LLMRuntime, LLMToolCall, LLMToolDefinition
from agentos.runtime import AgentToolLoop, AgentToolLoopConfig, ToolApprovalDecision
from agentos.tools import ToolRegistry, ToolRuntime


SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}


class OneResponseProvider:
    name = "offline-persistent-demo"
    default_model = "scripted-v1"

    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        response, self.response = self.response, None  # type: ignore[assignment]
        if response is None:
            raise AssertionError("provider request was unexpectedly repeated")
        return response


def make_loop(provider: OneResponseProvider, executions: list[str]) -> AgentToolLoop:
    registry = ToolRegistry()
    registry.register(
        name="publish_note", description="Publish an offline note.", parameters_schema=SCHEMA,
        func=lambda text: executions.append(text) or "published", parallel_safe=False,
    )
    return AgentToolLoop(
        llm_runtime=LLMRuntime(provider=provider), tool_runtime=ToolRuntime(registry=registry),
        config=AgentToolLoopConfig(),
    )


def main() -> None:
    executions: list[str] = []
    first_provider = OneResponseProvider(LLMResponse(
        "", "scripted-v1", "offline-persistent-demo", tool_calls=(
            LLMToolCall("publish-1", "publish_note", {"text": "restart-safe"}),
        ),
    ))
    run_one = make_loop(first_provider, executions)
    paused = run_one.run(
        [{"role": "user", "content": "Publish the note."}],
        [LLMToolDefinition("publish_note", "Publish an offline note.", SCHEMA)],
        require_tool_approval=True,
    )
    assert paused.checkpoint is not None and not executions

    with TemporaryDirectory() as directory:
        path = Path(directory) / "approval.checkpoint.json"
        save_checkpoint(paused.checkpoint, path)
        metadata = inspect_checkpoint(path)
        print(f"Run 1: saved {metadata.checkpoint_id}; pending={metadata.pending_approvals[0].tool_call_id}")

        # Run 2 owns entirely new provider/loop/runtime/registry instances.  Its
        # only scripted response is the continuation, proving the original
        # request cannot be repeated successfully.
        second_provider = OneResponseProvider(LLMResponse(
            "Published after restart.", "scripted-v1", "offline-persistent-demo"
        ))
        run_two = make_loop(second_provider, executions)
        completed = resume_checkpoint(
            run_two, path, [ToolApprovalDecision("publish-1", approved=True)]
        )
        assert completed.completed and executions == ["restart-safe"]
        assert len(first_provider.requests) == 1 and len(second_provider.requests) == 1
        assert not path.exists() and path.with_name(path.name + ".consumed").exists()
        print("Run 2: approved tool executed; continuation completed")
        print("Original provider requests: 1 (not repeated)")
        print(f"Final: {completed.final_response.content if completed.final_response else completed.reason}")


if __name__ == "__main__":
    main()
