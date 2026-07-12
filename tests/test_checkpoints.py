from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentos.checkpoints import CheckpointError, inspect_checkpoint, load_checkpoint, resume_checkpoint, save_checkpoint
from agentos.cli import main as cli_main
from agentos.llm import LLMResponse, LLMRuntime, LLMToolCall, LLMToolDefinition
from agentos.runtime import AgentToolLoop, ToolApprovalDecision, ToolResourceLimits
from agentos.tools import ToolRegistry, ToolRuntime
from tests.test_agent_tool_loop import ScriptedProvider, final_response, tool_response


SCHEMA = {"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}


def loop(provider, executions, *, schema=SCHEMA):
    registry=ToolRegistry()
    registry.register(name="add_numbers",description="Add two numbers.",parameters_schema=schema,func=lambda a,b: executions.append((a,b)) or a+b)
    return AgentToolLoop(llm_runtime=LLMRuntime(provider=provider),tool_runtime=ToolRuntime(registry=registry))


def pause(tmp_path, *, calls=None, limits=None):
    executions=[]; call=calls or (LLMToolCall("call-1","add_numbers",{"a":1,"b":2}),)
    provider=ScriptedProvider([tool_response(*call)])
    paused=loop(provider,executions).run([{"role":"user","content":"add"}],[LLMToolDefinition("add_numbers","Add two numbers.",SCHEMA)],require_tool_approval=True,stop_on_tool_error=False,tool_resource_limits=limits)
    path=tmp_path/"checkpoint.json"; save_checkpoint(paused.checkpoint,path)
    return path,paused,executions


def test_save_load_inspect_is_deterministic_and_sanitized(tmp_path: Path):
    path,paused,_=pause(tmp_path)
    first=path.read_bytes(); save_checkpoint(paused.checkpoint,path); assert path.read_bytes()==first
    loaded=load_checkpoint(path); assert loaded.persistent and loaded.response.tool_calls[0].arguments=={"a":1,"b":2}
    metadata=inspect_checkpoint(path); assert metadata.required_tools==("add_numbers",)
    assert not hasattr(metadata.pending_approvals[0],"arguments")


def test_cli_inspection_never_prints_arguments_or_message_content(tmp_path: Path, capsys):
    path,_,_=pause(tmp_path)
    assert cli_main(["checkpoint","inspect",str(path)])==0
    output=capsys.readouterr().out
    assert "call-1" in output and "add_numbers" in output
    assert '"a"' not in output and '"b"' not in output and "add" not in output.replace("add_numbers","")


def test_new_loop_resumes_without_repeating_original_request_and_consumes(tmp_path: Path):
    path,_,executions=pause(tmp_path,limits=ToolResourceLimits(max_tool_calls_per_loop=1))
    provider=ScriptedProvider([final_response("done")]); fresh=loop(provider,executions)
    result=resume_checkpoint(fresh,path,[ToolApprovalDecision("call-1",True)])
    assert result.completed and executions==[(1,2)] and len(provider.requests)==1
    assert not path.exists() and Path(str(path)+".consumed").exists()
    with pytest.raises(CheckpointError,match="consumed"): load_checkpoint(path)


def test_denial_never_executes_and_consumes(tmp_path: Path):
    path,_,executions=pause(tmp_path); fresh=loop(ScriptedProvider([final_response()]),executions)
    result=resume_checkpoint(fresh,path,[ToolApprovalDecision("call-1",False)])
    assert executions==[] and result.tool_results[-1].success is False and not path.exists()


def test_partial_decisions_preserve_file(tmp_path: Path):
    calls=(LLMToolCall("one","add_numbers",{"a":1,"b":2}),LLMToolCall("two","add_numbers",{"a":3,"b":4}))
    path,_,executions=pause(tmp_path,calls=calls); fresh=loop(ScriptedProvider([]),executions)
    result=resume_checkpoint(fresh,path,[ToolApprovalDecision("one",True)])
    assert result.reason=="approval_required" and path.exists() and executions==[]


def test_corruption_version_staleness_and_changed_tools_fail(tmp_path: Path):
    path,_,executions=pause(tmp_path)
    document=json.loads(path.read_text("utf-8")); document["payload"]["round_index"]=9; path.write_text(json.dumps(document),"utf-8")
    with pytest.raises(CheckpointError,match="integrity"): load_checkpoint(path)
    path,_,_=pause(tmp_path); document=json.loads(path.read_text("utf-8")); document["schema_version"]=99; path.write_text(json.dumps(document),"utf-8")
    with pytest.raises(CheckpointError,match="unsupported"): load_checkpoint(path)
    path,_,_=pause(tmp_path)
    with pytest.raises(CheckpointError,match="stale"): load_checkpoint(path,max_age_seconds=0)
    changed={"type":"object","properties":{}}
    with pytest.raises(CheckpointError,match="definition changed"):
        resume_checkpoint(loop(ScriptedProvider([]),executions,schema=changed),path,[ToolApprovalDecision("call-1",True)])


def test_secret_like_argument_is_refused(tmp_path: Path):
    call=(LLMToolCall("call-1","add_numbers",{"api_key":"never-write","a":1,"b":2}),)
    executions=[]; provider=ScriptedProvider([tool_response(*call)]); paused=loop(provider,executions).run([{"role":"user","content":"add"}],[LLMToolDefinition("add_numbers","Add two numbers.",SCHEMA)],require_tool_approval=True,stop_on_tool_error=False)
    with pytest.raises(CheckpointError,match="secret-like"): save_checkpoint(paused.checkpoint,tmp_path/"secret.json")
