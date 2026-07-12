"""Versioned, dependency-free persistence for paused agent tool loops."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from kernel.agent_tool_loop import (
    AgentToolLoop, AgentToolLoopCheckpoint, AgentToolLoopConfig, AgentToolLoopStep,
    PendingToolApproval, ToolApprovalDecision, ToolPermissionPolicy,
    ToolResourceLimits, _ToolOutcome,
)
from kernel.llm.types import (
    LLMMessage, LLMResponse, LLMToolCall, LLMToolDefinition, LLMToolResult, LLMUsage,
)

CHECKPOINT_SCHEMA_VERSION = 1
_FORMAT = "sulcus-agent-tool-loop-checkpoint"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SECRET_KEY = re.compile(r"(?:api[_-]?key|secret|password|passwd|token|authorization|credential)", re.I)
_SECRET_VALUE = re.compile(r"(?:\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|\bBearer\s+[A-Za-z0-9._~+/-]{12,})", re.I)


class CheckpointError(ValueError):
    """A checkpoint is unsafe, invalid, incompatible, stale, or consumed."""


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    checkpoint_id: str
    created_at: str
    status: str
    round_index: int
    pending_approvals: tuple[PendingToolApproval, ...]
    provider: str
    model: str
    execution_mode: str
    required_tools: tuple[str, ...]
    tool_schema_fingerprints: tuple[tuple[str, str], ...]


def save_checkpoint(checkpoint: AgentToolLoopCheckpoint, path: str | os.PathLike[str]) -> Path:
    """Atomically save a paused checkpoint as deterministic UTF-8 JSON."""
    if not isinstance(checkpoint, AgentToolLoopCheckpoint):
        raise TypeError("checkpoint must be an AgentToolLoopCheckpoint")
    if not checkpoint.pending_approvals:
        raise CheckpointError("checkpoint has no pending approvals")
    created_at = checkpoint.created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = _checkpoint_to_payload(replace(checkpoint, created_at=created_at))
    _reject_secrets_and_non_json(payload)
    document = {
        "format": _FORMAT,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "integrity": {"algorithm": "sha256", "digest": _digest(payload)},
        "payload": payload,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    data = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def load_checkpoint(path: str | os.PathLike[str], *, max_age_seconds: float | None = None) -> AgentToolLoopCheckpoint:
    """Load and structurally validate a checkpoint without runtime objects."""
    source = Path(path)
    document = _read_document(source)
    payload = _validate_document(document, max_age_seconds=max_age_seconds)
    try:
        checkpoint = _payload_to_checkpoint(payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError(f"invalid checkpoint payload: {exc}") from None
    _validate_checkpoint_state(checkpoint)
    return checkpoint


def inspect_checkpoint(path: str | os.PathLike[str], *, max_age_seconds: float | None = None) -> CheckpointMetadata:
    """Return sanitized metadata; tool arguments and message content are omitted."""
    checkpoint = load_checkpoint(path, max_age_seconds=max_age_seconds)
    fingerprints = tuple((tool.name, _schema_fingerprint(tool.parameters_schema)) for tool in checkpoint.tool_definitions)
    return CheckpointMetadata(
        CHECKPOINT_SCHEMA_VERSION, checkpoint.checkpoint_id, checkpoint.created_at or "", "pending",
        checkpoint.round_index, checkpoint.pending_approvals, checkpoint.provider, checkpoint.model,
        checkpoint.requested_execution_mode, tuple(sorted(checkpoint.allowed_tool_names)), fingerprints,
    )


def resume_checkpoint(
    loop: AgentToolLoop, checkpoint_or_path: AgentToolLoopCheckpoint | str | os.PathLike[str],
    approval_decisions: Sequence[ToolApprovalDecision], *, max_age_seconds: float | None = None,
):
    """Validate against a live loop, resume, and atomically consume a file."""
    if not isinstance(loop, AgentToolLoop):
        raise TypeError("loop must be an AgentToolLoop")
    path = None if isinstance(checkpoint_or_path, AgentToolLoopCheckpoint) else Path(checkpoint_or_path)
    checkpoint = checkpoint_or_path if path is None else load_checkpoint(path, max_age_seconds=max_age_seconds)
    assert isinstance(checkpoint, AgentToolLoopCheckpoint)
    _validate_runtime_compatibility(loop, checkpoint)
    pending_ids = {item.tool_call_id for item in checkpoint.pending_approvals}
    decided_ids = {item.tool_call_id for item in approval_decisions}
    if decided_ids != pending_ids:
        return loop.resume(checkpoint=checkpoint, approval_decisions=approval_decisions)
    claimed: Path | None = None
    if path is not None:
        claimed = _consumed_path(path)
        if claimed.exists():
            raise CheckpointError("checkpoint has already been consumed")
        try:
            os.replace(path, claimed)
        except FileNotFoundError:
            raise CheckpointError("checkpoint is missing or has already been consumed") from None
    try:
        return loop.resume(checkpoint=checkpoint, approval_decisions=approval_decisions)
    except Exception:
        if path is not None and claimed is not None and claimed.exists() and not path.exists():
            os.replace(claimed, path)
        raise


def _read_document(path: Path) -> Mapping[str, Any]:
    if not path.exists() and _consumed_path(path).exists():
        raise CheckpointError("checkpoint has already been consumed")
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise CheckpointError("checkpoint file does not exist") from None
    if len(data) > 16 * 1024 * 1024:
        raise CheckpointError("checkpoint file is too large")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise CheckpointError("checkpoint is not valid UTF-8 JSON") from None
    if not isinstance(value, Mapping):
        raise CheckpointError("checkpoint root must be an object")
    return value


def _validate_document(document: Mapping[str, Any], *, max_age_seconds: float | None) -> Mapping[str, Any]:
    if document.get("format") != _FORMAT:
        raise CheckpointError("not a Sulcus AgentToolLoop checkpoint")
    version = document.get("schema_version")
    if version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(f"unsupported checkpoint schema version: {version!r}")
    payload = document.get("payload")
    integrity = document.get("integrity")
    if not isinstance(payload, Mapping) or not isinstance(integrity, Mapping):
        raise CheckpointError("checkpoint payload or integrity metadata is missing")
    if integrity.get("algorithm") != "sha256" or integrity.get("digest") != _digest(payload):
        raise CheckpointError("checkpoint integrity check failed")
    compatibility = payload.get("compatibility")
    definitions = payload.get("tool_definitions")
    if not isinstance(compatibility, Mapping) or not isinstance(definitions, list):
        raise CheckpointError("checkpoint compatibility metadata is missing")
    expected_fingerprints = compatibility.get("tool_schema_fingerprints")
    try:
        actual_fingerprints = {
            item["name"]: _schema_fingerprint(item["parameters_schema"])
            for item in definitions if isinstance(item, Mapping)
        }
    except (KeyError, TypeError, ValueError):
        raise CheckpointError("checkpoint tool compatibility metadata is invalid") from None
    if expected_fingerprints != actual_fingerprints or len(actual_fingerprints) != len(definitions):
        raise CheckpointError("checkpoint tool compatibility metadata is inconsistent")
    created = _parse_timestamp(payload.get("created_at"))
    if max_age_seconds is not None:
        if isinstance(max_age_seconds, bool) or max_age_seconds < 0:
            raise ValueError("max_age_seconds must be nonnegative")
        age = (datetime.now(timezone.utc) - created).total_seconds()
        if age > max_age_seconds:
            raise CheckpointError("checkpoint is stale")
    return payload


def _checkpoint_to_payload(c: AgentToolLoopCheckpoint) -> dict[str, Any]:
    tools = [_tool_definition(t) for t in c.tool_definitions]
    return {
        "checkpoint_id": c.checkpoint_id, "loop_id": c.loop_id, "created_at": c.created_at,
        "round_index": c.round_index, "history": [_message(x) for x in c.history],
        "response": _response(c.response), "tool_definitions": tools,
        "compatibility": {"tool_schema_fingerprints": {t["name"]: _schema_fingerprint(t["parameters_schema"]) for t in tools}},
        "allowed_tool_names": sorted(c.allowed_tool_names), "config": _config(c.config),
        "steps": [_step(x) for x in c.steps], "tool_results": [_result(x) for x in c.tool_results],
        "pending_approvals": [asdict(x) for x in c.pending_approvals],
        "preflight_outcomes": [None if x is None else _outcome(x) for x in c.preflight_outcomes],
        "execution": {"requested_mode": c.requested_execution_mode, "effective_mode": c.effective_execution_mode,
            "fallback_reason": c.fallback_reason, "parallel_safe_tool_count": c.parallel_safe_tool_count,
            "unsafe_tool_count": c.unsafe_tool_count},
        "resources": {"total_requested": c.total_requested, "round_requested": [list(x) for x in c.round_requested],
            "tool_requested": [list(x) for x in c.tool_requested]},
        "route": {"provider": c.provider, "model": c.model},
    }


def _payload_to_checkpoint(p: Mapping[str, Any]) -> AgentToolLoopCheckpoint:
    execution, resources, route = p["execution"], p["resources"], p["route"]
    return AgentToolLoopCheckpoint(
        checkpoint_version=1, checkpoint_id=str(p["checkpoint_id"]), loop_id=str(p["loop_id"]),
        round_index=int(p["round_index"]), history=tuple(_load_message(x) for x in p["history"]),
        response=_load_response(p["response"]), tool_definitions=tuple(_load_definition(x) for x in p["tool_definitions"]),
        allowed_tool_names=frozenset(p["allowed_tool_names"]), config=_load_config(p["config"]),
        steps=tuple(_load_step(x) for x in p["steps"]), tool_results=tuple(_load_result(x) for x in p["tool_results"]),
        pending_approvals=tuple(PendingToolApproval(**x) for x in p["pending_approvals"]),
        preflight_outcomes=tuple(None if x is None else _load_outcome(x) for x in p["preflight_outcomes"]),
        requested_execution_mode=execution["requested_mode"], effective_execution_mode=execution["effective_mode"],
        fallback_reason=execution["fallback_reason"], parallel_safe_tool_count=int(execution["parallel_safe_tool_count"]),
        unsafe_tool_count=int(execution["unsafe_tool_count"]), total_requested=int(resources["total_requested"]),
        round_requested=tuple((int(a), int(b)) for a,b in resources["round_requested"]),
        tool_requested=tuple((str(a), int(b)) for a,b in resources["tool_requested"]),
        provider=str(route["provider"]), model=str(route["model"]), persistent=True, created_at=str(p["created_at"]),
    )


def _validate_checkpoint_state(c: AgentToolLoopCheckpoint) -> None:
    if not _ID.fullmatch(c.checkpoint_id): raise CheckpointError("invalid checkpoint ID")
    ids = [x.tool_call_id for x in c.pending_approvals]
    call_ids = [x.id for x in c.response.tool_calls]
    if not ids or len(ids) != len(set(ids)) or len(call_ids) != len(set(call_ids)) or not set(ids).issubset(call_ids):
        raise CheckpointError("pending tool-call IDs are invalid or inconsistent")
    if any(not _ID.fullmatch(x) for x in call_ids): raise CheckpointError("invalid tool-call ID")
    if len(c.preflight_outcomes) != len(c.response.tool_calls): raise CheckpointError("preflight state is inconsistent")
    if c.round_index < 0 or c.total_requested < 0: raise CheckpointError("checkpoint counters are invalid")


def _validate_runtime_compatibility(loop: AgentToolLoop, c: AgentToolLoopCheckpoint) -> None:
    if loop.config.tool_execution_mode != c.config.tool_execution_mode or loop.config.allow_parallel_tool_calls != c.config.allow_parallel_tool_calls:
        raise CheckpointError("current loop execution mode is incompatible with checkpoint")
    for stored in c.tool_definitions:
        current = loop.tool_runtime.registry.get(stored.name)
        if current is None: raise CheckpointError(f"required checkpoint tool is not registered: {stored.name}")
        live = current.to_llm_tool_definition()
        if _schema_fingerprint(live.parameters_schema) != _schema_fingerprint(stored.parameters_schema) or live.description != stored.description:
            raise CheckpointError(f"checkpoint tool definition changed: {stored.name}")
        if c.effective_execution_mode == "parallel" and not current.parallel_safe:
            raise CheckpointError(f"checkpoint tool is no longer parallel-safe: {stored.name}")
    runtime_provider = getattr(loop.llm_runtime, "provider", None)
    provider_name = getattr(runtime_provider, "name", None)
    default_model = getattr(runtime_provider, "default_model", None)
    if isinstance(provider_name, str) and provider_name and provider_name != c.provider:
        raise CheckpointError("current LLM provider is incompatible with checkpoint")
    if isinstance(default_model, str) and default_model and default_model != c.model:
        raise CheckpointError("current LLM model is incompatible with checkpoint")


def _message(x): return {"role":x.role,"content":x.content}
def _load_message(x): return LLMMessage(x["role"],x["content"])
def _call(x): return {"id":x.id,"name":x.name,"arguments":x.arguments,"provider":x.provider,"model":x.model}
def _load_call(x): return LLMToolCall(x["id"],x["name"],x["arguments"],x.get("provider",""),x.get("model",""))
def _usage(x): return None if x is None else asdict(x)
def _response(x): return {"content":x.content,"model":x.model,"provider":x.provider,"usage":_usage(x.usage),"tool_calls":[_call(y) for y in x.tool_calls]}
def _load_response(x): return LLMResponse(x["content"],x["model"],x["provider"],None if x.get("usage") is None else LLMUsage(**x["usage"]),{},tuple(_load_call(y) for y in x["tool_calls"]))
def _tool_definition(x): return {"name":x.name,"description":x.description,"parameters_schema":x.parameters_schema}
def _load_definition(x): return LLMToolDefinition(x["name"],x["description"],x["parameters_schema"])
def _result(x): return asdict(x)
def _load_result(x): return LLMToolResult(**x)
def _step(x): return {"index":x.index,"kind":x.kind,"response":None if x.response is None else _response(x.response),"tool_calls":[_call(y) for y in x.tool_calls],"tool_results":[_result(y) for y in x.tool_results],"success":x.success,"error_type":x.error_type,"error_category":x.error_category,"provider":x.provider,"model":x.model}
def _load_step(x): return AgentToolLoopStep(x["index"],x["kind"],None if x["response"] is None else _load_response(x["response"]),tuple(_load_call(y) for y in x["tool_calls"]),tuple(_load_result(y) for y in x["tool_results"]),x["success"],x["error_type"],x["error_category"],x["provider"],x["model"])
def _outcome(x): return {"llm_result":_result(x.llm_result),"success":x.success,"error_type":x.error_type,"error_category":x.error_category,"denied":x.denied,"resource_denied":x.resource_denied,"timed_out":x.timed_out,"limit_name":x.limit_name,"limit_value":x.limit_value}
def _load_outcome(x): return _ToolOutcome(_load_result(x["llm_result"]),x["success"],x["error_type"],x["error_category"],x["denied"],x["resource_denied"],x["timed_out"],x["limit_name"],x["limit_value"])


def _config(c):
    return {"max_steps":c.max_steps,"require_tool_approval":c.require_tool_approval,"stop_on_tool_error":c.stop_on_tool_error,"allow_parallel_tool_calls":c.allow_parallel_tool_calls,"tool_execution_mode":c.tool_execution_mode,"include_intermediate_steps":c.include_intermediate_steps,"tool_permission_policy":None if c.tool_permission_policy is None else {"allowed_tools":None if c.tool_permission_policy.allowed_tools is None else sorted(c.tool_permission_policy.allowed_tools),"denied_tools":None if c.tool_permission_policy.denied_tools is None else sorted(c.tool_permission_policy.denied_tools),"default_allow":c.tool_permission_policy.default_allow},"tool_resource_limits":None if c.tool_resource_limits is None else {**asdict(c.tool_resource_limits)}}
def _load_config(x):
    policy=x["tool_permission_policy"]; limits=x["tool_resource_limits"]
    return AgentToolLoopConfig(**{**x,"tool_permission_policy":None if policy is None else ToolPermissionPolicy(None if policy["allowed_tools"] is None else frozenset(policy["allowed_tools"]),None if policy["denied_tools"] is None else frozenset(policy["denied_tools"]),policy["default_allow"]),"tool_resource_limits":None if limits is None else ToolResourceLimits(**limits)})


def _schema_fingerprint(schema: Mapping[str, Any]) -> str: return hashlib.sha256(_canonical(schema)).hexdigest()
def _digest(payload: Mapping[str, Any]) -> str: return hashlib.sha256(_canonical(payload)).hexdigest()
def _canonical(value: Any) -> bytes: return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")
def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value,str): raise CheckpointError("checkpoint creation timestamp is invalid")
    try: parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError: raise CheckpointError("checkpoint creation timestamp is invalid") from None
    if parsed.tzinfo is None: raise CheckpointError("checkpoint creation timestamp must include a timezone")
    if parsed.timestamp() > datetime.now(timezone.utc).timestamp() + 1: raise CheckpointError("checkpoint creation timestamp is in the future")
    return parsed
def _consumed_path(path: Path) -> Path: return path.with_name(path.name+".consumed")
def _reject_secrets_and_non_json(value: Any, path: str="payload") -> None:
    if isinstance(value, Mapping):
        for key,item in value.items():
            if not isinstance(key,str): raise CheckpointError(f"non-string JSON key at {path}")
            if _SECRET_KEY.search(key): raise CheckpointError(f"refusing to serialize secret-like field: {key}")
            _reject_secrets_and_non_json(item,f"{path}.{key}")
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value): _reject_secrets_and_non_json(item,f"{path}[{index}]")
    elif isinstance(value, str) and _SECRET_VALUE.search(value):
        raise CheckpointError(f"refusing to serialize secret-like value at {path}")
    elif value is not None and not isinstance(value,(str,int,float,bool)):
        raise CheckpointError(f"value at {path} is not JSON-safe")
    try: _canonical(value)
    except (TypeError,ValueError): raise CheckpointError(f"value at {path} is not JSON-safe") from None


__all__ = ["CHECKPOINT_SCHEMA_VERSION","CheckpointError","CheckpointMetadata","save_checkpoint","load_checkpoint","inspect_checkpoint","resume_checkpoint"]
