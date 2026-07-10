"""Stable public Python API for Sulcus OS (imported as ``agentos``)."""

from agentos.loader import (
    AgentPermissions,
    ExternalAgentManifest,
    inspect_external_agent,
    load_external_agent,
)
from agentos._version import __version__
from agentos.ipc import (
    ControlMessage,
    ErrorMessage,
    EventMessage,
    HeartbeatMessage,
    IPCMessage,
    IPCProtocolError,
    TaskRequest,
    TaskResponse,
    make_error,
    make_message,
    parse_message,
)
from agentos.native import (
    NativeCoreUnavailableError,
    NativeCoreImportError,
    RuntimeCapabilities,
    get_runtime_capabilities,
    native_core_available,
    require_native_core,
)
from agentos.runtime import (
    AgentToolLoop,
    AgentToolLoopCheckpoint,
    AgentToolLoopConfig,
    AgentToolLoopResult,
    PendingToolApproval,
    ToolApprovalDecision,
    ToolPermissionPolicy,
    ToolResourceLimits,
)
from agentos.tools import ToolRegistry, ToolRuntime
from kernel.process import AgentProcess, ExecutionMode, RestartPolicy, SupervisorStrategy

__all__ = [
    "AgentProcess",
    "AgentToolLoop",
    "AgentToolLoopCheckpoint",
    "AgentToolLoopConfig",
    "AgentToolLoopResult",
    "AgentPermissions",
    "ControlMessage",
    "ErrorMessage",
    "EventMessage",
    "ExecutionMode",
    "ExternalAgentManifest",
    "HeartbeatMessage",
    "IPCMessage",
    "IPCProtocolError",
    "RestartPolicy",
    "SupervisorStrategy",
    "NativeCoreUnavailableError",
    "NativeCoreImportError",
    "RuntimeCapabilities",
    "get_runtime_capabilities",
    "native_core_available",
    "require_native_core",
    "TaskRequest",
    "TaskResponse",
    "PendingToolApproval",
    "ToolApprovalDecision",
    "ToolPermissionPolicy",
    "ToolRegistry",
    "ToolResourceLimits",
    "ToolRuntime",
    "__version__",
    "make_error",
    "make_message",
    "inspect_external_agent",
    "load_external_agent",
    "parse_message",
]
