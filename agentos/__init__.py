"""Stable public SDK for writing Agent OS processes."""

from agentos.loader import (
    AgentPermissions,
    ExternalAgentManifest,
    inspect_external_agent,
    load_external_agent,
)
from kernel.ipc_protocol import (
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
from kernel.process import AgentProcess, ExecutionMode, RestartPolicy, SupervisorStrategy
from kernel.native_core import (
    NativeCoreUnavailableError,
    NativeCoreImportError,
    RuntimeCapabilities,
    get_runtime_capabilities,
    native_core_available,
)

__all__ = [
    "AgentProcess",
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
    "TaskRequest",
    "TaskResponse",
    "make_error",
    "make_message",
    "inspect_external_agent",
    "load_external_agent",
    "parse_message",
]
