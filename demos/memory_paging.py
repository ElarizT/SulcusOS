from __future__ import annotations

from typing import Any

from kernel.events import RuntimeEvent


def build_demo_snapshot() -> dict[str, Any]:
    supervisor_pid = 300
    return {
        "status": "Memory Demo Complete",
        "process_rows": [
            {
                "pid": supervisor_pid,
                "name": "MemorySupervisor",
                "status": "running",
                "execution_mode": "demo",
                "child_count": 2,
                "restart_count": 0,
                "messages_sent": 0,
                "messages_received": 0,
                "message_errors": 0,
            },
            {
                "pid": 301,
                "name": "AgentA",
                "status": "running",
                "execution_mode": "demo",
                "supervisor_pid": supervisor_pid,
                "child_count": 0,
                "restart_count": 0,
                "memory_hot_tokens": 1,
                "memory_paged_count": 1,
                "messages_sent": 0,
                "messages_received": 0,
                "message_errors": 0,
            },
            {
                "pid": 302,
                "name": "AgentB",
                "status": "running",
                "execution_mode": "demo",
                "supervisor_pid": supervisor_pid,
                "child_count": 0,
                "restart_count": 0,
                "memory_hot_tokens": 2,
                "memory_paged_count": 0,
                "messages_sent": 0,
                "messages_received": 0,
                "message_errors": 0,
            },
        ],
        "hierarchy": {
            "supervisor": "MemorySupervisor",
            "children": ["AgentA", "AgentB"],
        },
        "page_tables": [
            {
                "agent": "Agent A",
                "pages": [
                    {"page": 0, "state": "active"},
                    {"page": 1, "state": "evicted"},
                ],
            },
            {
                "agent": "Agent B",
                "pages": [
                    {"page": 2, "state": "active"},
                    {"page": 3, "state": "active"},
                ],
            },
        ],
        "events": [
            RuntimeEvent.info(
                "MemorySupervisor", "page_allocated", "Allocated Page 0", {"page": 0, "agent": "AgentA"}
            ),
            RuntimeEvent.info(
                "MemorySupervisor", "page_allocated", "Allocated Page 1", {"page": 1, "agent": "AgentA"}
            ),
            RuntimeEvent.info(
                "MemorySupervisor", "page_allocated", "Allocated Page 2", {"page": 2, "agent": "AgentB"}
            ),
            RuntimeEvent.warning(
                "MemorySupervisor", "page_evicted", "Evicted Page 1", {"page": 1, "agent": "AgentA"}
            ),
            RuntimeEvent.info(
                "MemorySupervisor", "page_allocated", "Allocated Page 3", {"page": 3, "agent": "AgentB"}
            ),
        ],
    }
