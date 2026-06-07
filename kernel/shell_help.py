from __future__ import annotations

from pathlib import Path

DEMO_COMMANDS = frozenset({"demo", "demos"})
SUPERVISOR_RECOVERY_DEMO_PATH = "demos/supervisor_recovery"
MEMORY_PAGING_DEMO_PATH = "demos/memory_paging"


def is_supervisor_recovery_demo_path(raw_path: str) -> bool:
    return raw_path.replace("\\", "/").rstrip("/") == SUPERVISOR_RECOVERY_DEMO_PATH


def is_memory_paging_demo_path(raw_path: str) -> bool:
    return raw_path.replace("\\", "/").rstrip("/") == MEMORY_PAGING_DEMO_PATH


def format_demo_browser() -> str:
    return (
        "Available Demos\n"
        "\n"
        "research_team\n"
        "  Multi-agent research workflow with planner, researchers, synthesizer, and critic.\n"
        "  Run: run examples/research_team\n"
        "\n"
        "supervisor_recovery\n"
        "  Demonstrates supervised child termination detection and automatic restart.\n"
        f"  Run: run {SUPERVISOR_RECOVERY_DEMO_PATH}\n"
        "\n"
        "memory_paging\n"
        "  Demonstrates page allocation, page eviction, and context visualization.\n"
        f"  Run: run {MEMORY_PAGING_DEMO_PATH}"
    )


def format_shell_help(process_root: Path) -> str:
    return (
        "commands:\n"
        "  run <path>   start an AgentProcess script under "
        f"{process_root}\n"
        "  ps           list process registry status\n"
        "  kill <PID>   gracefully stop and unregister a process\n"
        "  demos        list available built-in demos\n"
        "  help         show this quick reference\n"
        "\n"
        "examples:\n"
        "  run examples/hello_agent.py\n"
        "  run examples/memory_agent.py\n"
        "  run examples/supervisor_quickstart.py\n"
        "  run examples/research_team\n"
        f"  run {SUPERVISOR_RECOVERY_DEMO_PATH}\n"
        f"  run {MEMORY_PAGING_DEMO_PATH}\n"
        "\n"
        "execution mode:\n"
        "  AGENT_OS_PROCESS_ISOLATION=in-process  trusted local mode (default)\n"
        "  AGENT_OS_PROCESS_ISOLATION=process     spawned child process isolation\n"
        "\n"
        "SDK guide: docs/sdk_quickstart.md"
    )
