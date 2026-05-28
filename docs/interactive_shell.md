# Interactive Shell MVP

Agent OS boots with only the control-plane mailboxes by default. Standalone
agent processes are loaded from the dashboard prompt.

## Commands

- `run <path-to-agent.py>` starts an `AgentProcess` script and assigns a PID.
- `ps` lists PID, name, lifecycle status, uptime, memory tokens, and mailbox usage.
- `kill <PID>` gracefully stops the process and unregisters its mailbox, memory
  table, and kernel capabilities.
- `help` prints the command summary in the dashboard log.

`run` is confined to `AGENT_OS_PROCESS_ROOT`, which defaults to the current
workspace. Paths outside that root are rejected after resolution, so `..`
traversal and arbitrary absolute paths are not accepted.

## Execution Modes

The default mode is trusted local development:

```powershell
$env:AGENT_OS_PROCESS_ISOLATION = "in-process"
```

To load dynamic agents in separate spawned child processes:

```powershell
$env:AGENT_OS_PROCESS_ISOLATION = "process"
```

Process mode uses Windows-safe `multiprocessing` spawn semantics, no
`shell=True`, a startup readiness handshake, and a timeout controlled by
`AGENT_OS_PROCESS_STARTUP_TIMEOUT` (default `5.0` seconds). This is process
isolation, not a full security sandbox.

## Agent Script Shape

Scripts should define an `AgentProcess` subclass:

```python
from kernel.process import AgentProcess


class EchoAgent(AgentProcess):
    name = "EchoAgent"
    capabilities = ("echo",)

    async def on_message(self, message):
        self.remember({"payload": message.payload}, 3)
```

For this MVP, imports are intentionally restricted. The preflight check only
allows `from kernel.process import AgentProcess` before executing the script.
