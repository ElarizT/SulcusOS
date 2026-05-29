# Persistent Memory Paging

Agent OS memory now has three tiers:

- Hot memory: active in-process context within the token budget.
- Warm memory: recently evicted records retained locally in memory.
- Cold memory: JSONL-backed records persisted on disk.

The default store path is configurable:

```powershell
$env:AGENT_OS_MEMORY_DIR = ".agent_os/memory"
```

The v0.1 backend is intentionally simple and inspectable:

- `memories.jsonl` stores cold paged records.
- `snapshots/*.json` stores process memory snapshots.
- No vector DB or embeddings are required.

## SDK API

Agents can use:

```python
self.remember({"fact": "Agent OS has structured IPC"}, importance=0.8, tags=["ipc"])
records = self.recall(query="structured", tags=["ipc"], limit=5)
self.forget(records[0]["memory_id"])
stats = self.memory_stats()
```

The retrieval policy is deterministic:

- tag filter
- substring filter
- importance descending
- recency descending

## Snapshots

The process registry exposes:

```python
snapshot_id = registry.snapshot_process(pid)
registry.restore_process_memory(pid, snapshot_id)
```

Snapshots include hot memory records, warm/cold references, token usage,
timestamp, PID, and process name metadata.

## Supervision

Supervised agents default to no memory restoration. Agents may opt in with:

```python
class Worker(AgentProcess):
    memory_restore_policy = "latest_snapshot"
```

Supported policies are:

- `none`
- `hot_only`
- `latest_snapshot`
- `persistent_recall`

## Events

Memory lifecycle events use structured IPC event messages when a parent process
exists:

- `memory_recorded`
- `memory_evicted`
- `memory_recalled`
- `memory_forgotten`
- `memory_snapshot_created`
- `memory_restored`
