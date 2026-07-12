# Persistent AgentToolLoop checkpoints

Checkpoint persistence is optional. Normal loop runs and in-memory resumes do
not serialize JSON.

```python
from agentos.checkpoints import save_checkpoint, load_checkpoint, resume_checkpoint
from agentos.runtime import ToolApprovalDecision

save_checkpoint(paused.checkpoint, "approval.checkpoint.json")
checkpoint = load_checkpoint("approval.checkpoint.json")  # inspectable data only
result = resume_checkpoint(
    newly_constructed_loop,
    "approval.checkpoint.json",
    [ToolApprovalDecision("call-1", approved=True)],
)
```

The caller supplies the current `AgentToolLoop`, LLM runtime, `ToolRuntime`,
registry, and registered functions. The stored response is reused, so resume
does not repeat the provider request that produced the tool calls.

## Schema and versioning

Version 1 is deterministic UTF-8 JSON. Its envelope has a format identifier,
schema version, payload, and SHA-256 digest of canonical payload JSON. The
payload holds the checkpoint ID and UTC creation time; message history and
pending response; pending calls/approvals; steps and prior results; round and
resource counters; execution/safety configuration; provider/model labels; and
tool names, definitions, and canonical JSON-schema fingerprints. Unknown
versions are rejected. The digest detects corruption but is not an
authentication signature.

No callables, runtimes, providers, API keys, arbitrary objects, raw exceptions,
stack traces, or approval comments are serialized. Runtime/message metadata is
omitted. Secret-like mapping keys are refused rather than persisted.

## Compatibility, staleness, and consumption

Load validates the envelope, version, digest, timestamp, IDs, counters, and
pending-state consistency. Applications can pass `max_age_seconds=` to load,
inspect, or resume. Resume requires every tool to remain registered under its
stable name with the same description and schema fingerprint. Execution mode
and parallel safety must remain compatible. Callable identity and module paths
are not compatibility keys.

Saving uses a temporary owner-restricted file, flush, and atomic replace where
the platform supports them. A complete decision set atomically renames the file
to `<name>.consumed` before tool execution, preventing accidental reuse. A
validation failure restores it; partial decisions leave it untouched and do not
execute tools. Restrictive permissions are best effort and depend on the host
filesystem.

## CLI

```text
sulcus checkpoint inspect approval.checkpoint.json
sulcus checkpoint resume approval.checkpoint.json \
  --runtime-factory my_app.runtime:make_loop \
  --approve call-1 --deny call-2
```

The explicit zero-argument factory constructs the caller's current loop and is
never stored in the checkpoint. Inspection is sanitized: it shows IDs, tool
names, fingerprints, route labels, and execution metadata, but no argument
values or message content. Expected failures have concise errors without
tracebacks. Incomplete decisions return status 3 and preserve the file.

## Privacy and limitations

Files contain user/model message content, tool argument values, and previous
tool-result content needed for resume. Protect them as sensitive application
data. Version 1 has no encryption, signature, automatic migration, distributed
locking, or cross-host consumed ledger. The caller must reconstruct a compatible
runtime and choose an appropriate staleness policy.
