# Agent OS IPC Protocol

Agent OS IPC protocol version `0.1` wraps the existing mailbox transport with
structured, validated agent-to-agent messages. The shell lifecycle commands
(`run`, `ps`, `kill`) still operate on process records and mailboxes; IPC uses
those records to route by PID.

## Message Types

All messages include:

- `protocol_version`
- `message_id`
- `correlation_id`
- `source_pid`
- `target_pid`
- `message_type`
- `timestamp`
- `priority`
- `payload`
- `expires_at`

Supported `message_type` values are:

- `task_request`
- `task_response`
- `event`
- `error`
- `heartbeat`
- `control`

Priorities are `low`, `normal`, `high`, and `critical`.

## SDK API

Agent authors use `AgentProcess` helpers:

```python
self.send(target_pid, {"cmd": "ping"})
message = await self.receive(timeout=1.0)
response = await self.request(target_pid, {"ping": True}, timeout=2.0)
self.reply(request_message, {"pong": True})
self.emit("ready", {"ok": True})
```

`request()` and `reply()` preserve `correlation_id` so callers can match
responses. Timeout and routing failures return structured `ErrorMessage`
objects where the SDK can create one locally.

## Validation

The protocol rejects:

- invalid source or target PIDs
- unsupported message types
- non-serializable payloads
- oversized messages above 64 KiB
- unsupported priorities
- missing correlation IDs for request/response flows
- expired messages

Protocol errors use these codes:

- `target_not_found`
- `mailbox_full`
- `timeout`
- `invalid_message`
- `process_dead`
- `payload_too_large`

## Observability

`ps` and the dashboard process table include IPC counters as `sent/received/errors`.
The registry also preserves mailbox depth and capacity so existing queue health
signals remain visible.

## Compatibility

The structured protocol is transported through the existing mailbox internals as
JSON payloads. Trusted in-process agents route directly through the registry.
Isolated process agents use Windows-safe `multiprocessing.Queue` bridges owned by
the parent registry, avoiding fork-only assumptions.
