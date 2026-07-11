# Small tool surfaces are easier to trust

An agent should receive only the tools needed for its current job. Registered
tools with schemas, bounded call counts, and safe event metadata make behavior
easier to inspect. Event records should describe tool names and argument keys,
not copy sensitive argument values.

Failures are normal runtime outcomes. A controlled failure can be returned to
the agent as structured tool feedback, allowing a bounded retry with a known
source instead of terminating the whole workflow.

