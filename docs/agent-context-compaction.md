# Agent Context Inspection and Compaction

ARK exposes provider-neutral context inspection and between-turn compaction
through `AgentService`. The first provider implementation is Codex; other
providers can implement the same optional capability without exposing their
native event or storage formats to applications.

## Public API

```python
from agent_runtime_kit.agent import AgentContextMaintenancePolicy

usage = agent_service.inspect_agent_context(agent_id)

result = agent_service.compact_agent_if_needed(
    agent_id,
    threshold=0.80,
    timeout_s=120,
)

forced = agent_service.compact_agent(agent_id, timeout_s=120)
```

`AgentContextUsage` reports the current provider context size and model context
window when both are available. `usage_ratio` is derived from those values. It
is distinct from cumulative billable token usage.

Conditional compaction returns a typed skipped result for a new session,
unavailable usage, a value below the threshold, or an unsupported provider.
A forced compaction reports an unsupported provider as an error.

## First-turn admission

`AgentService.start_agent()` accepts an optional
`context_maintenance_policy`. ARK admits the Agent as active first, performs
the inspection and any required compaction in the same worker, and starts the
provider turn only after compaction reaches a confirmed terminal state.

`AgentStep.prepare_agent_context_before_first_turn()` is a no-op extension
point. An application can override it and return an
`AgentContextMaintenancePolicy`. The policy is passed only to the first
`start_agent()` call in that Step run; AgentService auto-continue turns and
AgentStep retry turns do not compact again.

## Failure and recovery semantics

Compaction request acceptance is not treated as completion. The Codex adapter
requires new rollout evidence after a captured baseline and verifies that the
thread is idle before returning success.

ARK persists a `context_maintenance.json` journal beside the Agent truth. If a
request may have started but its terminal state is unknown, later Agent starts
fail closed with `AgentContextMaintenanceBlocked`. Operators can call
`reconcile_agent_context_maintenance()`; the block is removed only when the
provider confirms completion from persisted evidence.

The journal belongs to scope truth, so scope and runtime snapshots preserve the
same fail-closed state. Active manual or automatic compaction also participates
in the existing running-Agent snapshot barrier.

## Provider contract

A provider may implement these optional methods:

- `inspect_thread_context(...) -> ProviderContextUsage`
- `compact_thread(...) -> ProviderContextCompactionResult`
- `reconcile_thread_compaction(...) -> ProviderContextCompactionResult | None`

`compact_thread()` must wait for provider-specific completion and idle
evidence. Returning immediately after a native “start compaction” response does
not satisfy the ARK contract.
