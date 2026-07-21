# Agent Context Inspection and Compaction

ARK exposes provider-neutral context inspection and between-turn compaction
through `AgentService`. Codex and Claude Code implement this optional
capability without exposing their native event or storage formats to
applications.

## Public API

```python
from agent_runtime_kit.agent import AgentContextMaintenancePolicy

usage = agent_service.inspect_agent_context(agent_id)
expanded_usage = agent_service.inspect_agent_context_result(agent_id)

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
thread is idle before returning success. The Claude Code adapter requires a
terminal SDK `ResultMessage` plus a new persisted `compact_boundary`. A
successful Claude slash command that does not create a boundary—for example,
when the context is too small to compact—is not reported as completed
compaction.

ARK persists a `context_maintenance.json` journal beside the Agent truth. If a
request may have started but its terminal state is unknown, later Agent starts
fail closed with `AgentContextMaintenanceBlocked`. Operators can call
`reconcile_agent_context_maintenance()`; the block is removed only when the
provider confirms completion from persisted evidence.

The journal belongs to scope truth, so scope and runtime snapshots preserve the
same fail-closed state. Active manual or automatic compaction also participates
in the existing running-Agent snapshot barrier.

`inspect_agent_context()` retains the compact legacy projection used by
existing applications. `inspect_agent_context_result()` returns the expanded
provider-neutral model, including optional effective windows, remaining and
reserved tokens, categories, measurement quality, model identity, staleness,
and sanitized provider payload.

## Provider contract

A provider bundle exposes a `ProviderContextAdapter` with:

- `inspect(ProviderContextQuery) -> ProviderContextUsage`
- `compact(ProviderContextCompactionRequest) -> ProviderContextCompactionResult`
- `reconcile(ProviderContextReconcileRequest) -> ProviderContextCompactionResult | None`

`compact()` must wait for provider-specific completion and idle
evidence. Returning immediately after a native “start compaction” response does
not satisfy the ARK contract. The older `inspect_thread_context()`,
`compact_thread()`, and `reconcile_thread_compaction()` shapes are retained only
as a compatibility bridge for existing injected providers.
