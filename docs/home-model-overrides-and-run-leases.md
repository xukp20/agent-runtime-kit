# Home Model Overrides and Scheduler Run Leases

## Provider-neutral Home model settings

`HomeCreateSpec.model_config_overrides` carries the two model settings that an
embedding application commonly varies by Agent type:

```python
from agent_runtime_kit.agent.homes import HomeCreateSpec, ModelConfigOverrides

spec = HomeCreateSpec(
    cli_type="codex",
    home_id="PlanningAgent",
    base_config_path=base_config,
    model_config_overrides=ModelConfigOverrides(
        model="gpt-5.6-sol",
        reasoning_effort="high",
    ),
)
```

The fields are provider-neutral. A provider-specific Home renderer projects
them into its own configuration. The Codex renderer writes the top-level TOML
keys `model` and `model_reasoning_effort`, replaces an existing value exactly
once, preserves an unspecified base value, and remains idempotent across Home
refreshes. Providers without a renderer for these settings reject the override
instead of silently ignoring it.

This surface intentionally does not accept an arbitrary provider configuration
dictionary. Provider credentials, MCP servers, and other Home configuration
continue to use their existing typed inputs.

## Process-local semantic scheduler leases

`RuntimeScheduleService.configure_semantic_run(...)` creates a process-local
lease. Its `lease_id` is also present in `SchedulerRunControlView`. Applications
can observe it without changing scheduling semantics:

```python
lease = scheduler.get_run_lease(lease_id)
waited = scheduler.wait_run_lease(
    lease_id,
    after_version=lease.version,
    timeout_s=30,
)
```

`SchedulerRunLeaseView` records the policy, lifecycle status, monotonic version,
timestamps, completed action counts, action identifiers, terminal reason, and
the matching run-control view. Waiters use an internal condition and return
when the version changes, the lease becomes terminal, or the bounded timeout
expires. Multiple waiters may observe the same transition.

Lease state is deliberately not part of runtime snapshots. After a process
restart, the embedding application must treat an unknown old lease identifier
as lost, inspect current runtime truth, and start a new run plan if appropriate.
The application remains responsible for exposing HTTP or CLI read/wait APIs and
for deriving business-specific Flow, Step, Agent, or checkpoint locators.
