# Home Model Overrides and Scheduler Run Leases

## Provider-neutral Home model settings

`ProviderHomeSpec` carries provider-neutral backend identity and explicit
provider configuration overrides:

```python
from agent_runtime_kit.agent.provider_contracts import (
    BaseConfigSource,
    ModelBackendIdentity,
    ProviderHomeSpec,
)
from agent_runtime_kit.agent.providers.codex_home import CodexHomeOptions

spec = ProviderHomeSpec(
    provider_type="codex",
    home_id="PlanningAgent",
    base_config=BaseConfigSource(path=str(base_config)),
    config_overrides={
        "model": "gpt-5.6-sol",
        "model_reasoning_effort": "high",
    },
    model_config=ModelBackendIdentity(
        api_provider="openai",
        api_mode="responses",
        requested_model="gpt-5.6-sol",
        reasoning_effort="high",
    ),
    provider_options=CodexHomeOptions(auth_json_path=auth_json),
)
```

`model_config` describes backend/API-mode/model identity without changing the
Provider type. `config_overrides` is interpreted by the selected Home renderer;
the Codex renderer projects these keys into top-level TOML. Typed credentials,
skills, and other native resources live in `CodexHomeOptions` rather than in
the common record.

Every provider validates its supported overrides and fails rather than silently
ignoring an unsupported resource.

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
