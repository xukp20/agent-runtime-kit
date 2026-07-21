# Runtime Observation

ARK provides process-local, observation-only wait primitives for applications
that expose production monitoring APIs. These methods do not start work,
advance a Flow, consume a submission, bypass pause control, or mutate business
truth.

## Step terminal wait

`StepService.wait_step_terminal(step_id, timeout_s=...)` returns a
`StepTerminalWaitResult` containing the latest typed Step and:

- `terminal`: the Step is terminal and its worker has settled;
- `timed_out`: the requested observation window expired;
- `runner_state`: `active`, `not_started`, `lost`, or `settled`;
- `observed_at` and an optional warning.

A completed or failed Step with no active worker returns immediately as
`settled`. A created Step may wait for scheduler admission. A running Step with
an active worker waits for terminal handling and worker cleanup. Persisted
`running` truth without a process-local runner returns `lost` immediately; ARK
does not pretend that work survived a process restart.

The settled notification occurs after terminal-receipt validation and
`FlowService.handle_step_terminal()`. Application observers can therefore read
the persisted Step and the Flow's terminal consumption after a successful
wait. Multiple waiters are supported by a service-level condition.

## Agent status wait

`AgentService.wait_agent_status_change(agent_id, after_status=...,
timeout_s=...)` returns an `AgentStatusWaitResult` with the latest Agent,
`changed`, `timed_out`, and `observed_at`.

Unlike `wait_agent()`, this method does not interpret
completion-check failures or incomplete turns as exceptions. It observes only
the Agent lifecycle status. Agent start, terminal cleanup, synchronous context
maintenance, close, and explicit stale-record repair notify the shared status
condition.

## Application adapters

ARK does not provide an HTTP server. An embedding application may expose these
methods through bounded read-only endpoints. Synchronous waits must be moved
off an asynchronous HTTP event loop, for example with `asyncio.to_thread`, and
the client transport timeout must exceed the server-side wait window.

Provider rollout activity remains provider-native evidence. Step terminal wait
is the completion boundary; Agent status is a lifecycle signal and must not
replace application business gates.
