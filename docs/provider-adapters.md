# Provider Adapters and the Normalized Agent Runtime

ARK separates an application's Agent role from the harness that executes it.
`AgentType` owns prompts and completion policy. A provider bundle owns how a
Codex-, CLI-, subprocess-, or library-backed agent is configured, run,
queried, controlled, and snapshotted.

The bundled implementations are `codex` and `claude_code`. Additional
providers can be registered without changing Flow, Step, or snapshot
orchestration.

## Public Contract Namespace

Provider-neutral contracts are exported from:

```python
from agent_runtime_kit.agent.provider_contracts import (
    AgentProviderBundle,
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderRegistry,
    ProviderRunRequest,
)
```

The namespace contains:

- provider, backend/API-mode, model, session, turn, and artifact identities;
- Home specifications, materialization manifests, and execution contexts;
- runtime, query, context, and artifact protocols;
- normalized result, event, content, tool-call, usage, context, error, control,
  fork, and pagination models;
- payload sanitization and optional token-estimation/pricing protocols.

Unknown usage values remain `None`; ARK does not convert missing values to
zero. Cost is stored only when reported by the provider. A future pricing
resolver or tokenizer may provide explicitly marked estimates, but estimates
are not provider-reported truth.

## Provider Bundle

An `AgentProviderBundle` groups one descriptor with these extension points:

- `home_renderer`: validates and materializes provider resources, then builds
  a per-run `ProviderExecutionContext`;
- `runtime`: starts/resumes/forks sessions and returns a live
  `ProviderRunHandle` with wait, event, control, and terminal semantics;
- `query`: reconstructs provider-neutral sessions, turns, events, tool calls,
  and usage from provider-native evidence;
- `context`: inspects current context pressure and performs/reconciles compact
  operations when supported;
- `artifacts`: declares stable authoritative artifacts and owns capture,
  restore, and rebuildable-cache cleanup;
- `capability_resolver`: optionally resolves support from the effective Home,
  backend/API mode, and model instead of relying on static provider support.

Adapters convert native SDK or subprocess values at their boundary. Raw native
data may be retained in a bounded, secret-sanitized `ProviderPayload`, but it
is not the primary application contract.

## Capability Rules

Callers must treat capability resolution as authoritative. Unsupported or
unknown operations fail closed; ARK does not silently replace provider compact
with an application-owned summary, claim complete usage when only partial
usage is available, or treat a session fork as workspace isolation.

ARK's common fork meaning is:

```text
fork_mode = session_only
workspace_isolated = false
```

The provider creates an independently resumable conversation branch. Git
worktrees, file rollback, and workspace checkpoints remain application
responsibilities.

## Persistent Records and Snapshots

Agent record schema v2 stores `provider_type`, session/latest-turn/artifact
locators, and explicit fork information. During migration, Codex records also
write `cli_type`, `thread_id`, and `rollout_relpath`. Existing schema-v1 records
and scope snapshots remain readable.

`AgentSnapshotService` owns scope/runtime pausing, stable-point coordination,
archive integrity, and index rebuilding. It delegates every provider-specific
file decision to the bundle's Artifact adapter. The Codex adapter captures the
single-session rollout JSONL as authoritative resume evidence and discards its
rebuildable `state_5.sqlite*` cache during restore. The Claude Code adapter
captures one native session JSONL and records the matching Home
materialization manifest as a required external dependency.

## Claude Code Adapter

`ClaudeCodeProvider` uses `claude-agent-sdk==0.2.124` to control the Claude Code
CLI. Each run handle owns one thread, asyncio loop, and SDK client so interrupt
and terminal delivery remain on the client's native loop. The adapter exposes
the same normalized run, query, context, usage, fork, and artifact contracts as
Codex without treating the configured backend or model as the provider type.

Claude Code fork is session-only. File checkpointing is rejected by the first
adapter version because Claude's file-history artifacts are not yet included
in the Artifact Manifest. Context inspection and compact require a verified
CLI version with the context-control protocol; compact success additionally
requires a new persisted `compact_boundary` after the captured baseline.

See [Claude Code provider](claude-code-provider.md) for setup and operational
limits.

## Codex Compatibility

Existing application calls such as `wait_agent()`, legacy trace readers, and
Codex-shaped completion checkers keep their current behavior. New integrations
can use `wait_agent_result()`, `query_*()`, and
`inspect_agent_context_result()` for normalized values. Compatibility paths
are explicitly marked in source and have migration tests; they are not the
template for new provider implementations.
