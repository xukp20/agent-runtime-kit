# Provider Adapters and the Normalized Agent Runtime

ARK separates an application's Agent role from the harness that executes it.
`AgentType` owns prompts and completion policy. A provider bundle owns how a
Codex-, CLI-, subprocess-, or library-backed agent is configured, run,
queried, controlled, and snapshotted.

The bundled adapters are `codex`, `claude_code`, `pi`, `openai_agents`, and
`opencode`. Additional providers can be registered without
changing Flow, Step, or snapshot orchestration.

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

`AgentType.provider_type` and `AgentType.default_home_id` declare the normal
execution binding. `AgentStepState.provider_type` is an optional per-Step
override; when omitted, `AgentService` resolves the Provider and Home from
`AgentType`.

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

Agent record schema v3 stores `provider_type`, exact session/latest-turn/
artifact locators, and explicit fork information. ARK 0.3 reads and writes
schema v3 only. Pre-v3 Agent, Home, and snapshot records must be migrated by an
external one-time tool before this runtime is started; the runtime never
guesses missing provider identity or artifact ownership.

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

The Pi adapter captures one idle Pi v3 session JSONL. Its manifest also records
the hash of the ARK Home materialization manifest as a restore-time reference;
the Home is validated but not duplicated into every session snapshot. Pi
snapshot and fork operations do not capture or roll back workspace files.

## Standard AgentService Results

`wait_agent()` returns `AgentTurnResult`. Completion checkers receive that same
normalized object. Applications inspect `provider_result`, `session_locator`,
`turn_locator`, `final_text`, and `query_*()` results without depending on a
provider SDK object. Native evidence is retained only in sanitized
`ProviderPayload` or provider-private locators.

## Pi Adapter

Pi is integrated through its LF-delimited subprocess RPC protocol and native
v3 session JSONL. The adapter supplies Home, runtime, query, context, artifact,
snapshot, and dynamic capability implementations. Its compact operation is Pi
agent-owned history summarization and is independent of whether the selected
model backend uses Responses, Chat Completions, or Messages. MCP is projected
through an ARK-owned Pi extension rather than claimed as a Pi-native feature.

See [Pi provider](pi-provider.md) for configuration and exact limitations.

## OpenAI Agents Adapter

The OpenAI Agents adapter uses an application-owned resource registry for
non-serializable Agent factories and tools. Homes persist the factory
reference, backend/API-mode identity, MCP and skill resources, and SQLite
session policy without serializing Python callables or credentials. Responses
and Chat Completions are backend modes rather than distinct Provider types.

Responses Homes may opt into SDK input-history compaction. Chat Completions
Homes report compact as unsupported unless a separately designed compaction
strategy is configured; ARK does not silently substitute a summarizer. See
[OpenAI Agents provider](openai-agents-provider.md) for assembly and limits.

The low-level OpenAI Agents and OpenCode run handles can represent provider
approval/input boundaries, but ARK 0.3 does not expose a complete
`AgentService`/Flow `NEEDS_INPUT` lifecycle. Applications using those layers
must configure non-interactive operation; direct handle controls remain an
extension point for a later common lifecycle.

## OpenCode Adapter

Create the OpenCode bundle explicitly:

```python
from pathlib import Path

from agent_runtime_kit.agent.provider_contracts import ProviderRegistry
from agent_runtime_kit.agent.providers import build_opencode_provider_bundle

runtime_root = Path(".agent_runtime")
opencode = build_opencode_provider_bundle(
    runtime_root=runtime_root,
    binary_path="opencode",  # pinned and managed by the embedding application
)
registry = ProviderRegistry((opencode,))
```

The Home renderer writes `opencode.json`, `AGENTS.md`, skills, and MCP entries.
Sensitive config values must use OpenCode `{env:NAME}` references; inline API
keys and authorization values are rejected. Project-local config discovery,
OpenCode workspace snapshots, sharing, and automatic updates are disabled by
default.

Each Agent receives its own server process, XDG directories, and
`OPENCODE_DB`. The adapter subscribes to SSE before submitting a prompt and
does not report completion until it has both persisted assistant completion
evidence and a live idle status. Query results normalize OpenCode messages,
parts, tools, request usage, model identity, and provider-reported cost.

OpenCode `summarize` is exposed as model-backed OpenCode compaction. It is not
OpenAI Responses native compaction. Fork creates a session-only branch and
copies the source SQLite database into the target Agent runtime; it does not
isolate the workspace. Snapshot uses SQLite online backup and includes
referenced tool-output data, while credentials, logs, caches, and workspace
files remain excluded. Restore currently requires the same runtime root and
Agent path so OpenCode's absolute tool-output references remain valid.

The first adapter version queries through a live isolated OpenCode server and
does not claim offline artifact query or in-flight permission/question
snapshot support. Capability resolution must be checked for the effective
model backend and API mode.
