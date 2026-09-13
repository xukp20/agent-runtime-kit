# Provider Adapters and the Normalized Agent Runtime

ARK separates an application's Agent role from the harness that executes it.
`AgentType` owns prompts and completion policy. A provider bundle owns how a
Codex-, CLI-, subprocess-, or library-backed agent is configured, run,
queried, controlled, and snapshotted.

The bundled adapters are `codex`, `claude_code`, `pi`, `grok`, `openai_agents`,
and `opencode`. Additional providers can be registered without
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

Home initialization is fail-closed around the materialization manifest. ARK
validates the existing manifest before invoking a provider initializer. If the
initializer reports `materialization_changed`, ARK explicitly reseals the Home
before constructing the run execution context. This permits declared,
one-time provider initialization changes while later unsealed changes to
provider-managed files still fail hash validation.

A new-session runtime may also declare a trusted Home commit after the native
session exists but before its first turn starts. The Codex adapter uses this
boundary only for an SDK rewrite of `.codex/config.toml`: it diffs every
managed file, performs no write when nothing changed, and rejects any changed
path outside that declaration. Resume runs never receive this commit. Changes
made after the session-start callback, during a turn, or after terminal remain
unsealed and fail the next execution-context validation.
`ProviderRunRequest.session_start_home_commit` is the explicit callback for
this boundary; a supporting runtime must invoke it at most once and before it
starts the first turn or exposes any Agent tool execution.

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

`ClaudeCodeProvider` uses `claude-agent-sdk==0.2.152` to control the Claude Code
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

## Grok Adapter

The Grok adapter targets the pinned Grok Build CLI 1.0.30 binary and its ACP
v1 stdio protocol. Register it explicitly:

```python
from pathlib import Path

from agent_runtime_kit.agent.provider_contracts import ProviderRegistry
from agent_runtime_kit.agent.providers import build_grok_provider_bundle

runtime_root = Path(".agent_runtime")
grok = build_grok_provider_bundle(
    runtime_root=runtime_root,
    binary_path="/root/.grok/bin/grok",
)
registry = ProviderRegistry((grok,))
```

`GrokHomeOptions` selects the native auth reference, pinned binary, model,
reasoning effort, and curated tool set. The default auth reference is
`/root/.grok/auth.json`; ARK creates a symlink from the isolated native Home
and never copies, prints, or modifies the source secret. Both `HOME` and
`GROK_HOME` point inside the managed ARK Home and automatic updates are
disabled. Initialization verifies the 1.0.30 binary hash.

Tool selection is fail-closed. `GrokHomeOptions.tools=None` uses non-empty
string tools from `ProviderHomeSpec.tools`, or defaults to the read-only
`read_file`, `list_dir`, and `grep` set. An explicitly empty tuple is rejected.
Declaring tools in both places is ambiguous and rejected. The additional
verified tools are `run_terminal_cmd` and `search_replace`. A declared tool is
preauthorized only for its corresponding read/search, execute, or edit
permission kind; unknown permission kinds and incoming interactive requests
are denied.

Home instructions are written into the isolated agent profile. Per-run system
and developer instructions are added to the ACP prompt. Unmanaged and
inherited skills, native subagents, background work, plugins, extensions, raw
config overrides, and project executable configuration are not supported.
Before launch, the adapter scans the canonical cwd through its Git worktree
root (or all filesystem ancestors when no reliable Git boundary exists) and
rejects known Grok, MCP, Claude, Cursor, hook, plugin, and env configuration
markers.

Managed skills use explicit `SkillSpec` objects in `ProviderHomeSpec.skills`.
ARK writes their `SKILL.md` and resource files into the sealed Home and selects
their canonical names in the native profile. Skill names must use lowercase
letters, digits and hyphens. Skill files do not grant additional tool permissions.
When managed skills are enabled, project `.grok/skills`, `.grok/commands`,
`.agents/skills` and `.agents/commands` directories are rejected to prevent
project discovery from overriding the declared skill set. Compatibility skill
discovery remains disabled. Changes or additions to managed skill files fail
Home validation; native `skills-reload` responses are not ARK request responses.

Native MCP uses `ProviderHomeSpec.mcp_servers`. Grok 1.0.30 stdio and
streamable HTTP servers are supported. MCP Homes add the hidden
`search_tool`/`use_tool` pair to the curated profile and disable inherited
Claude, Cursor, marketplace, and managed gateway connectors. Required servers
must pass `grok mcp doctor --json` before a model prompt. MCP tool targets are
restricted to declared `server__tool` namespaces. Legacy SSE transport is
rejected: the validated 1.0.30 client attempted streamable HTTP semantics
against an SSE endpoint and could not establish the data plane.

Each turn owns a new process group. Completion follows the ACP
`session/prompt` stop reason: `end_turn` completes, `cancelled` cancels or
interrupts, and refusal/token/turn limits fail the turn. Cancel escalates from
the ACP notification through TERM and KILL, and success requires the complete
managed process group—including native tools and stdio MCP children—to be
gone. `run_options.max_turns` is rejected because this adapter has no verified
native mapping.

The adapter exposes live normalized text, tool, permission, terminal events,
and prompt-level aggregate token usage. Grok's `usage.modelCalls` is the turn
request count, but the CLI does not expose per-model-request records, so
`request_usages` and `AgentTurnUsage.requests` remain empty. Native
`costUsdTicks` is retained only in sanitized provider payload; it is not
converted into a fabricated currency amount.

Snapshot captures the complete stable native session directory while
excluding transient locks. Auth, caches, logs, cwd-level prompt history,
`session_search.sqlite`, and workspace files are not captured. Capture and
restore require no active or uncertain process group and are limited to the
same Home, canonical cwd, and session identity. Restore rewinds conversation
state only; it does not roll back workspace changes. In-process artifact copies
and restores are serialized against session maintenance and run admission.

`AgentService.fork_agent()` uses native `_x.ai/session/fork`. Fork is limited
to the latest persisted prompt, same Home and canonical cwd. The child has a
new session identity and independent conversation artifacts; workspace files
are shared. Historical-turn and cross-Home forks are unsupported.

`AgentService.compact_agent()` loads the idle session and invokes native
`_x.ai/compact_conversation`. The adapter records a baseline before submitting
the operation and confirms a persisted checkpoint followed by completion.
Lost responses leave the existing ARK maintenance journal unresolved; read-only
reconciliation requires new completion evidence and a clean process state.
The Context SPI accepts an optional `provider_options={"user_context": "..."}`.
Current-context token measurement is unavailable, so threshold-based ARK
compaction is not advertised. This is distinct from Grok's own automatic compaction.

`ProviderRunHandle.control(STEER)` uses native `_x.ai/interject` during an
active prompt. It rejects startup, cancellation and terminal boundaries.
An accepted receipt means queued, not proof of model consumption; delivery
racing a terminal response is reported as unconfirmed without hidden retries.
Live steering is a Provider SPI operation, not a new AgentService method.
Native concurrent prompts were observed to complete as separate prompt IDs,
terminal responses and usage records. This adapter closes its process after
one prompt, so queued follow-up is not exposed as a control that might outlive
that boundary. Queued follow-up and interactive input remain unsupported. To run a subsequent
ARK turn, wait for completion and call `start_agent()` again.

The extended real acceptance modes are `skills`, `fork`, `compact` and `steer`:

```bash
ARK_RUN_REAL_GROK=1 PYTHONPATH=src \
  /root/miniconda3/envs/benchmark/bin/python \
  tests/real/grok/run_extended_acceptance.py MODE
```

The opt-in real acceptance entrypoint is not collected by normal pytest runs:

```bash
ARK_RUN_REAL_GROK=1 PYTHONPATH=src \
  /root/miniconda3/envs/benchmark/bin/python \
  tests/real/grok/run_acceptance.py MODE
```

The HTTP MCP mode requires its fixture process to be started separately on
the default port 18976:

```bash
ARK_MCP_TRANSPORT=streamable-http \
  /root/miniconda3/envs/benchmark/bin/python tests/real/grok/mcp_fixture.py
```

The acceptance modes cover fresh/resume, Home and run
instructions, curated tools, stdio/HTTP MCP, cancellation and process-group
cleanup, and capture/advance/restore/resume behavior.

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

Each Agent receives its own server process, `OPENCODE_DB`, data, state, and
temporary directories. The materialized Home remains sealed and contains only
ARK-owned static configuration. OpenCode's mutable config and package runtime
is shared by Agents using the same exact Home materialization, while npm and
Bun download caches are shared at provider-runtime scope. A changed Home
manifest selects a new config runtime. The adapter subscribes to SSE before
submitting a prompt and does not report completion until it has both persisted
assistant completion evidence and a live idle status. Query results normalize
OpenCode messages, parts, tools, request usage, model identity, and
provider-reported cost.

OpenCode expands remote MCP environment-header references when the server
process starts. ARK therefore fingerprints the effective process environment
for each isolated server. A read-only session bootstrap may reuse an existing
server, but a run or context preflight carrying a new ARK runtime identity
restarts the server when that fingerprint changed. This preserves the SQLite
session while ensuring reused Agents send the current Flow, Step, and Agent
headers instead of stale or missing identities. Later read-only queries reuse
the identity-bound server and never interrupt an active turn merely to remove
those headers.

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
