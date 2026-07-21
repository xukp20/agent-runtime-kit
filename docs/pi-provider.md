# Pi Provider

ARK's `pi` adapter targets `@earendil-works/pi-coding-agent` 0.80.10. It keeps
the application-facing Agent API provider-neutral while using Pi's native Home,
LF-delimited JSONL RPC protocol, and v3 session artifacts underneath.

`pi` identifies the agent harness. OpenAI Codex OAuth, an OpenAI-compatible
endpoint, DeepSeek, Anthropic Messages, and other model services are backend
identities selected inside the Pi Home; they are not separate ARK providers.

## Registration and Home

Register the bundle explicitly when assembling `AgentService` and pass its Home
renderer to `HomeService` (or let `AgentService` use that registry):

```python
from pathlib import Path

from agent_runtime_kit.agent.homes import HomeService
from agent_runtime_kit.agent.provider_contracts import (
    ProviderHomeSpec,
    ProviderRegistry,
)
from agent_runtime_kit.agent.providers import PiHomeOptions, build_pi_provider_bundle

runtime_root = Path(".agent_runtime")
pi = build_pi_provider_bundle(runtime_root=runtime_root)
providers = ProviderRegistry((pi,))
homes = HomeService(runtime_root, renderers={"pi": pi.home_renderer})

home = homes.create_home(
    ProviderHomeSpec(
        provider_type="pi",
        home_id="pi-worker",
        required_env=("DEEPSEEK_API_KEY",),
        provider_options=PiHomeOptions(
            node_executable="/path/to/node",
            pi_cli_path=Path("/path/to/pi/dist/cli.js"),
            settings={
                "defaultProvider": "deepseek",
                "defaultModel": "deepseek-chat",
            },
        ),
    )
)
```

`PiHomeOptions` can project an existing `auth.json` or `models.json`, an inline
models mapping, settings, skills, extensions, tool allow-list, static
instructions, project-resource trust, offline mode, and restricted extra CLI
arguments. `ProviderHomeSpec.base_config`, `config_overrides`, fixed/run
environment, required environment names, and common resource fields retain the
same precedence and validation behavior as other provider Homes.

ARK materializes `.pi/settings.json`, optional auth/models/resources,
`.pi/sessions`, and an `.ark` runtime/materialization manifest. Secrets are
copied only when explicitly configured and are never embedded in normalized
results. The materialization hash is checked whenever an execution context is
built.

## Runtime Semantics

Each start or resume launches a managed Pi RPC subprocess. A prompt response
only means that Pi accepted the command; ARK waits for `agent_settled`, then
checks that streaming and compaction are idle before reading the persisted
session and returning a normalized result.

The runtime supports:

- start and resume by stable Pi session ID;
- event draining and terminal waits;
- interrupt/cancel, steer, and follow-up;
- independently resumable, session-only fork, including fork from a completed
  source turn;
- close and idle-state reporting for snapshot coordination.

Fork reports `fork_mode="session_only"` and `workspace_isolated=False`. It does
not create a Git worktree or restore files changed by either branch.

## Results, Usage, and Context

The v3 JSONL adapter projects the active parent-linked branch into normalized
sessions, turns, events, content blocks, tool calls, and request/turn/session
usage. It retains model provider, API mode, requested/resolved model, response
ID, stop reason, reasoning tokens, cache read/write fields, and Pi-reported USD
cost when present. Missing fields stay `None`; ARK does not calculate a price.

Pi session statistics expose an estimated trailing-context size. ARK marks the
measurement as `estimated`. After Pi compaction, Pi may report no usable token
count; ARK returns unavailable/stale context rather than zero until later model
evidence makes it available again.

Pi compact is agent-owned history summarization. ARK waits for both the compact
command response and `compaction_end`, verifies idle state, and records the
persisted compaction entry for reconciliation. This operation is available for
Responses, Chat Completions, and Messages backends and must not be interpreted
as a backend `responses.compact` call.

## MCP

Pi 0.80.10 does not supply ARK's common MCP Home semantics directly. When a
Home contains MCP server definitions, ARK materializes a Pi extension that:

- connects to stdio, Streamable HTTP, or SSE MCP transports;
- registers tools as `mcp__<server>__<tool>`;
- applies enabled/disabled tool filters and environment/header references;
- propagates cancellation and per-tool timeouts;
- reconnects once on a transport failure and closes clients at session
  shutdown;
- refreshes added or changed tools after MCP list-change notifications;
- fails Home initialization when the prepared MCP SDK runtime is absent.

The application supplies `PiHomeOptions.mcp_runtime_root`, whose
`node_modules` must contain `@modelcontextprotocol/sdk`. ARK packages the bridge
source but intentionally does not install Node dependencies during Home
materialization. Required server startup failures fail closed; optional server
failures omit that server's tools for the current session.

Pi 0.80.10 has no extension API for unregistering a tool definition during an
active session. If an MCP server removes a tool, the bridge updates its live
advertisement set and calls to the stale projected name fail closed, but the
name can remain visible to the model until the next Pi session starts.

## Snapshots and Limitations

At an idle stable point, the artifact adapter captures the single authoritative
session JSONL with its hash and stable session ID. It also records the current
Home materialization-manifest hash as a restore-time reference. Restore
preflights Home identity, paths, source hashes, and session ID before deleting
or writing any managed artifact, then replaces the session file atomically.

Version 1 intentionally does not support:

- in-flight process snapshots;
- workspace file checkpointing or rollback;
- interactive approval/input responses from Pi extensions;
- treating backend Responses compaction as Pi agent compaction.

Offline query and snapshot restore do not require model access. Starting,
resuming, compacting, or inspecting live Pi context requires the configured Pi
runtime and applicable backend credentials.
