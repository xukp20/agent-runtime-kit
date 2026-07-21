# Claude Code Provider

ARK's `claude_code` provider runs Claude Code through the Claude Agent SDK and
projects its native Home, stream messages, session JSONL, context controls, and
artifacts into the provider-neutral Agent contracts.

The provider identity is independent from the configured API backend and
model. Anthropic, an Anthropic-compatible service, or another backend exposed
through Claude Code remains a `claude_code` Agent with a separate
`ModelBackendIdentity`.

## Installation and Assembly

Install the optional pinned SDK dependency:

```bash
python -m pip install -e '.[claude]'
```

Register the provider explicitly and create a typed provider Home:

```python
from pathlib import Path

from agent_runtime_kit.agent.provider_contracts import (
    ModelBackendIdentity,
    ProviderHomeSpec,
)
from agent_runtime_kit.agent.providers import (
    ClaudeCodeHomeOptions,
    ClaudeCodeProvider,
)
from agent_runtime_kit.agent.service import AgentService

runtime_root = Path(".agent_runtime")
service = AgentService(
    runtime_root,
    providers={
        "claude_code": ClaudeCodeProvider(runtime_root=runtime_root),
    },
)
service.create_home(
    ProviderHomeSpec(
        provider_type="claude_code",
        home_id="worker",
        model_config=ModelBackendIdentity(
            api_provider="anthropic-compatible",
            api_mode="anthropic_messages",
            requested_model="configured-model",
        ),
        required_env=("ANTHROPIC_AUTH_TOKEN",),
        provider_options=ClaudeCodeHomeOptions(
            cli_path="claude",
            setting_sources=("user",),
        ),
    ),
    env={"ANTHROPIC_AUTH_TOKEN": "..."},
)
```

Pass credentials through `env` or another application-owned secret resolver.
Do not place credential values in committed settings, manifests, or snapshot
metadata.

## Isolated Home

The renderer creates:

```text
homes/claude_code/<home_id>/
├── .claude/
│   ├── settings.json
│   ├── skills/<name>/...
│   └── projects/.../<session_id>.jsonl
└── .ark/
    ├── claude_code_home.json
    ├── home_materialization.json
    └── claude_home_initialized.json
```

Each run sets `HOME` to the ARK Home root and `CLAUDE_CONFIG_DIR` to its
`.claude` directory. Base JSON settings and recursive overrides are sealed by
the Home materialization manifest. Instructions, skills, MCP definitions,
tools, model selection, permission mode, budgets, and allowed extra CLI flags
are converted to `ClaudeAgentOptions`. Unsupported or lossy MCP fields fail
Home validation instead of being silently ignored.

## Runtime and Queries

Start creates an explicit UUID session; resume uses the same session ID. A run
is terminal only after the SDK stream emits `ResultMessage`. Interrupt is sent
on the owning asyncio loop and is confirmed only after the terminal result is
observed.

The live stream is normalized into content blocks, tool calls, request usage,
reported cost, model identity, errors, and a final result. Offline `query_*`
APIs parse only the isolated session JSONL and do not start the CLI. Repeated
assistant records with the same message ID are deduplicated for request usage.
Missing token categories and costs remain `None`; ARK does not invent totals or
prices.

Claude slash-command records such as `/compact` are maintenance traffic, not
Agent turns. The query adapter excludes those records while retaining compact
boundaries as session evidence.

## Context, Compact, and Fork

Context inspection uses the Claude context-control response and reports the
provider's used, effective maximum, raw maximum, categories, model, and
auto-compact fields. The capability is available only when Home initialization
has verified a compatible CLI version.

Manual compact resumes the idle session, sends `/compact`, waits for terminal
SDK completion, and verifies a new persisted `compact_boundary`. If Claude
returns successfully without creating a boundary, ARK fails closed rather
than claiming that compaction occurred.

Fork uses Claude's native session fork and returns:

```text
fork_mode = session_only
workspace_isolated = false
```

The new session is independently resumable. ARK does not copy, reset, or fork
the working tree.

## Snapshot and Restore

The first adapter version requires an idle session and file checkpointing to
be disabled. Its Artifact Manifest contains:

- one copied provider-native session JSONL, required for resume;
- one external Home materialization manifest hash, required for restore.

Restore validates archive checksums and the live target Home dependency before
placing the transcript back in the provider-native location. It then verifies
that the restored transcript can be parsed offline. Rebuildable Claude caches
are not snapshot truth.

If Claude file checkpointing is enabled in a future version, every required
checkpoint/history artifact must first be represented in the Artifact
Manifest. Until then, Home validation rejects that configuration instead of
advertising incomplete snapshot support.

## Tests

Deterministic unit and fake-SDK integration tests run in the default suite.
Real tests are gated by `ARK_RUN_REAL_CLAUDE=1` and also require a usable Claude
Agent SDK, Claude Code CLI, backend configuration, and credentials. Optional
`ARK_CLAUDE_SETTINGS_PATH` and `ARK_CLAUDE_CLI_PATH` select the local test
configuration without embedding secrets in the test source.
