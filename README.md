# Agent Runtime Kit

`agent-runtime-kit` (ARK) is a lightweight Python runtime for provider-backed
agents and application-defined workflows. It provides the reusable execution
layer for applications that need isolated agent homes, persistent provider
threads, typed Flow/Step orchestration, bounded scheduling, MCP runtime
identity, and stable snapshot/restore.

ARK is intentionally application-neutral. It owns runtime mechanics and
persistence; the embedding application owns business services, concrete
agents, tools, permissions, and workflow semantics.

## What ARK Supports

The current implementation includes:

- code-defined `AgentType` templates with developer instructions, start and
  continuation prompts, completion checks, and auto-continue policy;
- isolated provider homes with configuration, credentials, environment
  requirements, MCP server definitions, and materialized skills;
- a provider-neutral contract layer with descriptors, capabilities, Home
  renderers, runtime handles, query/context/artifact adapters, and a registry;
- a Codex reference provider and an opt-in OpenCode 1.18.4 adapter; OpenCode
  runs an isolated local server per Agent and preserves its SQLite-backed
  resumable session artifacts;
- versioned Agent records with provider/session/turn/artifact locators plus
  compatibility aliases for existing Codex runtimes and snapshots;
- Agent start, wait, interrupt, session-only fork, close, stale-run
  reconciliation, normalized results, usage, and offline query APIs;
- provider-neutral context inspection and between-turn compaction, with Codex
  completion evidence, fail-closed recovery, and snapshot-safe maintenance;
- typed `BaseFlow` and `BaseStep` models with registries, JSON truth, SQLite
  indexes, lifecycle contexts, and transactional mutation;
- asynchronous Step execution and a scheduler that advances Flows separately
  from starting Steps, with concurrency limits and pause gates;
- numeric bounded runs and semantic run leases for controlled production
  advancement;
- standard `AgentStep` and `DispatchStep` implementations, including accepted
  submissions, child-Flow dispatch, callback continuation, and terminal
  handoff;
- MCP runtime identity resolution for Flow/Step/Agent relationships and a
  guarded helper for writing submissions to the current running Step;
- scope and runtime snapshots, selective scope refresh, restore validation,
  index/queue rebuilding, and stable-point checks across Agents and Steps;
- structured rollout trace readers and configurable JSON/Markdown trace report
  persistence.

## Architecture

```text
Embedding application
  ├─ AppServices: business services and tool handlers
  ├─ AgentType / Flow / Step subclasses
  └─ application MCP and admin surfaces
          │
          ▼
ARKServices
  ├─ AgentService ── Provider Registry, Home, run handles, completion, query
  ├─ FlowService  ── Flow lifecycle and child relationships
  ├─ StepService  ── asynchronous Step execution
  ├─ RuntimeScheduleService ── queues, limits, run control
  ├─ AgentSnapshotService ── scope/runtime orchestration over Artifact adapters
  └─ RuntimePauseController ── global and scope pause gates
```

The shared `ARKServices` container is deliberately mutable so applications can
assemble these services in two stages. Every runtime context carries both
`ctx.ark` and `ctx.app`, keeping framework services separate from application
services.

## Core Runtime Model

A typical execution path is:

```text
FlowRequest
  -> FlowService.start_flow(...)
  -> RuntimeScheduleService advances the Flow
  -> BaseFlow.create_next_step(...) creates a Step
  -> StepService runs the Step asynchronously
  -> BaseFlow.on_step_terminal(...) consumes the Step result
  -> the Flow completes, waits, fails, or creates another Step
```

For an `AgentStep`, ARK additionally:

1. creates or reuses an Agent bound to a role;
2. injects `ARK_FLOW_ID`, `ARK_STEP_ID`, and `ARK_AGENT_ID` into the provider
   environment;
3. optionally inspects and compacts an existing provider context before the
   first turn of the Step run;
4. starts or resumes the provider thread;
5. waits for an accepted typed submission, with bounded auto-continue;
6. converts that submission into a terminal Step result.

Applications expose their own MCP tools. ARK validates the runtime caller
identity and Step binding, but it does not define application tools, business
permissions, or domain context.

## Installation

ARK requires Python 3.11 or newer.

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e '.[dev]'
```

Codex support uses the OpenAI Codex Python SDK at runtime. The SDK may be
installed normally or supplied from a local Codex source checkout through the
provider's `sdk_python_root` option. Unit tests do not require a live Codex
session.

OpenCode support uses an external `opencode` 1.18.4 executable. Applications
register `build_opencode_provider_bundle(...)` explicitly and provide model
credentials through environment references; ARK does not bundle OpenCode or
copy credentials into a Home or snapshot.

## Runtime Assembly

Applications normally create one shared framework container and one
application container, register their concrete types, then attach the runtime
services:

```python
from pathlib import Path

from agent_runtime_kit.agent.service import AgentService
from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.flow import (
    FlowService,
    FlowTypeRegistry,
    RuntimeScheduleService,
    StepService,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePauseController

runtime_root = Path(".agent_runtime")
ark = ARKServices(pause_controller=RuntimePauseController())
app = AppServices()  # Replace with an application-specific subclass.

flow_types = FlowTypeRegistry()
step_types = StepTypeRegistry()
# flow_types.register(MyFlow)
# step_types.register(MyStep)

agent_service = AgentService(runtime_root, ark_services=ark, app_services=app)
FlowService(
    runtime_root,
    flow_registry=flow_types,
    step_registry=step_types,
    ark_services=ark,
    app_services=app,
)
StepService(
    runtime_root,
    step_registry=step_types,
    ark_services=ark,
    app_services=app,
)
RuntimeScheduleService(ark_services=ark, app_services=app)
AgentSnapshotService(
    runtime_root,
    store=agent_service.store,
    agent_service=agent_service,
    ark_services=ark,
    app_services=app,
)
```

Concrete Flow, Step, AgentType, provider Home, and MCP setup remain application
responsibilities. Tested examples are available under `tests/integration/`.

## Persistence Layout

By default, runtime state lives below the configured runtime root:

```text
.agent_runtime/
├── homes/                 # isolated provider homes and Home index
├── providers/             # provider-owned per-Agent runtime data (for example OpenCode SQLite/XDG)
├── scopes/                # scope-owned Agent, Flow, and Step truth
├── index/global.sqlite    # rebuildable global Agent/Flow/Step index
├── snapshots/
│   ├── scopes/
│   └── runtime/
└── reports/               # optional persisted trace reports
```

JSON files and provider-native artifacts identified by each provider's Artifact
Manifest are authoritative restorable truth. SQLite databases and scheduler
queues are rebuildable indexes or caches unless a provider explicitly declares
a database authoritative in its manifest.

## Boundaries

ARK does not provide:

- application-specific tools, ToolViews, authorization, or MCP endpoints;
- business Flow definitions or domain services;
- a web server, admin API, or production process supervisor;
- a replacement event model for provider-native thread and rollout truth;
- distributed scheduling across multiple ARK processes.

These boundaries keep the framework small enough for applications to own their
domain model without inheriting a second business abstraction layer.

## Testing

Run the deterministic unit and integration suites with:

```bash
python -m pytest -q tests/unit tests/integration
```

Real Codex tests are under `tests/real/` and require an explicitly configured
Codex SDK, CLI, Home, and credentials. They are intentionally separate from the
default regression suite.

The gated OpenCode integration tests use `ARK_OPENCODE_TEST_BINARY`. Real model
tests additionally require `ARK_OPENCODE_RUN_REAL_MODELS=1` and backend keys in
the documented environment variables. They never read a shared OpenCode Home.

## Documentation

- [`docs/README.md`](docs/README.md) is the public documentation entry point.
- [`docs/agent-context-compaction.md`](docs/agent-context-compaction.md)
  documents context usage, compaction admission, failure recovery, and the
  optional provider contract.
- [`docs/provider-adapters.md`](docs/provider-adapters.md) documents the
  provider-neutral contracts, capability rules, extension points, and Codex
  compatibility surface.

Maintainer checkouts may also contain a local `dev_docs/` tree with Chinese
design, implementation, and current-code reference material. It is intentionally
not part of the public documentation surface.

Public reusable documentation belongs in `README.md` and `docs/`. Local design
notes, implementation plans, audits, and development records belong in
`dev_docs/`.
