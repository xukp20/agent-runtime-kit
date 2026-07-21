# Agent Runtime Kit Documentation

This directory is the entry point for reusable, shareable ARK documentation.

Start with the repository [README](../README.md) for:

- the current project purpose and framework boundaries;
- implemented Agent, Flow/Step, scheduling, MCP, persistence, and snapshot
  capabilities;
- installation, runtime assembly, storage layout, and test commands.

## Documentation Layers

ARK keeps two documentation layers deliberately separate:

- `README.md` and `docs/` contain stable, public-facing English documentation
  for users and integrators;
- `dev_docs/` contains Chinese maintainer documentation, detailed architecture,
  design decisions, implementation plans, audits, and local development
  records.

Maintainer checkouts may include a local
`dev_docs/code_implementation_reference/` tree for the exact current source
structure. Historical design documents are useful for rationale, but current
source and current-code references take precedence when they differ.

## Current Public Surface

The package is organized into three primary namespaces:

- `agent_runtime_kit.agent`: Agent records, AgentType behavior, provider Homes,
  skills, traces, reports, and snapshots;
- `agent_runtime_kit.flow`: Flow/Step models, registries, stores, lifecycle
  services, scheduling, and standard Steps;
- `agent_runtime_kit.runtime`: shared service containers, pause control,
  runtime contexts, and MCP identity/submission helpers.

The project does not currently publish generated API reference pages. Until
that surface is added, the package exports, type hints, tests, and maintainer
current-code reference are the authoritative API guides.

## Feature Guides

- [Runtime observation](runtime-observation.md): observation-only Step terminal
  and Agent status waits, settled/lost semantics, timeout behavior, and web
  adapter guidance.
- [Agent context inspection and compaction](agent-context-compaction.md):
  provider-neutral usage and compaction APIs, first-turn admission,
  fail-closed recovery, snapshot behavior, and the optional provider contract.
- [Provider adapters and normalized Agent runtime](provider-adapters.md):
  Provider Registry, capabilities, Home/runtime/query/context/artifact SPI,
  normalized records and results, and migration compatibility.
- [Claude Code provider](claude-code-provider.md): isolated Claude Homes,
  Claude Agent SDK runtime assembly, normalized queries, context/compact,
  session-only fork, and transcript snapshot/restore.
- [Pi provider](pi-provider.md): Pi 0.80.10 Home and model configuration,
  JSONL RPC lifecycle, agent-owned compaction, MCP bridge, and snapshots.
