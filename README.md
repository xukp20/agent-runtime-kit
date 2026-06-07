# Agent Runtime Kit

`agent-runtime-kit` is a lightweight Python runtime kit for provider-backed
agents, scoped persistence, snapshots, and simple orchestration primitives.

It is the planned successor to the heavier `agent-profile-runtime` design. The
goal is to keep the useful core ideas, such as isolated provider homes,
AgentType templates, scoped thread artifacts, and snapshot-friendly runtime
state, while avoiding unnecessary Session / Turn / Workflow machinery in the
first implementation.

## Installation

Install locally in editable mode:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e .[dev]
```

## Current Scope

The initial design focuses on:

- code-defined `AgentType` templates for instructions and prompts
- provider homes for Codex and future provider backends
- Agent records as lightweight wrappers around provider threads
- scoped runtime persistence and snapshot restoration
- simple synchronous run / wait / pause interfaces
- room for later, simplified step and workflow primitives

## Repository Layout

```text
agent-runtime-kit/
├── src/
│   └── agent_runtime_kit/
├── tests/
├── docs/
├── dev_docs/
├── data/
├── configs/
├── README.md
├── AGENTS.md
└── .gitignore
```

Public, reusable documentation belongs in `docs/`. Local design notes,
working plans, and daily records belong in `dev_docs/`.
