# OpenAI Agents Provider

ARK's `openai_agents` adapter embeds the OpenAI Agents Python SDK behind the
same Home, run, query, context, usage, fork, and snapshot contracts used by the
other Providers. Install the pinned optional dependencies with:

```bash
python -m pip install -e '.[openai-agents]'
```

## Assembly

Python Agent graphs and tools are application resources, not serializable Home
data. Register a stable factory reference, build the Provider, and pass its
bundle through `ProviderRegistry`:

```python
from agents import Agent

from agent_runtime_kit.agent.provider_contracts import ProviderRegistry
from agent_runtime_kit.agent.providers import OpenAIAgentsProvider

provider = OpenAIAgentsProvider()
provider.registry.register_agent_factory(
    "my_app/default",
    lambda context: Agent(
        name="my-app-agent",
        instructions=context.instructions,
        model=context.model,
    ),
)
registry = ProviderRegistry((provider.build_bundle(runtime_root=runtime_root),))
```

The corresponding `ProviderHomeSpec` must use
`OpenAIAgentsHomeOptions(agent_factory_ref="my_app/default", ...)` and a
`ModelBackendIdentity`. `api_provider` identifies the configured endpoint;
`api_mode` is `responses` or `chat_completions`. The selected model does not
change the Provider type.

API keys are named by environment reference and are never copied into the
Home or snapshot. A custom base URL can be supplied directly when it is not a
secret, or through `base_url_env`.

## Sessions, Queries, and Snapshots

Each session uses an ARK-owned SQLite file. The adapter normalizes SDK items,
tool calls, request usage, cache/reasoning token fields when present, model
identity, and errors. Unknown values remain absent rather than being inferred
as zero.

Fork copies an idle session into a new SQLite session and records
`fork_mode=session_only` and `workspace_isolated=false`. Snapshot uses SQLite
online backup and restores the provider-owned session database; it does not
capture workspace files or credentials.

## Context and Compaction

`compaction_mode="input_history"` is valid for a compatible Responses backend
and uses the SDK's input-history compaction session. Chat Completions does not
provide Responses-native compact, so its resolved compact capability is
unsupported unless a future, explicitly configured strategy is added.

Context-window and maximum-output values may be declared in Home options when
the endpoint does not report them. Such configured limits remain distinct from
provider-reported token usage.

## Interaction Boundary

The Provider run handle can persist and respond to SDK approval boundaries.
ARK 0.2 does not yet keep a complete `NEEDS_INPUT` lifecycle through
`AgentService`, Flow, and Step. Applications using those upper layers must use
non-interactive tools and approval policies. This limitation is explicit; a
Provider boundary must not be treated as successful completion.
