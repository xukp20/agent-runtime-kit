from __future__ import annotations

import json
from pathlib import Path

from agent_runtime_kit.agent.homes import ProviderHomeSpec
from agent_runtime_kit.agent.provider_contracts import (
    AgentArtifactLocator,
    BaseConfigSource,
    CapabilityKey,
    CapabilityStatus,
    ProviderRegistry,
    ProviderSessionLocator,
)
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.providers.codex_bundle import build_codex_provider_bundle
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry


class _QueryAgentType(AgentType):
    agent_type = "query-agent"
    start_prompt_template = "query"


def test_codex_standard_query_projects_rollout(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(_QueryAgentType())
    provider = CodexProvider(runtime_root=runtime_root)
    service = AgentService(
        runtime_root,
        agent_types=registry,
        provider_registry=ProviderRegistry(
            (build_codex_provider_bundle(provider, runtime_root=runtime_root),)
        ),
    )
    service.home_service.create_home(ProviderHomeSpec(provider_type="codex", home_id="query-agent"))
    agent = service.create_agent("scope", "query-agent")
    rollout_relpath = "sessions/trace.jsonl"
    session = ProviderSessionLocator(
        provider_type="codex",
        session_id="thread-1",
        home_id="query-agent",
        created_at="2026-07-22T00:00:00Z",
        native_locator={"rollout_relpath": rollout_relpath},
    )
    service.store.update_session_locators(
        agent.agent_id,
        session_locator=session,
        artifact_locator=AgentArtifactLocator(
            provider_type="codex",
            home_id="query-agent",
            session_id="thread-1",
            adapter_version="1",
            native_primary_ref=rollout_relpath,
        ),
    )
    rollout = runtime_root / "homes" / "codex" / "query-agent" / ".codex" / rollout_relpath
    rollout.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "type": "session_meta",
            "payload": {
                "id": "thread-1",
                "timestamp": "2026-07-21T10:00:00Z",
            },
        },
        {
            "type": "turn_context",
            "payload": {
                "turn_id": "turn-1",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "lean_check",
                "call_id": "call-1",
                "arguments": {"file": "Main.lean"},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": {"ok": True},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": "turn-1",
                "last_agent_message": "done",
            },
        },
    ]
    rollout.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    sessions = service.query_sessions(provider_type="codex", home_id="query-agent")
    turns = service.query_turns(agent.agent_id)
    latest = service.query_turn(agent.agent_id, latest=True)
    first_events = service.query_events(agent.agent_id, limit=2)
    second_events = service.query_events(agent.agent_id, cursor=first_events.next_cursor, limit=2)
    tools = service.query_tool_calls(agent.agent_id)
    usage = service.query_usage(agent.agent_id, latest=True)

    assert len(sessions.items) == 1
    assert sessions.items[0].locator.session_id == "thread-1"
    assert sessions.items[0].locator.created_at == "2026-07-21T10:00:00Z"
    assert sessions.items[0].locator.native_locator == {
        "rollout_relpath": "sessions/trace.jsonl"
    }
    assert len(turns.items) == 1
    assert latest.locator.turn_id == "turn-1"
    assert latest.result.final_text == "done"
    assert [event.sequence for event in first_events.items] == [0, 1]
    assert [event.sequence for event in second_events.items] == [2, 3]
    assert len(tools.items) == 1
    assert tools.items[0].call_id == "call-1"
    assert tools.items[0].tool_name == "lean_check"
    assert tools.items[0].result == {"ok": True}
    assert usage.token_usage.total_tokens == 15

    report = service.build_trace_report(agent.agent_id)
    assert report.latest_turn.locator.turn_id == "turn-1"
    assert report.tool_calls[0].tool_name == "lean_check"


def test_codex_capabilities_distinguish_native_adapted_and_unavailable(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    provider = CodexProvider(runtime_root=runtime_root)
    service = AgentService(
        runtime_root,
        provider_registry=ProviderRegistry(
            (build_codex_provider_bundle(provider, runtime_root=runtime_root),)
        ),
    )
    capabilities = service.provider_registry.get("codex").descriptor.static_capabilities

    assert capabilities is not None
    assert capabilities.get(CapabilityKey.SESSION_CREATE).status is CapabilityStatus.NATIVE
    assert capabilities.get(CapabilityKey.SESSION_LIST).status is CapabilityStatus.ADAPTABLE
    assert capabilities.get(CapabilityKey.ARTIFACT_RESTORE).status is CapabilityStatus.ADAPTABLE
    request_usage = capabilities.get(CapabilityKey.QUERY_REQUEST_USAGE)
    assert request_usage.status is CapabilityStatus.UNSUPPORTED
    assert request_usage.available is False

    service.home_service.create_home(ProviderHomeSpec(provider_type="codex", home_id="responses"))
    responses = service.resolve_provider_capabilities(
        provider_type="codex",
        home_id="responses",
    )
    assert responses.available(CapabilityKey.MODEL_RESPONSES)
    assert not responses.available(CapabilityKey.MODEL_CHAT_COMPLETIONS)

    config_path = tmp_path / "deepseek-chat.toml"
    config_path.write_text(
        """
model = "deepseek-chat"
model_provider = "deepseek"

[model_providers.deepseek]
base_url = "https://api.deepseek.com"
wire_api = "chat"
""".lstrip(),
        encoding="utf-8",
    )
    service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="codex",
            home_id="chat",
            base_config=BaseConfigSource(path=str(config_path)),
        )
    )
    chat = service.resolve_provider_capabilities(provider_type="codex", home_id="chat")
    assert chat.resolved_for_backend == "deepseek:chat_completions"
    assert chat.available(CapabilityKey.MODEL_CHAT_COMPLETIONS)
    assert not chat.available(CapabilityKey.MODEL_RESPONSES)
