from __future__ import annotations

from pathlib import Path

from agent_runtime_kit.agent.context import (
    AgentContextCompactionStatus,
    ProviderContextCompactionResult,
    ProviderContextUsage,
)
from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry


class _ContextAgentType(AgentType):
    agent_type = "context-agent"
    start_prompt_template = "context"


def test_codex_context_adapter_preserves_standard_and_legacy_shapes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    provider = CodexProvider(runtime_root=runtime_root)
    registry = AgentTypeRegistry()
    registry.register(_ContextAgentType())
    service = AgentService(
        runtime_root,
        agent_types=registry,
        providers={"codex": provider},
    )
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="context-agent"))
    agent = service.create_agent("scope", "context-agent")
    service.store.update_thread_locator(
        agent.agent_id,
        thread_id="session-1",
        rollout_relpath="sessions/session-1.jsonl",
    )

    def inspect_thread_context(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["env"]["CODEX_HOME"].endswith("/.codex")
        return ProviderContextUsage(
            session_id="session-1",
            total_tokens=80,
            context_window=100,
            observed_at="2026-07-21T00:00:00Z",
            source="artifact",
            available=True,
        )

    def compact_thread(**kwargs):  # noqa: ANN003, ANN202
        kwargs["on_compaction_started"]({"event_count": 4}, "compact-1")
        return ProviderContextCompactionResult(
            session_id="session-1",
            usage_after=ProviderContextUsage(
                session_id="session-1",
                total_tokens=20,
                context_window=100,
                observed_at="2026-07-21T00:01:00Z",
                source="artifact",
                available=True,
            ),
            started_at="2026-07-21T00:00:30Z",
            completed_at="2026-07-21T00:01:00Z",
            provider_operation_id="compact-1",
        )

    monkeypatch.setattr(provider, "inspect_thread_context", inspect_thread_context)
    monkeypatch.setattr(provider, "compact_thread", compact_thread)

    standard = service.inspect_agent_context_result(agent.agent_id)
    legacy = service.inspect_agent_context(agent.agent_id)
    compacted = service.compact_agent_if_needed(agent.agent_id, threshold=0.8)

    assert standard.used_tokens == 80
    assert standard.context_window_tokens == 100
    assert standard.remaining_tokens == 20
    assert standard.measurement == "provider_artifact"
    assert legacy.total_tokens == 80
    assert legacy.context_window == 100
    assert compacted.status is AgentContextCompactionStatus.COMPACTED
    assert compacted.usage_after is not None
    assert compacted.usage_after.total_tokens == 20
    assert service.store.read_context_maintenance(agent.agent_id).provider_operation_id == "compact-1"
