from __future__ import annotations

from dataclasses import asdict

import pytest

from agent_runtime_kit.agent.context import (
    AgentContextCompactionStatus,
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
    AgentContextMaintenancePolicy,
    AgentContextUsage,
    ProviderContextUsage,
)
from agent_runtime_kit.agent import AgentContextUsage as ExportedAgentContextUsage
from agent_runtime_kit.agent.models import to_jsonable


def test_context_usage_computes_ratio() -> None:
    usage = AgentContextUsage(
        agent_id="agent-1",
        provider_type="codex",
        session_id="thread-1",
        total_tokens=80,
        context_window=100,
        observed_at="2026-07-21T00:00:00Z",
        source="artifact",
        available=True,
    )

    assert usage.usage_ratio == 0.8
    assert to_jsonable(usage)["usage_ratio"] == 0.8


def test_context_usage_is_exported_from_agent_package() -> None:
    assert ExportedAgentContextUsage is AgentContextUsage


@pytest.mark.parametrize(
    ("total_tokens", "context_window", "available"),
    [(-1, 100, True), (1, 0, True), (None, 100, True), (1, None, True)],
)
def test_context_usage_rejects_invalid_values(
    total_tokens: int | None,
    context_window: int | None,
    available: bool,
) -> None:
    with pytest.raises(ValueError):
        AgentContextUsage(
            agent_id="agent-1",
            provider_type="codex",
            session_id="thread-1",
            total_tokens=total_tokens,
            context_window=context_window,
            observed_at="2026-07-21T00:00:00Z",
            source="artifact",
            available=available,
        )


def test_unavailable_context_usage_allows_partial_values() -> None:
    usage = ProviderContextUsage(
        session_id="thread-1",
        total_tokens=None,
        context_window=100,
        observed_at="2026-07-21T00:00:00Z",
        source="artifact",
        available=False,
        reason="token_count_missing",
    ).for_agent(agent_id="agent-1", provider_type="codex")

    assert usage.usage_ratio is None
    assert usage.reason == "token_count_missing"


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1])
def test_context_maintenance_policy_validates_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        AgentContextMaintenancePolicy(threshold=threshold)


def test_to_jsonable_serializes_context_status_enum() -> None:
    assert to_jsonable(AgentContextCompactionStatus.COMPACTED) == "compacted"


def test_context_maintenance_journal_round_trip() -> None:
    journal = AgentContextMaintenanceJournal(
        agent_id="agent-1",
        provider_type="codex",
        session_id="thread-1",
        status=AgentContextMaintenanceJournalStatus.STARTED,
        trigger="threshold_preflight",
        prepared_at="2026-07-21T00:00:00Z",
        started_at="2026-07-21T00:00:01Z",
        baseline={"offset": 12},
    )

    restored = AgentContextMaintenanceJournal.from_dict(journal.to_dict())

    assert restored == journal
    assert restored.unresolved
    assert asdict(restored)["status"] is AgentContextMaintenanceJournalStatus.STARTED
