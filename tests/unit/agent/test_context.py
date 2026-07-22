from __future__ import annotations

import pytest

from agent_runtime_kit.agent.context import (
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
    AgentContextMaintenancePolicy,
)
from agent_runtime_kit.agent.provider_contracts import AgentContextUsage, ProviderContextUsage


def test_standard_context_usage_computes_ratio_without_alias_fields() -> None:
    usage = AgentContextUsage(
        agent_id="agent-1",
        provider_type="codex",
        session_id="session-1",
        observed_at="2026-07-22T00:00:00Z",
        source="artifact",
        available=True,
        used_tokens=80,
        effective_context_window_tokens=100,
        measurement="provider_artifact",
    )

    assert usage.usage_ratio == 0.8
    assert not hasattr(usage, "total_tokens")
    assert not hasattr(usage, "context_window")


def test_provider_usage_for_agent_preserves_standard_fields() -> None:
    usage = ProviderContextUsage(
        session_id="session-1",
        observed_at="2026-07-22T00:00:00Z",
        source="provider",
        available=True,
        used_tokens=20,
        context_window_tokens=100,
        remaining_tokens=80,
        measurement="provider_reported",
    )

    projected = usage.for_agent(agent_id="agent-1", provider_type="fake")

    assert projected.agent_id == "agent-1"
    assert projected.used_tokens == 20
    assert projected.remaining_tokens == 80


def test_context_policy_and_journal_validation() -> None:
    with pytest.raises(ValueError):
        AgentContextMaintenancePolicy(threshold=0)
    journal = AgentContextMaintenanceJournal(
        agent_id="agent-1",
        provider_type="codex",
        session_id="session-1",
        status=AgentContextMaintenanceJournalStatus.STARTED,
        trigger="manual",
        prepared_at="2026-07-22T00:00:00Z",
    )
    assert AgentContextMaintenanceJournal.from_dict(journal.to_dict()) == journal
    assert journal.unresolved
