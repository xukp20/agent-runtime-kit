from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime_kit.agent.context import (
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
)
from agent_runtime_kit.agent.homes import ProviderHomeSpec
from agent_runtime_kit.agent.models import AgentContextMaintenanceBlocked
from agent_runtime_kit.agent.provider_contracts import (
    ProviderContextCompactionResult,
    ProviderRegistry,
    ProviderSessionLocator,
)
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from tests.integration.agent.provider_contract_harness import make_contract_fake_bundle


class _ReconcileAgentType(AgentType):
    agent_type = "reconcile-agent"
    provider_type = "contract_fake"
    default_home_id = "contract-home"
    start_prompt_template = "work"


class _ContextAdapter:
    provider_type = "contract_fake"

    def __init__(self) -> None:
        self.result: ProviderContextCompactionResult | None = None
        self.requests = []

    def inspect(self, _request):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def compact(self, _request):  # noqa: ANN001, ANN201
        raise NotImplementedError

    def reconcile(self, request):  # noqa: ANN001, ANN201
        self.requests.append(request)
        return self.result


def _service(
    tmp_path: Path,
) -> tuple[AgentService, _ContextAdapter, str]:
    context = _ContextAdapter()
    bundle = replace(make_contract_fake_bundle(), context=context)
    registry = AgentTypeRegistry()
    registry.register(_ReconcileAgentType())
    service = AgentService(
        tmp_path / ".agent_runtime",
        agent_types=registry,
        provider_registry=ProviderRegistry((bundle,)),
    )
    service.home_service.create_home(
        ProviderHomeSpec(provider_type="contract_fake", home_id="contract-home")
    )
    agent = service.create_agent("scope", "reconcile-agent")
    session = ProviderSessionLocator(
        provider_type="contract_fake",
        session_id="session-1",
        home_id="contract-home",
        created_at="2026-09-12T00:00:00Z",
    )
    service.store.update_session_locators(agent.agent_id, session_locator=session)
    service.store.write_context_maintenance(
        agent.agent_id,
        AgentContextMaintenanceJournal(
            agent_id=agent.agent_id,
            provider_type="contract_fake",
            session_id=session.session_id,
            status=AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL,
            trigger="before_agent_step",
            prepared_at="2026-09-12T00:00:00Z",
            started_at="2026-09-12T00:00:01Z",
            provider_operation_id="compact-1",
            baseline={"event_count": 1},
            error_type="AgentContextCompactionTimeout",
        ),
    )
    return service, context, agent.agent_id


def test_context_maintenance_preview_is_sanitized_and_tokenized(tmp_path: Path) -> None:
    service, _, agent_id = _service(tmp_path)

    preview = service.inspect_agent_context_maintenance(agent_id)

    assert preview is not None
    assert preview.agent_id == agent_id
    assert preview.session_id == "session-1"
    assert preview.status is AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL
    assert preview.unresolved is True
    assert len(preview.reconciliation_token) == 64
    assert not hasattr(preview, "baseline")


def test_context_reconciliation_rejects_stale_token_before_agent_mutation(
    tmp_path: Path,
) -> None:
    service, context, agent_id = _service(tmp_path)
    preview = service.inspect_agent_context_maintenance(agent_id)
    assert preview is not None
    before_agent = service.get_agent(agent_id)
    journal = service.store.read_context_maintenance(agent_id)
    assert journal is not None
    service.store.write_context_maintenance(agent_id, replace(journal, trigger="changed"))

    with pytest.raises(AgentContextMaintenanceBlocked, match="token changed"):
        service.reconcile_agent_context_maintenance(
            agent_id,
            expected_reconciliation_token=preview.reconciliation_token,
        )

    assert service.get_agent(agent_id) == before_agent
    assert context.requests == []


def test_context_reconciliation_unconfirmed_preserves_business_truth(tmp_path: Path) -> None:
    service, _, agent_id = _service(tmp_path)
    preview = service.inspect_agent_context_maintenance(agent_id)
    assert preview is not None
    before_journal = service.store.read_context_maintenance(agent_id)

    with pytest.raises(AgentContextMaintenanceBlocked, match="has not confirmed"):
        service.reconcile_agent_context_maintenance(
            agent_id,
            expected_reconciliation_token=preview.reconciliation_token,
        )

    assert service.store.read_context_maintenance(agent_id) == before_journal
    assert service.get_agent(agent_id).status == "idle"


@pytest.mark.parametrize(
    ("status", "session_id"),
    [
        ("ambiguous", "session-1"),
        ("compacted", "different-session"),
    ],
)
def test_context_reconciliation_rejects_unconfirmed_or_wrong_session_result(
    tmp_path: Path,
    status: str,
    session_id: str,
) -> None:
    service, context, agent_id = _service(tmp_path)
    preview = service.inspect_agent_context_maintenance(agent_id)
    assert preview is not None
    before_journal = service.store.read_context_maintenance(agent_id)
    context.result = ProviderContextCompactionResult(
        session_id=session_id,
        status=status,
        reason="provider evidence is not authoritative",
        started_at="2026-09-12T00:00:01Z",
        completed_at="2026-09-12T00:00:10Z",
        provider_operation_id="compact-1",
    )

    with pytest.raises(AgentContextMaintenanceBlocked):
        service.reconcile_agent_context_maintenance(
            agent_id,
            expected_reconciliation_token=preview.reconciliation_token,
        )

    assert service.store.read_context_maintenance(agent_id) == before_journal
    assert service.get_agent(agent_id).status == "idle"


def test_context_reconciliation_confirms_exact_admitted_journal(tmp_path: Path) -> None:
    service, context, agent_id = _service(tmp_path)
    preview = service.inspect_agent_context_maintenance(agent_id)
    assert preview is not None
    context.result = ProviderContextCompactionResult(
        session_id="session-1",
        status="compacted",
        reason="confirmed",
        started_at="2026-09-12T00:00:01Z",
        completed_at="2026-09-12T00:00:10Z",
        provider_operation_id="compact-1",
    )

    reconciled = service.reconcile_agent_context_maintenance(
        agent_id,
        expected_reconciliation_token=preview.reconciliation_token,
    )

    assert reconciled is not None
    assert reconciled.status is AgentContextMaintenanceJournalStatus.CONFIRMED
    assert reconciled.baseline == {"event_count": 1}
    assert service.get_agent(agent_id).status == "idle"
    after = service.inspect_agent_context_maintenance(agent_id)
    assert after is not None and after.unresolved is False
    assert after.reconciliation_token != preview.reconciliation_token
