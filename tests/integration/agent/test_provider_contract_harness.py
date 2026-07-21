from __future__ import annotations

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    AgentProviderBundle,
    ModelBackendIdentity,
    ProviderContextUsage,
    ProviderControlAction,
    ProviderControlRequest,
    ProviderExecutionKind,
    ProviderRegistry,
    ProviderRunState,
    ProviderSessionLocator,
)
from agent_runtime_kit.agent.context import AgentContextCompactionStatus
from agent_runtime_kit.agent.homes import ProviderHomeSpec
from agent_runtime_kit.agent.models import AgentContextMaintenanceUnsupported
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry

from .provider_contract_harness import (
    assert_immediate_terminal_contract,
    assert_interrupt_contract,
    make_contract_fake_bundle,
    standard_request,
)


@pytest.mark.parametrize(
    "execution_kind",
    [
        ProviderExecutionKind.SDK,
        ProviderExecutionKind.SUBPROCESS_RPC,
        ProviderExecutionKind.PYTHON_LIBRARY,
    ],
)
def test_same_contract_accepts_sdk_subprocess_and_library_shapes(execution_kind) -> None:  # noqa: ANN001
    bundle = make_contract_fake_bundle(execution_kind=execution_kind)
    ProviderRegistry((bundle,))
    assert_immediate_terminal_contract(bundle)


def test_interrupt_contract_waits_for_terminal_barrier() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.RUNNING)
    assert_interrupt_contract(bundle)


def test_timeout_does_not_implicitly_cancel_active_run() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.RUNNING)
    handle = bundle.runtime.start(standard_request())
    with pytest.raises(TimeoutError):
        handle.wait_terminal(timeout_s=0.001)
    assert handle.poll_state() is ProviderRunState.RUNNING


def test_needs_input_is_resumable_through_neutral_control_action() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.NEEDS_INPUT)
    handle = bundle.runtime.start(standard_request())
    waiting = handle.wait_terminal(timeout_s=0.1)
    assert waiting.status is ProviderRunState.NEEDS_INPUT

    control = handle.control(
        ProviderControlRequest(
            action=ProviderControlAction.RESPOND_INPUT,
            requested_at="2026-07-21T10:00:00Z",
            content={"answer": "continue"},
        )
    )
    assert control.accepted
    assert control.terminal_confirmed
    assert handle.wait_terminal(timeout_s=0.1).status is ProviderRunState.COMPLETED


def test_event_cursor_is_monotonic_and_does_not_repeat_events() -> None:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.RUNNING)
    handle = bundle.runtime.start(standard_request())
    first = handle.drain_events()
    second = handle.drain_events(after_cursor=first.next_cursor)
    assert [event.sequence for event in first.events] == [0]
    assert second.events == ()


class _ContractAgentType(AgentType):
    agent_type = "contract-agent"
    start_prompt_template = "work"


class _InspectOnlyContextAdapter:
    provider_type = "contract_fake"

    def __init__(self) -> None:
        self.compact_calls = 0

    def inspect(self, request):  # noqa: ANN001, ANN201
        return ProviderContextUsage(
            session_id=request.session.session_id,
            observed_at="2026-07-21T10:00:00Z",
            source="contract_fake",
            available=True,
            used_tokens=90,
            effective_context_window_tokens=100,
            measurement="provider_reported",
            model_identity=request.session.backend_identity,
        )

    def compact(self, request):  # noqa: ANN001, ANN201
        self.compact_calls += 1
        raise AssertionError("compact must be capability-gated before adapter invocation")

    def reconcile(self, request):  # noqa: ANN001, ANN201
        return None


def _contract_agent_service(tmp_path, bundle: AgentProviderBundle) -> AgentService:  # noqa: ANN001
    agent_types = AgentTypeRegistry()
    agent_types.register(_ContractAgentType())
    return AgentService(
        tmp_path / ".agent_runtime",
        agent_types=agent_types,
        provider_registry=ProviderRegistry((bundle,)),
    )


def test_agent_service_resume_preserves_exact_provider_session_locator(tmp_path) -> None:  # noqa: ANN001
    bundle = make_contract_fake_bundle()
    service = _contract_agent_service(tmp_path, bundle)
    service.home_service.create_home(
        ProviderHomeSpec(provider_type="contract_fake", home_id="contract-home")
    )
    agent = service.create_agent(
        "scope-contract",
        "contract-agent",
        provider_type="contract_fake",
        home_id="contract-home",
    )
    exact = ProviderSessionLocator(
        provider_type="contract_fake",
        session_id="session-exact",
        home_id="contract-home",
        created_at="2026-07-21T10:00:00Z",
        native_locator={"database": "agents/a-1/opencode.db", "opaque": {"partition": 7}},
        backend_identity=ModelBackendIdentity(
            api_provider="backend-a",
            api_mode="responses",
            requested_model="model-a",
        ),
    )
    service.store.update_session_locators(
        agent.agent_id,
        session_locator=exact,
    )

    service.start_agent(agent.agent_id)
    result = service.wait_agent(agent.agent_id, timeout_s=2)

    runtime = bundle.runtime
    assert runtime.handles[-1].request.session_locator == exact
    assert result.session_locator == exact
    assert service.get_agent(agent.agent_id).session_locator == exact


def test_agent_service_compact_fails_closed_when_context_exists_without_capability(
    tmp_path,  # noqa: ANN001
) -> None:
    base = make_contract_fake_bundle()
    context = _InspectOnlyContextAdapter()
    bundle = AgentProviderBundle(
        descriptor=base.descriptor,
        runtime=base.runtime,
        home_renderer=base.home_renderer,
        context=context,
    )
    service = _contract_agent_service(tmp_path, bundle)
    service.home_service.create_home(
        ProviderHomeSpec(provider_type="contract_fake", home_id="contract-home")
    )
    agent = service.create_agent(
        "scope-contract",
        "contract-agent",
        provider_type="contract_fake",
        home_id="contract-home",
    )
    service.store.update_session_locators(
        agent.agent_id,
        session_locator=ProviderSessionLocator(
            provider_type="contract_fake",
            session_id="session-1",
            home_id="contract-home",
            created_at="2026-07-21T10:00:00Z",
        ),
    )

    with pytest.raises(AgentContextMaintenanceUnsupported, match="control.compact"):
        service.compact_agent(agent.agent_id)
    conditional = service.compact_agent_if_needed(agent.agent_id, threshold=0.5)

    assert conditional.status is AgentContextCompactionStatus.UNSUPPORTED
    assert context.compact_calls == 0
