from types import SimpleNamespace

import pytest

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    AgentRoleBindings,
    AgentStepState,
    FlowRequest,
    FlowStatus,
    FlowStepValidationError,
    StepStatus,
)
from agent_runtime_kit.flow.provider_switch import (
    apply_provider_switch,
    plan_provider_switch,
)
from test_agent_step_restart import RestartAgentStep, _service


def setup(tmp_path, *, source_provider="grok"):
    old = Agent("old", "scope", "ReviewerAgent", source_provider, "ReviewerAgent")
    service, ark, schedule, agents = _service(tmp_path, agent=old)
    spec = SimpleNamespace(
        agent_type="ReviewerAgent",
        provider_type="codex",
        default_home_id="ReviewerLuna",
    )
    agents.agent_types = SimpleNamespace(list=lambda: [spec])
    agents.list_agents = lambda: list(agents.agents.values())
    agents.home_service = SimpleNamespace(
        get_home=lambda *a: SimpleNamespace(
            status="active",
            materialization_manifest_hash="home-v1",
            base_config_fingerprint="config-v1",
        ),
        build_execution_context=lambda *a: None,
    )
    return service, ark, schedule, agents


def step(service, name, status=StepStatus.SUSPENDED, *, bound=True, scope="scope"):
    flow_id = service.start_flow(
        FlowRequest(flow_type="restart_flow", scope_id=scope, params={}), enqueue=False
    )
    record = RestartAgentStep(
        step_id=name,
        flow_id=flow_id,
        scope_id=scope,
        status=status,
        state=AgentStepState(
            agent_role="reviewer",
            agent_type="ReviewerAgent",
            provider_type="grok",
            home_id="ReviewerAgent",
        ),
        agent_bindings=AgentRoleBindings(by_role={"reviewer": "old"} if bound else {}),
    )
    service.store.create_step(record)

    def attach(flow):
        flow.status = FlowStatus.RUNNING
        flow.current_step_id = name
        flow.step_ids.append(name)
        if bound:
            flow.agent_bindings.by_role["reviewer"] = "old"

    service.store.update_flow_record(flow_id, attach)
    return record


def plan(service):
    return plan_provider_switch(
        service, scope_id="scope", source_agent_ids=["old"], target_provider="codex"
    )


def apply(service, preview=None):
    return apply_provider_switch(
        service,
        scope_id="scope",
        source_agent_ids=["old"],
        target_provider="codex",
        expected_plan_hash=(preview or plan(service))["plan_hash"],
    )


@pytest.mark.parametrize("source_provider", ["grok", "codex"])
def test_shared_identity_and_suspended_history(tmp_path, source_provider):
    service, _, schedule, agents = setup(tmp_path, source_provider=source_provider)
    original = [
        step(service, "first"),
        step(service, "second"),
        step(service, "created", StepStatus.CREATED),
    ]
    preimages = [
        service.get_step(s.step_id).model_dump(mode="json") for s in original[:2]
    ]
    result = apply(service)
    assert result["complete"], result
    assert len(agents.created) == 1
    replacement = agents.created[0]
    assert (replacement.provider_type, replacement.home_id) == ("codex", "ReviewerLuna")
    assert replacement.session_locator is None
    assert agents.get_agent("old").status == "closed"
    assert schedule.step_ids == []
    for index, source in enumerate(original):
        flow = service.get_flow(source.flow_id)
        assert flow.agent_bindings.get("reviewer") == replacement.agent_id
        assert (
            service.get_step(flow.current_step_id).agent_bindings.get("reviewer")
            == replacement.agent_id
        )
        if index < 2:
            assert (
                service.get_step(source.step_id).model_dump(mode="json")
                == preimages[index]
            )
            assert (
                service.get_step(flow.current_step_id).state.restart_of_step_id
                == source.step_id
            )
    assert apply(service)[
        "complete"
    ]  # idempotent retirement, never create another identity
    assert len(agents.created) == 1


def test_pure_history_and_unbound_future_step(tmp_path):
    service, _, _, agents = setup(tmp_path)
    pending = step(service, "unbound", StepStatus.CREATED, bound=False)
    assert apply(service)["complete"]
    state = service.get_step(pending.step_id).state
    assert (state.provider_type, state.home_id) == ("codex", "ReviewerLuna")
    assert agents.created == []
    assert agents.get_agent("old").status == "closed"


def test_stale_or_active_rejected_without_creating(tmp_path):
    service, ark, _, agents = setup(tmp_path)
    step(service, "one")
    preview = plan(service)
    step(service, "two")
    with pytest.raises(FlowStepValidationError, match="changed"):
        apply(service, preview)
    ark.schedule_service.active_flow_advances = ["running"]
    with pytest.raises(FlowStepValidationError, match="zero active"):
        apply(service)
    assert not agents.created


def test_post_commit_retirement_failure_can_be_retried(tmp_path):
    service, _, _, agents = setup(tmp_path)
    step(service, "one")
    close = agents.close_agent
    agents.close_agent = lambda key: (_ for _ in ()).throw(RuntimeError("retirement"))
    result = apply(service)
    assert not result["complete"]
    assert result["receipts"][0]["bindings_committed"]
    agents.close_agent = close
    assert apply(service)["complete"]
    assert len(agents.created) == 1


def test_tx_failure_preserves_source_and_closes_candidate(tmp_path, monkeypatch):
    service, _, _, agents = setup(tmp_path)
    source = step(service, "one")
    before = service.get_flow(source.flow_id).model_dump(mode="json")
    monkeypatch.setattr(
        service,
        "_build_replacement_step",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("injected")),
    )
    result = apply(service)
    assert not result["complete"]
    assert service.get_flow(source.flow_id).model_dump(mode="json") == before
    assert agents.get_agent("old").status == "idle"
    assert agents.created[0].status == "closed"


@pytest.mark.parametrize("state", ["closed", "missing"])
def test_step_only_unselected_identity_blocks(tmp_path, state):
    service, _, _, agents = setup(tmp_path)
    pending = step(service, "one", StepStatus.CREATED, bound=False)
    if state == "closed":
        agents.agents["other"] = Agent(
            "other", "scope", "ReviewerAgent", "codex", "ReviewerLuna", status="closed"
        )
    service.store.update_step_record(
        pending.step_id, lambda s: s.agent_bindings.by_role.update(reviewer="other")
    )
    assert plan(service)["blockers"]
    with pytest.raises(FlowStepValidationError):
        apply(service)
    assert not agents.created


def test_external_step_reference_blocks(tmp_path):
    service, _, _, _ = setup(tmp_path)
    pending = step(service, "external", StepStatus.CREATED, bound=False, scope="other")
    service.store.update_step_record(
        pending.step_id, lambda s: s.agent_bindings.by_role.update(reviewer="old")
    )
    assert any("cross_scope_reference" in item for item in plan(service)["blockers"])


def test_actual_store_failure_rolls_back_multiple_objects(tmp_path, monkeypatch):
    service, _, _, agents = setup(tmp_path)
    original = [step(service, "one"), step(service, "two")]
    before = [service.get_flow(s.flow_id).model_dump(mode="json") for s in original]
    write = service.store._write_flow
    count = 0

    def fail_second(flow):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected disk error")
        write(flow)

    monkeypatch.setattr(service.store, "_write_flow", fail_second)
    result = apply(service)
    assert not result["complete"]
    assert [
        service.get_flow(s.flow_id).model_dump(mode="json") for s in original
    ] == before
    assert agents.created[0].status == "closed"
    assert len(service.store.list_steps()) == 2


def test_dispatch_reference_migrates_without_rewriting_input_or_submission(tmp_path):
    from agent_runtime_kit.flow.standard_steps.dispatch_step import (
        DispatchStep,
        DispatchStepState,
    )

    service, _, _, agents = setup(tmp_path)
    service.step_registry.register(DispatchStep)
    flow_id = service.start_flow(
        FlowRequest(flow_type="restart_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    pending = DispatchStep(
        step_id="dispatch",
        flow_id=flow_id,
        scope_id="scope",
        state=DispatchStepState(
            source_step_id="submitted",
            source_submission_id="trusted",
            requests=[
                FlowRequest(
                    flow_type="restart_flow",
                    scope_id="scope",
                    params={"agent_id": "old"},
                )
            ],
        ),
    )
    service.store.create_step(pending)
    assert apply(service)["complete"]
    state = service.get_step("dispatch").state
    assert state.requests[0].params["agent_id"] == agents.created[0].agent_id
    assert state.source_submission_id == "trusted"
    assert state.created_children == []


def test_preflight_failure_before_any_agent_creation(tmp_path):
    service, _, _, agents = setup(tmp_path)
    step(service, "one")
    agents.home_service.build_execution_context = lambda *a: (_ for _ in ()).throw(
        RuntimeError("missing credential")
    )
    with pytest.raises(RuntimeError):
        apply(service)
    assert not agents.created


def test_default_update_failure_retains_retirement_receipts(tmp_path, monkeypatch):
    service, _, _, agents = setup(tmp_path)
    pending = step(service, "future", StepStatus.CREATED, bound=False)
    monkeypatch.setattr(
        service.store, "_write_step", lambda *a: (_ for _ in ()).throw(OSError("disk"))
    )
    result = apply(service)
    assert not result["complete"]
    assert result["receipts"][0]["retired"]
    assert agents.get_agent("old").status == "closed"
    assert result["plan"]["unbound_created_steps"] == [pending.step_id]


def test_newer_unrelated_flow_does_not_hide_failed_execution(tmp_path):
    service, _, _, _ = setup(tmp_path)
    failed = step(service, "failed")
    service.store.update_flow_record(
        failed.flow_id,
        lambda f: (
            setattr(f, "status", FlowStatus.FAILED),
            setattr(f, "current_step_id", None),
        ),
    )
    step(service, "newer")
    assert any("failed_current_flow" in x for x in plan(service)["blockers"])


def test_explicit_app_proven_historical_failure_is_in_plan_identity(tmp_path):
    service, _, _, _ = setup(tmp_path)
    failed = step(service, 'old_failed')
    service.store.update_flow_record(failed.flow_id, lambda f: (
        setattr(f, 'status', FlowStatus.FAILED), setattr(f, 'current_step_id', None)))
    blocked = plan(service)
    admitted = plan_provider_switch(service, scope_id='scope', source_agent_ids=['old'],
                                    target_provider='codex', historical_failed_flow_ids=(failed.flow_id,))
    assert blocked['blockers'] and not admitted['blockers']
    assert admitted['historical_failed_flow_ids'] == [failed.flow_id]
    assert admitted['plan_hash'] != blocked['plan_hash']


def test_migrate_executed_suspended_step_in_created_flow(tmp_path):
    service, _, schedule, _ = setup(tmp_path)
    source = step(service, "first")
    service.store.update_step_record(source.step_id, lambda s: setattr(s, "started_at", "2026-09-14T00:00:00Z"))
    service.store.update_flow_record(source.flow_id, lambda f: setattr(f, "status", FlowStatus.CREATED))
    before = service.get_step(source.step_id).model_dump(mode="json")
    assert apply(service)["complete"]
    flow = service.get_flow(source.flow_id)
    assert flow.status is FlowStatus.RUNNING
    assert flow.current_step_id != source.step_id
    assert service.get_step(source.step_id).model_dump(mode="json") == before
    assert schedule.step_ids == []
