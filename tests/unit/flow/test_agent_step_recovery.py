from dataclasses import replace
from pathlib import Path
from threading import Thread
from time import sleep
from typing import ClassVar

import pytest

from agent_runtime_kit.flow import (
    AgentRoleBindings,
    AgentStepState,
    BaseStepError,
    FlowStatus,
    FlowStepValidationError,
    FlowRequest,
    LostStepSubmissionFinalizeUnavailableError,
    StepStatus,
    StepNotFoundError,
)

from test_agent_step_restart import RestartAgentStep, RestartFlow, _failed_step, _service
from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.agent.provider_contracts import (
    AgentTurnView,
    ProviderRunState,
    ProviderSessionLocator,
    ProviderTurnLocator,
    ProviderTurnResult,
)
from agent_runtime_kit.flow import BaseSubmission


class FinalizableRestartAgentStep(RestartAgentStep):
    step_type: ClassVar[str] = "finalizable_restart_agent_step"
    offline_submission_finalize_supported: ClassVar[bool] = True


def _lost_step(
    service,
    *,
    agent_id: str,
    submission=None,
    step_cls=RestartAgentStep,
) -> tuple[str, str]:  # noqa: ANN001
    flow_id = service.start_flow(
        FlowRequest(flow_type="restart_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    step = step_cls(
        step_id="lost-reviewer",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.RUNNING,
        state=AgentStepState(agent_role="reviewer", agent_type="ReviewerAgent"),
        submission=submission,
        agent_bindings=AgentRoleBindings(by_role={"reviewer": agent_id}),
    )
    service.store.create_step(step)

    def attach(flow):  # noqa: ANN001
        flow.status = FlowStatus.RUNNING
        flow.step_ids.append(step.step_id)
        flow.current_step_id = step.step_id
        flow.agent_bindings.by_role["reviewer"] = agent_id

    service.store.update_flow_record(flow_id, attach)
    return flow_id, step.step_id


def test_recover_suspended_agent_step_preserves_complete_source_preimage(tmp_path: Path) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, schedule, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    service.store.update_step_record(
        step_id,
        lambda step: (
            setattr(step, "status", StepStatus.SUSPENDED),
            setattr(step, "error", BaseStepError(error_type="agent_provider_failure", message="offline")),
        ),
    )
    source_preimage = service.get_step(step_id).model_dump(mode="json")
    preview = service.inspect_agent_step_recovery(step_id)

    receipt = service.recover_agent_step(
        step_id=step_id,
        expected_status=StepStatus.SUSPENDED,
        expected_recovery_token=preview.recovery_token,
        action="resume_suspended",
        agent_mode="fresh",
    )

    assert service.get_step(step_id).model_dump(mode="json") == source_preimage
    assert service.get_step(receipt.replacement_step_id).status is StepStatus.CREATED
    assert service.get_flow(flow_id).current_step_id == receipt.replacement_step_id
    assert schedule.step_ids == [receipt.replacement_step_id]


def test_explicit_fresh_recovery_does_not_inspect_old_context_journal(
    tmp_path: Path,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    service.store.update_step_record(
        step_id,
        lambda step: (
            setattr(step, "status", StepStatus.SUSPENDED),
            setattr(
                step,
                "error",
                BaseStepError(
                    error_type="agent_context_maintenance_blocked",
                    message="blocked",
                ),
            ),
        ),
    )

    def reject_inspection(_agent_id: str):
        raise AssertionError("fresh recovery must not inspect the old context journal")

    agents.inspect_agent_context_maintenance = reject_inspection
    preview = service.inspect_agent_step_recovery(step_id)

    receipt = service.recover_agent_step(
        step_id=step_id,
        expected_status=StepStatus.SUSPENDED,
        expected_recovery_token=preview.recovery_token,
        action="resume_suspended",
        agent_mode="fresh",
    )

    assert receipt.agent_reused is False
    assert receipt.replacement_agent_id == "fresh-1"
    assert service.get_flow(flow_id).current_step_id == receipt.replacement_step_id


def test_suspended_agent_step_auto_uses_fresh_agent_when_context_is_unresolved(
    tmp_path: Path,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    service.store.update_step_record(
        step_id,
        lambda step: (
            setattr(step, "status", StepStatus.SUSPENDED),
            setattr(step, "error", BaseStepError(error_type="agent_context_maintenance_blocked", message="blocked")),
        ),
    )
    agents.unresolved_agent_ids.add(agent.agent_id)
    preview = service.inspect_agent_step_recovery(step_id)

    receipt = service.recover_agent_step(
        step_id=step_id,
        expected_status=StepStatus.SUSPENDED,
        expected_recovery_token=preview.recovery_token,
        action="resume_suspended",
        agent_mode="auto",
    )

    assert receipt.agent_reused is False
    assert receipt.replacement_agent_id == "fresh-1"
    assert service.get_flow(flow_id).current_step_id == receipt.replacement_step_id


@pytest.mark.parametrize("agent_mode", ["reuse", "fork_current"])
def test_suspended_agent_step_rejects_session_reuse_when_context_is_unresolved_without_mutation(
    tmp_path: Path,
    agent_mode: str,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, schedule, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    service.store.update_step_record(
        step_id,
        lambda step: (
            setattr(step, "status", StepStatus.SUSPENDED),
            setattr(step, "error", BaseStepError(error_type="agent_context_maintenance_blocked", message="blocked")),
        ),
    )
    agents.unresolved_agent_ids.add(agent.agent_id)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    preview = service.inspect_agent_step_recovery(step_id)

    with pytest.raises(FlowStepValidationError, match="context maintenance is unresolved"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.SUSPENDED,
            expected_recovery_token=preview.recovery_token,
            action="resume_suspended",
            agent_mode=agent_mode,
        )

    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow
    assert agents.created == []
    assert schedule.step_ids == []


@pytest.mark.parametrize("agent_mode", ["reuse", "fork_current"])
def test_lost_running_step_rejects_unresolved_session_mode_before_agent_repair(
    tmp_path: Path,
    agent_mode: str,
) -> None:
    agent = Agent(
        "reviewer",
        "scope",
        "ReviewerAgent",
        "codex",
        "ReviewerAgent",
        status="running",
    )
    service, _, schedule, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    agents.unresolved_agent_ids.add(agent.agent_id)
    before_agent = replace(agents.get_agent(agent.agent_id))
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    preview = service.inspect_agent_step_recovery(step_id)

    with pytest.raises(FlowStepValidationError, match="context maintenance is unresolved"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.RUNNING,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            agent_mode=agent_mode,
        )

    assert agents.get_agent(agent.agent_id) == before_agent
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow
    assert agents.created == []
    assert schedule.step_ids == []


def test_reconcile_suspended_agent_step_context_preserves_step_and_flow(
    tmp_path: Path,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, ark, schedule, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)

    def suspend(step):  # noqa: ANN001
        step.status = StepStatus.SUSPENDED
        step.error = BaseStepError(
            error_type="agent_context_maintenance_blocked",
            message="blocked",
        )
        step.state.env_overrides = {"CUSTOM": "value"}
        step.state.workdir_override = "/tmp/recovery-workdir"

    service.store.update_step_record(step_id, suspend)
    agents.unresolved_agent_ids.add(agent.agent_id)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    original_reconcile = agents.reconcile_agent_context_maintenance
    lock_observation: dict[str, bool] = {}

    def reconcile_with_lock_observation(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        lock_observation.update(
            pause=ark.pause_controller._lock._is_owned(),  # noqa: SLF001
            agent=agents.lock._is_owned(),  # noqa: SLF001
            flow=service.lock._is_owned(),  # noqa: SLF001
        )
        return original_reconcile(*args, **kwargs)

    agents.reconcile_agent_context_maintenance = reconcile_with_lock_observation

    preview = service.inspect_agent_step_context_maintenance(step_id)
    assert preview is not None and preview.unresolved is True
    reconciled = service.reconcile_agent_step_context_maintenance(
        step_id=step_id,
        expected_reconciliation_token=preview.reconciliation_token,
    )

    assert reconciled.unresolved is False
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow
    assert lock_observation == {"pause": True, "agent": True, "flow": True}
    assert agents.reconcile_calls == [
        {
            "agent_id": agent.agent_id,
            "env": {
                "CUSTOM": "value",
                "ARK_STEP_ID": step_id,
                "ARK_FLOW_ID": flow_id,
                "ARK_AGENT_ID": agent.agent_id,
            },
            "workdir": "/tmp/recovery-workdir",
        }
    ]

    recovery = service.inspect_agent_step_recovery(step_id)
    receipt = service.recover_agent_step(
        step_id=step_id,
        expected_status=StepStatus.SUSPENDED,
        expected_recovery_token=recovery.recovery_token,
        action="resume_suspended",
        agent_mode="reuse",
    )

    assert receipt.agent_reused is True
    assert receipt.replacement_agent_id == agent.agent_id
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).current_step_id == receipt.replacement_step_id
    assert schedule.step_ids == [receipt.replacement_step_id]


def test_reconcile_agent_step_context_requires_global_pause_without_mutation(
    tmp_path: Path,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, ark, _, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    service.store.update_step_record(
        step_id,
        lambda step: (
            setattr(step, "status", StepStatus.SUSPENDED),
            setattr(step, "error", BaseStepError(error_type="agent_context_maintenance_blocked", message="blocked")),
        ),
    )
    agents.unresolved_agent_ids.add(agent.agent_id)
    preview = service.inspect_agent_step_context_maintenance(step_id)
    assert preview is not None
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    ark.pause_controller.resume(None)
    ark.pause_controller.pause("scope")

    with pytest.raises(FlowStepValidationError, match="global runtime pause"):
        service.reconcile_agent_step_context_maintenance(
            step_id=step_id,
            expected_reconciliation_token=preview.reconciliation_token,
        )

    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow
    assert agents.reconcile_calls == []


def test_suspended_step_with_accepted_submission_rejects_resume_without_mutation(tmp_path: Path) -> None:
    from agent_runtime_kit.flow import BaseSubmission

    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(
        service,
        agent_id=agent.agent_id,
        submission=BaseSubmission(submission_id="accepted", submission_type="result", tool_name="submit"),
    )
    service.store.update_step_record(step_id, lambda step: setattr(step, "status", StepStatus.SUSPENDED))
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    preview = service.inspect_agent_step_recovery(step_id)

    assert preview.available_actions == []
    with pytest.raises(FlowStepValidationError):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.SUSPENDED,
            expected_recovery_token=preview.recovery_token,
            action="resume_suspended",
        )
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow


def test_recovery_token_rejects_flow_drift_without_mutation(tmp_path: Path) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _failed_step(service, agent_id=agent.agent_id)
    preview = service.inspect_agent_step_recovery(step_id)
    service.store.update_flow_record(flow_id, lambda flow: setattr(flow.state, "position_marker", "changed"))

    with pytest.raises(FlowStepValidationError, match="token changed"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
        )


def test_recover_lost_agent_step_without_submission_marks_runner_lost_and_restarts_atomically(
    tmp_path: Path,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    preview = service.inspect_agent_step_recovery(step_id)

    receipt = service.recover_agent_step(
        step_id=step_id,
        expected_status=StepStatus.RUNNING,
        expected_recovery_token=preview.recovery_token,
        action="restart",
    )

    source = service.get_step(step_id)
    assert source.status is StepStatus.FAILED
    assert source.error is not None and source.error.error_type == "runner_lost"
    assert service.get_flow(flow_id).current_step_id == receipt.replacement_step_id


def test_recover_failed_agent_step_uses_recovery_fork(tmp_path: Path) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, agents = _service(tmp_path, agent=agent)
    _, step_id = _failed_step(service, agent_id=agent.agent_id)
    preview = service.inspect_agent_step_recovery(step_id)

    receipt = service.recover_agent_step(
        step_id=step_id,
        expected_status=StepStatus.FAILED,
        expected_recovery_token=preview.recovery_token,
        action="restart",
        agent_mode="fork_current",
    )

    assert receipt.replacement_agent_id == "fork-1"
    assert service.get_step(receipt.replacement_step_id).agent_bindings.get("reviewer") == "fork-1"
    assert agents.created[0].agent_id == "fork-1"


def test_recovery_mutator_write_then_raise_compensates_without_ark_mutation(tmp_path: Path) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _failed_step(service, agent_id=agent.agent_id)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    events: list[str] = []
    preview = service.inspect_agent_step_recovery(step_id)

    def mutator(*_args):  # noqa: ANN002
        events.append("written")
        raise RuntimeError("LC write failed")

    with pytest.raises(RuntimeError, match="LC write failed"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            boundary_mutator=mutator,
            boundary_compensator=lambda: events.append("compensated"),
        )

    assert events == ["written", "compensated"]
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow


def test_recovery_enqueue_failure_does_not_compensate_committed_ark_truth(tmp_path: Path) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, schedule, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _failed_step(service, agent_id=agent.agent_id)
    events: list[str] = []
    preview = service.inspect_agent_step_recovery(step_id)

    def fail_enqueue(_step_id: str) -> None:
        raise RuntimeError("queue unavailable")

    schedule.enqueue_step = fail_enqueue
    with pytest.raises(RuntimeError, match="queue unavailable"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            boundary_mutator=lambda *_args: events.append("written"),
            boundary_compensator=lambda: events.append("compensated"),
        )

    flow = service.get_flow(flow_id)
    assert events == ["written"]
    assert flow.current_step_id is not None and flow.current_step_id != step_id
    assert service.get_step(flow.current_step_id).status is StepStatus.CREATED


def test_recovery_flow_transaction_commit_failure_compensates_and_preserves_preimages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_runtime_kit.flow.store import FlowStepMutationSession

    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _failed_step(service, agent_id=agent.agent_id)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    events: list[str] = []
    preview = service.inspect_agent_step_recovery(step_id)

    def fail_flush(_session):  # noqa: ANN001
        raise RuntimeError("flow transaction failed")

    monkeypatch.setattr(FlowStepMutationSession, "flush", fail_flush)
    with pytest.raises(RuntimeError, match="flow transaction failed"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            agent_mode="reuse",
            boundary_mutator=lambda *_args: events.append("written"),
            boundary_compensator=lambda: events.append("compensated"),
        )

    assert events == ["written", "compensated"]
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow


def test_recovery_replacement_write_failure_rolls_back_mid_flush_and_compensates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _failed_step(service, agent_id=agent.agent_id)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    events: list[str] = []
    preview = service.inspect_agent_step_recovery(step_id)
    original_write_step = service.store._write_step
    replacement_ids: list[str] = []

    def fail_replacement_once(step):  # noqa: ANN001
        if step.step_id != step_id and not replacement_ids:
            replacement_ids.append(step.step_id)
            raise OSError("replacement write failed")
        return original_write_step(step)

    monkeypatch.setattr(service.store, "_write_step", fail_replacement_once)
    with pytest.raises(OSError, match="replacement write failed"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            agent_mode="reuse",
            boundary_mutator=lambda *_args: events.append("written"),
            boundary_compensator=lambda: events.append("compensated"),
        )

    assert events == ["written", "compensated"]
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow
    assert len(replacement_ids) == 1
    with pytest.raises(StepNotFoundError):
        service.get_step(replacement_ids[0])
    service.store.assert_restorable_truth(scope_id="scope")


def test_recovery_late_replacement_index_failure_rolls_back_all_written_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _failed_step(service, agent_id=agent.agent_id)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    events: list[str] = []
    preview = service.inspect_agent_step_recovery(step_id)
    original_upsert = service.store._upsert_step_global_index
    failed = False
    replacement_ids: list[str] = []
    monkeypatch.setattr(service.store, "_step_exists", lambda _step_id: False)

    def fail_replacement_index_once(step, flow):  # noqa: ANN001
        nonlocal failed
        if step.step_id != step_id:
            replacement_ids.append(step.step_id)
        if step.step_id != step_id and not failed:
            failed = True
            raise OSError("replacement index failed")
        return original_upsert(step, flow)

    monkeypatch.setattr(service.store, "_upsert_step_global_index", fail_replacement_index_once)
    with pytest.raises(OSError, match="replacement index failed"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            agent_mode="reuse",
            boundary_mutator=lambda *_args: events.append("written"),
            boundary_compensator=lambda: events.append("compensated"),
        )

    assert events == ["written", "compensated"]
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow
    assert replacement_ids
    with pytest.raises(StepNotFoundError):
        service.get_step(replacement_ids[0])
    service.store.assert_restorable_truth(scope_id="scope")


def test_finalize_unavailable_preserves_exact_step_and_flow_preimages(tmp_path: Path) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, _ = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(
        service,
        agent_id=agent.agent_id,
        submission=BaseSubmission(submission_id="accepted", submission_type="result", tool_name="submit"),
    )
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    preview = service.inspect_agent_step_recovery(step_id)

    with pytest.raises(LostStepSubmissionFinalizeUnavailableError):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.RUNNING,
            expected_recovery_token=preview.recovery_token,
            action="finalize_submission",
        )

    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow


def test_finalize_submission_builds_deterministic_result_and_calls_stable_hook_once(
    tmp_path: Path,
) -> None:
    session = ProviderSessionLocator("codex", "session-1", "ReviewerAgent", "2026-09-10T00:00:00Z")
    turn = ProviderTurnLocator(session, "turn-1")
    agent = Agent(
        "reviewer",
        "scope",
        "ReviewerAgent",
        "codex",
        "ReviewerAgent",
        session_locator=session,
        latest_turn_locator=turn,
    )
    service, _, _, agents = _service(tmp_path, agent=agent)
    service.step_registry.register(FinalizableRestartAgentStep)
    flow_id, step_id = _lost_step(
        service,
        agent_id=agent.agent_id,
        submission=BaseSubmission(
            submission_id="accepted",
            submission_type="result",
            tool_name="submit",
            submitted_by_agent_id=agent.agent_id,
            summary="deterministic",
        ),
        step_cls=FinalizableRestartAgentStep,
    )
    result = ProviderTurnResult(
        provider_type="codex",
        run_id="run-1",
        session_locator=session,
        turn_locator=turn,
        status=ProviderRunState.COMPLETED,
        started_at="2026-09-10T00:00:00Z",
        completed_at="2026-09-10T00:01:00Z",
    )
    agents.provider_turn = AgentTurnView(locator=turn, result=result)
    hook_calls: list[str] = []
    original_hook = RestartFlow.after_step_terminal_stable
    RestartFlow.after_step_terminal_stable = lambda self, ctx: hook_calls.append(ctx.step.step_id)
    try:
        preview = service.inspect_agent_step_recovery(step_id)
        assert preview.available_actions == ["finalize_submission"]
        receipt = service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.RUNNING,
            expected_recovery_token=preview.recovery_token,
            action="finalize_submission",
        )
        with pytest.raises(FlowStepValidationError, match="status changed"):
            service.recover_agent_step(
                step_id=step_id,
                expected_status=StepStatus.RUNNING,
                expected_recovery_token=preview.recovery_token,
                action="finalize_submission",
            )
    finally:
        RestartFlow.after_step_terminal_stable = original_hook

    finalized = service.get_step(step_id)
    assert receipt.submission_disposition == "accepted_finalized"
    assert finalized.status is StepStatus.COMPLETED
    assert finalized.result is not None
    assert finalized.result.summary == "deterministic"
    assert hook_calls == [step_id]
    assert service.get_flow(flow_id).current_step_id is None


def test_post_hook_provider_turn_drift_rejects_with_exact_preimages(tmp_path: Path) -> None:
    session = ProviderSessionLocator("codex", "session-1", "ReviewerAgent", "2026-09-10T00:00:00Z")
    turn_1 = ProviderTurnLocator(session, "turn-1")
    turn_2 = ProviderTurnLocator(session, "turn-2")
    agent = Agent(
        "reviewer",
        "scope",
        "ReviewerAgent",
        "codex",
        "ReviewerAgent",
        session_locator=session,
        latest_turn_locator=turn_1,
    )
    service, _, _, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _failed_step(service, agent_id=agent.agent_id)

    def view(turn: ProviderTurnLocator) -> AgentTurnView:
        return AgentTurnView(
            locator=turn,
            result=ProviderTurnResult(
                provider_type="codex",
                run_id=f"run-{turn.turn_id}",
                session_locator=session,
                turn_locator=turn,
                status=ProviderRunState.COMPLETED,
                started_at="2026-09-10T00:00:00Z",
                completed_at="2026-09-10T00:01:00Z",
            ),
        )

    agents.provider_turn = view(turn_1)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    preview = service.inspect_agent_step_recovery(step_id)

    with pytest.raises(FlowStepValidationError, match="before commit"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            boundary_mutator=lambda *_args: setattr(agents, "provider_turn", view(turn_2)),
        )

    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow


def test_close_after_lost_agent_repair_rejects_without_step_flow_or_hook_mutation(
    tmp_path: Path,
) -> None:
    agent = Agent(
        "reviewer",
        "scope",
        "ReviewerAgent",
        "codex",
        "ReviewerAgent",
        status="running",
    )
    service, _, _, agents = _service(tmp_path, agent=agent)
    flow_id, step_id = _lost_step(service, agent_id=agent.agent_id)
    before_step = service.get_step(step_id).model_dump(mode="json")
    before_flow = service.get_flow(flow_id).model_dump(mode="json")
    events: list[str] = []
    preview = service.inspect_agent_step_recovery(step_id)

    def close_after_repair(*_args):  # noqa: ANN002
        events.append("hook")
        agents.agents[agent.agent_id].status = "closed"

    with pytest.raises(FlowStepValidationError, match="before commit"):
        service.recover_agent_step(
            step_id=step_id,
            expected_status=StepStatus.RUNNING,
            expected_recovery_token=preview.recovery_token,
            action="restart",
            boundary_mutator=close_after_repair,
            boundary_compensator=lambda: events.append("compensated"),
        )

    assert events == ["hook", "compensated"]
    assert service.get_step(step_id).model_dump(mode="json") == before_step
    assert service.get_flow(flow_id).model_dump(mode="json") == before_flow


def test_recovery_holds_agent_boundary_until_flow_commit_against_close(tmp_path: Path) -> None:
    agent = Agent("reviewer", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    service, _, _, agents = _service(tmp_path, agent=agent)
    _, step_id = _failed_step(service, agent_id=agent.agent_id)
    preview = service.inspect_agent_step_recovery(step_id)
    close_thread: Thread | None = None

    def contend_with_close(*_args):  # noqa: ANN002
        nonlocal close_thread
        close_thread = Thread(target=lambda: agents.close_agent(agent.agent_id))
        close_thread.start()
        sleep(0.05)
        assert close_thread.is_alive()

    receipt = service.recover_agent_step(
        step_id=step_id,
        expected_status=StepStatus.FAILED,
        expected_recovery_token=preview.recovery_token,
        action="restart",
        agent_mode="reuse",
        boundary_mutator=contend_with_close,
    )

    assert close_thread is not None
    close_thread.join(2)
    assert not close_thread.is_alive()
    assert service.get_flow(receipt.flow_id).current_step_id == receipt.replacement_step_id
