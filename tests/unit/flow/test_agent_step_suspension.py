from pathlib import Path
from typing import ClassVar

import pytest

from agent_runtime_kit.agent.models import AgentProviderTurnFailed
from agent_runtime_kit.flow import (
    AgentStep,
    AgentStepState,
    BaseStep,
    BaseStepError,
    BaseStepResult,
    BaseStepState,
    FlowStatus,
    StepStatus,
    StepSuspensionReceipt,
)

from test_agent_step import attach_agent_step, create_flow, make_services


def test_agent_step_suspends_on_standard_provider_terminal_failure(tmp_path: Path) -> None:
    flow_service, step_service, agent_service = make_services(tmp_path / ".agent_runtime")
    flow_id = create_flow(flow_service)
    flow_service.store.update_flow_record(flow_id, lambda flow: setattr(flow, "status", FlowStatus.RUNNING))
    step = AgentStep(
        step_id="provider-failure",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(agent_role="worker"),
    )
    step.agent_bindings.by_role["worker"] = "agent-bound"
    attach_agent_step(flow_service, flow_id, step)

    def fail_wait(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AgentProviderTurnFailed(
            provider_type="scripted",
            provider_error_type="provider_rate_limit",
            code="429",
            retryable=True,
            run_id="run-1",
            session_id="session-1",
            turn_id="turn-1",
        )

    agent_service.wait_agent = fail_wait
    step_service.run_step(step.step_id)

    persisted = flow_service.get_step(step.step_id)
    flow = flow_service.get_flow(flow_id)
    assert persisted.status is StepStatus.SUSPENDED
    assert persisted.error is not None
    assert persisted.error.error_type == "agent_provider_turn_failed"
    assert persisted.error.details == {
        "provider_type": "scripted",
        "provider_error_type": "provider_rate_limit",
        "retryable": True,
        "operator_action_required": False,
        "run_id": "run-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
    }
    assert flow.status is FlowStatus.RUNNING
    assert flow.current_step_id == step.step_id


def test_agent_step_completion_exception_still_fails_flow(tmp_path: Path) -> None:
    class CheckerBugStep(AgentStep):
        step_type: ClassVar[str] = "checker_bug_agent_step"

        def check_completion(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            raise TypeError("checker bug")

    flow_service, step_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_service.step_registry.register(CheckerBugStep)
    flow_id = create_flow(flow_service)
    flow_service.store.update_flow_record(flow_id, lambda flow: setattr(flow, "status", FlowStatus.RUNNING))
    step = CheckerBugStep(
        step_id="checker-bug",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(agent_role="worker"),
    )
    step.agent_bindings.by_role["worker"] = "agent-bound"
    attach_agent_step(flow_service, flow_id, step)

    step_service.run_step(step.step_id)

    persisted = flow_service.get_step(step.step_id)
    assert persisted.status is StepStatus.FAILED
    assert persisted.error is not None and persisted.error.error_type == "step_run_exception"


def test_invalid_step_suspension_receipt_is_failed_and_terminal_handled(tmp_path: Path) -> None:
    class InvalidSuspensionStep(BaseStep):
        step_type: ClassVar[str] = "invalid_suspension_step"
        State: ClassVar[type[BaseStepState]] = BaseStepState

        def run(self, ctx):  # noqa: ANN001, ANN201
            return StepSuspensionReceipt(
                step_id=ctx.step_id,
                flow_id=ctx.flow_id,
                scope_id=ctx.scope_id,
                error_type="agent_provider_turn_failed",
                finished_at="2026-09-10T00:00:00Z",
            )

    flow_service, step_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_service.step_registry.register(InvalidSuspensionStep)
    flow_id = create_flow(flow_service)
    flow_service.store.update_flow_record(flow_id, lambda flow: setattr(flow, "status", FlowStatus.RUNNING))
    step = InvalidSuspensionStep(step_id="invalid-suspend", flow_id=flow_id, scope_id="scope")
    attach_agent_step(flow_service, flow_id, step)

    step_service.run_step(step.step_id)

    persisted = flow_service.get_step(step.step_id)
    assert persisted.status is StepStatus.FAILED
    assert persisted.error is not None
    assert persisted.error.error_type == "invalid_suspension_receipt"
    assert flow_service.get_flow(flow_id).current_step_id is None


@pytest.mark.parametrize("malformation", ["result", "error_type", "finished_at"])
def test_persisted_suspension_evidence_must_exactly_match_receipt(
    tmp_path: Path,
    malformation: str,
) -> None:
    class MalformedPersistedSuspensionStep(BaseStep):
        step_type: ClassVar[str] = "malformed_persisted_suspension_step"
        State: ClassVar[type[BaseStepState]] = BaseStepState

        def run(self, ctx):  # noqa: ANN001, ANN201
            receipt = ctx.suspend_step(
                BaseStepError(error_type="provider_failure", message="offline")
            )

            def corrupt(step):  # noqa: ANN001
                if malformation == "result":
                    step.result = BaseStepResult(result_type="impossible")
                elif malformation == "error_type":
                    step.error.error_type = "different_failure"
                else:
                    step.finished_at = "2026-09-10T00:00:00Z"

            ctx.update_step(corrupt)
            return receipt

    flow_service, step_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_service.step_registry.register(MalformedPersistedSuspensionStep)
    flow_id = create_flow(flow_service)
    flow_service.store.update_flow_record(flow_id, lambda flow: setattr(flow, "status", FlowStatus.RUNNING))
    step = MalformedPersistedSuspensionStep(step_id="malformed", flow_id=flow_id, scope_id="scope")
    attach_agent_step(flow_service, flow_id, step)

    step_service.run_step(step.step_id)

    persisted = flow_service.get_step(step.step_id)
    assert persisted.status is StepStatus.FAILED
    assert persisted.error is not None and persisted.error.error_type == "invalid_suspension_receipt"
