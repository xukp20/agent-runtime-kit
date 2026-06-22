from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
    BaseStepError,
    BaseStepResult,
    BaseStepState,
    BaseSubmission,
    FlowContext,
    FlowStatus,
    FlowStepStore,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepRunContext,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class CtxFlowParams(BaseModel):
    target: str = "target"


class CtxFlowState(BaseFlowState):
    state_type: str = "ctx_flow_state"
    marker: str = "initial"


class CtxFlow(BaseFlow):
    flow_type: ClassVar[str] = "ctx_flow"
    Params: ClassVar[type[BaseModel]] = CtxFlowParams
    State: ClassVar[type[BaseFlowState]] = CtxFlowState

    @classmethod
    def build_from_request(cls, ctx: object) -> "CtxFlow":
        raise NotImplementedError


class CtxStepState(BaseStepState):
    state_type: str = "ctx_step_state"
    marker: str = "initial"


class CtxStep(BaseStep):
    step_type: ClassVar[str] = "ctx_step"
    State: ClassVar[type[BaseStepState]] = CtxStepState


class StrictCtxSubmission(BaseSubmission):
    submission_type: str = "strict_submit"
    payload: str


class StrictCtxStep(CtxStep):
    step_type: ClassVar[str] = "strict_ctx_step"
    Submission: ClassVar[type[BaseSubmission] | None] = None
    Submissions: ClassVar[dict[str, type[BaseSubmission]]] = {
        "strict_submit": StrictCtxSubmission,
    }
    SubmitTools: ClassVar[set[str] | None] = {"submit_strict"}


@dataclass
class FakeFlowService:
    store: FlowStepStore


def make_store(runtime_root: Path) -> FlowStepStore:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(CtxFlow)
    step_registry.register(CtxStep)
    step_registry.register(StrictCtxStep)
    return FlowStepStore(runtime_root, flow_registry=flow_registry, step_registry=step_registry)


def make_ark(store: FlowStepStore) -> ARKServices:
    return ARKServices(flow_service=FakeFlowService(store))


def seed_running_step(store: FlowStepStore) -> None:
    store.create_flow(CtxFlow(flow_id="flow-1", scope_id="scope", state=CtxFlowState()))
    store.create_step(
        CtxStep(
            step_id="step-1",
            flow_id="flow-1",
            scope_id="scope",
            state=CtxStepState(),
            status=StepStatus.RUNNING,
        )
    )


def seed_strict_running_step(store: FlowStepStore) -> None:
    store.create_flow(CtxFlow(flow_id="flow-1", scope_id="scope", state=CtxFlowState()))
    store.create_step(
        StrictCtxStep(
            step_id="step-1",
            flow_id="flow-1",
            scope_id="scope",
            state=CtxStepState(),
            status=StepStatus.RUNNING,
        )
    )


def test_flow_context_create_step_updates_flow_history(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    store.create_flow(CtxFlow(flow_id="flow-1", scope_id="scope"))

    with store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update("flow-1")
        ctx = FlowContext(
            ark=make_ark(store),
            app=AppServices(),
            flow=flow,
            tx=tx,
        )
        step_id = ctx.create_step(CtxStep(step_id="step-1", flow_id="flow-1", scope_id="scope"))

    restored_flow = store.get_flow("flow-1")
    assert step_id == "step-1"
    assert restored_flow.step_ids == ["step-1"]
    assert restored_flow.current_step_id == "step-1"
    assert store.get_step("step-1").flow_id == "flow-1"


def test_flow_context_sets_result_and_waiting_state(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    store.create_flow(CtxFlow(flow_id="flow-1", scope_id="scope"))

    with store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update("flow-1")
        ctx = FlowContext(ark=make_ark(store), app=AppServices(), flow=flow, tx=tx)
        ctx.set_flow_waiting()

    assert store.get_flow("flow-1").status is FlowStatus.WAITING

    with store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update("flow-1")
        ctx = FlowContext(ark=make_ark(store), app=AppServices(), flow=flow, tx=tx)
        ctx.set_flow_result(BaseFlowResult(result_type="done", summary="finished"))

    completed = store.get_flow("flow-1")
    assert completed.status is FlowStatus.COMPLETED
    assert completed.result is not None
    assert completed.result.result_type == "done"
    assert completed.finished_at is not None


def test_step_run_context_accepts_submission_and_complete_preserves_it(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_running_step(store)
    ctx = StepRunContext(ark=make_ark(store), app=AppServices(), step_id="step-1", flow_id="flow-1", scope_id="scope")

    submitted = ctx.accept_step_submission(
        BaseSubmission(submission_id="sub-1", submission_type="result", tool_name="submit_ready")
    )
    receipt = ctx.complete_step(BaseStepResult(result_type="done", summary="ok"))
    completed = ctx.load_step()

    assert submitted.submission is not None
    assert receipt.status == "completed"
    assert receipt.result_type == "done"
    assert completed.status is StepStatus.COMPLETED
    assert completed.result is not None
    assert completed.submission is not None
    assert completed.submission.submission_id == "sub-1"


def test_step_run_context_rejects_submission_from_wrong_agent(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_running_step(store)
    ctx = StepRunContext(ark=make_ark(store), app=AppServices(), step_id="step-1", flow_id="flow-1", scope_id="scope")

    with pytest.raises(FlowStepValidationError):
        ctx.accept_step_submission(
            BaseSubmission(
                submission_id="sub-1",
                submission_type="result",
                tool_name="submit_ready",
                submitted_by_agent_id="agent-a",
            ),
            expected_agent_id="agent-b",
        )

    assert ctx.load_step().submission is None


def test_step_run_context_rejects_disallowed_submission_type_and_tool(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_strict_running_step(store)
    ctx = StepRunContext(ark=make_ark(store), app=AppServices(), step_id="step-1", flow_id="flow-1", scope_id="scope")

    with pytest.raises(FlowStepValidationError):
        ctx.accept_step_submission(
            BaseSubmission(submission_id="sub-1", submission_type="result", tool_name="submit_strict")
        )
    with pytest.raises(FlowStepValidationError):
        ctx.accept_step_submission(
            StrictCtxSubmission(
                submission_id="sub-2",
                tool_name="wrong_tool",
                payload="ok",
            )
        )

    accepted = ctx.accept_step_submission(
        StrictCtxSubmission(
            submission_id="sub-3",
            tool_name="submit_strict",
            payload="ok",
        )
    )

    assert isinstance(accepted.submission, StrictCtxSubmission)
    assert accepted.submission.payload == "ok"


def test_step_run_context_rejects_complete_for_non_running_step(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    store.create_flow(CtxFlow(flow_id="flow-1", scope_id="scope"))
    store.create_step(CtxStep(step_id="step-1", flow_id="flow-1", scope_id="scope", status=StepStatus.CREATED))
    ctx = StepRunContext(ark=make_ark(store), app=AppServices(), step_id="step-1", flow_id="flow-1", scope_id="scope")

    with pytest.raises(FlowStepValidationError):
        ctx.complete_step(BaseStepResult(result_type="done"))


def test_step_run_context_rejects_duplicate_terminal_write(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_running_step(store)
    ctx = StepRunContext(ark=make_ark(store), app=AppServices(), step_id="step-1", flow_id="flow-1", scope_id="scope")

    ctx.complete_step(BaseStepResult(result_type="done"))

    with pytest.raises(FlowStepValidationError):
        ctx.complete_step(BaseStepResult(result_type="done_again"))


def test_step_run_context_fail_step_writes_framework_error(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_running_step(store)
    ctx = StepRunContext(ark=make_ark(store), app=AppServices(), step_id="step-1", flow_id="flow-1", scope_id="scope")

    receipt = ctx.fail_step(BaseStepError(error_type="runtime", message="boom"))
    failed = store.get_step("step-1")

    assert receipt.status == "failed"
    assert receipt.error_type == "runtime"
    assert failed.status is StepStatus.FAILED
    assert failed.error is not None
    assert failed.error.message == "boom"


def test_step_run_context_update_helpers_use_latest_truth(tmp_path: Path) -> None:
    store = make_store(tmp_path / ".agent_runtime")
    seed_running_step(store)
    ctx = StepRunContext(ark=make_ark(store), app=AppServices(), step_id="step-1", flow_id="flow-1", scope_id="scope")

    ctx.update_step(lambda step: setattr(step.state, "marker", "step-updated"))
    ctx.update_flow(lambda flow: setattr(flow.state, "marker", "flow-updated"))

    step = store.get_step("step-1")
    flow = store.get_flow("flow-1")
    assert isinstance(step.state, CtxStepState)
    assert step.state.marker == "step-updated"
    assert isinstance(flow.state, CtxFlowState)
    assert flow.state.marker == "flow-updated"
