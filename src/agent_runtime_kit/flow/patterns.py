from __future__ import annotations

import uuid

from .contexts import FlowContext, FlowReadContext, FlowStepContext
from .models import (
    BaseFlow,
    ChildFlowDispatchSubmission,
    FlowStatus,
    FlowStepValidationError,
    StepStatus,
    utc_now_iso,
)
from .standard_steps import (
    AgentStep,
    DispatchStep,
    DispatchStepResult,
    DispatchStepState,
    build_followup_agent_step_from_dispatch,
)


def create_dispatch_step_from_agent_submission(
    ctx: FlowContext,
    *,
    source_agent_step_id: str,
) -> str:
    source_step = _get_step_for_current_flow(ctx, source_agent_step_id)
    if not isinstance(source_step, AgentStep):
        raise FlowStepValidationError(f"source step {source_agent_step_id} is not an AgentStep")
    if source_step.status is not StepStatus.COMPLETED:
        raise FlowStepValidationError(f"source step {source_agent_step_id} is not completed")
    if not isinstance(source_step.submission, ChildFlowDispatchSubmission):
        raise FlowStepValidationError(f"source step {source_agent_step_id} has no child flow dispatch submission")
    if not source_step.submission.requests:
        raise FlowStepValidationError(f"source step {source_agent_step_id} dispatch submission has no requests")

    existing = _find_dispatch_step_for_submission(ctx, source_step.submission.submission_id)
    if existing is not None:
        return existing.step_id

    step_id = f"dispatch_{uuid.uuid4().hex}"
    dispatch_step = DispatchStep(
        step_id=step_id,
        flow_id=ctx.flow.flow_id,
        scope_id=ctx.flow.scope_id,
        state=DispatchStepState(
            source_step_id=source_agent_step_id,
            source_submission_id=source_step.submission.submission_id,
            requests=source_step.submission.requests,
        ),
    )
    return ctx.create_step(dispatch_step)


def create_followup_agent_step_from_dispatch(
    ctx: FlowContext,
    *,
    source_agent_step_id: str,
    dispatch_step_id: str,
) -> str:
    dispatch_step = _get_step_for_current_flow(ctx, dispatch_step_id)
    if not isinstance(dispatch_step, DispatchStep):
        raise FlowStepValidationError(f"dispatch step {dispatch_step_id} is not a DispatchStep")
    if dispatch_step.status is not StepStatus.COMPLETED:
        raise FlowStepValidationError(f"dispatch step {dispatch_step_id} is not completed")
    if not isinstance(dispatch_step.result, DispatchStepResult) or dispatch_step.result.outcome != "dispatched":
        raise FlowStepValidationError(f"dispatch step {dispatch_step_id} did not dispatch child flows")
    if not isinstance(dispatch_step.state, DispatchStepState) or not dispatch_step.state.created_children:
        raise FlowStepValidationError(f"dispatch step {dispatch_step_id} has no created child flows")
    if not _all_children_terminal(ctx, dispatch_step):
        raise FlowStepValidationError(f"dispatch step {dispatch_step_id} still has non-terminal child flows")

    step_id = f"agent_callback_{uuid.uuid4().hex}"
    followup = build_followup_agent_step_from_dispatch(
        ctx,
        step_id=step_id,
        source_agent_step_id=source_agent_step_id,
        dispatch_step_id=dispatch_step_id,
    )
    return ctx.create_step(followup)


def create_standard_next_step_if_applicable(flow: BaseFlow, ctx: FlowContext) -> str | None:
    latest_step = _latest_terminal_step(flow, ctx)
    if isinstance(latest_step, AgentStep) and isinstance(latest_step.submission, ChildFlowDispatchSubmission):
        return create_dispatch_step_from_agent_submission(ctx, source_agent_step_id=latest_step.step_id)
    if isinstance(latest_step, DispatchStep):
        if _dispatch_ready_for_callback(ctx, latest_step):
            assert isinstance(latest_step.state, DispatchStepState)
            return create_followup_agent_step_from_dispatch(
                ctx,
                source_agent_step_id=latest_step.state.source_step_id,
                dispatch_step_id=latest_step.step_id,
            )
    return None


def handle_standard_step_terminal(flow: BaseFlow, ctx: FlowStepContext) -> bool:
    if isinstance(ctx.step, AgentStep) and isinstance(ctx.step.submission, ChildFlowDispatchSubmission):
        flow.current_step_id = None
        flow.status = FlowStatus.RUNNING
        flow.updated_at = utc_now_iso()
        return True
    if isinstance(ctx.step, DispatchStep):
        if not isinstance(ctx.step.result, DispatchStepResult) or ctx.step.result.outcome != "dispatched":
            return False
        if not hasattr(flow.state, "waiting_dispatch_step_id"):
            raise FlowStepValidationError(
                f"flow {flow.flow_id} state has no waiting_dispatch_step_id for standard dispatch wait"
            )
        setattr(flow.state, "waiting_dispatch_step_id", ctx.step.step_id)
        flow.current_step_id = None
        flow.status = FlowStatus.WAITING
        flow.updated_at = utc_now_iso()
        return True
    return False


def can_exit_standard_dispatch_wait(flow: BaseFlow, ctx: FlowContext | FlowReadContext) -> bool:
    dispatch_step_id = getattr(flow.state, "waiting_dispatch_step_id", None)
    if dispatch_step_id is None:
        return False
    dispatch_step = _get_step_for_current_flow(ctx, dispatch_step_id)
    if not isinstance(dispatch_step, DispatchStep):
        raise FlowStepValidationError(f"waiting dispatch step {dispatch_step_id} is not a DispatchStep")
    return _all_children_terminal(ctx, dispatch_step)


def on_exit_standard_dispatch_wait(flow: BaseFlow, ctx: FlowContext) -> bool:
    dispatch_step_id = getattr(flow.state, "waiting_dispatch_step_id", None)
    if dispatch_step_id is None:
        return False
    setattr(flow.state, "waiting_dispatch_step_id", None)
    flow.status = FlowStatus.RUNNING
    flow.updated_at = utc_now_iso()
    return True


def _get_step_for_current_flow(ctx: FlowContext | FlowReadContext, step_id: str):
    tx = getattr(ctx, "tx", None)
    if tx is not None:
        step = tx.load_step_for_update(step_id)
    else:
        flow_service = ctx.ark.flow_service
        if flow_service is None:
            raise FlowStepValidationError("ark.flow_service is not registered")
        step = flow_service.get_step(step_id)
    if step.flow_id != ctx.flow.flow_id:
        raise FlowStepValidationError(f"step {step_id} does not belong to flow {ctx.flow.flow_id}")
    return step


def _find_dispatch_step_for_submission(ctx: FlowContext, submission_id: str) -> DispatchStep | None:
    for step_id in ctx.flow.step_ids:
        step = _get_step_for_current_flow(ctx, step_id)
        if not isinstance(step, DispatchStep) or not isinstance(step.state, DispatchStepState):
            continue
        if step.state.source_submission_id == submission_id:
            return step
    return None


def _latest_terminal_step(flow: BaseFlow, ctx: FlowContext):
    for step_id in reversed(flow.step_ids):
        step = _get_step_for_current_flow(ctx, step_id)
        if step.status in {StepStatus.COMPLETED, StepStatus.FAILED}:
            return step
    return None


def _dispatch_ready_for_callback(ctx: FlowContext, step) -> bool:
    return (
        isinstance(step, DispatchStep)
        and step.status is StepStatus.COMPLETED
        and isinstance(step.result, DispatchStepResult)
        and step.result.outcome == "dispatched"
        and isinstance(step.state, DispatchStepState)
        and bool(step.state.created_children)
        and _all_children_terminal(ctx, step)
    )


def _all_children_terminal(ctx: FlowContext | FlowReadContext, dispatch_step: DispatchStep) -> bool:
    if not isinstance(dispatch_step.state, DispatchStepState):
        raise FlowStepValidationError(f"dispatch step {dispatch_step.step_id} does not have DispatchStepState")
    if not dispatch_step.state.created_children:
        return False
    flow_service = ctx.ark.flow_service
    if flow_service is None:
        raise FlowStepValidationError("ark.flow_service is not registered")
    for child in dispatch_step.state.created_children:
        child_flow = flow_service.get_flow(child.child_flow_id)
        if child_flow.status not in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
            return False
    return True
