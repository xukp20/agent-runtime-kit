from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from agent_runtime_kit.flow.contexts import StepRunContext
from agent_runtime_kit.flow.models import (
    BaseStep,
    BaseStepResult,
    BaseStepState,
    CreatedChildFlow,
    DispatchRequestFailure,
    FlowRequest,
    FlowStepValidationError,
    StepTerminalReceipt,
)


class DispatchStepState(BaseStepState):
    state_type: str = "dispatch_step"
    source_step_id: str
    source_submission_id: str
    requests: list[FlowRequest]
    created_children: list[CreatedChildFlow] = Field(default_factory=list)
    failed_requests: list[DispatchRequestFailure] = Field(default_factory=list)


class DispatchStepResult(BaseStepResult):
    result_type: str = "dispatch_step"
    outcome: Literal["dispatched", "empty", "failed"]
    source_step_id: str
    source_submission_id: str
    child_flow_ids: list[str] = Field(default_factory=list)
    failed_request_indices: list[int] = Field(default_factory=list)


class DispatchStep(BaseStep):
    step_type: ClassVar[str] = "dispatch_step"
    State: ClassVar[type[BaseStepState]] = DispatchStepState
    Result: ClassVar[type[BaseStepResult]] = DispatchStepResult

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        latest = self._latest_dispatch_step(ctx)
        state = self._dispatch_state(latest)
        if not state.requests:
            return ctx.complete_step(
                DispatchStepResult(
                    outcome="empty",
                    source_step_id=state.source_step_id,
                    source_submission_id=state.source_submission_id,
                    summary="no child flow requests",
                )
            )

        failed_requests = self._validate_requests(ctx, state.requests)
        if failed_requests:
            ctx.update_step(lambda step: self._set_failed_requests(step, failed_requests))
            return ctx.complete_step(
                DispatchStepResult(
                    outcome="failed",
                    source_step_id=state.source_step_id,
                    source_submission_id=state.source_submission_id,
                    failed_request_indices=[failure.request_index for failure in failed_requests],
                    summary="one or more child flow requests failed validation",
                )
            )

        child_flow_ids = self._flow_service(ctx).start_flows_batch(
            state.requests,
            parent_flow_id=ctx.flow_id,
            parent_dispatch_step_id=ctx.step_id,
            enqueue=True,
        )
        for index, child_flow_id in enumerate(child_flow_ids):
            child = CreatedChildFlow(request_index=index, child_flow_id=child_flow_id)
            ctx.update_step(lambda step, child=child: self._append_created_child(step, child))

        latest = self._latest_dispatch_step(ctx)
        latest_state = self._dispatch_state(latest)
        child_flow_ids = [child.child_flow_id for child in latest_state.created_children]
        return ctx.complete_step(
            DispatchStepResult(
                outcome="dispatched",
                source_step_id=latest_state.source_step_id,
                source_submission_id=latest_state.source_submission_id,
                child_flow_ids=child_flow_ids,
                summary=f"dispatched {len(child_flow_ids)} child flows",
            )
        )

    def _validate_requests(self, ctx: StepRunContext, requests: list[FlowRequest]) -> list[DispatchRequestFailure]:
        flow_service = self._flow_service(ctx)
        failures: list[DispatchRequestFailure] = []
        for index, request in enumerate(requests):
            try:
                flow_service.flow_registry.validate_request_params(request)
            except Exception as exc:
                failures.append(
                    DispatchRequestFailure(
                        request_index=index,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
        return failures

    def _append_created_child(self, step: BaseStep, child: CreatedChildFlow) -> None:
        if not isinstance(step.state, DispatchStepState):
            raise FlowStepValidationError(f"step {step.step_id} does not have DispatchStepState")
        if child.request_index not in {existing.request_index for existing in step.state.created_children}:
            step.state.created_children.append(child)

    def _set_failed_requests(self, step: BaseStep, failures: list[DispatchRequestFailure]) -> None:
        if not isinstance(step.state, DispatchStepState):
            raise FlowStepValidationError(f"step {step.step_id} does not have DispatchStepState")
        step.state.failed_requests = failures

    def _latest_dispatch_step(self, ctx: StepRunContext) -> "DispatchStep":
        latest = ctx.load_step()
        if not isinstance(latest, DispatchStep):
            raise FlowStepValidationError(f"step {ctx.step_id} is not a DispatchStep")
        return latest

    def _dispatch_state(self, step: "DispatchStep") -> DispatchStepState:
        if not isinstance(step.state, DispatchStepState):
            raise FlowStepValidationError(f"step {step.step_id} does not have DispatchStepState")
        return step.state

    def _flow_service(self, ctx: StepRunContext):
        flow_service = ctx.ark.flow_service
        if flow_service is None:
            raise FlowStepValidationError("ark.flow_service is not registered")
        return flow_service
