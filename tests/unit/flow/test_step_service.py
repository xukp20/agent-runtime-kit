from pathlib import Path
from threading import Event
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepResult,
    BaseStepState,
    ActiveStepRun,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepRunContext,
    StepService,
    StepStatus,
    StepTerminalReceipt,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class StepSvcFlowParams(BaseModel):
    pass


class StepSvcFlowState(BaseFlowState):
    state_type: str = "step_svc_flow_state"


class StepSvcFlow(BaseFlow):
    flow_type: ClassVar[str] = "step_svc_flow"
    Params: ClassVar[type[BaseModel]] = StepSvcFlowParams
    State: ClassVar[type[BaseFlowState]] = StepSvcFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "StepSvcFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=StepSvcFlowState())


class StepSvcStepState(BaseStepState):
    state_type: str = "step_svc_step_state"


class CompleteStep(BaseStep):
    step_type: ClassVar[str] = "complete_step"
    State: ClassVar[type[BaseStepState]] = StepSvcStepState
    observed_running: ClassVar[bool] = False

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        CompleteStep.observed_running = ctx.ark.step_service.has_running_steps(ctx.scope_id)
        return ctx.complete_step(BaseStepResult(result_type="complete_step_result", summary="done"))


class ErrorStep(BaseStep):
    step_type: ClassVar[str] = "error_step"
    State: ClassVar[type[BaseStepState]] = StepSvcStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        raise RuntimeError("boom")


class InvalidReceiptStep(BaseStep):
    step_type: ClassVar[str] = "invalid_receipt_step"
    State: ClassVar[type[BaseStepState]] = StepSvcStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return StepTerminalReceipt(
            step_id="wrong-step",
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            status="completed",
            result_type="wrong",
            finished_at="2026-06-22T00:00:00Z",
        )


class NoReceiptStep(BaseStep):
    step_type: ClassVar[str] = "no_receipt_step"
    State: ClassVar[type[BaseStepState]] = StepSvcStepState

    def run(self, ctx: StepRunContext):  # intentionally invalid for StepService coverage
        return None


class BlockingStep(BaseStep):
    step_type: ClassVar[str] = "blocking_step"
    State: ClassVar[type[BaseStepState]] = StepSvcStepState
    release_event: ClassVar[Event] = Event()
    started_events: ClassVar[dict[str, Event]] = {}

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        BlockingStep.started_events.setdefault(ctx.step_id, Event()).set()
        BlockingStep.release_event.wait(timeout=5)
        return ctx.complete_step(BaseStepResult(result_type="blocking_step_result", summary="done"))


class FakeScheduleService:
    def __init__(self) -> None:
        self.step_ids: list[str] = []

    def enqueue_step(self, step_id: str) -> None:
        self.step_ids.append(step_id)


class FakePauseController:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused

    def is_paused(self, scope_id: str | None = None) -> bool:
        return self.paused


def make_services(
    runtime_root: Path,
    *,
    schedule: FakeScheduleService | None = None,
    pause: FakePauseController | None = None,
) -> tuple[FlowService, StepService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(StepSvcFlow)
    for step_cls in [CompleteStep, ErrorStep, InvalidReceiptStep, NoReceiptStep, BlockingStep]:
        step_registry.register(step_cls)
    ark = ARKServices(schedule_service=schedule, pause_controller=pause)
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    step_service = StepService(
        runtime_root,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    return flow_service, step_service


def start_flow(flow_service: FlowService) -> str:
    return flow_service.start_flow(FlowRequest(flow_type="step_svc_flow", scope_id="scope", params={}), enqueue=False)


def attach_step(flow_service: FlowService, flow_id: str, step: BaseStep) -> str:
    flow_service.store.create_step(step)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (flow.step_ids.append(step.step_id), setattr(flow, "current_step_id", step.step_id)),
    )
    return step.step_id


def test_create_step_can_enqueue_for_management_use(tmp_path: Path) -> None:
    schedule = FakeScheduleService()
    flow_service, step_service = make_services(tmp_path / ".agent_runtime", schedule=schedule)
    flow_id = start_flow(flow_service)

    step_id = step_service.create_step(
        CompleteStep(step_id="created-step", flow_id=flow_id, scope_id="scope"),
        enqueue=True,
    )

    assert step_id == "created-step"
    assert schedule.step_ids == ["created-step"]


def test_run_step_completes_step_tracks_active_and_calls_flow_terminal(tmp_path: Path) -> None:
    CompleteStep.observed_running = False
    flow_service, step_service = make_services(tmp_path / ".agent_runtime")
    flow_id = start_flow(flow_service)
    step_id = attach_step(flow_service, flow_id, CompleteStep(step_id="step-1", flow_id=flow_id, scope_id="scope"))

    step_service.run_step(step_id)

    step = step_service.wait_step(step_id)
    flow = flow_service.get_flow(flow_id)
    assert CompleteStep.observed_running is True
    assert step.status is StepStatus.COMPLETED
    assert step.result is not None
    assert step.result.result_type == "complete_step_result"
    assert flow.current_step_id is None
    assert step_service.has_running_steps("scope") is False


def test_start_step_is_nonblocking_and_allows_multiple_active_steps(tmp_path: Path) -> None:
    BlockingStep.release_event = Event()
    BlockingStep.started_events = {}
    flow_service, step_service = make_services(tmp_path / ".agent_runtime")
    first_flow_id = start_flow(flow_service)
    second_flow_id = start_flow(flow_service)
    first_step_id = attach_step(
        flow_service,
        first_flow_id,
        BlockingStep(step_id="blocking-1", flow_id=first_flow_id, scope_id="scope"),
    )
    second_step_id = attach_step(
        flow_service,
        second_flow_id,
        BlockingStep(step_id="blocking-2", flow_id=second_flow_id, scope_id="scope"),
    )
    BlockingStep.started_events[first_step_id] = Event()
    BlockingStep.started_events[second_step_id] = Event()

    first_active = step_service.start_step(first_step_id)
    second_active = step_service.start_step(second_step_id)

    assert first_active.done_event is not None
    assert second_active.done_event is not None
    assert BlockingStep.started_events[first_step_id].wait(timeout=2)
    assert BlockingStep.started_events[second_step_id].wait(timeout=2)
    assert {first_step_id, second_step_id}.issubset(set(step_service.list_running_steps("scope")))

    BlockingStep.release_event.set()
    assert step_service.wait_step(first_step_id, timeout_s=2).status is StepStatus.COMPLETED
    assert step_service.wait_step(second_step_id, timeout_s=2).status is StepStatus.COMPLETED
    assert step_service.has_running_steps("scope") is False


def test_run_step_catches_exception_and_writes_failed_terminal(tmp_path: Path) -> None:
    flow_service, step_service = make_services(tmp_path / ".agent_runtime")
    flow_id = start_flow(flow_service)
    step_id = attach_step(flow_service, flow_id, ErrorStep(step_id="step-1", flow_id=flow_id, scope_id="scope"))

    step_service.run_step(step_id)

    step = step_service.wait_step(step_id)
    flow = flow_service.get_flow(flow_id)
    assert step.status is StepStatus.FAILED
    assert step.error is not None
    assert step.error.error_type == "step_run_exception"
    assert flow.current_step_id is None


def test_run_step_invalid_or_missing_receipt_becomes_failed(tmp_path: Path) -> None:
    flow_service, step_service = make_services(tmp_path / ".agent_runtime")
    invalid_flow_id = start_flow(flow_service)
    missing_flow_id = start_flow(flow_service)
    invalid_step_id = attach_step(
        flow_service,
        invalid_flow_id,
        InvalidReceiptStep(step_id="invalid", flow_id=invalid_flow_id, scope_id="scope"),
    )
    missing_step_id = attach_step(
        flow_service,
        missing_flow_id,
        NoReceiptStep(step_id="missing", flow_id=missing_flow_id, scope_id="scope"),
    )

    step_service.run_step(invalid_step_id)
    step_service.run_step(missing_step_id)

    invalid = step_service.wait_step(invalid_step_id)
    missing = step_service.wait_step(missing_step_id)
    assert invalid.status is StepStatus.FAILED
    assert invalid.error is not None
    assert invalid.error.error_type == "invalid_terminal_receipt"
    assert missing.status is StepStatus.FAILED
    assert missing.error is not None
    assert missing.error.error_type == "step_not_terminal"


def test_pause_and_flow_current_step_gate_block_run(tmp_path: Path) -> None:
    paused_flow, paused_step_service = make_services(tmp_path / "paused", pause=FakePauseController(paused=True))
    paused_flow_id = start_flow(paused_flow)
    paused_step_id = attach_step(
        paused_flow,
        paused_flow_id,
        CompleteStep(step_id="paused-step", flow_id=paused_flow_id, scope_id="scope"),
    )
    assert paused_step_service.can_run_step(paused_step_id) is False
    with pytest.raises(FlowStepValidationError):
        paused_step_service.run_step(paused_step_id)

    flow_service, step_service = make_services(tmp_path / "mismatch")
    flow_id = start_flow(flow_service)
    step_id = "mismatch-step"
    flow_service.store.create_step(CompleteStep(step_id=step_id, flow_id=flow_id, scope_id="scope"))
    assert step_service.can_run_step(step_id) is False


def test_list_helpers_report_running_and_created_steps(tmp_path: Path) -> None:
    flow_service, step_service = make_services(tmp_path / ".agent_runtime")
    flow_id = start_flow(flow_service)
    flow_service.store.create_step(CompleteStep(step_id="created", flow_id=flow_id, scope_id="scope"))
    flow_service.store.create_step(CompleteStep(step_id="running", flow_id=flow_id, scope_id="scope", status=StepStatus.RUNNING))
    step_service.active_steps["active"] = ActiveStepRun(step_id="active", flow_id=flow_id, scope_id="scope", started_at="now")

    assert step_service.list_created_steps("scope") == ["created"]
    assert step_service.has_running_steps("scope") is True
    assert step_service.list_running_steps("scope") == ["active", "running"]
