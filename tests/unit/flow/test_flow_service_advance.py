from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
    BaseStepResult,
    BaseStepState,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class AdvanceFlowParams(BaseModel):
    wait_ready: bool = False


class AdvanceFlowState(BaseFlowState):
    state_type: str = "advance_flow_state"
    wait_ready: bool = False
    exited_waiting: bool = False
    next_step_count: int = 0


class AdvanceStepState(BaseStepState):
    state_type: str = "advance_step_state"


class AdvanceStep(BaseStep):
    step_type: ClassVar[str] = "advance_step"
    State: ClassVar[type[BaseStepState]] = AdvanceStepState


class AdvanceFlow(BaseFlow):
    flow_type: ClassVar[str] = "advance_flow"
    Params: ClassVar[type[BaseModel]] = AdvanceFlowParams
    State: ClassVar[type[BaseFlowState]] = AdvanceFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "AdvanceFlow":
        params = ctx.params
        assert isinstance(params, AdvanceFlowParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            state=AdvanceFlowState(wait_ready=params.wait_ready),
        )

    def can_exit_waiting(self, ctx: FlowContext) -> bool:
        assert isinstance(self.state, AdvanceFlowState)
        return self.state.wait_ready

    def on_exit_waiting(self, ctx: FlowContext) -> None:
        assert isinstance(self.state, AdvanceFlowState)
        self.state.exited_waiting = True
        super().on_exit_waiting(ctx)

    def create_next_step(self, ctx: FlowContext) -> str | None:
        assert isinstance(self.state, AdvanceFlowState)
        self.state.next_step_count += 1
        step_id = f"step-{self.state.next_step_count}"
        return ctx.create_step(AdvanceStep(step_id=step_id, flow_id=self.flow_id, scope_id=self.scope_id))


class FakeScheduleService:
    def __init__(self) -> None:
        self.flow_ids: list[str] = []
        self.step_ids: list[str] = []

    def enqueue_flow(self, flow_id: str) -> None:
        self.flow_ids.append(flow_id)

    def enqueue_step(self, step_id: str) -> None:
        self.step_ids.append(step_id)


class FakePauseController:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused

    def is_paused(self, scope_id: str | None = None) -> bool:
        return self.paused


def make_service(
    runtime_root: Path,
    *,
    schedule: FakeScheduleService | None = None,
    pause: FakePauseController | None = None,
) -> FlowService:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(AdvanceFlow)
    step_registry.register(AdvanceStep)
    ark = ARKServices(schedule_service=schedule, pause_controller=pause)
    return FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )


def start_flow(service: FlowService, *, wait_ready: bool = False) -> str:
    return service.start_flow(
        FlowRequest(flow_type="advance_flow", scope_id="scope", params={"wait_ready": wait_ready}),
        enqueue=False,
    )


def test_created_and_running_flows_can_advance_but_terminal_flows_cannot(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    created_id = start_flow(service)
    running_id = start_flow(service)
    completed_id = start_flow(service)
    failed_id = start_flow(service)
    service.store.update_flow_record(running_id, lambda flow: setattr(flow, "status", FlowStatus.RUNNING))
    service.store.update_flow_record(completed_id, lambda flow: setattr(flow, "status", FlowStatus.COMPLETED))
    service.store.update_flow_record(failed_id, lambda flow: setattr(flow, "status", FlowStatus.FAILED))

    assert service.can_advance_flow(created_id) is True
    assert service.can_advance_flow(running_id) is True
    assert service.can_advance_flow(completed_id) is False
    assert service.can_advance_flow(failed_id) is False


def test_pause_manual_pause_and_current_step_block_advance(tmp_path: Path) -> None:
    paused_service = make_service(tmp_path / "paused", pause=FakePauseController(paused=True))
    paused_id = start_flow(paused_service)
    assert paused_service.can_advance_flow(paused_id) is False

    service = make_service(tmp_path / "manual")
    manual_id = start_flow(service)
    current_step_id = start_flow(service)
    service.store.update_flow_record(manual_id, lambda flow: setattr(flow.manual_pause, "active", True))
    service.store.create_step(AdvanceStep(step_id="current", flow_id=current_step_id, scope_id="scope"))
    service.store.update_flow_record(current_step_id, lambda flow: setattr(flow, "current_step_id", "current"))

    assert service.can_advance_flow(manual_id) is False
    assert service.can_advance_flow(current_step_id) is False


def test_waiting_flow_exits_waiting_when_condition_is_ready(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    not_ready_id = start_flow(service, wait_ready=False)
    ready_id = start_flow(service, wait_ready=True)
    service.store.update_flow_record(not_ready_id, lambda flow: setattr(flow, "status", FlowStatus.WAITING))
    service.store.update_flow_record(ready_id, lambda flow: setattr(flow, "status", FlowStatus.WAITING))

    assert service.can_advance_flow(not_ready_id) is False
    assert service.can_advance_flow(ready_id) is True

    ready = service.get_flow(ready_id)
    assert ready.status is FlowStatus.WAITING
    assert isinstance(ready.state, AdvanceFlowState)
    assert ready.state.exited_waiting is False

    assert service.prepare_flow_for_advance(ready_id) is True
    ready = service.get_flow(ready_id)
    assert ready.status is FlowStatus.RUNNING
    assert isinstance(ready.state, AdvanceFlowState)
    assert ready.state.exited_waiting is True


def test_advance_flow_creates_step_and_enqueues_it(tmp_path: Path) -> None:
    schedule = FakeScheduleService()
    service = make_service(tmp_path / ".agent_runtime", schedule=schedule)
    flow_id = start_flow(service)

    step_id = service.advance_flow(flow_id)
    flow = service.get_flow(flow_id)
    step = service.get_step(step_id or "")

    assert step_id == "step-1"
    assert flow.current_step_id == "step-1"
    assert flow.step_ids == ["step-1"]
    assert step.flow_id == flow_id
    assert schedule.step_ids == ["step-1"]


def test_advance_flow_rejects_unadvanceable_flow(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = start_flow(service)
    service.store.update_flow_record(flow_id, lambda flow: setattr(flow, "status", FlowStatus.COMPLETED))

    with pytest.raises(FlowStepValidationError):
        service.advance_flow(flow_id)


def test_advance_flow_marks_no_progress_flow_failed(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = start_flow(service)

    def make_noop(flow: BaseFlow) -> None:
        assert isinstance(flow.state, AdvanceFlowState)
        flow.state.next_step_count = -999

    service.store.update_flow_record(flow_id, make_noop)

    def noop_create_next_step(self, ctx):  # type: ignore[no-untyped-def]
        return None

    original = AdvanceFlow.create_next_step
    AdvanceFlow.create_next_step = noop_create_next_step  # type: ignore[method-assign]
    try:
        with pytest.raises(FlowStepValidationError):
            service.advance_flow(flow_id)
    finally:
        AdvanceFlow.create_next_step = original  # type: ignore[method-assign]

    failed = service.get_flow(flow_id)
    assert failed.status is FlowStatus.FAILED
    assert failed.error is not None
    assert failed.error.error_type == "flow_no_progress"
