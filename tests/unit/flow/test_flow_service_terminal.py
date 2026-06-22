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
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepContext,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class TerminalFlowParams(BaseModel):
    complete_on_step: bool = False
    hook_raises: bool = False


class TerminalFlowState(BaseFlowState):
    state_type: str = "terminal_flow_state"
    complete_on_step: bool = False
    hook_raises: bool = False
    terminal_seen: bool = False
    hook_called: bool = False
    hook_attempted: bool = False


class TerminalStepState(BaseStepState):
    state_type: str = "terminal_step_state"


class TerminalStep(BaseStep):
    step_type: ClassVar[str] = "terminal_step"
    State: ClassVar[type[BaseStepState]] = TerminalStepState


class TerminalFlow(BaseFlow):
    flow_type: ClassVar[str] = "terminal_flow"
    Params: ClassVar[type[BaseModel]] = TerminalFlowParams
    State: ClassVar[type[BaseFlowState]] = TerminalFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "TerminalFlow":
        params = ctx.params
        assert isinstance(params, TerminalFlowParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            state=TerminalFlowState(complete_on_step=params.complete_on_step, hook_raises=params.hook_raises),
        )

    def on_step_terminal(self, ctx: FlowStepContext) -> None:
        assert isinstance(self.state, TerminalFlowState)
        self.state.terminal_seen = True
        if self.state.complete_on_step:
            self.result = BaseFlowResult(result_type="terminal_flow_done", summary="done")
        super().on_step_terminal(ctx)

    def after_step_terminal_stable(self, ctx: FlowStepContext) -> None:
        assert not hasattr(ctx, "tx")
        assert isinstance(self.state, TerminalFlowState)
        self.state.hook_called = True
        if self.state.hook_raises:
            self.state.hook_attempted = True
            raise RuntimeError("hook failed")


class FakeScheduleService:
    def __init__(self) -> None:
        self.flow_ids: list[str] = []

    def enqueue_flow(self, flow_id: str) -> None:
        self.flow_ids.append(flow_id)


def make_service(runtime_root: Path, *, schedule: FakeScheduleService | None = None) -> FlowService:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(TerminalFlow)
    step_registry.register(TerminalStep)
    ark = ARKServices(schedule_service=schedule)
    return FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )


def start_flow(service: FlowService, *, complete_on_step: bool = False, hook_raises: bool = False) -> str:
    return service.start_flow(
        FlowRequest(
            flow_type="terminal_flow",
            scope_id="scope",
            params={"complete_on_step": complete_on_step, "hook_raises": hook_raises},
        ),
        enqueue=False,
    )


def attach_terminal_step(service: FlowService, flow_id: str, *, status: StepStatus = StepStatus.COMPLETED) -> str:
    step_id = f"step-{flow_id[-6:]}"
    service.store.create_step(
        TerminalStep(
            step_id=step_id,
            flow_id=flow_id,
            scope_id="scope",
            status=status,
            result=BaseStepResult(result_type="step_done") if status is StepStatus.COMPLETED else None,
        )
    )
    service.store.update_flow_record(
        flow_id,
        lambda flow: (flow.step_ids.append(step_id), setattr(flow, "current_step_id", step_id)),
    )
    return step_id


def test_handle_step_terminal_absorbs_step_and_enqueues_non_terminal_flow(tmp_path: Path) -> None:
    schedule = FakeScheduleService()
    service = make_service(tmp_path / ".agent_runtime", schedule=schedule)
    flow_id = start_flow(service)
    step_id = attach_terminal_step(service, flow_id)

    service.handle_step_terminal(step_id)

    flow = service.get_flow(flow_id)
    assert flow.current_step_id is None
    assert flow.status is FlowStatus.RUNNING
    assert isinstance(flow.state, TerminalFlowState)
    assert flow.state.terminal_seen is True
    assert flow.state.hook_called is False
    assert schedule.flow_ids == [flow_id]


def test_handle_step_terminal_can_complete_flow_without_reenqueue(tmp_path: Path) -> None:
    schedule = FakeScheduleService()
    service = make_service(tmp_path / ".agent_runtime", schedule=schedule)
    flow_id = start_flow(service, complete_on_step=True)
    step_id = attach_terminal_step(service, flow_id)

    service.handle_step_terminal(step_id)

    flow = service.get_flow(flow_id)
    assert flow.status is FlowStatus.COMPLETED
    assert flow.result is not None
    assert flow.result.result_type == "terminal_flow_done"
    assert schedule.flow_ids == []


def test_handle_step_terminal_rejects_non_terminal_step(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = start_flow(service)
    step_id = attach_terminal_step(service, flow_id, status=StepStatus.CREATED)

    with pytest.raises(FlowStepValidationError):
        service.handle_step_terminal(step_id)


def test_handle_step_terminal_is_noop_when_already_absorbed(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = start_flow(service)
    step_id = attach_terminal_step(service, flow_id)
    service.store.update_flow_record(flow_id, lambda flow: setattr(flow, "current_step_id", None))

    service.handle_step_terminal(step_id)

    assert service.get_flow(flow_id).current_step_id is None


def test_after_step_terminal_hook_error_does_not_rollback_persisted_truth(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = start_flow(service, hook_raises=True)
    step_id = attach_terminal_step(service, flow_id)

    service.handle_step_terminal(step_id)

    flow = service.get_flow(flow_id)
    assert flow.current_step_id is None
    assert flow.status is FlowStatus.RUNNING
    assert isinstance(flow.state, TerminalFlowState)
    assert flow.state.terminal_seen is True
    assert flow.state.hook_called is False
    assert service.stable_hook_errors == [
        {
            "flow_id": flow_id,
            "step_id": step_id,
            "error_type": "RuntimeError",
            "message": "hook failed",
        }
    ]
