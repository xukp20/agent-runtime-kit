from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepResult,
    BaseStepState,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    RuntimeScheduleService,
    StepRunContext,
    StepService,
    StepStatus,
    StepTerminalReceipt,
    FlowTypeRegistry,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePauseController


class PauseFlowParams(BaseModel):
    pass


class PauseFlowState(BaseFlowState):
    state_type: str = "pause_flow_state"


class PauseStepState(BaseStepState):
    state_type: str = "pause_step_state"


class PauseStep(BaseStep):
    step_type: ClassVar[str] = "pause_step"
    State: ClassVar[type[BaseStepState]] = PauseStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return ctx.complete_step(BaseStepResult(result_type="pause_step_done"))


class PauseFlow(BaseFlow):
    flow_type: ClassVar[str] = "pause_flow"
    Params: ClassVar[type[BaseModel]] = PauseFlowParams
    State: ClassVar[type[BaseFlowState]] = PauseFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "PauseFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=PauseFlowState())

    def create_next_step(self, ctx: FlowContext) -> str | None:
        return ctx.create_step(
            PauseStep(
                step_id=f"{self.flow_id}-step",
                flow_id=self.flow_id,
                scope_id=self.scope_id,
            )
        )


def make_services(runtime_root: Path) -> tuple[FlowService, StepService, RuntimeScheduleService, RuntimePauseController]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(PauseFlow)
    step_registry.register(PauseStep)
    pause_controller = RuntimePauseController()
    ark = ARKServices(pause_controller=pause_controller)
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
    scheduler = RuntimeScheduleService(ark_services=ark, app_services=AppServices())
    return flow_service, step_service, scheduler, pause_controller


def test_runtime_pause_blocks_scheduler_and_resume_continues(tmp_path: Path) -> None:
    flow_service, step_service, scheduler, pause = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.start_flow(FlowRequest(flow_type="pause_flow", scope_id="scope", params={}), enqueue=True)

    pause.pause(None)
    blocked_tick = scheduler.schedule_ready()
    assert blocked_tick.reason == "no_runnable_candidate"
    assert flow_id in scheduler.queued_flow_ids

    pause.resume(None)
    resumed_tick = scheduler.schedule_ready()
    step_id = f"{flow_id}-step"
    assert resumed_tick.advanced_flow_ids == [flow_id]
    assert resumed_tick.started_step_ids == [step_id]
    assert step_service.wait_step(step_id).status is StepStatus.COMPLETED


def test_scope_pause_blocks_only_matching_scope(tmp_path: Path) -> None:
    flow_service, _, scheduler, pause = make_services(tmp_path / ".agent_runtime")
    paused_flow_id = flow_service.start_flow(
        FlowRequest(flow_type="pause_flow", scope_id="scope-a", params={}),
        enqueue=True,
    )
    runnable_flow_id = flow_service.start_flow(
        FlowRequest(flow_type="pause_flow", scope_id="scope-b", params={}),
        enqueue=True,
    )

    pause.pause("scope-a")
    tick = scheduler.schedule_ready()

    assert paused_flow_id not in tick.advanced_flow_ids
    assert runnable_flow_id in tick.advanced_flow_ids
    assert paused_flow_id in scheduler.queued_flow_ids


def test_step_service_bypass_pause_starts_one_step_without_resuming_runtime(tmp_path: Path) -> None:
    flow_service, step_service, _, pause = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.start_flow(FlowRequest(flow_type="pause_flow", scope_id="scope", params={}), enqueue=False)
    step_id = flow_service.advance_flow(flow_id)
    assert step_id is not None

    pause.pause(None)
    with pytest.raises(Exception, match="step cannot run"):
        step_service.run_step(step_id)

    step_service.run_step(step_id, bypass_pause=True)

    assert pause.is_paused()
    assert step_service.wait_step(step_id).status is StepStatus.COMPLETED
