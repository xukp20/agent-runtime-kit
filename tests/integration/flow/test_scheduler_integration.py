from pathlib import Path
from typing import ClassVar

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
    RuntimeScheduleService,
    SchedulerRunBudget,
    StepRunContext,
    StepService,
    StepStatus,
    StepTerminalReceipt,
    FlowTypeRegistry,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePauseController


class SchedulerIntegrationFlowParams(BaseModel):
    pass


class SchedulerIntegrationFlowState(BaseFlowState):
    state_type: str = "scheduler_integration_flow_state"


class SchedulerIntegrationStepState(BaseStepState):
    state_type: str = "scheduler_integration_step_state"


class SchedulerIntegrationStep(BaseStep):
    step_type: ClassVar[str] = "scheduler_integration_step"
    State: ClassVar[type[BaseStepState]] = SchedulerIntegrationStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return ctx.complete_step(BaseStepResult(result_type="scheduler_integration_step_done"))


class SchedulerIntegrationFlow(BaseFlow):
    flow_type: ClassVar[str] = "scheduler_integration_flow"
    Params: ClassVar[type[BaseModel]] = SchedulerIntegrationFlowParams
    State: ClassVar[type[BaseFlowState]] = SchedulerIntegrationFlowState
    stable_hook_seen: ClassVar[bool] = False

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "SchedulerIntegrationFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=SchedulerIntegrationFlowState())

    def create_next_step(self, ctx: FlowContext) -> str | None:
        if not self.step_ids:
            return ctx.create_step(
                SchedulerIntegrationStep(
                    step_id=f"{self.flow_id}-step",
                    flow_id=self.flow_id,
                    scope_id=self.scope_id,
                )
            )
        ctx.set_flow_result(BaseFlowResult(result_type="scheduler_integration_flow_done"))
        return None

    def after_step_terminal_stable(self, ctx) -> None:
        assert ctx.ark.pause_controller is None or not ctx.ark.pause_controller.is_paused(self.scope_id)
        SchedulerIntegrationFlow.stable_hook_seen = True


def make_services(
    runtime_root: Path,
) -> tuple[FlowService, StepService, RuntimeScheduleService, RuntimePauseController]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(SchedulerIntegrationFlow)
    step_registry.register(SchedulerIntegrationStep)
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


def test_scheduler_ticks_drive_flow_step_and_flow_completion(tmp_path: Path) -> None:
    flow_service, step_service, scheduler, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="scheduler_integration_flow", scope_id="scope", params={}),
        enqueue=True,
    )

    first_tick = scheduler.schedule_ready()
    step = step_service.wait_step(f"{flow_id}-step")
    second_tick = scheduler.schedule_ready()

    flow = flow_service.get_flow(flow_id)
    assert first_tick.advanced_flow_ids == [flow_id]
    assert first_tick.started_step_ids == [f"{flow_id}-step"]
    assert second_tick.advanced_flow_ids == [flow_id]
    assert second_tick.started_step_ids == []
    assert step.result is not None
    assert step.result.result_type == "scheduler_integration_step_done"
    assert flow.status is FlowStatus.COMPLETED
    assert flow.result is not None
    assert flow.result.result_type == "scheduler_integration_flow_done"


def test_bounded_scheduler_auto_pauses_only_after_stable_terminal_hook(tmp_path: Path) -> None:
    flow_service, step_service, scheduler, pause = make_services(tmp_path / ".agent_runtime")
    SchedulerIntegrationFlow.stable_hook_seen = False
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="scheduler_integration_flow", scope_id="scope", params={}),
        enqueue=True,
    )
    pause.pause(None)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=1, step_starts=1))
    pause.resume(None)

    first_tick = scheduler.schedule_ready()
    step_id = f"{flow_id}-step"
    assert first_tick.advanced_flow_ids == [flow_id]
    assert first_tick.started_step_ids == [step_id]
    assert step_service.wait_step(step_id, timeout_s=2).status is StepStatus.COMPLETED
    assert SchedulerIntegrationFlow.stable_hook_seen is True

    terminal_tick = first_tick if first_tick.auto_paused else scheduler.schedule_ready()
    assert terminal_tick.auto_paused is True
    assert terminal_tick.run_control is not None
    assert terminal_tick.run_control.pause_reason == "budget_exhausted"
    assert pause.is_paused()
