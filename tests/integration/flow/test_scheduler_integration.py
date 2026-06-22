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
    StepRunContext,
    StepService,
    StepTerminalReceipt,
    FlowTypeRegistry,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


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


def make_services(runtime_root: Path) -> tuple[FlowService, StepService, RuntimeScheduleService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(SchedulerIntegrationFlow)
    step_registry.register(SchedulerIntegrationStep)
    ark = ARKServices()
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
    return flow_service, step_service, scheduler


def test_scheduler_ticks_drive_flow_step_and_flow_completion(tmp_path: Path) -> None:
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime")
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
