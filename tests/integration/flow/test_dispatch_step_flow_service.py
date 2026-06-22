from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowTypeRegistry,
    RuntimeScheduleService,
    StepService,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.flow.standard_steps import DispatchStep, DispatchStepState
from agent_runtime_kit.runtime import ARKServices, AppServices


class IntegrationParentParams(BaseModel):
    pass


class IntegrationParentState(BaseFlowState):
    state_type: str = "dispatch_integration_parent_state"


class IntegrationParentFlow(BaseFlow):
    flow_type: ClassVar[str] = "dispatch_integration_parent"
    Params: ClassVar[type[BaseModel]] = IntegrationParentParams
    State: ClassVar[type[BaseFlowState]] = IntegrationParentState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "IntegrationParentFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=IntegrationParentState())


class IntegrationChildParams(BaseModel):
    name: str


class IntegrationChildState(BaseFlowState):
    state_type: str = "dispatch_integration_child_state"


class IntegrationChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "dispatch_integration_child"
    Params: ClassVar[type[BaseModel]] = IntegrationChildParams
    State: ClassVar[type[BaseFlowState]] = IntegrationChildState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "IntegrationChildFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=IntegrationChildState())


def make_services(runtime_root: Path) -> tuple[FlowService, StepService, RuntimeScheduleService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(IntegrationParentFlow)
    flow_registry.register(IntegrationChildFlow)
    step_registry.register(DispatchStep)
    ark = ARKServices()
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    step_service = StepService(runtime_root, step_registry=step_registry, ark_services=ark, app_services=AppServices())
    scheduler = RuntimeScheduleService(ark_services=ark, app_services=AppServices())
    return flow_service, step_service, scheduler


def test_scheduler_runs_dispatch_step_and_creates_child_flows(tmp_path: Path) -> None:
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime")
    parent_id = flow_service.start_flow(
        FlowRequest(flow_type="dispatch_integration_parent", scope_id="scope", params={}),
        enqueue=False,
    )
    step = DispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        state=DispatchStepState(
            source_step_id="source-step",
            source_submission_id="submission-1",
            requests=[
                FlowRequest(flow_type="dispatch_integration_child", scope_id="scope", params={"name": "a"}),
                FlowRequest(flow_type="dispatch_integration_child", scope_id="scope", params={"name": "b"}),
            ],
        ),
    )
    step_service.create_step(step, enqueue=False)
    flow_service.store.update_flow_record(
        parent_id,
        lambda flow: (flow.step_ids.append(step.step_id), setattr(flow, "current_step_id", step.step_id)),
    )
    scheduler.enqueue_step(step.step_id)

    tick = scheduler.schedule_ready()

    terminal = step_service.wait_step(step.step_id)
    children = flow_service.store.list_child_flows(parent_flow_id=parent_id, parent_dispatch_step_id=step.step_id)
    assert tick.started_step_ids == [step.step_id]
    assert terminal.status is StepStatus.COMPLETED
    assert terminal.result is not None
    assert terminal.result.outcome == "dispatched"
    assert terminal.result.child_flow_ids == [child.flow_id for child in children]
    assert len(children) == 2
