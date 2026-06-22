from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowTypeRegistry,
    StepService,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.flow.standard_steps import DispatchStep, DispatchStepState
from agent_runtime_kit.runtime import ARKServices, AppServices


class ParentFlowParams(BaseModel):
    pass


class ParentFlowState(BaseFlowState):
    state_type: str = "dispatch_parent_flow_state"


class ParentFlow(BaseFlow):
    flow_type: ClassVar[str] = "dispatch_parent_flow"
    Params: ClassVar[type[BaseModel]] = ParentFlowParams
    State: ClassVar[type[BaseFlowState]] = ParentFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "ParentFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=ParentFlowState())


class ChildFlowParams(BaseModel):
    value: int


class ChildFlowState(BaseFlowState):
    state_type: str = "dispatch_child_flow_state"


class ChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "dispatch_child_flow"
    Params: ClassVar[type[BaseModel]] = ChildFlowParams
    State: ClassVar[type[BaseFlowState]] = ChildFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "ChildFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=ChildFlowState())


class ExplodingFlow(ChildFlow):
    flow_type: ClassVar[str] = "exploding_flow"

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "ExplodingFlow":
        raise RuntimeError("cannot build child flow")


class FakeScheduleService:
    def __init__(self) -> None:
        self.flow_ids: list[str] = []

    def enqueue_flow(self, flow_id: str) -> None:
        self.flow_ids.append(flow_id)


def make_services(
    runtime_root: Path,
    *,
    register_exploding: bool = False,
) -> tuple[FlowService, StepService, FakeScheduleService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(ParentFlow)
    flow_registry.register(ChildFlow)
    if register_exploding:
        flow_registry.register(ExplodingFlow)
    step_registry.register(DispatchStep)
    schedule = FakeScheduleService()
    ark = ARKServices(schedule_service=schedule)
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    step_service = StepService(runtime_root, step_registry=step_registry, ark_services=ark, app_services=AppServices())
    return flow_service, step_service, schedule


def start_parent(flow_service: FlowService) -> str:
    return flow_service.start_flow(
        FlowRequest(flow_type="dispatch_parent_flow", scope_id="scope", params={}),
        enqueue=False,
    )


def attach_dispatch_step(flow_service: FlowService, parent_id: str, step: DispatchStep) -> str:
    flow_service.store.create_step(step)
    flow_service.store.update_flow_record(
        parent_id,
        lambda flow: (flow.step_ids.append(step.step_id), setattr(flow, "current_step_id", step.step_id)),
    )
    return step.step_id


def make_dispatch_step(parent_id: str, requests: list[FlowRequest]) -> DispatchStep:
    return DispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        state=DispatchStepState(
            source_step_id="source-step",
            source_submission_id="submission-1",
            requests=requests,
        ),
    )


def test_empty_requests_complete_with_empty_result(tmp_path: Path) -> None:
    flow_service, step_service, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    step_id = attach_dispatch_step(flow_service, parent_id, make_dispatch_step(parent_id, []))

    step_service.run_step(step_id)

    step = step_service.wait_step(step_id)
    assert step.status is StepStatus.COMPLETED
    assert step.result is not None
    assert step.result.outcome == "empty"
    assert step.result.child_flow_ids == []


def test_invalid_request_fails_result_without_creating_children(tmp_path: Path) -> None:
    flow_service, step_service, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    request = FlowRequest(flow_type="dispatch_child_flow", scope_id="scope", params={"value": "not-int"})
    step_id = attach_dispatch_step(flow_service, parent_id, make_dispatch_step(parent_id, [request]))

    step_service.run_step(step_id)

    step = step_service.wait_step(step_id)
    assert step.status is StepStatus.COMPLETED
    assert step.result is not None
    assert step.result.outcome == "failed"
    assert step.result.failed_request_indices == [0]
    assert isinstance(step.state, DispatchStepState)
    assert step.state.failed_requests[0].request_index == 0
    assert flow_service.store.list_child_flows(parent_flow_id=parent_id, parent_dispatch_step_id=step_id) == []


def test_valid_requests_create_child_flows_with_parent_links_in_order(tmp_path: Path) -> None:
    flow_service, step_service, schedule = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    requests = [
        FlowRequest(flow_type="dispatch_child_flow", scope_id="scope", params={"value": 1}),
        FlowRequest(flow_type="dispatch_child_flow", scope_id="scope", params={"value": 2}),
    ]
    step_id = attach_dispatch_step(flow_service, parent_id, make_dispatch_step(parent_id, requests))

    step_service.run_step(step_id)

    step = step_service.wait_step(step_id)
    children = flow_service.store.list_child_flows(parent_flow_id=parent_id, parent_dispatch_step_id=step_id)
    child_ids = [child.flow_id for child in children]
    assert step.status is StepStatus.COMPLETED
    assert step.result is not None
    assert step.result.outcome == "dispatched"
    assert step.result.child_flow_ids == child_ids
    assert isinstance(step.state, DispatchStepState)
    assert [child.child_flow_id for child in step.state.created_children] == child_ids
    assert [child.request_index for child in step.state.created_children] == [0, 1]
    assert [child.parent_flow_id for child in children] == [parent_id, parent_id]
    assert [child.parent_dispatch_step_id for child in children] == [step_id, step_id]
    assert schedule.flow_ids[:2] == child_ids
    assert parent_id in schedule.flow_ids


def test_start_flow_exception_becomes_step_failed(tmp_path: Path) -> None:
    flow_service, step_service, _ = make_services(tmp_path / ".agent_runtime", register_exploding=True)
    parent_id = start_parent(flow_service)
    request = FlowRequest(flow_type="exploding_flow", scope_id="scope", params={"value": 1})
    step_id = attach_dispatch_step(flow_service, parent_id, make_dispatch_step(parent_id, [request]))

    step_service.run_step(step_id)

    step = step_service.wait_step(step_id)
    assert step.status is StepStatus.FAILED
    assert step.error is not None
    assert step.error.error_type == "step_run_exception"
    assert "cannot build child flow" in step.error.message
    assert flow_service.store.list_child_flows(parent_flow_id=parent_id, parent_dispatch_step_id=step_id) == []
    assert flow_service.get_flow(parent_id).status is FlowStatus.RUNNING
