from pathlib import Path
from typing import ClassVar, Literal

import pytest
from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowInput,
    BaseStepResult,
    BaseFlowState,
    BaseStep,
    BaseStepState,
    BaseSubmission,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepTypeError,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.agent.store_utils import read_json, write_json_atomic
from agent_runtime_kit.runtime import ARKServices, AppServices


class ServiceFlowParams(BaseModel):
    target: str


class ServiceFlowInput(BaseFlowInput):
    input_type: Literal["service_flow"] = "service_flow"
    target: str

    def render_for_agent(self, ctx) -> str:
        return self.target


class ServiceFlowState(BaseFlowState):
    state_type: str = "service_flow_state"
    target: str


class ServiceFlow(BaseFlow):
    flow_type: ClassVar[str] = "service_flow"
    Params: ClassVar[type[BaseModel]] = ServiceFlowParams
    State: ClassVar[type[BaseFlowState]] = ServiceFlowState
    Input: ClassVar[type[BaseFlowInput]] = ServiceFlowInput
    requires_callback_input: ClassVar[bool] = True

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "ServiceFlow":
        params = ctx.params
        assert isinstance(params, ServiceFlowParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=ServiceFlowInput(target=params.target),
            state=ServiceFlowState(target=params.target),
        )


class ServiceStepState(BaseStepState):
    state_type: str = "service_step_state"


class ServiceStep(BaseStep):
    step_type: ClassVar[str] = "service_step"
    State: ClassVar[type[BaseStepState]] = ServiceStepState


class StrictServiceResult(BaseStepResult):
    result_type: Literal["strict_service_result"] = "strict_service_result"


class StrictServiceStep(BaseStep):
    step_type: ClassVar[str] = "strict_service_step"
    State: ClassVar[type[BaseStepState]] = ServiceStepState
    Result: ClassVar[type[BaseStepResult]] = StrictServiceResult
    Submission: ClassVar[type[BaseSubmission] | None] = None


class WrongTypeFlow(ServiceFlow):
    flow_type: ClassVar[str] = "wrong_type_flow"

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> BaseFlow:
        return ServiceFlow.build_from_request(ctx)


class TerminalFlow(ServiceFlow):
    flow_type: ClassVar[str] = "terminal_flow"

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "TerminalFlow":
        params = ctx.params
        assert isinstance(params, ServiceFlowParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=ServiceFlowInput(target=params.target),
            state=ServiceFlowState(target=params.target),
            status=FlowStatus.COMPLETED,
        )


class FakeScheduleService:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def enqueue_flow(self, flow_id: str) -> None:
        self.enqueued.append(flow_id)


def make_service(runtime_root: Path, *, schedule_service: object | None = None) -> FlowService:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    for flow_cls in [ServiceFlow, WrongTypeFlow, TerminalFlow]:
        flow_registry.register(flow_cls)
    step_registry.register(ServiceStep)
    step_registry.register(StrictServiceStep)
    ark = ARKServices(schedule_service=schedule_service)
    return FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )


def test_start_flow_creates_flow_through_request_and_registers_service(tmp_path: Path) -> None:
    schedule = FakeScheduleService()
    service = make_service(tmp_path / ".agent_runtime", schedule_service=schedule)

    flow_id = service.start_flow(FlowRequest(flow_type="service_flow", scope_id="scope", params={"target": "T"}))
    flow = service.get_flow(flow_id)

    assert service.ark.flow_service is service
    assert flow.flow_id == flow_id
    assert getattr(flow, "flow_type") == "service_flow"
    assert flow.scope_id == "scope"
    assert isinstance(flow.input, ServiceFlowInput)
    assert flow.input.target == "T"
    assert isinstance(flow.state, ServiceFlowState)
    assert flow.state.target == "T"
    assert schedule.enqueued == [flow_id]


def test_start_flow_can_skip_enqueue(tmp_path: Path) -> None:
    schedule = FakeScheduleService()
    service = make_service(tmp_path / ".agent_runtime", schedule_service=schedule)

    flow_id = service.start_flow(
        FlowRequest(flow_type="service_flow", scope_id="scope", params={"target": "T"}),
        enqueue=False,
    )

    assert service.get_flow(flow_id).flow_id == flow_id
    assert schedule.enqueued == []


def test_start_flow_writes_parent_links_for_child_flow(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")

    child_id = service.start_flow(
        FlowRequest(flow_type="service_flow", scope_id="child", params={"target": "T"}),
        parent_flow_id="parent-flow",
        parent_dispatch_step_id="dispatch-step",
        enqueue=False,
    )
    child = service.get_flow(child_id)

    assert child.parent_flow_id == "parent-flow"
    assert child.parent_dispatch_step_id == "dispatch-step"


def test_start_flow_rejects_invalid_params_and_unknown_type(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")

    with pytest.raises(FlowStepValidationError):
        service.start_flow(FlowRequest(flow_type="service_flow", scope_id="scope", params={}), enqueue=False)
    with pytest.raises(FlowStepTypeError):
        service.start_flow(FlowRequest(flow_type="missing", scope_id="scope", params={}), enqueue=False)


def test_start_flow_rejects_wrong_returned_flow_type(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")

    with pytest.raises(FlowStepValidationError):
        service.start_flow(FlowRequest(flow_type="wrong_type_flow", scope_id="scope", params={"target": "T"}), enqueue=False)


def test_start_flow_rejects_terminal_new_flow(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")

    with pytest.raises(FlowStepValidationError):
        service.start_flow(FlowRequest(flow_type="terminal_flow", scope_id="scope", params={"target": "T"}), enqueue=False)


def test_assert_restorable_flows_rejects_running_steps(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = service.start_flow(
        FlowRequest(flow_type="service_flow", scope_id="scope", params={"target": "T"}),
        enqueue=False,
    )
    service.store.create_step(
        ServiceStep(
            step_id="running-step",
            flow_id=flow_id,
            scope_id="scope",
            status=StepStatus.RUNNING,
        )
    )

    with pytest.raises(FlowStepValidationError):
        service.assert_restorable_flows(scope_id="scope")


def test_assert_restorable_flows_rejects_bad_terminal_step_result(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = service.start_flow(
        FlowRequest(flow_type="service_flow", scope_id="scope", params={"target": "T"}),
        enqueue=False,
    )
    step = StrictServiceStep(
        step_id="strict-step",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.COMPLETED,
        result=StrictServiceResult(summary="done"),
    )
    service.store.create_step(step)
    path = service.store.resolve_step_path(step.step_id)
    payload = read_json(path)
    payload["result"]["result_type"] = "bad_result"
    write_json_atomic(path, payload)

    with pytest.raises(FlowStepValidationError):
        service.assert_restorable_flows(scope_id="scope")


def test_assert_restorable_flows_rejects_bad_terminal_step_submission(tmp_path: Path) -> None:
    service = make_service(tmp_path / ".agent_runtime")
    flow_id = service.start_flow(
        FlowRequest(flow_type="service_flow", scope_id="scope", params={"target": "T"}),
        enqueue=False,
    )
    step = StrictServiceStep(
        step_id="strict-step",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.COMPLETED,
        result=StrictServiceResult(summary="done"),
    )
    service.store.create_step(step)
    path = service.store.resolve_step_path(step.step_id)
    payload = read_json(path)
    payload["submission"] = {
        "submission_id": "sub-1",
        "submission_type": "bad_submission",
        "tool_name": "submit_bad",
        "submitted_by_agent_id": "agent",
    }
    write_json_atomic(path, payload)

    with pytest.raises(FlowStepValidationError):
        service.assert_restorable_flows(scope_id="scope")
