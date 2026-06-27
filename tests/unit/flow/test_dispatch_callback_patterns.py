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
    ChildFlowDispatchSubmission,
    CreatedChildFlow,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepContext,
    FlowStepValidationError,
    RuntimeScheduleService,
    StepService,
    StepStatus,
    StepTypeRegistry,
    FlowTypeRegistry,
    can_exit_standard_dispatch_wait,
    create_dispatch_step_from_agent_submission,
    create_dispatch_step_from_step_submission,
    create_followup_agent_step_from_dispatch,
    create_standard_next_step_if_applicable,
    handle_standard_step_terminal,
    on_exit_standard_dispatch_wait,
)
from agent_runtime_kit.flow.standard_steps import (
    AgentStep,
    AgentStepResult,
    AgentStepState,
    DispatchStep,
    DispatchStepResult,
    DispatchStepState,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class PatternParentParams(BaseModel):
    pass


class PatternParentState(BaseFlowState):
    state_type: str = "pattern_parent_state"
    waiting_dispatch_step_id: str | None = None


class PatternParentFlow(BaseFlow):
    flow_type: ClassVar[str] = "pattern_parent"
    Params: ClassVar[type[BaseModel]] = PatternParentParams
    State: ClassVar[type[BaseFlowState]] = PatternParentState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "PatternParentFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=PatternParentState())


class PatternChildParams(BaseModel):
    pass


class PatternChildState(BaseFlowState):
    state_type: str = "pattern_child_state"


class PatternChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "pattern_child"
    Params: ClassVar[type[BaseModel]] = PatternChildParams
    State: ClassVar[type[BaseFlowState]] = PatternChildState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "PatternChildFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=PatternChildState())


class PatternLogicDispatchSourceStep(BaseStep):
    step_type: ClassVar[str] = "logic_dispatch_source"
    State: ClassVar[type[BaseStepState]] = BaseStepState
    Submissions: ClassVar[dict[str, type[ChildFlowDispatchSubmission]]] = {
        "child_flow_dispatch": ChildFlowDispatchSubmission
    }


def make_services(runtime_root: Path) -> tuple[FlowService, StepService, RuntimeScheduleService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(PatternParentFlow)
    flow_registry.register(PatternChildFlow)
    step_registry.register(AgentStep)
    step_registry.register(DispatchStep)
    step_registry.register(PatternLogicDispatchSourceStep)
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


def start_parent(flow_service: FlowService) -> str:
    return flow_service.start_flow(FlowRequest(flow_type="pattern_parent", scope_id="scope", params={}), enqueue=False)


def start_child(flow_service: FlowService, *, terminal: bool = False) -> str:
    child_id = flow_service.start_flow(FlowRequest(flow_type="pattern_child", scope_id="scope", params={}), enqueue=False)
    if terminal:
        flow_service.store.update_flow_record(
            child_id,
            lambda flow: (setattr(flow, "status", FlowStatus.COMPLETED), setattr(flow, "result", BaseFlowResult(result_type="done"))),
        )
    return child_id


def attach_step(flow_service: FlowService, flow_id: str, step) -> str:
    flow_service.store.create_step(step)
    flow_service.store.update_flow_record(flow_id, lambda flow: flow.step_ids.append(step.step_id))
    return step.step_id


def make_dispatch_submission(*, continuation: str = "wait_for_callback") -> ChildFlowDispatchSubmission:
    return ChildFlowDispatchSubmission(
        submission_id="dispatch-submission",
        tool_name="submit_child_flows",
        continuation=continuation,
        requests=[FlowRequest(flow_type="pattern_child", scope_id="scope", params={})],
        summary="dispatch child",
    )


def attach_source_agent_step(flow_service: FlowService, flow_id: str, *, continuation: str = "wait_for_callback") -> AgentStep:
    source = AgentStep(
        step_id="source-agent-step",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.COMPLETED,
        state=AgentStepState(agent_role="planner", agent_type="planner_type"),
        submission=make_dispatch_submission(continuation=continuation),
        result=AgentStepResult(result_type="agent_step_submission", outcome="submitted", submission_id="dispatch-submission"),
    )
    source.agent_bindings.by_role["planner"] = "agent-planner"
    attach_step(flow_service, flow_id, source)
    return source


def attach_source_base_step(flow_service: FlowService, flow_id: str, *, continuation: str = "terminal_handoff") -> BaseStep:
    source = PatternLogicDispatchSourceStep(
        step_id="source-logic-step",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.COMPLETED,
        submission=make_dispatch_submission(continuation=continuation),
        result=BaseStepResult(result_type="logic_dispatch_source_done"),
    )
    attach_step(flow_service, flow_id, source)
    return source


def flow_ctx(flow_service: FlowService, flow_id: str):
    tx = flow_service.store.edit_session("scope")
    tx.__enter__()
    flow = tx.load_flow_for_update(flow_id)
    return tx, FlowContext(ark=flow_service.ark, app=AppServices(), flow=flow, tx=tx)


def test_agent_dispatch_submission_creates_dispatch_step_once(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_agent_step(flow_service, parent_id)

    tx, ctx = flow_ctx(flow_service, parent_id)
    try:
        first_id = create_dispatch_step_from_step_submission(ctx, source_step_id=source.step_id)
        second_id = create_dispatch_step_from_step_submission(ctx, source_step_id=source.step_id)
    finally:
        tx.__exit__(None, None, None)

    flow = flow_service.get_flow(parent_id)
    dispatch_steps = [
        flow_service.get_step(step_id)
        for step_id in flow.step_ids
        if isinstance(flow_service.get_step(step_id), DispatchStep)
    ]
    assert first_id == second_id
    assert len(dispatch_steps) == 1
    dispatch = dispatch_steps[0]
    assert isinstance(dispatch.state, DispatchStepState)
    assert dispatch.state.source_submission_id == "dispatch-submission"
    assert dispatch.state.continuation == "wait_for_callback"


def test_agent_dispatch_submission_wrapper_remains_available(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_agent_step(flow_service, parent_id)

    tx, ctx = flow_ctx(flow_service, parent_id)
    try:
        dispatch_id = create_dispatch_step_from_agent_submission(ctx, source_agent_step_id=source.step_id)
    finally:
        tx.__exit__(None, None, None)

    dispatch = flow_service.get_step(dispatch_id)
    assert isinstance(dispatch, DispatchStep)
    assert isinstance(dispatch.state, DispatchStepState)
    assert dispatch.state.source_step_id == source.step_id


def test_agent_dispatch_submission_copies_terminal_handoff_continuation(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_agent_step(flow_service, parent_id, continuation="terminal_handoff")

    tx, ctx = flow_ctx(flow_service, parent_id)
    try:
        dispatch_id = create_dispatch_step_from_agent_submission(ctx, source_agent_step_id=source.step_id)
    finally:
        tx.__exit__(None, None, None)

    dispatch = flow_service.get_step(dispatch_id)
    assert isinstance(dispatch, DispatchStep)
    assert isinstance(dispatch.state, DispatchStepState)
    assert dispatch.state.continuation == "terminal_handoff"


def test_non_agent_step_dispatch_submission_creates_terminal_handoff_dispatch_step(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_base_step(flow_service, parent_id, continuation="terminal_handoff")

    tx, ctx = flow_ctx(flow_service, parent_id)
    try:
        dispatch_id = create_dispatch_step_from_step_submission(ctx, source_step_id=source.step_id)
    finally:
        tx.__exit__(None, None, None)

    dispatch = flow_service.get_step(dispatch_id)
    assert isinstance(dispatch, DispatchStep)
    assert isinstance(dispatch.state, DispatchStepState)
    assert dispatch.state.source_step_id == source.step_id
    assert dispatch.state.source_submission_id == "dispatch-submission"
    assert dispatch.state.continuation == "terminal_handoff"
    assert len(dispatch.state.requests) == 1


def test_standard_next_step_accepts_non_agent_dispatch_source(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_base_step(flow_service, parent_id, continuation="terminal_handoff")

    tx, ctx = flow_ctx(flow_service, parent_id)
    try:
        dispatch_id = create_standard_next_step_if_applicable(ctx.flow, ctx)
    finally:
        tx.__exit__(None, None, None)

    assert dispatch_id is not None
    dispatch = flow_service.get_step(dispatch_id)
    assert isinstance(dispatch, DispatchStep)
    assert isinstance(dispatch.state, DispatchStepState)
    assert dispatch.state.source_step_id == source.step_id
    assert dispatch.state.continuation == "terminal_handoff"


def test_standard_terminal_helper_accepts_non_agent_dispatch_source(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_base_step(flow_service, parent_id, continuation="terminal_handoff")

    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        flow.current_step_id = source.step_id
        step = tx.load_step_for_update(source.step_id)
        ctx = FlowStepContext(ark=flow_service.ark, app=AppServices(), flow=flow, step=step, tx=tx)
        assert handle_standard_step_terminal(flow, ctx) is True
        assert flow.current_step_id is None
        assert flow.status is FlowStatus.RUNNING


def test_create_followup_waits_until_all_child_flows_terminal(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_agent_step(flow_service, parent_id)
    child_id = start_child(flow_service, terminal=False)
    dispatch = DispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        status=StepStatus.COMPLETED,
        state=DispatchStepState(
            source_step_id=source.step_id,
            source_submission_id="dispatch-submission",
            requests=[],
            created_children=[CreatedChildFlow(request_index=0, child_flow_id=child_id)],
        ),
        result=DispatchStepResult(
            outcome="dispatched",
            source_step_id=source.step_id,
            source_submission_id="dispatch-submission",
            child_flow_ids=[child_id],
        ),
    )
    attach_step(flow_service, parent_id, dispatch)

    tx, ctx = flow_ctx(flow_service, parent_id)
    try:
        assert create_standard_next_step_if_applicable(ctx.flow, ctx) is None
    finally:
        tx.__exit__(None, None, None)

    flow_service.store.update_flow_record(
        child_id,
        lambda flow: (setattr(flow, "status", FlowStatus.COMPLETED), setattr(flow, "result", BaseFlowResult(result_type="done"))),
    )
    tx, ctx = flow_ctx(flow_service, parent_id)
    try:
        callback_id = create_followup_agent_step_from_dispatch(
            ctx,
            source_agent_step_id=source.step_id,
            dispatch_step_id=dispatch.step_id,
        )
    finally:
        tx.__exit__(None, None, None)

    callback = flow_service.get_step(callback_id)
    assert isinstance(callback, AgentStep)
    assert isinstance(callback.state, AgentStepState)
    assert callback.agent_bindings.by_role["planner"] == "agent-planner"
    assert callback.state.prompt_mode == "callback"
    assert callback.state.callback_dispatch_step_id == dispatch.step_id


def test_standard_terminal_and_waiting_helpers(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_agent_step(flow_service, parent_id)
    child_id = start_child(flow_service, terminal=False)
    dispatch = DispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        status=StepStatus.COMPLETED,
        state=DispatchStepState(
            source_step_id=source.step_id,
            source_submission_id="dispatch-submission",
            requests=[],
            created_children=[CreatedChildFlow(request_index=0, child_flow_id=child_id)],
        ),
        result=DispatchStepResult(
            outcome="dispatched",
            source_step_id=source.step_id,
            source_submission_id="dispatch-submission",
            child_flow_ids=[child_id],
        ),
    )
    attach_step(flow_service, parent_id, dispatch)

    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        step = tx.load_step_for_update(dispatch.step_id)
        ctx = FlowStepContext(ark=flow_service.ark, app=AppServices(), flow=flow, step=step, tx=tx)
        assert handle_standard_step_terminal(flow, ctx) is True
        assert flow.status is FlowStatus.WAITING
        assert isinstance(flow.state, PatternParentState)
        assert flow.state.waiting_dispatch_step_id == dispatch.step_id

    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        ctx = FlowContext(ark=flow_service.ark, app=AppServices(), flow=flow, tx=tx)
        assert can_exit_standard_dispatch_wait(flow, ctx) is False

    flow_service.store.update_flow_record(
        child_id,
        lambda flow: (setattr(flow, "status", FlowStatus.FAILED), setattr(flow, "result", None)),
    )
    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        ctx = FlowContext(ark=flow_service.ark, app=AppServices(), flow=flow, tx=tx)
        assert can_exit_standard_dispatch_wait(flow, ctx) is True
        assert on_exit_standard_dispatch_wait(flow, ctx) is True
        assert flow.status is FlowStatus.RUNNING
        assert isinstance(flow.state, PatternParentState)
        assert flow.state.waiting_dispatch_step_id is None


def test_terminal_handoff_dispatch_is_not_handled_by_standard_waiting_helpers(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = attach_source_agent_step(flow_service, parent_id, continuation="terminal_handoff")
    child_id = start_child(flow_service, terminal=True)
    dispatch = DispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        status=StepStatus.COMPLETED,
        state=DispatchStepState(
            source_step_id=source.step_id,
            source_submission_id="dispatch-submission",
            requests=[],
            continuation="terminal_handoff",
            created_children=[CreatedChildFlow(request_index=0, child_flow_id=child_id)],
        ),
        result=DispatchStepResult(
            outcome="dispatched",
            continuation="terminal_handoff",
            source_step_id=source.step_id,
            source_submission_id="dispatch-submission",
            child_flow_ids=[child_id],
        ),
    )
    attach_step(flow_service, parent_id, dispatch)

    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        step = tx.load_step_for_update(dispatch.step_id)
        ctx = FlowStepContext(ark=flow_service.ark, app=AppServices(), flow=flow, step=step, tx=tx)
        assert handle_standard_step_terminal(flow, ctx) is False
        assert isinstance(flow.state, PatternParentState)
        assert flow.state.waiting_dispatch_step_id is None

    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        ctx = FlowContext(ark=flow_service.ark, app=AppServices(), flow=flow, tx=tx)
        assert create_standard_next_step_if_applicable(ctx.flow, ctx) is None
        with pytest.raises(FlowStepValidationError):
            create_followup_agent_step_from_dispatch(
                ctx,
                source_agent_step_id=source.step_id,
                dispatch_step_id=dispatch.step_id,
            )

    flow_service.store.update_flow_record(
        parent_id,
        lambda flow: setattr(flow.state, "waiting_dispatch_step_id", dispatch.step_id),
    )
    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        ctx = FlowContext(ark=flow_service.ark, app=AppServices(), flow=flow, tx=tx)
        with pytest.raises(FlowStepValidationError):
            can_exit_standard_dispatch_wait(flow, ctx)
