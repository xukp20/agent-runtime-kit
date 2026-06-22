from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, Field

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowError,
    BaseFlowInput,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
    BaseStepState,
    CreatedChildFlow,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepRunContext,
    StepTerminalReceipt,
    StepTypeRegistry,
)
from agent_runtime_kit.flow.standard_steps import (
    AgentStep,
    AgentStepState,
    build_followup_agent_step_from_dispatch,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class ParentFlowParams(BaseModel):
    pass


class ParentFlowState(BaseFlowState):
    state_type: str = "callback_parent_flow_state"


class ParentFlow(BaseFlow):
    flow_type: ClassVar[str] = "callback_parent_flow"
    Params: ClassVar[type[BaseModel]] = ParentFlowParams
    State: ClassVar[type[BaseFlowState]] = ParentFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "ParentFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=ParentFlowState())


class ChildFlowParams(BaseModel):
    title: str
    result_text: str


class ChildFlowInput(BaseFlowInput):
    input_type: str = "callback_child_input"
    title: str

    def render_for_agent(self, ctx) -> str:
        return f"Input: {self.title}"


class ChildFlowResult(BaseFlowResult):
    result_type: str = "callback_child_result"
    result_text: str

    def render_for_agent(self, ctx) -> str:
        return f"Result: {self.result_text}"


class ChildFlowState(BaseFlowState):
    state_type: str = "callback_child_flow_state"


class ChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "callback_child_flow"
    Params: ClassVar[type[BaseModel]] = ChildFlowParams
    State: ClassVar[type[BaseFlowState]] = ChildFlowState
    Input: ClassVar[type[BaseFlowInput]] = ChildFlowInput
    Result: ClassVar[type[BaseFlowResult]] = ChildFlowResult

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "ChildFlow":
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=ChildFlowInput(title=ctx.params.title),
            state=ChildFlowState(),
        )


class FakeDispatchStepState(BaseStepState):
    state_type: str = "fake_dispatch_step_state"
    created_children: list[CreatedChildFlow] = Field(default_factory=list)


class FakeDispatchStep(BaseStep):
    step_type: ClassVar[str] = "fake_dispatch_step"
    State: ClassVar[type[BaseStepState]] = FakeDispatchStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        raise NotImplementedError


def make_service(runtime_root: Path) -> FlowService:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(ParentFlow)
    flow_registry.register(ChildFlow)
    step_registry.register(AgentStep)
    step_registry.register(FakeDispatchStep)
    ark = ARKServices()
    return FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )


def start_parent(flow_service: FlowService) -> str:
    return flow_service.start_flow(
        FlowRequest(flow_type="callback_parent_flow", scope_id="scope", params={}),
        enqueue=False,
    )


def start_child(flow_service: FlowService, *, title: str, result_text: str) -> str:
    flow_id = flow_service.start_flow(
        FlowRequest(
            flow_type="callback_child_flow",
            scope_id="scope",
            params={"title": title, "result_text": result_text},
        ),
        enqueue=False,
    )

    def complete(flow: BaseFlow) -> None:
        flow.status = FlowStatus.COMPLETED
        flow.result = ChildFlowResult(result_text=result_text)

    flow_service.store.update_flow_record(flow_id, complete)
    return flow_id


def attach_step(flow_service: FlowService, flow_id: str, step: BaseStep) -> str:
    flow_service.store.create_step(step)
    flow_service.store.update_flow_record(flow_id, lambda flow: flow.step_ids.append(step.step_id))
    return step.step_id


def make_callback_step(flow_id: str, dispatch_step_id: str) -> AgentStep:
    return AgentStep(
        step_id="callback-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="planner",
            prompt_mode="callback",
            callback_dispatch_step_id=dispatch_step_id,
        ),
    )


def make_ctx(flow_service: FlowService, flow_id: str, step_id: str) -> StepRunContext:
    return StepRunContext(ark=flow_service.ark, app=AppServices(), step_id=step_id, flow_id=flow_id, scope_id="scope")


def test_callback_prompt_renders_child_inputs_and_results_without_runtime_ids(tmp_path: Path) -> None:
    flow_service = make_service(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    child_a = start_child(flow_service, title="alpha source", result_text="alpha result")
    child_b = start_child(flow_service, title="beta source", result_text="beta result")
    dispatch = FakeDispatchStep(
        step_id="dispatch-secret",
        flow_id=parent_id,
        scope_id="scope",
        state=FakeDispatchStepState(
            created_children=[
                CreatedChildFlow(request_index=0, child_flow_id=child_a),
                CreatedChildFlow(request_index=1, child_flow_id=child_b),
            ]
        ),
    )
    attach_step(flow_service, parent_id, dispatch)
    callback = make_callback_step(parent_id, dispatch.step_id)
    attach_step(flow_service, parent_id, callback)

    prompt = callback.build_start_prompt(make_ctx(flow_service, parent_id, callback.step_id), "agent-1")

    assert prompt is not None
    assert "Input: alpha source" in prompt
    assert "Result: alpha result" in prompt
    assert prompt.index("Input: alpha source") < prompt.index("Input: beta source")
    assert child_a not in prompt
    assert child_b not in prompt
    assert "dispatch-secret" not in prompt
    assert "callback_child_flow" not in prompt


def test_callback_prompt_uses_runtime_error_fallback_for_failed_child(tmp_path: Path) -> None:
    flow_service = make_service(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    child = start_child(flow_service, title="failed source", result_text="unused")

    def fail(flow: BaseFlow) -> None:
        flow.status = FlowStatus.FAILED
        flow.result = None
        flow.error = BaseFlowError(error_type="child_failed", message="child exploded")

    flow_service.store.update_flow_record(child, fail)
    dispatch = FakeDispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        state=FakeDispatchStepState(created_children=[CreatedChildFlow(request_index=0, child_flow_id=child)]),
    )
    attach_step(flow_service, parent_id, dispatch)
    callback = make_callback_step(parent_id, dispatch.step_id)
    attach_step(flow_service, parent_id, callback)

    prompt = callback.build_start_prompt(make_ctx(flow_service, parent_id, callback.step_id), "agent-1")

    assert prompt is not None
    assert "Input: failed source" in prompt
    assert "Runtime error:" in prompt
    assert "child exploded" in prompt


def test_callback_prompt_requires_child_input_and_completed_result(tmp_path: Path) -> None:
    flow_service = make_service(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    missing_input = start_child(flow_service, title="missing input", result_text="result")
    missing_result = start_child(flow_service, title="missing result", result_text="result")
    flow_service.store.update_flow_record(missing_input, lambda flow: setattr(flow, "input", None))
    flow_service.store.update_flow_record(missing_result, lambda flow: setattr(flow, "result", None))

    dispatch = FakeDispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        state=FakeDispatchStepState(
            created_children=[CreatedChildFlow(request_index=0, child_flow_id=missing_input)]
        ),
    )
    attach_step(flow_service, parent_id, dispatch)
    callback = make_callback_step(parent_id, dispatch.step_id)
    attach_step(flow_service, parent_id, callback)

    with pytest.raises(FlowStepValidationError, match="no renderable input"):
        callback.build_start_prompt(make_ctx(flow_service, parent_id, callback.step_id), "agent-1")

    flow_service.store.update_step_record(
        dispatch.step_id,
        lambda step: setattr(
            step.state,
            "created_children",
            [CreatedChildFlow(request_index=0, child_flow_id=missing_result)],
        ),
    )
    with pytest.raises(FlowStepValidationError, match="no renderable result"):
        callback.build_start_prompt(make_ctx(flow_service, parent_id, callback.step_id), "agent-1")


def test_build_followup_agent_step_from_dispatch_reuses_source_agent_binding(tmp_path: Path) -> None:
    flow_service = make_service(tmp_path / ".agent_runtime")
    parent_id = start_parent(flow_service)
    source = AgentStep(
        step_id="source-agent-step",
        flow_id=parent_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="planner",
            agent_type="planner_type",
            variables={"goal": "main"},
            env_overrides={"CUSTOM": "1"},
            workdir_override="/tmp/work",
            max_auto_continue_turns=3,
        ),
    )
    source.agent_bindings.by_role["planner"] = "agent-source"
    attach_step(flow_service, parent_id, source)
    dispatch = FakeDispatchStep(
        step_id="dispatch-step",
        flow_id=parent_id,
        scope_id="scope",
        state=FakeDispatchStepState(),
    )
    attach_step(flow_service, parent_id, dispatch)
    with flow_service.store.edit_session("scope") as tx:
        flow = tx.load_flow_for_update(parent_id)
        ctx = FlowContext(ark=flow_service.ark, app=AppServices(), flow=flow, tx=tx)
        followup = build_followup_agent_step_from_dispatch(
            ctx,
            step_id="followup-step",
            source_agent_step_id=source.step_id,
            dispatch_step_id=dispatch.step_id,
        )

    assert followup.agent_bindings.by_role["planner"] == "agent-source"
    assert isinstance(followup.state, AgentStepState)
    assert followup.state.prompt_mode == "callback"
    assert followup.state.followup_of_step_id == source.step_id
    assert followup.state.callback_dispatch_step_id == dispatch.step_id
    assert followup.state.create_agent_if_missing is False
    assert followup.state.variables == {"goal": "main"}
    assert followup.state.env_overrides == {"CUSTOM": "1"}
    assert followup.state.workdir_override == "/tmp/work"
    assert followup.state.max_auto_continue_turns == 3
