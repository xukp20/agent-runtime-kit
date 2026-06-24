from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from pydantic import BaseModel, Field

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowInput,
    BaseFlowResult,
    BaseFlowState,
    ChildFlowDispatchSubmission,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepContext,
    FlowTypeRegistry,
    RuntimeScheduleService,
    StepService,
    StepStatus,
    StepTypeRegistry,
    create_standard_next_step_if_applicable,
)
from agent_runtime_kit.flow.rendering import RenderContext
from agent_runtime_kit.flow.standard_steps import AgentStep, AgentStepState, DispatchStep, DispatchStepResult
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimeMcpToolGateway


class HandoffParentParams(BaseModel):
    pass


class HandoffParentInput(BaseFlowInput):
    input_type: str = "handoff_parent_input"
    summary: str | None = "handoff parent input"

    def render_for_agent(self, ctx: RenderContext) -> str:
        return "Input: terminal handoff parent"


class HandoffParentState(BaseFlowState):
    state_type: str = "handoff_parent_state"
    handoff_dispatch_step_id: str | None = None
    handoff_child_flow_ids: list[str] = Field(default_factory=list)


class HandoffParentResult(BaseFlowResult):
    result_type: str = "handoff_parent_done"
    child_flow_ids: list[str]

    def render_for_agent(self, ctx: RenderContext) -> str:
        return "Result: handed off " + ",".join(self.child_flow_ids)


class HandoffParentFlow(BaseFlow):
    flow_type: ClassVar[str] = "handoff_parent"
    Params: ClassVar[type[BaseModel]] = HandoffParentParams
    Input: ClassVar[type[BaseFlowInput]] = HandoffParentInput
    Result: ClassVar[type[BaseFlowResult]] = HandoffParentResult
    State: ClassVar[type[BaseFlowState]] = HandoffParentState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "HandoffParentFlow":
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=HandoffParentInput(),
            state=HandoffParentState(),
        )

    def create_next_step(self, ctx: FlowContext) -> str | None:
        if not self.step_ids:
            return ctx.create_step(
                AgentStep(
                    step_id=f"{self.flow_id}-initial-agent",
                    flow_id=self.flow_id,
                    scope_id=self.scope_id,
                    state=AgentStepState(
                        agent_role="planner",
                        agent_type="handoff_planner",
                        create_agent_if_missing=True,
                    ),
                )
            )
        standard_step_id = create_standard_next_step_if_applicable(self, ctx)
        if standard_step_id is not None:
            return standard_step_id
        return None

    def on_step_terminal(self, ctx: FlowStepContext) -> None:
        if isinstance(ctx.step, DispatchStep) and isinstance(ctx.step.result, DispatchStepResult):
            if ctx.step.result.outcome == "dispatched" and ctx.step.result.continuation == "terminal_handoff":
                assert isinstance(self.state, HandoffParentState)
                self.state.handoff_dispatch_step_id = ctx.step.step_id
                self.state.handoff_child_flow_ids = list(ctx.step.result.child_flow_ids)
                self.result = HandoffParentResult(
                    summary="handoff dispatched",
                    child_flow_ids=list(ctx.step.result.child_flow_ids),
                )
                super().on_step_terminal(ctx)
                return
        super().on_step_terminal(ctx)


class HandoffChildParams(BaseModel):
    name: str


class HandoffChildInput(BaseFlowInput):
    input_type: str = "handoff_child_input"
    name: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Input: handoff child {self.name}"


class HandoffChildState(BaseFlowState):
    state_type: str = "handoff_child_state"


class HandoffChildResult(BaseFlowResult):
    result_type: str = "handoff_child_done"
    name: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Result: handoff child {self.name} done"


class HandoffChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "handoff_child"
    Params: ClassVar[type[BaseModel]] = HandoffChildParams
    Input: ClassVar[type[BaseFlowInput]] = HandoffChildInput
    Result: ClassVar[type[BaseFlowResult]] = HandoffChildResult
    State: ClassVar[type[BaseFlowState]] = HandoffChildState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "HandoffChildFlow":
        params = ctx.params
        assert isinstance(params, HandoffChildParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=HandoffChildInput(name=params.name),
            state=HandoffChildState(),
        )

    def create_next_step(self, ctx: FlowContext) -> str | None:
        assert isinstance(self.input, HandoffChildInput)
        ctx.set_flow_result(HandoffChildResult(summary="handoff child done", name=self.input.name))
        return None


class HandoffAgentService:
    def __init__(self, ark: ARKServices) -> None:
        self.ark = ark
        self.gateway = RuntimeMcpToolGateway(ark_services=ark, app_services=AppServices())
        self.agents: dict[str, Agent] = {}
        self.started: list[dict[str, object]] = []

    def create_agent(
        self,
        scope_id: str,
        agent_type: str,
        cli_type: str = "codex",
        home_id: str | None = None,
    ) -> Agent:
        agent = Agent(
            agent_id="handoff-agent",
            scope_id=scope_id,
            agent_type=agent_type,
            cli_type=cli_type,
            home_id=home_id or agent_type,
        )
        self.agents[agent.agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Agent:
        return self.agents[agent_id]

    def start_agent(
        self,
        agent_id: str,
        *,
        variables: dict[str, object] | None = None,
        prompt: str | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> Agent:
        assert env is not None
        runtime_ctx = self.gateway.resolve_context_from_env(
            env,
            require_running_step=True,
            allowed_submit_tool_name="submit_child_flows",
        )
        self.started.append({"agent_id": agent_id, "step_id": runtime_ctx.step.step_id})
        submission = ChildFlowDispatchSubmission(
            submission_id="handoff-dispatch-submission",
            tool_name="submit_child_flows",
            submitted_by_agent_id=agent_id,
            continuation="terminal_handoff",
            summary="handoff child flow",
            requests=[
                FlowRequest(flow_type="handoff_child", scope_id=runtime_ctx.scope_id, params={"name": "alpha"})
            ],
        )
        self.gateway.accept_step_submission(runtime_ctx, submission)
        return self.get_agent(agent_id)

    def wait_agent(self, agent_id: str, timeout_s: float | None = None) -> object:
        return SimpleNamespace(id=f"turn-{len(self.started)}")


def make_services(tmp_path: Path) -> tuple[FlowService, StepService, RuntimeScheduleService, HandoffAgentService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(HandoffParentFlow)
    flow_registry.register(HandoffChildFlow)
    step_registry.register(AgentStep)
    step_registry.register(DispatchStep)
    ark = ARKServices()
    flow_service = FlowService(
        tmp_path / ".agent_runtime",
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    step_service = StepService(
        tmp_path / ".agent_runtime",
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    scheduler = RuntimeScheduleService(ark_services=ark, app_services=AppServices())
    agent_service = HandoffAgentService(ark)
    ark.agent_service = agent_service
    return flow_service, step_service, scheduler, agent_service


def test_terminal_handoff_dispatch_completes_parent_without_callback(tmp_path: Path) -> None:
    flow_service, step_service, scheduler, agent_service = make_services(tmp_path)
    parent_id = flow_service.start_flow(
        FlowRequest(flow_type="handoff_parent", scope_id="scope", params={}),
        enqueue=True,
    )

    for _ in range(10):
        tick = scheduler.schedule_ready()
        for step_id in tick.started_step_ids:
            step_service.wait_step(step_id, timeout_s=2)
        parent = flow_service.get_flow(parent_id)
        children = flow_service.store.list_child_flows(parent_flow_id=parent_id)
        if parent.status is FlowStatus.COMPLETED and children:
            break

    parent = flow_service.get_flow(parent_id)
    children = flow_service.store.list_child_flows(parent_flow_id=parent_id)
    steps = [flow_service.get_step(step_id) for step_id in parent.step_ids]

    assert parent.status is FlowStatus.COMPLETED
    assert isinstance(parent.result, HandoffParentResult)
    assert parent.result.child_flow_ids == [child.flow_id for child in children]
    assert isinstance(parent.state, HandoffParentState)
    assert parent.state.handoff_dispatch_step_id == steps[1].step_id
    assert parent.state.handoff_child_flow_ids == [child.flow_id for child in children]
    assert [type(step).__name__ for step in steps] == ["AgentStep", "DispatchStep"]
    assert steps[0].status is StepStatus.COMPLETED
    assert steps[1].status is StepStatus.COMPLETED
    assert getattr(steps[1].result, "continuation", None) == "terminal_handoff"
    assert len(children) == 1
    assert children[0].parent_flow_id == parent_id
    assert children[0].parent_dispatch_step_id == steps[1].step_id
    assert agent_service.started == [{"agent_id": "handoff-agent", "step_id": steps[0].step_id}]

    for _ in range(5):
        tick = scheduler.schedule_ready()
        if not tick.advanced_flow_ids and not tick.started_step_ids:
            break

    child = flow_service.get_flow(children[0].flow_id)
    assert child.status is FlowStatus.COMPLETED
    assert isinstance(child.result, HandoffChildResult)
