from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowInput,
    BaseFlowResult,
    BaseFlowState,
    BaseSubmission,
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
    can_exit_standard_dispatch_wait,
    create_standard_next_step_if_applicable,
    handle_standard_step_terminal,
    on_exit_standard_dispatch_wait,
)
from agent_runtime_kit.flow.rendering import RenderContext
from agent_runtime_kit.flow.standard_steps import AgentStep, AgentStepState, DispatchStep
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimeMcpToolGateway


class CallbackParentParams(BaseModel):
    pass


class CallbackParentInput(BaseFlowInput):
    input_type: str = "callback_parent_input"
    summary: str | None = "parent input"

    def render_for_agent(self, ctx: RenderContext) -> str:
        return "Input: parent dispatch callback task"


class CallbackParentResult(BaseFlowResult):
    result_type: str = "callback_parent_done"
    summary: str | None = "parent completed after callback"

    def render_for_agent(self, ctx: RenderContext) -> str:
        return "Result: parent completed after callback"


class CallbackParentState(BaseFlowState):
    state_type: str = "callback_parent_state"
    waiting_dispatch_step_id: str | None = None


class CallbackParentFlow(BaseFlow):
    flow_type: ClassVar[str] = "callback_parent"
    Params: ClassVar[type[BaseModel]] = CallbackParentParams
    Input: ClassVar[type[BaseFlowInput]] = CallbackParentInput
    Result: ClassVar[type[BaseFlowResult]] = CallbackParentResult
    State: ClassVar[type[BaseFlowState]] = CallbackParentState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "CallbackParentFlow":
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=CallbackParentInput(),
            state=CallbackParentState(),
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
                        agent_type="planner_type",
                        create_agent_if_missing=True,
                    ),
                )
            )
        standard_step_id = create_standard_next_step_if_applicable(self, ctx)
        if standard_step_id is not None:
            return standard_step_id
        latest_step = ctx.tx.load_step_for_update(self.step_ids[-1])
        if isinstance(latest_step, AgentStep) and latest_step.submission is not None:
            ctx.set_flow_result(CallbackParentResult())
        return None

    def on_step_terminal(self, ctx: FlowStepContext) -> None:
        if handle_standard_step_terminal(self, ctx):
            return
        super().on_step_terminal(ctx)

    def can_exit_waiting(self, ctx: FlowContext) -> bool:
        return can_exit_standard_dispatch_wait(self, ctx)

    def on_exit_waiting(self, ctx: FlowContext) -> None:
        if on_exit_standard_dispatch_wait(self, ctx):
            return
        super().on_exit_waiting(ctx)


class CallbackChildParams(BaseModel):
    name: str


class CallbackChildInput(BaseFlowInput):
    input_type: str = "callback_child_input"
    name: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Input: child {self.name}"


class CallbackChildResult(BaseFlowResult):
    result_type: str = "callback_child_done"
    name: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Result: child {self.name} done"


class CallbackChildState(BaseFlowState):
    state_type: str = "callback_child_state"


class CallbackChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "callback_child"
    Params: ClassVar[type[BaseModel]] = CallbackChildParams
    Input: ClassVar[type[BaseFlowInput]] = CallbackChildInput
    Result: ClassVar[type[BaseFlowResult]] = CallbackChildResult
    State: ClassVar[type[BaseFlowState]] = CallbackChildState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "CallbackChildFlow":
        params = ctx.params
        assert isinstance(params, CallbackChildParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=CallbackChildInput(name=params.name),
            state=CallbackChildState(),
        )

    def create_next_step(self, ctx: FlowContext) -> str | None:
        assert isinstance(self.input, CallbackChildInput)
        ctx.set_flow_result(CallbackChildResult(name=self.input.name))
        return None


class CallbackAgentService:
    def __init__(self, ark: ARKServices) -> None:
        self.ark = ark
        self.gateway = RuntimeMcpToolGateway(ark_services=ark, app_services=AppServices())
        self.started: list[dict[str, object]] = []
        self.agents: dict[str, Agent] = {}

    def create_agent(
        self,
        scope_id: str,
        agent_type: str,
        cli_type: str = "codex",
        home_id: str | None = None,
    ) -> Agent:
        agent = Agent(
            agent_id="callback-agent",
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
        runtime_ctx = self.gateway.resolve_context_from_env(env, require_running_step=True)
        step = runtime_ctx.step
        self.started.append(
            {
                "agent_id": agent_id,
                "step_id": step.step_id,
                "prompt": prompt,
                "prompt_mode": getattr(step.state, "prompt_mode", None),
            }
        )

        if step.submission is None:
            if getattr(step.state, "prompt_mode", None) == "callback":
                submission = BaseSubmission(
                    submission_id="callback-submission",
                    submission_type="result",
                    tool_name="submit_callback_result",
                    submitted_by_agent_id=agent_id,
                    summary="callback consumed child result",
                )
            else:
                submission = ChildFlowDispatchSubmission(
                    submission_id="dispatch-submission",
                    tool_name="submit_child_flows",
                    submitted_by_agent_id=agent_id,
                    requests=[
                        FlowRequest(flow_type="callback_child", scope_id="scope", params={"name": "alpha"}),
                    ],
                    summary="dispatch child flow",
                )
            self.gateway.accept_step_submission(runtime_ctx, submission)
        return self.get_agent(agent_id)

    def wait_agent(self, agent_id: str, timeout_s: float | None = None) -> object:
        return SimpleNamespace(id=f"turn-{len(self.started)}")


def make_services(runtime_root: Path) -> tuple[FlowService, StepService, RuntimeScheduleService, CallbackAgentService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(CallbackParentFlow)
    flow_registry.register(CallbackChildFlow)
    step_registry.register(AgentStep)
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
    agent_service = CallbackAgentService(ark)
    ark.agent_service = agent_service
    return flow_service, step_service, scheduler, agent_service


def test_standard_dispatch_callback_pattern_runs_end_to_end(tmp_path: Path) -> None:
    flow_service, step_service, scheduler, agent_service = make_services(tmp_path / ".agent_runtime")
    parent_id = flow_service.start_flow(
        FlowRequest(flow_type="callback_parent", scope_id="scope", params={}),
        enqueue=True,
    )

    for _ in range(10):
        tick = scheduler.schedule_ready()
        for step_id in tick.started_step_ids:
            step_service.wait_step(step_id, timeout_s=2)
        if flow_service.get_flow(parent_id).status is FlowStatus.COMPLETED:
            break

    parent = flow_service.get_flow(parent_id)
    steps = [flow_service.get_step(step_id) for step_id in parent.step_ids]
    children = flow_service.store.list_child_flows(parent_flow_id=parent_id)

    assert parent.status is FlowStatus.COMPLETED
    assert isinstance(parent.result, CallbackParentResult)
    assert [type(step).__name__ for step in steps] == ["AgentStep", "DispatchStep", "AgentStep"]
    assert steps[0].status is StepStatus.COMPLETED
    assert steps[1].status is StepStatus.COMPLETED
    assert getattr(steps[1].result, "continuation", None) == "wait_for_callback"
    assert steps[2].status is StepStatus.COMPLETED
    assert len(children) == 1
    assert children[0].status is FlowStatus.COMPLETED
    assert isinstance(children[0].result, CallbackChildResult)
    assert agent_service.started[0]["prompt_mode"] == "initial"
    assert agent_service.started[1]["prompt_mode"] == "callback"
    assert "Result: child alpha done" in str(agent_service.started[1]["prompt"])
