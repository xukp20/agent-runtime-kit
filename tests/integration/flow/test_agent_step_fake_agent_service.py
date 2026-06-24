from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseSubmission,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowTypeRegistry,
    RuntimeScheduleService,
    StepService,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.flow.standard_steps import AgentStep, AgentStepState
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimeMcpToolGateway


class AgentStepIntegrationFlowParams(BaseModel):
    pass


class AgentStepIntegrationFlowState(BaseFlowState):
    state_type: str = "agent_step_integration_flow_state"


class AgentStepIntegrationFlow(BaseFlow):
    flow_type: ClassVar[str] = "agent_step_integration_flow"
    Params: ClassVar[type[BaseModel]] = AgentStepIntegrationFlowParams
    State: ClassVar[type[BaseFlowState]] = AgentStepIntegrationFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "AgentStepIntegrationFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=AgentStepIntegrationFlowState())

    def create_next_step(self, ctx: FlowContext) -> str | None:
        if self.step_ids:
            return None
        return ctx.create_step(
            AgentStep(
                step_id=f"{self.flow_id}-agent-step",
                flow_id=self.flow_id,
                scope_id=self.scope_id,
                state=AgentStepState(
                    agent_role="worker",
                    agent_type="worker_type",
                    create_agent_if_missing=True,
                    variables={"goal": "integration"},
                ),
            )
        )


class SubmittingAgentService:
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
            agent_id="agent-1",
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
        self.started.append({"agent_id": agent_id, "variables": variables or {}, "env": env or {}})
        assert env is not None
        runtime_ctx = self.gateway.resolve_context_from_env(
            env,
            require_running_step=True,
            allowed_submit_tool_name="submit_result",
        )
        if runtime_ctx.step.submission is None:
            self.gateway.accept_step_submission(
                runtime_ctx,
                BaseSubmission(
                    submission_id="integration-submission",
                    submission_type="result",
                    tool_name="submit_result",
                    submitted_by_agent_id=agent_id,
                    summary="integration submitted",
                ),
            )
        return self.get_agent(agent_id)

    def wait_agent(self, agent_id: str) -> object:
        return SimpleNamespace(id="turn-1")


def make_services(runtime_root: Path) -> tuple[FlowService, StepService, RuntimeScheduleService, SubmittingAgentService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(AgentStepIntegrationFlow)
    step_registry.register(AgentStep)
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
    agent_service = SubmittingAgentService(ark)
    ark.agent_service = agent_service
    return flow_service, step_service, scheduler, agent_service


def test_scheduler_runs_agent_step_until_submission_and_flow_absorbs_terminal(tmp_path: Path) -> None:
    flow_service, step_service, scheduler, agent_service = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="agent_step_integration_flow", scope_id="scope", params={}),
        enqueue=True,
    )

    tick = scheduler.schedule_ready()

    step_id = f"{flow_id}-agent-step"
    step = step_service.wait_step(step_id)
    flow = flow_service.get_flow(flow_id)
    assert tick.advanced_flow_ids == [flow_id]
    assert tick.started_step_ids == [step_id]
    assert step.status is StepStatus.COMPLETED
    assert step.result is not None
    assert step.result.result_type == "agent_step_submission"
    assert step.result.submission_id == "integration-submission"
    assert step.submission is not None
    assert flow.status is FlowStatus.RUNNING
    assert flow.current_step_id is None
    assert agent_service.started[0]["variables"] == {"goal": "integration"}
