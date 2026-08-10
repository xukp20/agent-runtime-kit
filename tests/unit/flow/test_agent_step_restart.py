from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    AgentRoleBindings,
    AgentStep,
    AgentStepState,
    BaseFlow,
    BaseFlowError,
    BaseFlowState,
    BaseStepError,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepValidationError,
    FlowTypeRegistry,
    StepRunContext,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.flow.standard_steps.agent_step import AGENT_STEP_RESTART_PROMPT_SUFFIX
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePauseController


class RestartFlowParams(BaseModel):
    pass


class RestartFlowState(BaseFlowState):
    state_type: str = "restart_flow_state"
    position_marker: str = "reviewer"


class RestartFlow(BaseFlow):
    flow_type: ClassVar[str] = "restart_flow"
    Params: ClassVar[type[BaseModel]] = RestartFlowParams
    State: ClassVar[type[BaseFlowState]] = RestartFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "RestartFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=RestartFlowState())


class RestartAgentStep(AgentStep):
    step_type: ClassVar[str] = "restart_agent_step"


class FakeScheduleService:
    def __init__(self) -> None:
        self.step_ids: list[str] = []

    def enqueue_step(self, step_id: str) -> None:
        self.step_ids.append(step_id)


class FakeAgentService:
    def __init__(self, agents: list[Agent]) -> None:
        self.agents = {agent.agent_id: agent for agent in agents}
        self.created: list[Agent] = []

    def get_agent(self, agent_id: str) -> Agent:
        if agent_id not in self.agents:
            raise FileNotFoundError(agent_id)
        return self.agents[agent_id]

    def create_agent(
        self,
        scope_id: str,
        agent_type: str,
        provider_type: str | None = None,
        home_id: str | None = None,
    ) -> Agent:
        agent = Agent(
            agent_id=f"fresh-{len(self.created) + 1}",
            scope_id=scope_id,
            agent_type=agent_type,
            provider_type=provider_type or "codex",
            home_id=home_id or agent_type,
        )
        self.agents[agent.agent_id] = agent
        self.created.append(agent)
        return agent

    def has_running_agents(self, scope_id: str | None = None) -> bool:
        return any(
            agent.status == "running" and (scope_id is None or agent.scope_id == scope_id)
            for agent in self.agents.values()
        )


def _service(tmp_path: Path, *, agent: Agent) -> tuple[FlowService, ARKServices, FakeScheduleService, FakeAgentService]:
    flow_registry = FlowTypeRegistry()
    flow_registry.register(RestartFlow)
    step_registry = StepTypeRegistry()
    step_registry.register(RestartAgentStep)
    schedule = FakeScheduleService()
    agents = FakeAgentService([agent])
    ark = ARKServices(
        agent_service=agents,
        schedule_service=schedule,
        pause_controller=RuntimePauseController(global_paused=True),
    )
    service = FlowService(
        tmp_path / ".agent_runtime",
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    return service, ark, schedule, agents


def _failed_step(service: FlowService, *, agent_id: str) -> tuple[str, str]:
    flow_id = service.start_flow(
        FlowRequest(flow_type="restart_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    step_id = "failed-reviewer"
    step = RestartAgentStep(
        step_id=step_id,
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.FAILED,
        state=AgentStepState(
            agent_role="reviewer",
            agent_type="ReviewerAgent",
            provider_type="codex",
            home_id="ReviewerAgent",
            create_agent_if_missing=True,
            bind_created_agent_to="flow",
            variables={"round_id": "round-1"},
            prompt_override="Review the current declaration batch.",
        ),
        error=BaseStepError(error_type="step_run_exception", message="stream disconnected"),
        agent_bindings=AgentRoleBindings(by_role={"reviewer": agent_id}),
    )
    service.store.create_step(step)

    def mark_failed(flow: BaseFlow) -> None:
        flow.status = FlowStatus.FAILED
        flow.error = BaseFlowError(error_type="round_step_failed", message="stream disconnected")
        flow.step_ids.append(step_id)
        flow.current_step_id = None
        flow.finished_at = "2026-08-10T00:00:00Z"
        flow.agent_bindings.by_role["reviewer"] = agent_id

    service.store.update_flow_record(flow_id, mark_failed)
    return flow_id, step_id


def test_restart_failed_agent_step_reuses_agent_and_preserves_failed_evidence(tmp_path: Path) -> None:
    agent = Agent(
        agent_id="reviewer-agent",
        scope_id="scope",
        agent_type="ReviewerAgent",
        provider_type="codex",
        home_id="ReviewerAgent",
    )
    service, ark, schedule, agents = _service(tmp_path, agent=agent)
    flow_id, failed_step_id = _failed_step(service, agent_id=agent.agent_id)

    receipt = service.restart_failed_agent_step(failed_step_id)

    assert receipt.failed_step_id == failed_step_id
    assert receipt.flow_id == flow_id
    assert receipt.agent_id == agent.agent_id
    assert receipt.agent_reused is True
    assert receipt.enqueued is True
    assert agents.created == []
    assert schedule.step_ids == [receipt.replacement_step_id]

    old_step = service.get_step(failed_step_id)
    replacement = service.get_step(receipt.replacement_step_id)
    flow = service.get_flow(flow_id)
    assert old_step.status is StepStatus.FAILED
    assert old_step.error is not None
    assert isinstance(replacement, RestartAgentStep)
    assert replacement.status is StepStatus.CREATED
    assert replacement.error is None
    assert replacement.submission is None
    assert replacement.result is None
    assert isinstance(replacement.state, AgentStepState)
    assert replacement.state.restart_of_step_id == failed_step_id
    assert replacement.state.variables == {"round_id": "round-1"}
    assert replacement.agent_bindings.get("reviewer") == agent.agent_id
    assert flow.status is FlowStatus.RUNNING
    assert flow.error is None
    assert flow.finished_at is None
    assert flow.current_step_id == replacement.step_id
    assert flow.step_ids == [failed_step_id, replacement.step_id]
    assert isinstance(flow.state, RestartFlowState)
    assert flow.state.position_marker == "reviewer"

    prompt = replacement.build_start_prompt(
        StepRunContext(
            ark=ark,
            app=AppServices(),
            step_id=replacement.step_id,
            flow_id=flow_id,
            scope_id="scope",
        ),
        agent.agent_id,
    )
    assert prompt == f"Review the current declaration batch.\n\n{AGENT_STEP_RESTART_PROMPT_SUFFIX}"


def test_restart_failed_agent_step_replaces_closed_agent(tmp_path: Path) -> None:
    agent = Agent(
        agent_id="closed-reviewer",
        scope_id="scope",
        agent_type="ReviewerAgent",
        provider_type="opencode",
        home_id="ClosedReviewerHome",
        status="closed",
    )
    service, _, _, agents = _service(tmp_path, agent=agent)
    flow_id, failed_step_id = _failed_step(service, agent_id=agent.agent_id)

    receipt = service.restart_failed_agent_step(failed_step_id)

    assert receipt.agent_reused is False
    assert receipt.agent_id == "fresh-1"
    assert len(agents.created) == 1
    assert agents.created[0].agent_type == "ReviewerAgent"
    assert agents.created[0].provider_type == "opencode"
    assert agents.created[0].home_id == "ClosedReviewerHome"
    assert service.get_step(receipt.replacement_step_id).agent_bindings.get("reviewer") == "fresh-1"
    assert service.get_flow(flow_id).agent_bindings.get("reviewer") == "fresh-1"


def test_restart_failed_agent_step_requires_paused_scope(tmp_path: Path) -> None:
    agent = Agent(
        agent_id="reviewer-agent",
        scope_id="scope",
        agent_type="ReviewerAgent",
        provider_type="codex",
        home_id="ReviewerAgent",
    )
    service, ark, _, _ = _service(tmp_path, agent=agent)
    _, failed_step_id = _failed_step(service, agent_id=agent.agent_id)
    assert isinstance(ark.pause_controller, RuntimePauseController)
    ark.pause_controller.resume()

    with pytest.raises(FlowStepValidationError, match="requires a paused runtime scope"):
        service.restart_failed_agent_step(failed_step_id)
