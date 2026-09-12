from pathlib import Path
from contextlib import contextmanager
from dataclasses import replace
from threading import RLock
from types import SimpleNamespace
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
        self.flow_ids: list[str] = []

    def enqueue_step(self, step_id: str) -> None:
        self.step_ids.append(step_id)

    def enqueue_flow(self, flow_id: str) -> None:
        self.flow_ids.append(flow_id)


class FakeAgentService:
    def __init__(self, agents: list[Agent]) -> None:
        self.agents = {agent.agent_id: agent for agent in agents}
        self.created: list[Agent] = []
        self.unresolved_agent_ids: set[str] = set()
        self.confirmed_agent_ids: set[str] = set()
        self.reconcile_calls: list[dict[str, object]] = []
        self.lock = RLock()
        self.provider_turn = None

    @contextmanager
    def hold_agent_boundary(self):
        with self.lock:
            yield

    def get_agent(self, agent_id: str) -> Agent:
        if agent_id not in self.agents:
            raise FileNotFoundError(agent_id)
        return self.agents[agent_id]

    def close_agent(self, agent_id: str) -> Agent:
        with self.lock:
            self.agents[agent_id].status = "closed"
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

    def fork_agent_for_recovery(self, source_agent_id: str, *, target_scope_id: str | None = None) -> Agent:
        source = self.get_agent(source_agent_id)
        agent = replace(
            source,
            agent_id=f"fork-{len(self.created) + 1}",
            scope_id=target_scope_id or source.scope_id,
            status="idle",
        )
        self.agents[agent.agent_id] = agent
        self.created.append(agent)
        return agent

    def has_running_agents(self, scope_id: str | None = None) -> bool:
        return any(
            agent.status == "running" and (scope_id is None or agent.scope_id == scope_id)
            for agent in self.agents.values()
        )

    def list_running_agents(self, scope_id: str | None = None) -> list[Agent]:
        return [
            agent
            for agent in self.agents.values()
            if agent.status == "running" and (scope_id is None or agent.scope_id == scope_id)
        ]

    def audit_running_agents(self, scope_id: str | None = None):  # noqa: ANN201
        return [
            SimpleNamespace(agent_id=agent.agent_id, classification="safe_to_mark_idle")
            for agent in self.list_running_agents(scope_id)
        ]

    def repair_running_agent(self, agent_id: str, **_kwargs):  # noqa: ANN003, ANN201
        self.agents[agent_id].status = "idle"
        return SimpleNamespace(repaired=True)

    def query_turn(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
        return self.provider_turn

    def inspect_agent_context_maintenance(self, agent_id: str):  # noqa: ANN201
        if agent_id not in self.unresolved_agent_ids | self.confirmed_agent_ids:
            return None
        return SimpleNamespace(
            unresolved=agent_id in self.unresolved_agent_ids,
            reconciliation_token="a" * 64,
        )

    def reconcile_agent_context_maintenance(
        self,
        agent_id: str,
        *,
        expected_reconciliation_token: str,
        env: dict[str, str],
        workdir: str | None,
    ) -> None:
        if expected_reconciliation_token != "a" * 64:
            raise ValueError("context maintenance reconciliation token changed")
        self.reconcile_calls.append(
            {"agent_id": agent_id, "env": env, "workdir": workdir}
        )
        self.unresolved_agent_ids.remove(agent_id)
        self.confirmed_agent_ids.add(agent_id)


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


def test_recover_failed_agent_step_preserves_complete_source_preimage(tmp_path: Path) -> None:
    agent = Agent(
        agent_id="reviewer-agent",
        scope_id="scope",
        agent_type="ReviewerAgent",
        provider_type="codex",
        home_id="ReviewerAgent",
    )
    service, ark, schedule, agents = _service(tmp_path, agent=agent)
    flow_id, failed_step_id = _failed_step(service, agent_id=agent.agent_id)

    source_preimage = service.get_step(failed_step_id).model_dump(mode="json")
    preview = service.inspect_agent_step_recovery(failed_step_id)
    receipt = service.recover_agent_step(
        step_id=failed_step_id,
        expected_status=StepStatus.FAILED,
        expected_recovery_token=preview.recovery_token,
        action="restart",
        agent_mode="reuse",
    )

    assert receipt.source_step_id == failed_step_id
    assert receipt.flow_id == flow_id
    assert receipt.replacement_agent_id == agent.agent_id
    assert receipt.agent_reused is True
    assert receipt.enqueued is True
    assert agents.created == []
    assert schedule.step_ids == [receipt.replacement_step_id]

    old_step = service.get_step(failed_step_id)
    replacement = service.get_step(receipt.replacement_step_id)
    flow = service.get_flow(flow_id)
    assert old_step.model_dump(mode="json") == source_preimage
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


def test_restart_prompt_appends_persisted_operator_instruction_after_restart_suffix(
    tmp_path: Path,
) -> None:
    agent = Agent(
        agent_id="reviewer-agent",
        scope_id="scope",
        agent_type="ReviewerAgent",
        provider_type="codex",
        home_id="ReviewerAgent",
    )
    service, ark, _, _ = _service(tmp_path, agent=agent)
    flow_id, failed_step_id = _failed_step(service, agent_id=agent.agent_id)
    service.store.update_step_record(
        failed_step_id,
        lambda step: setattr(
            step.state,
            "operator_instruction",
            "Use the current workspace truth.",
        ),
    )
    preview = service.inspect_agent_step_recovery(failed_step_id)

    receipt = service.recover_agent_step(
        step_id=failed_step_id,
        expected_status=StepStatus.FAILED,
        expected_recovery_token=preview.recovery_token,
        action="restart",
        agent_mode="reuse",
    )
    replacement = service.get_step(receipt.replacement_step_id)
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

    assert replacement.state.operator_instruction == "Use the current workspace truth."
    assert prompt.endswith("Use the current workspace truth.")
    assert prompt.count("Use the current workspace truth.") == 1
    assert prompt.index(AGENT_STEP_RESTART_PROMPT_SUFFIX) < prompt.index(
        "Use the current workspace truth."
    )


def test_recover_failed_agent_step_auto_replaces_closed_agent(tmp_path: Path) -> None:
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

    preview = service.inspect_agent_step_recovery(failed_step_id)
    receipt = service.recover_agent_step(
        step_id=failed_step_id,
        expected_status=StepStatus.FAILED,
        expected_recovery_token=preview.recovery_token,
        action="restart",
    )

    assert receipt.agent_reused is False
    assert receipt.replacement_agent_id == "fresh-1"
    assert len(agents.created) == 1
    assert agents.created[0].agent_type == "ReviewerAgent"
    assert agents.created[0].provider_type == "opencode"
    assert agents.created[0].home_id == "ClosedReviewerHome"
    assert service.get_step(receipt.replacement_step_id).agent_bindings.get("reviewer") == "fresh-1"
    assert service.get_flow(flow_id).agent_bindings.get("reviewer") == "fresh-1"


def test_recover_failed_agent_step_requires_paused_scope(tmp_path: Path) -> None:
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

    preview = service.inspect_agent_step_recovery(failed_step_id)
    with pytest.raises(RuntimeError, match="not paused"):
        service.recover_agent_step(
            step_id=failed_step_id,
            expected_status=StepStatus.FAILED,
            expected_recovery_token=preview.recovery_token,
            action="restart",
        )
