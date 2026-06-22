from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseSubmission,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowTypeRegistry,
    FlowStepValidationError,
    StepRunContext,
    StepService,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.flow.standard_steps import (
    AgentStep,
    AgentStepState,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class AgentHostFlowParams(BaseModel):
    pass


class AgentHostFlowState(BaseFlowState):
    state_type: str = "agent_host_flow_state"


class AgentHostFlow(BaseFlow):
    flow_type: ClassVar[str] = "agent_host_flow"
    Params: ClassVar[type[BaseModel]] = AgentHostFlowParams
    State: ClassVar[type[BaseFlowState]] = AgentHostFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "AgentHostFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=AgentHostFlowState())


class FakeAgentService:
    def __init__(
        self,
        ark: ARKServices,
        *,
        submit_on_start: bool = False,
        timeout_on_wait: bool = False,
    ) -> None:
        self.ark = ark
        self.submit_on_start = submit_on_start
        self.timeout_on_wait = timeout_on_wait
        self.created: list[dict[str, object]] = []
        self.start_calls: list[dict[str, object]] = []
        self.wait_calls: list[dict[str, object]] = []
        self.next_agent = 1

    def create_agent(
        self,
        scope_id: str,
        agent_type: str,
        cli_type: str = "codex",
        home_id: str | None = None,
    ) -> Agent:
        agent = Agent(
            agent_id=f"agent-{self.next_agent}",
            scope_id=scope_id,
            agent_type=agent_type,
            cli_type=cli_type,
            home_id=home_id or agent_type,
        )
        self.next_agent += 1
        self.created.append(
            {
                "scope_id": scope_id,
                "agent_type": agent_type,
                "cli_type": cli_type,
                "home_id": home_id,
                "agent_id": agent.agent_id,
            }
        )
        return agent

    def start_agent(
        self,
        agent_id: str,
        *,
        variables: dict[str, object] | None = None,
        prompt: str | None = None,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
    ) -> Agent:
        call = {
            "agent_id": agent_id,
            "variables": variables or {},
            "prompt": prompt,
            "env": env or {},
            "workdir": workdir,
        }
        self.start_calls.append(call)
        if self.submit_on_start:
            self._accept_submission(call)
        return Agent(agent_id=agent_id, scope_id="scope", agent_type="worker", cli_type="codex", home_id="worker")

    def wait_agent(self, agent_id: str, timeout_s: float | None = None) -> object:
        self.wait_calls.append({"agent_id": agent_id, "timeout_s": timeout_s})
        if self.timeout_on_wait:
            raise TimeoutError(agent_id)
        return SimpleNamespace(id=f"turn-{len(self.start_calls)}", agent_id=agent_id)

    def _accept_submission(self, call: dict[str, object]) -> None:
        env = call["env"]
        assert isinstance(env, dict)
        step_id = str(env["ARK_STEP_ID"])
        agent_id = str(env["ARK_AGENT_ID"])
        store = self.ark.flow_service.store

        def write_submission(step):
            assert step.status is StepStatus.RUNNING
            if step.submission is None:
                step.submission = BaseSubmission(
                    submission_id="submission-1",
                    submission_type="result",
                    tool_name="submit_result",
                    submitted_by_agent_id=agent_id,
                    summary="submitted",
                )

        store.update_step_record(step_id, write_submission)


def make_services(
    runtime_root: Path,
    *,
    submit_on_start: bool = False,
    timeout_on_wait: bool = False,
) -> tuple[FlowService, StepService, FakeAgentService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(AgentHostFlow)
    step_registry.register(AgentStep)
    ark = ARKServices()
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    step_service = StepService(
        runtime_root,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )
    agent_service = FakeAgentService(ark, submit_on_start=submit_on_start, timeout_on_wait=timeout_on_wait)
    ark.agent_service = agent_service
    return flow_service, step_service, agent_service


def create_flow(flow_service: FlowService) -> str:
    return flow_service.start_flow(
        FlowRequest(flow_type="agent_host_flow", scope_id="scope", params={}),
        enqueue=False,
    )


def attach_agent_step(flow_service: FlowService, flow_id: str, step: AgentStep) -> str:
    flow_service.store.create_step(step)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (flow.step_ids.append(step.step_id), setattr(flow, "current_step_id", step.step_id)),
    )
    return step.step_id


def make_ctx(step_id: str, flow_id: str, ark: ARKServices) -> StepRunContext:
    return StepRunContext(ark=ark, app=AppServices(), step_id=step_id, flow_id=flow_id, scope_id="scope")


def test_prepare_agent_uses_step_binding(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = create_flow(flow_service)
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(agent_role="worker"),
    )
    step.agent_bindings.by_role["worker"] = "agent-step-bound"
    step_id = attach_agent_step(flow_service, flow_id, step)

    agent_id = step.prepare_agent(make_ctx(step_id, flow_id, flow_service.ark))

    assert agent_id == "agent-step-bound"


def test_prepare_agent_uses_flow_binding_and_records_step_binding(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = create_flow(flow_service)
    flow_service.store.update_flow_record(flow_id, lambda flow: flow.agent_bindings.by_role.__setitem__("planner", "agent-flow"))
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(agent_role="planner"),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)

    agent_id = step.prepare_agent(make_ctx(step_id, flow_id, flow_service.ark))
    latest = flow_service.get_step(step_id)

    assert agent_id == "agent-flow"
    assert latest.agent_bindings.by_role["planner"] == "agent-flow"


def test_prepare_agent_can_create_and_bind_to_flow(tmp_path: Path) -> None:
    flow_service, _, agent_service = make_services(tmp_path / ".agent_runtime")
    flow_id = create_flow(flow_service)
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="worker",
            agent_type="worker_type",
            home_id="worker_home",
            create_agent_if_missing=True,
            bind_created_agent_to="flow",
        ),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)

    agent_id = step.prepare_agent(make_ctx(step_id, flow_id, flow_service.ark))
    latest_step = flow_service.get_step(step_id)
    latest_flow = flow_service.get_flow(flow_id)

    assert agent_id == "agent-1"
    assert agent_service.created[0]["agent_type"] == "worker_type"
    assert latest_step.agent_bindings.by_role["worker"] == "agent-1"
    assert latest_flow.agent_bindings.by_role["worker"] == "agent-1"


def test_build_agent_env_preserves_identity_over_overrides(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = create_flow(flow_service)
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="worker",
            env_overrides={
                "CUSTOM": "value",
                "ARK_STEP_ID": "bad",
                "ARK_FLOW_ID": "bad",
                "ARK_AGENT_ID": "bad",
            },
        ),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)

    env = step.build_agent_env(make_ctx(step_id, flow_id, flow_service.ark), "agent-1")

    assert env == {
        "CUSTOM": "value",
        "ARK_STEP_ID": step_id,
        "ARK_FLOW_ID": flow_id,
        "ARK_AGENT_ID": "agent-1",
    }


def test_agent_step_submission_must_match_bound_agent(tmp_path: Path) -> None:
    flow_service, _, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = create_flow(flow_service)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: flow.agent_bindings.by_role.__setitem__("worker", "agent-bound"),
    )
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(agent_role="worker"),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)
    flow_service.store.update_step_record(step_id, lambda step: setattr(step, "status", StepStatus.RUNNING))
    ctx = make_ctx(step_id, flow_id, flow_service.ark)

    with pytest.raises(FlowStepValidationError):
        ctx.accept_step_submission(
            BaseSubmission(
                submission_id="sub-1",
                submission_type="result",
                tool_name="submit_result",
                submitted_by_agent_id="agent-other",
            ),
            expected_agent_id="agent-other",
        )

    accepted = ctx.accept_step_submission(
        BaseSubmission(
            submission_id="sub-2",
            submission_type="result",
            tool_name="submit_result",
            submitted_by_agent_id="agent-bound",
        ),
        expected_agent_id="agent-bound",
    )

    assert accepted.submission is not None
    assert accepted.submission.submission_id == "sub-2"


def test_run_retries_and_completes_with_incomplete_result(tmp_path: Path) -> None:
    flow_service, step_service, agent_service = make_services(tmp_path / ".agent_runtime")
    flow_id = create_flow(flow_service)
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="worker",
            agent_type="worker_type",
            create_agent_if_missing=True,
            prompt_override="start prompt",
            max_auto_continue_turns=1,
        ),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)

    step_service.run_step(step_id)

    latest = step_service.wait_step(step_id)
    assert latest.status is StepStatus.COMPLETED
    assert latest.result is not None
    assert latest.result.result_type == "agent_step_incomplete"
    assert latest.result.outcome == "incomplete"
    assert latest.result.attempts == 1
    assert len(agent_service.start_calls) == 2
    assert agent_service.start_calls[0]["prompt"] == "start prompt"
    assert "previous turn did not complete" in str(agent_service.start_calls[1]["prompt"])


def test_run_completes_from_latest_submission_truth(tmp_path: Path) -> None:
    flow_service, step_service, agent_service = make_services(tmp_path / ".agent_runtime", submit_on_start=True)
    flow_id = create_flow(flow_service)
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="worker",
            agent_type="worker_type",
            create_agent_if_missing=True,
            variables={"goal": "demo"},
            workdir_override="/tmp/demo",
        ),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)

    step_service.run_step(step_id)

    latest = step_service.wait_step(step_id)
    assert latest.status is StepStatus.COMPLETED
    assert latest.result is not None
    assert latest.result.result_type == "agent_step_submission"
    assert latest.result.outcome == "submitted"
    assert latest.result.submission_id == "submission-1"
    assert latest.submission is not None
    assert agent_service.start_calls[0]["variables"] == {"goal": "demo"}
    assert agent_service.start_calls[0]["workdir"] == "/tmp/demo"


def test_run_passes_agent_wait_timeout_to_agent_service(tmp_path: Path) -> None:
    flow_service, step_service, agent_service = make_services(tmp_path / ".agent_runtime", submit_on_start=True)
    flow_id = create_flow(flow_service)
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="worker",
            agent_type="worker_type",
            create_agent_if_missing=True,
            agent_wait_timeout_s=1.25,
        ),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)

    step_service.run_step(step_id)

    assert agent_service.wait_calls == [{"agent_id": "agent-1", "timeout_s": 1.25}]


def test_run_timeout_marks_step_failed(tmp_path: Path) -> None:
    flow_service, step_service, _agent_service = make_services(
        tmp_path / ".agent_runtime",
        timeout_on_wait=True,
    )
    flow_id = create_flow(flow_service)
    step = AgentStep(
        step_id="agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="worker",
            agent_type="worker_type",
            create_agent_if_missing=True,
            agent_wait_timeout_s=0.01,
        ),
    )
    step_id = attach_agent_step(flow_service, flow_id, step)

    step_service.run_step(step_id)

    latest = step_service.wait_step(step_id)
    assert latest.status is StepStatus.FAILED
    assert latest.error is not None
    assert latest.error.error_type == "step_run_exception"
    assert latest.error.details["exception_type"] == "TimeoutError"
