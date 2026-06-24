from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    AgentRoleBindings,
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepState,
    BaseSubmission,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowTypeRegistry,
    StepService,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices
from agent_runtime_kit.runtime import RuntimeMcpToolGateway as ExportedRuntimeMcpToolGateway
from agent_runtime_kit.runtime import RuntimeToolIdentity as ExportedRuntimeToolIdentity
from agent_runtime_kit.runtime.mcp_tool_gateway import (
    RuntimeMcpContextResolver,
    RuntimeMcpToolGateway,
    RuntimeToolContextError,
    RuntimeToolIdentity,
)


class GatewayFlowParams(BaseModel):
    pass


class GatewayFlowState(BaseFlowState):
    state_type: str = "gateway_flow_state"


class GatewayFlow(BaseFlow):
    flow_type: ClassVar[str] = "gateway_flow"
    Params: ClassVar[type[BaseModel]] = GatewayFlowParams
    State: ClassVar[type[BaseFlowState]] = GatewayFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "GatewayFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=GatewayFlowState())


class GatewayStepState(BaseStepState):
    state_type: str = "gateway_step_state"


class GatewayStep(BaseStep):
    step_type: ClassVar[str] = "gateway_step"
    State: ClassVar[type[BaseStepState]] = GatewayStepState
    SubmitTools: ClassVar[set[str] | None] = {"submit_result"}


class FakeAgentService:
    def __init__(self, agents: dict[str, Agent]) -> None:
        self.agents = agents

    def get_agent(self, agent_id: str) -> Agent:
        return self.agents[agent_id]


class StaticFlowService:
    def __init__(self, *, step: BaseStep, flow: BaseFlow) -> None:
        self.step = step
        self.flow = flow

    def get_step(self, step_id: str) -> BaseStep:
        if step_id != self.step.step_id:
            raise KeyError(step_id)
        return self.step

    def get_flow(self, flow_id: str) -> BaseFlow:
        if flow_id != self.flow.flow_id:
            raise KeyError(flow_id)
        return self.flow


def make_services(
    runtime_root: Path,
    *,
    step_agent_id: str = "agent-1",
    flow_agent_id: str | None = None,
) -> tuple[ARKServices, FlowService, StepService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(GatewayFlow)
    step_registry.register(GatewayStep)
    ark = ARKServices()
    app = AppServices()
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=app,
    )
    step_service = StepService(runtime_root, step_registry=step_registry, ark_services=ark, app_services=app)
    agents = {
        "agent-1": Agent(
            agent_id="agent-1",
            scope_id="scope",
            agent_type="gateway_agent",
            cli_type="codex",
            home_id="gateway_agent",
        ),
        "agent-2": Agent(
            agent_id="agent-2",
            scope_id="scope",
            agent_type="gateway_agent",
            cli_type="codex",
            home_id="gateway_agent",
        ),
    }
    ark.agent_service = FakeAgentService(agents)
    flow_id = flow_service.start_flow(FlowRequest(flow_type="gateway_flow", scope_id="scope", params={}), enqueue=False)
    step = GatewayStep(
        step_id="step-1",
        flow_id=flow_id,
        scope_id="scope",
        state=GatewayStepState(),
        agent_bindings=AgentRoleBindings(by_role={"worker": step_agent_id}),
    )
    if flow_agent_id is not None:
        flow_service.store.update_flow_record(
            flow_id,
            lambda flow: flow.agent_bindings.by_role.__setitem__("worker", flow_agent_id),
        )
    flow_service.store.create_step(step)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (flow.step_ids.append("step-1"), setattr(flow, "current_step_id", "step-1")),
    )
    return ark, flow_service, step_service


def assert_context_error(exc_info: pytest.ExceptionInfo[RuntimeToolContextError], code: str) -> None:
    assert exc_info.value.code == code
    assert exc_info.value.message


def test_runtime_package_exports_gateway_types() -> None:
    assert ExportedRuntimeMcpToolGateway is RuntimeMcpToolGateway
    assert ExportedRuntimeToolIdentity is RuntimeToolIdentity


def test_identity_from_http_headers_is_case_insensitive(tmp_path: Path) -> None:
    ark, _, _ = make_services(tmp_path / ".agent_runtime")
    resolver = RuntimeMcpContextResolver(ark_services=ark)

    identity = resolver.identity_from_http_headers(
        {
            "X-Ark-Step-Id": " step-1 ",
            "x-ark-flow-id": "flow-1",
            "X-ARK-AGENT-ID": "agent-1",
        }
    )

    assert identity == RuntimeToolIdentity(step_id="step-1", flow_id="flow-1", agent_id="agent-1")


def test_identity_from_env_and_missing_identity(tmp_path: Path) -> None:
    ark, _, _ = make_services(tmp_path / ".agent_runtime")
    resolver = RuntimeMcpContextResolver(ark_services=ark)

    assert resolver.identity_from_env(
        {"ARK_STEP_ID": "step-1", "ARK_FLOW_ID": "flow-1", "ARK_AGENT_ID": "agent-1"}
    ) == RuntimeToolIdentity(step_id="step-1", flow_id="flow-1", agent_id="agent-1")
    with pytest.raises(RuntimeToolContextError) as exc_info:
        resolver.identity_from_env({"ARK_STEP_ID": "", "ARK_FLOW_ID": "flow-1"})
    assert_context_error(exc_info, "missing_identity")
    assert exc_info.value.details["missing"] == "step_id,agent_id"


def test_resolve_validates_step_flow_agent_and_submit_tool(tmp_path: Path) -> None:
    ark, flow_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.list_flows()[0].flow_id
    resolver = RuntimeMcpContextResolver(ark_services=ark)

    ctx = resolver.resolve_from_identity(
        RuntimeToolIdentity(step_id="step-1", flow_id=flow_id, agent_id="agent-1"),
        allowed_submit_tool_name="submit_result",
    )

    assert ctx.step.step_id == "step-1"
    assert ctx.flow.flow_id == flow_id
    assert ctx.agent.agent_id == "agent-1"
    assert ctx.scope_id == "scope"


def test_resolve_rejects_missing_objects_and_relation_mismatches(tmp_path: Path) -> None:
    ark, flow_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.list_flows()[0].flow_id
    resolver = RuntimeMcpContextResolver(ark_services=ark)

    cases = [
        (RuntimeToolIdentity(step_id="missing", flow_id=flow_id, agent_id="agent-1"), "step_not_found"),
        (RuntimeToolIdentity(step_id="step-1", flow_id="missing", agent_id="agent-1"), "flow_not_found"),
        (RuntimeToolIdentity(step_id="step-1", flow_id=flow_id, agent_id="missing"), "agent_not_found"),
        (RuntimeToolIdentity(step_id="step-1", flow_id=flow_id, agent_id="agent-2"), "agent_not_bound_to_step"),
    ]
    for identity, code in cases:
        with pytest.raises(RuntimeToolContextError) as exc_info:
            resolver.resolve_from_identity(identity)
        assert_context_error(exc_info, code)

    other_flow_id = flow_service.start_flow(
        FlowRequest(flow_type="gateway_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    with pytest.raises(RuntimeToolContextError) as exc_info:
        resolver.resolve_from_identity(RuntimeToolIdentity(step_id="step-1", flow_id=other_flow_id, agent_id="agent-1"))
    assert_context_error(exc_info, "step_flow_mismatch")

    agent = Agent(
        agent_id="agent-1",
        scope_id="scope-a",
        agent_type="gateway_agent",
        cli_type="codex",
        home_id="gateway_agent",
    )
    malformed_step = GatewayStep(
        step_id="malformed-step",
        flow_id="malformed-flow",
        scope_id="scope-a",
        state=GatewayStepState(),
        agent_bindings=AgentRoleBindings(by_role={"worker": "agent-1"}),
    )
    malformed_flow = GatewayFlow(
        flow_id="malformed-flow",
        scope_id="scope-b",
        state=GatewayFlowState(),
    )
    malformed_ark = ARKServices(
        flow_service=StaticFlowService(step=malformed_step, flow=malformed_flow),
        agent_service=FakeAgentService({"agent-1": agent}),
    )
    with pytest.raises(RuntimeToolContextError) as exc_info:
        RuntimeMcpContextResolver(ark_services=malformed_ark).resolve_from_identity(
            RuntimeToolIdentity(step_id="malformed-step", flow_id="malformed-flow", agent_id="agent-1")
        )
    assert_context_error(exc_info, "step_scope_mismatch")


def test_resolve_requires_running_step_and_allowed_tool(tmp_path: Path) -> None:
    ark, flow_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.list_flows()[0].flow_id
    resolver = RuntimeMcpContextResolver(ark_services=ark)
    identity = RuntimeToolIdentity(step_id="step-1", flow_id=flow_id, agent_id="agent-1")

    with pytest.raises(RuntimeToolContextError) as exc_info:
        resolver.resolve_from_identity(identity, require_running_step=True)
    assert_context_error(exc_info, "step_not_running")

    flow_service.store.update_step_record("step-1", lambda step: setattr(step, "status", StepStatus.RUNNING))
    with pytest.raises(RuntimeToolContextError) as exc_info:
        resolver.resolve_from_identity(identity, allowed_submit_tool_name="wrong_tool")
    assert_context_error(exc_info, "tool_not_allowed_for_step")


def test_gateway_accepts_submission_and_rejects_duplicates_or_wrong_agent(tmp_path: Path) -> None:
    ark, flow_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.list_flows()[0].flow_id
    flow_service.store.update_step_record("step-1", lambda step: setattr(step, "status", StepStatus.RUNNING))
    gateway = RuntimeMcpToolGateway(ark_services=ark, app_services=AppServices())
    ctx = gateway.resolve_context_from_env(
        {"ARK_STEP_ID": "step-1", "ARK_FLOW_ID": flow_id, "ARK_AGENT_ID": "agent-1"},
        require_running_step=True,
        allowed_submit_tool_name="submit_result",
    )

    submission = BaseSubmission(
        submission_id="sub-1",
        submission_type="result",
        tool_name="submit_result",
        submitted_by_agent_id="agent-1",
        summary="done",
    )
    updated = gateway.accept_step_submission(ctx, submission)

    assert updated.submission is not None
    assert updated.submission.submission_id == "sub-1"

    with pytest.raises(RuntimeToolContextError) as exc_info:
        gateway.accept_step_submission(ctx, submission)
    assert_context_error(exc_info, "tool_not_allowed_for_step")

    fresh_ark, fresh_flow_service, _ = make_services(tmp_path / "fresh_runtime")
    fresh_flow_id = fresh_flow_service.list_flows()[0].flow_id
    fresh_flow_service.store.update_step_record("step-1", lambda step: setattr(step, "status", StepStatus.RUNNING))
    fresh_gateway = RuntimeMcpToolGateway(ark_services=fresh_ark, app_services=AppServices())
    fresh_ctx = fresh_gateway.resolve_context_from_env(
        {"ARK_STEP_ID": "step-1", "ARK_FLOW_ID": fresh_flow_id, "ARK_AGENT_ID": "agent-1"},
        require_running_step=True,
    )
    with pytest.raises(RuntimeToolContextError) as exc_info:
        fresh_gateway.accept_step_submission(
            fresh_ctx,
            BaseSubmission(
                submission_id="sub-2",
                submission_type="result",
                tool_name="submit_result",
                submitted_by_agent_id="agent-2",
            ),
        )
    assert_context_error(exc_info, "submission_agent_mismatch")
    assert fresh_flow_service.get_step("step-1").submission is None
