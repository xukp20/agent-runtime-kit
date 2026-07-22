from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.agent.service import AgentCompletionContext, AgentService
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStepContext,
    FlowTypeRegistry,
    RuntimeScheduleService,
    StepRunContext,
    StepService,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimeContext


class RuntimeCtxFlowParams(BaseModel):
    pass


class RuntimeCtxFlowState(BaseFlowState):
    state_type: str = "runtime_ctx_flow_state"


class RuntimeCtxFlow(BaseFlow):
    flow_type: ClassVar[str] = "runtime_ctx_flow"
    Params: ClassVar[type[BaseModel]] = RuntimeCtxFlowParams
    State: ClassVar[type[BaseFlowState]] = RuntimeCtxFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "RuntimeCtxFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=RuntimeCtxFlowState())


def test_ark_services_defaults_and_app_validate() -> None:
    ark = ARKServices()
    app = AppServices()

    assert ark.agent_service is None
    assert ark.flow_service is None
    assert ark.step_service is None
    assert ark.schedule_service is None
    assert ark.snapshot_service is None
    assert ark.pause_controller is None
    assert app.validate() is None


def test_services_register_into_shared_ark_object(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    ark = ARKServices()
    app = AppServices()
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(RuntimeCtxFlow)

    agent_service = AgentService(runtime_root, ark_services=ark, app_services=app)
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=app,
    )
    step_service = StepService(runtime_root, step_registry=step_registry, ark_services=ark, app_services=app)
    schedule_service = RuntimeScheduleService(ark_services=ark, app_services=app)

    assert ark.agent_service is agent_service
    assert ark.flow_service is flow_service
    assert ark.step_service is step_service
    assert ark.schedule_service is schedule_service
    assert ark.pause_controller is agent_service.pause_controller
    assert agent_service.ark_services is ark
    assert flow_service.ark is ark
    assert step_service.ark is ark
    assert schedule_service.ark is ark
    assert agent_service.app_services is app
    assert flow_service.app is app
    assert step_service.app is app
    assert schedule_service.app is app


def test_runtime_context_subclasses_share_base_contract() -> None:
    for ctx_cls in [
        AgentCompletionContext,
        FlowBuildContext,
        FlowContext,
        FlowStepContext,
        StepRunContext,
    ]:
        assert issubclass(ctx_cls, RuntimeContext)


def test_flow_service_start_flow_tolerates_missing_schedule_service(tmp_path: Path) -> None:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(RuntimeCtxFlow)
    ark = ARKServices()
    flow_service = FlowService(
        tmp_path / ".agent_runtime",
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=AppServices(),
    )

    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="runtime_ctx_flow", scope_id="scope", params={}),
        enqueue=True,
    )

    assert flow_service.get_flow(flow_id).flow_id == flow_id
    assert ark.schedule_service is None
