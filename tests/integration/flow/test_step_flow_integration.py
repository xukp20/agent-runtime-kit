from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowState,
    BaseStep,
    BaseStepResult,
    BaseStepState,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowTypeRegistry,
    StepRunContext,
    StepService,
    StepTerminalReceipt,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class IntegrationFlowParams(BaseModel):
    pass


class IntegrationFlowState(BaseFlowState):
    state_type: str = "integration_flow_state"
    terminal_seen: bool = False


class IntegrationStepState(BaseStepState):
    state_type: str = "integration_step_state"


class IntegrationStep(BaseStep):
    step_type: ClassVar[str] = "integration_step"
    State: ClassVar[type[BaseStepState]] = IntegrationStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return ctx.complete_step(BaseStepResult(result_type="integration_step_done"))


class IntegrationFlow(BaseFlow):
    flow_type: ClassVar[str] = "integration_flow"
    Params: ClassVar[type[BaseModel]] = IntegrationFlowParams
    State: ClassVar[type[BaseFlowState]] = IntegrationFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "IntegrationFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=IntegrationFlowState())

    def create_next_step(self, ctx: FlowContext) -> str | None:
        return ctx.create_step(IntegrationStep(step_id="integration-step", flow_id=self.flow_id, scope_id=self.scope_id))

    def on_step_terminal(self, ctx) -> None:
        assert isinstance(self.state, IntegrationFlowState)
        self.state.terminal_seen = True
        super().on_step_terminal(ctx)


def make_services(runtime_root: Path) -> tuple[FlowService, StepService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(IntegrationFlow)
    step_registry.register(IntegrationStep)
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
    return flow_service, step_service


def test_flow_advance_step_run_and_flow_terminal_absorption(tmp_path: Path) -> None:
    flow_service, step_service = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.start_flow(FlowRequest(flow_type="integration_flow", scope_id="scope", params={}), enqueue=False)

    step_id = flow_service.advance_flow(flow_id)
    step_service.run_step(step_id or "")

    flow = flow_service.get_flow(flow_id)
    step = step_service.wait_step(step_id or "")
    assert step.result is not None
    assert step.result.result_type == "integration_step_done"
    assert flow.current_step_id is None
    assert isinstance(flow.state, IntegrationFlowState)
    assert flow.state.terminal_seen is True
