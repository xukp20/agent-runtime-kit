from __future__ import annotations

import importlib.util
import json
import os
import shutil
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

try:
    from mcp.server.fastmcp import Context, FastMCP
except ImportError:  # pragma: no cover - real Codex tests skip when MCP is unavailable.
    Context = Any  # type: ignore[misc, assignment]
    FastMCP = None  # type: ignore[assignment]

from agent_runtime_kit.agent.homes import HomeCreateSpec, McpServerSpec
from agent_runtime_kit.agent.models import CompletionDecision
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.service import AgentCompletionContext, AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.store import AgentStoreService
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowInput,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
    BaseStepResult,
    BaseStepState,
    BaseSubmission,
    ChildFlowDispatchSubmission,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowStepContext,
    FlowStepValidationError,
    FlowTypeRegistry,
    RuntimeScheduleService,
    StepRunContext,
    StepService,
    StepStatus,
    StepTerminalReceipt,
    StepTypeRegistry,
    can_exit_standard_dispatch_wait,
    create_standard_next_step_if_applicable,
    handle_standard_step_terminal,
    on_exit_standard_dispatch_wait,
)
from agent_runtime_kit.flow.rendering import RenderContext
from agent_runtime_kit.flow.standard_steps import AgentStep, AgentStepState, DispatchStep
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePauseController


pytestmark = pytest.mark.real_codex


REAL_AGENT_TYPE = "real_flow_submit_agent"
SUBMIT_HOME_ID = "real_flow_submit_agent"
NO_TOOL_HOME_ID = "real_flow_no_tools"
DEFAULT_AGENT_TIMEOUT_S = 600.0


class RealFlowSubmitAgentType(AgentType):
    agent_type = REAL_AGENT_TYPE
    developer_instructions_template = (
        "You are running an agent-runtime-kit real Flow/Step end-to-end test. "
        "Use the exact MCP tool requested by the prompt. After the tool call succeeds, "
        "reply briefly with the tool result."
    )
    start_prompt_template = "{{prompt}}"
    continue_prompt_template = "{{prompt}}"

    def check_completion(self, ctx: AgentCompletionContext) -> CompletionDecision:
        if not getattr(ctx.turn_result, "id", None):
            return CompletionDecision(complete=False, reason="turn result has no id")
        return CompletionDecision(complete=True)


class RealSingleAgentParams(BaseModel):
    prompt: str
    expected_summary: str = "real single agent submitted"
    home_id: str = SUBMIT_HOME_ID
    max_auto_continue_turns: int = 1


class RealSingleAgentInput(BaseFlowInput):
    input_type: str = "real_single_agent_input"
    prompt: str
    home_id: str = SUBMIT_HOME_ID
    max_auto_continue_turns: int = 1

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Input: {self.prompt}"


class RealSingleAgentState(BaseFlowState):
    state_type: str = "real_single_agent_state"


class RealSingleAgentResult(BaseFlowResult):
    result_type: str = "real_single_agent_done"
    submitted_summary: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Result: {self.submitted_summary}"


class RealSingleAgentFlow(BaseFlow):
    flow_type: ClassVar[str] = "real_single_agent"
    Params: ClassVar[type[BaseModel]] = RealSingleAgentParams
    Input: ClassVar[type[BaseFlowInput]] = RealSingleAgentInput
    Result: ClassVar[type[BaseFlowResult]] = RealSingleAgentResult
    State: ClassVar[type[BaseFlowState]] = RealSingleAgentState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "RealSingleAgentFlow":
        params = ctx.params
        assert isinstance(params, RealSingleAgentParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=RealSingleAgentInput(
                prompt=params.prompt,
                home_id=params.home_id,
                max_auto_continue_turns=params.max_auto_continue_turns,
            ),
            state=RealSingleAgentState(),
        )

    def create_next_step(self, ctx: FlowContext) -> str | None:
        params = _single_params_from_input(self.input)
        if not self.step_ids:
            return ctx.create_step(
                AgentStep(
                    step_id=f"{self.flow_id}-agent",
                    flow_id=self.flow_id,
                    scope_id=self.scope_id,
                    state=AgentStepState(
                        agent_role="worker",
                        agent_type=REAL_AGENT_TYPE,
                        home_id=params.home_id,
                        create_agent_if_missing=True,
                        bind_created_agent_to="flow",
                        variables={"prompt": params.prompt},
                        prompt_override=params.prompt,
                        max_auto_continue_turns=params.max_auto_continue_turns,
                        agent_wait_timeout_s=DEFAULT_AGENT_TIMEOUT_S,
                    ),
                )
            )
        latest_step = ctx.tx.load_step_for_update(self.step_ids[-1])
        if latest_step.status is StepStatus.COMPLETED and latest_step.submission is not None:
            ctx.set_flow_result(
                RealSingleAgentResult(
                    summary="single real AgentStep completed",
                    submitted_summary=latest_step.submission.summary or "",
                )
            )
        elif latest_step.status is StepStatus.COMPLETED and latest_step.result is not None:
            ctx.set_flow_result(
                RealSingleAgentResult(
                    summary="single real AgentStep completed without submission",
                    submitted_summary=latest_step.result.summary or "",
                )
            )
        return None


def _single_params_from_input(input_model: BaseFlowInput | None) -> RealSingleAgentParams:
    if not isinstance(input_model, RealSingleAgentInput):
        raise FlowStepValidationError("RealSingleAgentFlow has no RealSingleAgentInput")
    return RealSingleAgentParams(
        prompt=input_model.prompt,
        home_id=input_model.home_id,
        max_auto_continue_turns=input_model.max_auto_continue_turns,
    )


class RealDispatchParentParams(BaseModel):
    names_csv: str = "alpha,beta"
    child_flow_type: str = "real_logic_child"


class RealDispatchParentInput(BaseFlowInput):
    input_type: str = "real_dispatch_parent_input"
    names_csv: str
    child_flow_type: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Input: dispatch {self.names_csv} as {self.child_flow_type}"


class RealDispatchParentState(BaseFlowState):
    state_type: str = "real_dispatch_parent_state"
    waiting_dispatch_step_id: str | None = None


class RealDispatchParentResult(BaseFlowResult):
    result_type: str = "real_dispatch_parent_done"
    callback_summary: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Result: callback submitted {self.callback_summary}"


class RealDispatchParentFlow(BaseFlow):
    flow_type: ClassVar[str] = "real_dispatch_parent"
    Params: ClassVar[type[BaseModel]] = RealDispatchParentParams
    Input: ClassVar[type[BaseFlowInput]] = RealDispatchParentInput
    Result: ClassVar[type[BaseFlowResult]] = RealDispatchParentResult
    State: ClassVar[type[BaseFlowState]] = RealDispatchParentState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "RealDispatchParentFlow":
        params = ctx.params
        assert isinstance(params, RealDispatchParentParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=RealDispatchParentInput(names_csv=params.names_csv, child_flow_type=params.child_flow_type),
            state=RealDispatchParentState(),
        )

    def create_next_step(self, ctx: FlowContext) -> str | None:
        if not isinstance(self.input, RealDispatchParentInput):
            raise FlowStepValidationError("RealDispatchParentFlow has no input")
        if not self.step_ids:
            prompt = (
                "Call MCP tool ark_submit_child_flows exactly once with "
                f"names_csv='{self.input.names_csv}' and child_flow_type='{self.input.child_flow_type}'. "
                "Reply with exactly the returned tool result."
            )
            return ctx.create_step(
                AgentStep(
                    step_id=f"{self.flow_id}-initial-agent",
                    flow_id=self.flow_id,
                    scope_id=self.scope_id,
                    state=AgentStepState(
                        agent_role="planner",
                        agent_type=REAL_AGENT_TYPE,
                        home_id=SUBMIT_HOME_ID,
                        create_agent_if_missing=True,
                        bind_created_agent_to="flow",
                        variables={"prompt": prompt},
                        prompt_override=prompt,
                        max_auto_continue_turns=1,
                        agent_wait_timeout_s=DEFAULT_AGENT_TIMEOUT_S,
                    ),
                )
            )
        standard_step_id = create_standard_next_step_if_applicable(self, ctx)
        if standard_step_id is not None:
            return standard_step_id
        latest_step = ctx.tx.load_step_for_update(self.step_ids[-1])
        if isinstance(latest_step, AgentStep) and latest_step.submission is not None:
            ctx.set_flow_result(
                RealDispatchParentResult(
                    summary="dispatch parent completed after callback",
                    callback_summary=latest_step.submission.summary or "",
                )
            )
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


class RealLogicChildParams(BaseModel):
    name: str


class RealLogicChildInput(BaseFlowInput):
    input_type: str = "real_logic_child_input"
    name: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Input: logic child {self.name}"


class RealLogicChildState(BaseFlowState):
    state_type: str = "real_logic_child_state"


class RealLogicChildResult(BaseFlowResult):
    result_type: str = "real_logic_child_done"
    name: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Result: logic child {self.name} done"


class RealLogicChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "real_logic_child"
    Params: ClassVar[type[BaseModel]] = RealLogicChildParams
    Input: ClassVar[type[BaseFlowInput]] = RealLogicChildInput
    Result: ClassVar[type[BaseFlowResult]] = RealLogicChildResult
    State: ClassVar[type[BaseFlowState]] = RealLogicChildState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "RealLogicChildFlow":
        params = ctx.params
        assert isinstance(params, RealLogicChildParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=RealLogicChildInput(name=params.name),
            state=RealLogicChildState(),
        )

    def create_next_step(self, ctx: FlowContext) -> str | None:
        if not isinstance(self.input, RealLogicChildInput):
            raise FlowStepValidationError("RealLogicChildFlow has no input")
        ctx.set_flow_result(
            RealLogicChildResult(summary=f"logic child {self.input.name} done", name=self.input.name)
        )
        return None


class RealAgentChildParams(BaseModel):
    name: str


class RealAgentChildInput(BaseFlowInput):
    input_type: str = "real_agent_child_input"
    name: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Input: agent child {self.name}"


class RealAgentChildState(BaseFlowState):
    state_type: str = "real_agent_child_state"


class RealAgentChildResult(BaseFlowResult):
    result_type: str = "real_agent_child_done"
    name: str
    submitted_summary: str

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Result: agent child {self.name} submitted {self.submitted_summary}"


class RealAgentChildFlow(BaseFlow):
    flow_type: ClassVar[str] = "real_agent_child"
    Params: ClassVar[type[BaseModel]] = RealAgentChildParams
    Input: ClassVar[type[BaseFlowInput]] = RealAgentChildInput
    Result: ClassVar[type[BaseFlowResult]] = RealAgentChildResult
    State: ClassVar[type[BaseFlowState]] = RealAgentChildState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "RealAgentChildFlow":
        params = ctx.params
        assert isinstance(params, RealAgentChildParams)
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            input=RealAgentChildInput(name=params.name),
            state=RealAgentChildState(),
        )

    def create_next_step(self, ctx: FlowContext) -> str | None:
        if not isinstance(self.input, RealAgentChildInput):
            raise FlowStepValidationError("RealAgentChildFlow has no input")
        if not self.step_ids:
            prompt = (
                "Call MCP tool ark_submit_result exactly once with "
                f"summary='agent child {self.input.name} done'. Reply with exactly the returned tool result."
            )
            return ctx.create_step(
                AgentStep(
                    step_id=f"{self.flow_id}-agent-child",
                    flow_id=self.flow_id,
                    scope_id=self.scope_id,
                    state=AgentStepState(
                        agent_role="child_worker",
                        agent_type=REAL_AGENT_TYPE,
                        home_id=SUBMIT_HOME_ID,
                        create_agent_if_missing=True,
                        variables={"prompt": prompt},
                        prompt_override=prompt,
                        max_auto_continue_turns=1,
                        agent_wait_timeout_s=DEFAULT_AGENT_TIMEOUT_S,
                    ),
                )
            )
        latest_step = ctx.tx.load_step_for_update(self.step_ids[-1])
        if latest_step.status is StepStatus.COMPLETED and latest_step.submission is not None:
            ctx.set_flow_result(
                RealAgentChildResult(
                    summary=f"agent child {self.input.name} done",
                    name=self.input.name,
                    submitted_summary=latest_step.submission.summary or "",
                )
            )
        return None


class RealMarkerStepState(BaseStepState):
    state_type: str = "real_marker_step_state"
    marker: str = "marker"


class RealMarkerStepResult(BaseStepResult):
    result_type: str = "real_marker_step_done"

    def render_for_agent(self, ctx: RenderContext) -> str:
        return f"Result: {self.summary or 'marker done'}"


class RealMarkerStep(BaseStep):
    step_type: ClassVar[str] = "real_marker_step"
    State: ClassVar[type[BaseStepState]] = RealMarkerStepState
    Result: ClassVar[type[BaseStepResult]] = RealMarkerStepResult

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return ctx.complete_step(RealMarkerStepResult(summary="marker step completed"))


def build_real_flow_registries() -> tuple[FlowTypeRegistry, StepTypeRegistry]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(RealSingleAgentFlow)
    flow_registry.register(RealDispatchParentFlow)
    flow_registry.register(RealLogicChildFlow)
    flow_registry.register(RealAgentChildFlow)
    step_registry.register(AgentStep)
    step_registry.register(DispatchStep)
    step_registry.register(RealMarkerStep)
    return flow_registry, step_registry


@dataclass
class RealFlowRuntime:
    runtime_root: Path
    ark: ARKServices
    app: AppServices
    agent_service: AgentService
    flow_service: FlowService
    step_service: StepService
    schedule_service: RuntimeScheduleService
    snapshot_service: AgentSnapshotService
    submit_bridge: "RealFlowSubmitBridge"

    def close(self) -> None:
        self.submit_bridge.close()
        self.agent_service.close()


class RealFlowSubmitBridge:
    def __init__(self, *, ark: ARKServices, app: AppServices) -> None:
        self.ark = ark
        self.app = app
        self.host = "127.0.0.1"
        self.port = _free_tcp_port()
        self.url = f"http://{self.host}:{self.port}/mcp"
        self.call_log: list[dict[str, Any]] = []
        self.lock = threading.RLock()
        self.sleep_started = threading.Event()
        self._server: Any | None = None
        self._thread: threading.Thread | None = None

    def as_mcp_server_spec(self) -> McpServerSpec:
        return McpServerSpec(
            name="ark_flow_submit",
            url=self.url,
            required=True,
            env_http_headers={
                "X-Ark-Step-Id": "ARK_STEP_ID",
                "X-Ark-Flow-Id": "ARK_FLOW_ID",
                "X-Ark-Agent-Id": "ARK_AGENT_ID",
            },
        )

    def start(self) -> None:
        import uvicorn

        if FastMCP is None:
            pytest.skip("mcp is not installed")
        bridge = self
        mcp = FastMCP(
            "ark-flow-submit",
            host=self.host,
            port=self.port,
            streamable_http_path="/mcp",
            stateless_http=True,
            log_level="ERROR",
        )

        @mcp.tool()
        def ark_read_identity(ctx: Context) -> str:
            identity = bridge._identity_from_context(ctx)
            bridge._record_call("ark_read_identity", identity, {})
            return f"ARK_IDENTITY::{identity.step_id}::{identity.flow_id}::{identity.agent_id}"

        @mcp.tool()
        def ark_submit_result(summary: str, ctx: Context) -> str:
            identity = bridge._identity_from_context(ctx)
            submission = BaseSubmission(
                submission_id=f"sub_{uuid.uuid4().hex}",
                submission_type="result",
                tool_name="ark_submit_result",
                submitted_by_agent_id=identity.agent_id,
                summary=summary,
            )
            bridge._accept_submission(identity, submission)
            bridge._record_call("ark_submit_result", identity, {"summary": summary})
            return f"ARK_SUBMIT_ACCEPTED::{identity.step_id}::{submission.submission_id}"

        @mcp.tool()
        def ark_submit_child_flows(names_csv: str, ctx: Context, child_flow_type: str = "real_logic_child") -> str:
            identity = bridge._identity_from_context(ctx)
            step = bridge._flow_service().get_step(identity.step_id)
            names = [name.strip() for name in names_csv.split(",") if name.strip()]
            submission = ChildFlowDispatchSubmission(
                submission_id=f"dispatch_{uuid.uuid4().hex}",
                tool_name="ark_submit_child_flows",
                submitted_by_agent_id=identity.agent_id,
                summary=f"dispatch {','.join(names)}",
                requests=[
                    FlowRequest(flow_type=child_flow_type, scope_id=step.scope_id, params={"name": name})
                    for name in names
                ],
            )
            bridge._accept_submission(identity, submission)
            bridge._record_call(
                "ark_submit_child_flows",
                identity,
                {"names_csv": names_csv, "child_flow_type": child_flow_type},
            )
            return f"ARK_DISPATCH_ACCEPTED::{identity.step_id}::{submission.submission_id}"

        @mcp.tool()
        def ark_sleep_then_submit(seconds: float, summary: str, ctx: Context) -> str:
            identity = bridge._identity_from_context(ctx)
            bridge.sleep_started.set()
            time.sleep(float(seconds))
            submission = BaseSubmission(
                submission_id=f"sleep_{uuid.uuid4().hex}",
                submission_type="result",
                tool_name="ark_sleep_then_submit",
                submitted_by_agent_id=identity.agent_id,
                summary=summary,
            )
            bridge._accept_submission(identity, submission)
            bridge._record_call("ark_sleep_then_submit", identity, {"seconds": seconds, "summary": summary})
            return f"ARK_SLEEP_SUBMIT_ACCEPTED::{identity.step_id}::{submission.submission_id}"

        self._server = uvicorn.Server(
            uvicorn.Config(
                mcp.streamable_http_app(),
                host=self.host,
                port=self.port,
                log_level="error",
            )
        )
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._wait_until_ready()

    def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.05)
        raise TimeoutError(f"MCP bridge did not start on {self.host}:{self.port}")

    def _identity_from_context(self, ctx: Any) -> "_BridgeIdentity":
        headers = _headers_from_context(ctx)
        step_id = _header_value(headers, "x-ark-step-id")
        flow_id = _header_value(headers, "x-ark-flow-id")
        agent_id = _header_value(headers, "x-ark-agent-id")
        if not step_id or not flow_id or not agent_id:
            raise FlowStepValidationError(f"missing ARK identity headers: {headers!r}")
        return _BridgeIdentity(step_id=step_id, flow_id=flow_id, agent_id=agent_id, headers=headers)

    def _accept_submission(self, identity: "_BridgeIdentity", submission: BaseSubmission) -> None:
        step = self._flow_service().get_step(identity.step_id)
        ctx = StepRunContext(
            ark=self.ark,
            app=self.app,
            step_id=identity.step_id,
            flow_id=identity.flow_id,
            scope_id=step.scope_id,
        )
        ctx.accept_step_submission(submission)

    def _record_call(self, tool_name: str, identity: "_BridgeIdentity", args: dict[str, Any]) -> None:
        with self.lock:
            self.call_log.append(
                {
                    "tool_name": tool_name,
                    "step_id": identity.step_id,
                    "flow_id": identity.flow_id,
                    "agent_id": identity.agent_id,
                    "args": args,
                    "headers": identity.headers,
                }
            )

    def _flow_service(self) -> FlowService:
        flow_service = self.ark.flow_service
        if not isinstance(flow_service, FlowService):
            raise FlowStepValidationError("bridge has no FlowService")
        return flow_service


@dataclass(frozen=True)
class _BridgeIdentity:
    step_id: str
    flow_id: str
    agent_id: str
    headers: dict[str, str]


def make_real_flow_runtime(
    tmp_path: Path,
    *,
    max_concurrent_flow_advances: int = 1,
    max_concurrent_steps: int = 1,
) -> RealFlowRuntime:
    ensure_real_codex_enabled()
    runtime_root = tmp_path / "project" / ".agent_runtime"
    ark = ARKServices()
    app = AppServices()
    ark.pause_controller = RuntimePauseController()
    flow_registry, step_registry = build_real_flow_registries()
    agent_types = AgentTypeRegistry()
    agent_types.register(RealFlowSubmitAgentType())
    provider = CodexProvider(
        runtime_root=runtime_root,
        codex_bin=os.environ.get("ARK_CODEX_BIN") or shutil.which("codex"),
        sdk_python_root=sdk_python_root(),
        model=os.environ.get("ARK_REAL_CODEX_MODEL"),
    )
    agent_service = AgentService(
        runtime_root,
        agent_types=agent_types,
        providers={"codex": provider},
        ark_services=ark,
        app_services=app,
    )
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_registry,
        step_registry=step_registry,
        ark_services=ark,
        app_services=app,
    )
    step_service = StepService(
        runtime_root,
        step_registry=step_registry,
        ark_services=ark,
        app_services=app,
    )
    schedule_service = RuntimeScheduleService(
        ark_services=ark,
        app_services=app,
        max_concurrent_flow_advances=max_concurrent_flow_advances,
        max_concurrent_steps=max_concurrent_steps,
    )
    snapshot_service = AgentSnapshotService(
        runtime_root,
        store=AgentStoreService(runtime_root),
        agent_service=agent_service,
        ark_services=ark,
        app_services=app,
    )
    submit_bridge = RealFlowSubmitBridge(ark=ark, app=app)
    submit_bridge.start()
    config_dir = codex_config_dir()
    agent_service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id=SUBMIT_HOME_ID,
            base_config_path=config_dir / "config.toml",
            auth_json_path=config_dir / "auth.json",
            mcp_servers=[submit_bridge.as_mcp_server_spec()],
        ),
        initialize_provider_home=True,
    )
    agent_service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id=NO_TOOL_HOME_ID,
            base_config_path=config_dir / "config.toml",
            auth_json_path=config_dir / "auth.json",
        ),
        initialize_provider_home=True,
    )
    return RealFlowRuntime(
        runtime_root=runtime_root,
        ark=ark,
        app=app,
        agent_service=agent_service,
        flow_service=flow_service,
        step_service=step_service,
        schedule_service=schedule_service,
        snapshot_service=snapshot_service,
        submit_bridge=submit_bridge,
    )


def run_scheduler_until_idle(runtime: RealFlowRuntime, *, max_ticks: int = 50) -> None:
    idle_ticks = 0
    for _ in range(max_ticks):
        tick = runtime.schedule_service.schedule_ready()
        for step_id in tick.started_step_ids:
            runtime.step_service.wait_step(step_id, timeout_s=DEFAULT_AGENT_TIMEOUT_S)
        if tick.advanced_flow_ids or tick.started_step_ids:
            idle_ticks = 0
            continue
        idle_ticks += 1
        if idle_ticks >= 2:
            return
    raise TimeoutError("scheduler did not become idle")


def run_flow_until_terminal(runtime: RealFlowRuntime, flow_id: str, *, max_ticks: int = 100) -> BaseFlow:
    for _ in range(max_ticks):
        flow = runtime.flow_service.get_flow(flow_id)
        if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
            return flow
        tick = runtime.schedule_service.schedule_ready()
        for step_id in tick.started_step_ids:
            runtime.step_service.wait_step(step_id, timeout_s=DEFAULT_AGENT_TIMEOUT_S)
    flow = runtime.flow_service.get_flow(flow_id)
    raise TimeoutError(f"flow did not become terminal: {flow_id} status={flow.status}")


def run_step_in_thread(runtime: RealFlowRuntime, step_id: str) -> threading.Thread:
    thread = threading.Thread(target=runtime.step_service.run_step, args=(step_id,), daemon=True)
    thread.start()
    return thread


def ensure_real_codex_enabled() -> None:
    if os.environ.get("ARK_RUN_REAL_CODEX") != "1":
        pytest.skip("set ARK_RUN_REAL_CODEX=1 to run real Codex SDK tests")
    if shutil.which("codex") is None and not os.environ.get("ARK_CODEX_BIN"):
        pytest.skip("codex binary is not available")
    root = sdk_python_root()
    if importlib.util.find_spec("openai_codex") is None and root is None:
        pytest.skip("openai_codex is not installed and no local SDK root is available")


def sdk_python_root() -> Path | None:
    value = os.environ.get("ARK_CODEX_SDK_PYTHON_ROOT")
    root = Path(value) if value else Path("/root/code/tools/codex/sdk/python")
    if not root.exists():
        return None
    src = root / "src"
    if not (src / "openai_codex").exists():
        pytest.skip(f"invalid Codex SDK Python root: {root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def codex_config_dir() -> Path:
    path = Path(os.environ.get("ARK_CODEX_CONFIG_DIR", "data/configs/codex"))
    if not path.exists():
        pytest.skip(f"Codex config dir does not exist: {path}")
    if not (path / "config.toml").exists() or not (path / "auth.json").exists():
        pytest.skip(f"Codex config dir must contain config.toml and auth.json: {path}")
    return path


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _headers_from_context(ctx: Any) -> dict[str, str]:
    request_context = getattr(ctx, "request_context", None)
    request = getattr(request_context, "request", None)
    raw_headers = getattr(request, "headers", {}) or {}
    try:
        items = raw_headers.items()
    except AttributeError:
        items = []
    return {str(key).lower(): str(value) for key, value in items}


def _header_value(headers: dict[str, str], name: str) -> str | None:
    return headers.get(name.lower())
