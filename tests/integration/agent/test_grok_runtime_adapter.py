from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.agent.homes import HomeService
from agent_runtime_kit.agent.provider_contracts import (
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
    ProviderControlAction,
    ProviderControlRequest,
    ProviderHomeSpec,
    ProviderRegistry,
    ProviderRunOptions,
    ProviderRunRequest,
    ProviderRunState,
)
from agent_runtime_kit.agent.providers import GrokHomeOptions, build_grok_provider_bundle
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.flow import (
    AgentStep,
    AgentStepState,
    BaseFlow,
    BaseFlowState,
    FlowBuildContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    FlowTypeRegistry,
    StepService,
    StepStatus,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


FIXTURE = Path(__file__).parents[2] / "fixtures" / "grok_acp_fixture.py"


def _fixture_binary(tmp_path: Path) -> Path:
    wrapper = tmp_path / "grok-fixture"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {FIXTURE} \"$@\"\n", encoding="utf-8")
    wrapper.chmod(0o755)
    return wrapper


def _setup(tmp_path: Path, *, tools=None):  # noqa: ANN001, ANN202
    binary = _fixture_binary(tmp_path)
    runtime_root = tmp_path / "runtime"
    bundle = build_grok_provider_bundle(runtime_root=runtime_root, binary_path=binary)
    homes = HomeService(runtime_root, renderers={"grok": bundle.home_renderer})
    homes.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id="demo",
            provider_options=GrokHomeOptions(auth_json_path=None, tools=tools),
        )
    )
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    context = homes.build_execution_context("grok", "demo", workdir=str(workdir))
    return bundle, context, workdir


def _request(context, workdir: Path, *, prompt: str, session=None, max_turns=None):  # noqa: ANN001, ANN202
    return ProviderRunRequest(
        agent_id="agent",
        scope_id="scope",
        agent_type="worker",
        provider_type="grok",
        home_id="demo",
        prompt=prompt,
        session_locator=session,
        developer_instructions="RUN_DEVELOPER_MARKER",
        workdir=str(workdir),
        run_options=ProviderRunOptions(timeout_s=10, max_turns=max_turns),
        execution_context=context,
    )


def test_grok_runtime_fresh_resume_events_usage_and_artifact_restore(tmp_path: Path) -> None:
    bundle, context, workdir = _setup(tmp_path)
    first = bundle.runtime.start(_request(context, workdir, prompt="first")).wait_terminal(20)
    assert first.status is ProviderRunState.COMPLETED
    assert first.final_text == "ARK_GROK_DONE"
    assert first.turn_usage is not None
    assert first.turn_usage.token_usage.total_tokens == 10
    assert first.turn_usage.request_count == 3
    assert first.turn_usage.requests == ()
    assert first.request_usages == ()
    assert first.turn_usage.reported_costs == ()
    assert first.provider_payload is not None
    assert "costUsdTicks" in str(first.provider_payload.sanitized_data)
    events = bundle.runtime._unstable_sessions  # ensure successful cleanup was recorded as stable
    assert first.session_locator.session_id not in events

    assert bundle.artifacts is not None
    snapshot_root = tmp_path / "snapshot"
    captured = bundle.artifacts.capture(
        ArtifactCaptureRequest(session=first.session_locator, snapshot_root=str(snapshot_root))
    )
    session_ref = Path(str(first.artifact_locator.native_primary_ref))
    history = bundle.runtime.runtime_root / session_ref / "chat_history.jsonl"
    original = history.read_bytes()

    second = bundle.runtime.resume(
        _request(context, workdir, prompt="second", session=first.session_locator)
    ).wait_terminal(20)
    assert second.session_locator.session_id == first.session_locator.session_id
    assert history.read_bytes() != original
    bundle.artifacts.prepare_restore(
        ArtifactRestoreRequest(manifest=captured.manifest, snapshot_root=str(snapshot_root))
    )
    bundle.artifacts.restore(
        ArtifactRestoreRequest(manifest=captured.manifest, snapshot_root=str(snapshot_root))
    )
    assert history.read_bytes() == original
    third = bundle.runtime.resume(
        _request(context, workdir, prompt="third", session=first.session_locator)
    ).wait_terminal(20)
    assert third.status is ProviderRunState.COMPLETED


def test_grok_runtime_refusal_is_failed_and_max_turns_is_explicitly_unsupported(tmp_path: Path) -> None:
    bundle, context, workdir = _setup(tmp_path)
    failed = bundle.runtime.start(_request(context, workdir, prompt="refuse")).wait_terminal(20)
    assert failed.status is ProviderRunState.FAILED
    assert failed.error is not None and failed.error.code == "refusal"
    with pytest.raises(ValueError, match="max_turns"):
        bundle.runtime.start(_request(context, workdir, prompt="first", max_turns=2))


def test_grok_runtime_interrupt_reaps_entire_process_group(tmp_path: Path) -> None:
    bundle, context, workdir = _setup(
        tmp_path,
        tools=("read_file", "list_dir", "grep", "run_terminal_cmd"),
    )
    handle = bundle.runtime.start(_request(context, workdir, prompt="cancel"))
    deadline = time.monotonic() + 5
    while handle.session_locator() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    control = handle.interrupt(5)
    result = handle.wait_terminal(10)
    assert control.accepted and control.terminal_confirmed
    assert result.status is ProviderRunState.INTERRUPTED
    assert handle.cleanup_confirmed


def test_grok_runtime_rejects_starting_max_turn_before_spawning(tmp_path: Path) -> None:
    bundle, context, workdir = _setup(tmp_path)
    with pytest.raises(ValueError, match="max_turns"):
        bundle.runtime.start(_request(context, workdir, prompt="never", max_turns=1))


def test_grok_runtime_cancel_during_required_mcp_startup_never_submits_prompt(tmp_path: Path) -> None:
    bundle, context, workdir = _setup(tmp_path)
    assert isinstance(context.runtime_payload, dict)
    context.runtime_payload["required_mcp_server_names"] = ["slow"]
    context.process_environment["GROK_FIXTURE_DOCTOR_DELAY"] = "5"
    handle = bundle.runtime.start(_request(context, workdir, prompt="never submit"))
    control = handle.control(
        ProviderControlRequest(
            action=ProviderControlAction.CANCEL,
            requested_at="2026-09-13T00:00:00Z",
            run_id=handle.run_id,
            options={"timeout_s": 2},
        )
    )
    assert control.accepted and control.terminal_confirmed
    with pytest.raises(Exception, match="cancelled"):
        handle.wait_terminal(2)
    assert handle.session_locator() is None


def test_grok_control_never_confirms_terminal_when_process_group_cleanup_failed(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    bundle, context, workdir = _setup(tmp_path)

    def fail_cleanup(self, **_kwargs):  # noqa: ANN001, ANN202
        raise RuntimeError("fixture cleanup uncertainty")

    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.grok_runtime.GrokAcpProcess.close_process_group",
        fail_cleanup,
    )
    handle = bundle.runtime.start(_request(context, workdir, prompt="first"))
    with pytest.raises(RuntimeError, match="cleanup uncertainty"):
        handle.wait_terminal(10)
    assert handle.session_locator() is not None

    control = handle.interrupt()
    assert not control.terminal_confirmed
    assert control.resulting_state is None
    assert "without confirmed" in str(control.reason)
    closed = bundle.runtime.close_session(handle.session_locator())
    assert not closed.terminal_confirmed
    assert "unconfirmed" in str(closed.reason)


def test_grok_doctor_cleanup_failure_without_acp_transport_stays_unconfirmed(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    bundle, context, workdir = _setup(tmp_path)
    assert isinstance(context.runtime_payload, dict)
    context.runtime_payload["required_mcp_server_names"] = ["required"]

    def fail_doctor(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        from agent_runtime_kit.agent.providers.grok_runtime import GrokProcessCleanupError

        raise GrokProcessCleanupError("doctor cleanup uncertainty")

    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.grok_runtime._verify_required_mcp",
        fail_doctor,
    )
    handle = bundle.runtime.start(_request(context, workdir, prompt="never"))
    with pytest.raises(RuntimeError, match="doctor cleanup uncertainty"):
        handle.wait_terminal(5)
    control = handle.control(
        ProviderControlRequest(
            action=ProviderControlAction.STEER,
            requested_at="2026-09-13T00:00:00Z",
            run_id=handle.run_id,
            content="unused",
        )
    )
    assert not control.terminal_confirmed
    assert control.resulting_state is None


class _GrokStepAgent(AgentType):
    agent_type = "grok-step-agent"
    provider_type = "grok"
    default_home_id = "grok-step-agent"
    start_prompt_template = "complete fake Grok step"


class _GrokFlowParams(BaseModel):
    pass


class _GrokFlowState(BaseFlowState):
    state_type: str = "grok_flow_state"


class _GrokFlow(BaseFlow):
    flow_type: ClassVar[str] = "grok_flow"
    Params: ClassVar[type[BaseModel]] = _GrokFlowParams
    State: ClassVar[type[BaseFlowState]] = _GrokFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "_GrokFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=_GrokFlowState())


def test_grok_fake_provider_runs_through_agent_step(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    binary = _fixture_binary(tmp_path)
    digest = __import__("hashlib").sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr("agent_runtime_kit.agent.providers.grok_home.GROK_BINARY_SHA256", digest)
    runtime_root = tmp_path / "runtime"
    bundle = build_grok_provider_bundle(runtime_root=runtime_root, binary_path=binary)
    agent_types = AgentTypeRegistry()
    agent_types.register(_GrokStepAgent())
    flow_types = FlowTypeRegistry()
    flow_types.register(_GrokFlow)
    step_types = StepTypeRegistry()
    step_types.register(AgentStep)
    ark = ARKServices()
    app = AppServices()
    flow_service = FlowService(
        runtime_root,
        flow_registry=flow_types,
        step_registry=step_types,
        ark_services=ark,
        app_services=app,
    )
    step_service = StepService(runtime_root, step_registry=step_types, ark_services=ark, app_services=app)
    agent_service = AgentService(
        runtime_root,
        agent_types=agent_types,
        provider_registry=ProviderRegistry((bundle,)),
        ark_services=ark,
        app_services=app,
    )
    agent_service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id="grok-step-agent",
            provider_options=GrokHomeOptions(auth_json_path=None),
        )
    )
    workdir = tmp_path / "workspace"
    workdir.mkdir()
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="grok_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: setattr(flow, "status", FlowStatus.RUNNING),
    )
    step = AgentStep(
        step_id="grok-agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="worker",
            agent_type="grok-step-agent",
            provider_type="grok",
            home_id="grok-step-agent",
            create_agent_if_missing=True,
            prompt_override="complete fake Grok step",
            workdir_override=str(workdir),
            require_submission=False,
            max_auto_continue_turns=0,
        ),
    )
    flow_service.store.create_step(step)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (flow.step_ids.append(step.step_id), setattr(flow, "current_step_id", step.step_id)),
    )

    step_service.run_step(step.step_id)

    persisted = flow_service.get_step(step.step_id)
    assert persisted.status is StepStatus.COMPLETED
    assert persisted.result is not None and persisted.result.outcome == "incomplete"
