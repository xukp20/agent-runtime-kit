from pathlib import Path
from contextlib import contextmanager
from threading import Barrier, Event, Thread, Timer
from typing import ClassVar

import pytest
from pydantic import BaseModel

from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.store import AgentStoreService
from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
    BaseStepError,
    BaseStepResult,
    BaseStepState,
    FlowBuildContext,
    FlowContext,
    FlowRequest,
    FlowService,
    FlowStatus,
    RuntimeScheduleService,
    StepRunContext,
    StepService,
    StepStatus,
    StepTerminalReceipt,
    StepTypeRegistry,
    FlowTypeRegistry,
    AgentRoleBindings,
    AgentStep,
    AgentStepState,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class SnapshotFlowParams(BaseModel):
    pass


class SnapshotFlowState(BaseFlowState):
    state_type: str = "snapshot_flow_state"


class SnapshotStepState(BaseStepState):
    state_type: str = "snapshot_step_state"


class SnapshotStep(BaseStep):
    step_type: ClassVar[str] = "snapshot_step"
    State: ClassVar[type[BaseStepState]] = SnapshotStepState

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        return ctx.complete_step(BaseStepResult(result_type="snapshot_step_done", summary="step done"))


class BlockingSnapshotStep(BaseStep):
    step_type: ClassVar[str] = "blocking_snapshot_step"
    State: ClassVar[type[BaseStepState]] = SnapshotStepState
    release_event: ClassVar[Event] = Event()
    started_event: ClassVar[Event] = Event()

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        BlockingSnapshotStep.started_event.set()
        BlockingSnapshotStep.release_event.wait(timeout=5)
        return ctx.complete_step(BaseStepResult(result_type="blocking_snapshot_step_done", summary="step done"))


class SnapshotAgentStep(AgentStep):
    step_type: ClassVar[str] = "snapshot_agent_step"


class SnapshotFlow(BaseFlow):
    flow_type: ClassVar[str] = "snapshot_flow"
    Params: ClassVar[type[BaseModel]] = SnapshotFlowParams
    State: ClassVar[type[BaseFlowState]] = SnapshotFlowState

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "SnapshotFlow":
        return cls(flow_id=ctx.flow_id, scope_id=ctx.scope_id, state=SnapshotFlowState())

    def create_next_step(self, ctx: FlowContext) -> str | None:
        if not self.step_ids:
            return ctx.create_step(
                SnapshotStep(
                    step_id=f"{self.flow_id}-snapshot-step",
                    flow_id=self.flow_id,
                    scope_id=self.scope_id,
                    state=SnapshotStepState(),
                )
            )
        ctx.set_flow_result(BaseFlowResult(result_type="snapshot_flow_done", summary="flow done"))
        return None


def make_services(
    runtime_root: Path,
) -> tuple[FlowService, StepService, RuntimeScheduleService, AgentSnapshotService, ARKServices]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(SnapshotFlow)
    step_registry.register(SnapshotStep)
    step_registry.register(BlockingSnapshotStep)
    step_registry.register(SnapshotAgentStep)
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
    snapshot_service = AgentSnapshotService(
        runtime_root,
        store=AgentStoreService(runtime_root),
        ark_services=ark,
        app_services=AppServices(),
    )
    return flow_service, step_service, scheduler, snapshot_service, ark


class RecoveryAgents:
    def __init__(self, source: Agent) -> None:
        self.agents = {source.agent_id: source}

    @contextmanager
    def hold_agent_boundary(self):
        yield

    def get_agent(self, agent_id):  # noqa: ANN001, ANN201
        return self.agents[agent_id]

    def list_running_agents(self, _scope_id=None):  # noqa: ANN001, ANN201
        return []

    def query_turn(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
        return None

    def create_agent(self, scope_id, agent_type, provider_type=None, home_id=None):  # noqa: ANN001, ANN201
        created = Agent(f"fresh-{len(self.agents)}", scope_id, agent_type, provider_type, home_id)
        self.agents[created.agent_id] = created
        return created


def seed_recoverable_agent_step(
    flow_service: FlowService,
    step_service: StepService,
    ark: ARKServices,
    *,
    scope_id: str = "scope",
) -> str:
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="snapshot_flow", scope_id=scope_id, params={}),
        enqueue=False,
    )
    source = Agent("source-agent", scope_id, "ReviewerAgent", "codex", "ReviewerAgent")
    step = SnapshotAgentStep(
        step_id="suspended-agent-step",
        flow_id=flow_id,
        scope_id=scope_id,
        status=StepStatus.SUSPENDED,
        state=AgentStepState(
            agent_role="reviewer",
            agent_type="ReviewerAgent",
            provider_type="codex",
            home_id="ReviewerAgent",
        ),
        error=BaseStepError(error_type="agent_provider_turn_failed", message="offline"),
        agent_bindings=AgentRoleBindings(by_role={"reviewer": source.agent_id}),
    )
    step_service.create_step(step, enqueue=False)

    def attach(flow):  # noqa: ANN001
        flow.status = FlowStatus.RUNNING
        flow.step_ids.append(step.step_id)
        flow.current_step_id = step.step_id
        flow.agent_bindings.by_role["reviewer"] = source.agent_id

    flow_service.store.update_flow_record(flow_id, attach)
    ark.agent_service = RecoveryAgents(source)
    return step.step_id


def test_scope_snapshot_blocks_when_scope_has_running_step(tmp_path: Path) -> None:
    flow_service, step_service, _, snapshot_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_id = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope", params={}), enqueue=False)
    step = SnapshotStep(
        step_id="running-step",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.RUNNING,
        state=SnapshotStepState(),
    )
    step_service.create_step(step, enqueue=False)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (flow.step_ids.append(step.step_id), setattr(flow, "current_step_id", step.step_id)),
    )

    result = snapshot_service.create_scope_snapshot("scope")

    assert result.status == "blocked"
    assert result.running_step_ids == ("running-step",)


def test_scope_snapshot_restore_preserves_suspended_step(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    flow_service, step_service, _, snapshot_service, _ = make_services(runtime_root)
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="snapshot_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    step = SnapshotStep(
        step_id="suspended-step",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.SUSPENDED,
        state=SnapshotStepState(),
        error=BaseStepError(error_type="agent_provider_failure", message="offline"),
    )
    step_service.create_step(step, enqueue=False)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (
            setattr(flow, "status", FlowStatus.RUNNING),
            flow.step_ids.append(step.step_id),
            setattr(flow, "current_step_id", step.step_id),
        ),
    )

    snapshot = snapshot_service.create_scope_snapshot("scope")
    assert snapshot.status == "created"
    flow_service.store.update_step_record(step.step_id, lambda target: setattr(target, "status", StepStatus.FAILED))
    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    assert flow_service.get_step(step.step_id).status is StepStatus.SUSPENDED
    assert flow_service.get_flow(flow_id).current_step_id == step.step_id


def test_scope_snapshot_restore_preserves_agent_step_operator_instruction(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    flow_service, step_service, _, snapshot_service, _ = make_services(runtime_root)
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="snapshot_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    step = SnapshotAgentStep(
        step_id="operator-agent-step",
        flow_id=flow_id,
        scope_id="scope",
        state=AgentStepState(
            agent_role="reviewer",
            operator_instruction="Inspect the current snapshot truth.",
        ),
    )
    step_service.create_step(step, enqueue=False)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (
            setattr(flow, "status", FlowStatus.RUNNING),
            flow.step_ids.append(step.step_id),
            setattr(flow, "current_step_id", step.step_id),
        ),
    )

    snapshot = snapshot_service.create_scope_snapshot("scope")
    assert snapshot.status == "created"
    flow_service.store.update_step_record(
        step.step_id,
        lambda target: setattr(target.state, "operator_instruction", None),
    )
    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    assert (
        flow_service.get_step(step.step_id).state.operator_instruction
        == "Inspect the current snapshot truth."
    )


def test_scope_snapshot_restore_then_fresh_resume_suspended_agent_step(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    flow_service, step_service, _, snapshot_service, ark = make_services(runtime_root)
    flow_id = flow_service.start_flow(
        FlowRequest(flow_type="snapshot_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    source = Agent("source-agent", "scope", "ReviewerAgent", "codex", "ReviewerAgent")
    step = SnapshotAgentStep(
        step_id="suspended-agent-step",
        flow_id=flow_id,
        scope_id="scope",
        status=StepStatus.SUSPENDED,
        state=AgentStepState(
            agent_role="reviewer",
            agent_type="ReviewerAgent",
            provider_type="codex",
            home_id="ReviewerAgent",
        ),
        error=BaseStepError(error_type="agent_provider_turn_failed", message="offline"),
        agent_bindings=AgentRoleBindings(by_role={"reviewer": source.agent_id}),
    )
    step_service.create_step(step, enqueue=False)

    def attach(flow):  # noqa: ANN001
        flow.status = FlowStatus.RUNNING
        flow.step_ids.append(step.step_id)
        flow.current_step_id = step.step_id
        flow.agent_bindings.by_role["reviewer"] = source.agent_id

    flow_service.store.update_flow_record(flow_id, attach)
    snapshot = snapshot_service.create_scope_snapshot("scope")
    assert snapshot.snapshot_id is not None
    flow_service.store.update_step_record(step.step_id, lambda target: setattr(target, "status", StepStatus.FAILED))
    snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    class RecoveryAgents:
        def __init__(self) -> None:
            self.agents = {source.agent_id: source}

        @contextmanager
        def hold_agent_boundary(self):
            yield

        def get_agent(self, agent_id):  # noqa: ANN001, ANN201
            return self.agents[agent_id]

        def list_running_agents(self, _scope_id=None):  # noqa: ANN001, ANN201
            return []

        def query_turn(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN201
            return None

        def create_agent(self, scope_id, agent_type, provider_type=None, home_id=None):  # noqa: ANN001, ANN201
            created = Agent("fresh-agent", scope_id, agent_type, provider_type, home_id)
            self.agents[created.agent_id] = created
            return created

    ark.agent_service = RecoveryAgents()
    preview = flow_service.inspect_agent_step_recovery(step.step_id)
    receipt = flow_service.recover_agent_step(
        step_id=step.step_id,
        expected_status=StepStatus.SUSPENDED,
        expected_recovery_token=preview.recovery_token,
        action="resume_suspended",
        agent_mode="fresh",
    )

    assert receipt.replacement_agent_id == "fresh-agent"
    assert flow_service.get_step(step.step_id).status is StepStatus.SUSPENDED
    assert flow_service.get_step(receipt.replacement_step_id).status is StepStatus.CREATED
    assert flow_service.get_flow(flow_id).current_step_id == receipt.replacement_step_id


def test_scope_restore_rebuilds_flow_step_indexes_queue_and_leaves_paused(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    flow_service, step_service, scheduler, snapshot_service, ark = make_services(runtime_root)
    flow_id = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope", params={}), enqueue=False)
    step_id = flow_service.advance_flow(flow_id)
    assert step_id is not None

    snapshot = snapshot_service.create_scope_snapshot("scope")
    assert snapshot.status == "created"
    assert snapshot.snapshot_id is not None

    step_service.run_step(step_id)
    flow_service.store.update_flow_record(
        flow_id,
        lambda flow: (setattr(flow, "status", FlowStatus.COMPLETED), setattr(flow, "result", BaseFlowResult(result_type="mutated"))),
    )

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    assert ark.pause_controller is not None
    assert ark.pause_controller.is_paused("scope") is True
    restored_flow = flow_service.get_flow(flow_id)
    restored_step = step_service.wait_step(step_id)
    assert restored_flow.status is FlowStatus.CREATED
    assert restored_flow.current_step_id == step_id
    assert restored_step.status is StepStatus.CREATED

    paused_tick = scheduler.schedule_ready()
    assert paused_tick.started_step_ids == []
    assert paused_tick.advanced_flow_ids == []

    ark.pause_controller.resume("scope")
    resumed_tick = scheduler.schedule_ready()
    assert resumed_tick.started_step_ids == [step_id]
    assert step_service.wait_step(step_id).status is StepStatus.COMPLETED


def test_scope_snapshot_under_global_pause_does_not_leak_direct_scope_pause(tmp_path: Path) -> None:
    _, _, _, snapshot_service, ark = make_services(tmp_path / ".agent_runtime")
    assert ark.pause_controller is not None
    ark.pause_controller.pause(None)

    result = snapshot_service.create_scope_snapshot("scope")

    assert result.status == "created"
    assert ark.pause_controller.is_paused("scope") is True
    assert ark.pause_controller.is_scope_directly_paused("scope") is False
    ark.pause_controller.resume(None)
    assert ark.pause_controller.is_paused("scope") is False


def test_scope_restore_under_global_pause_does_not_leak_direct_scope_pause_when_unpaused(tmp_path: Path) -> None:
    _, _, _, snapshot_service, ark = make_services(tmp_path / ".agent_runtime")
    snapshot = snapshot_service.create_scope_snapshot("scope")
    assert snapshot.snapshot_id is not None
    assert ark.pause_controller is not None
    ark.pause_controller.pause(None)

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id, leave_paused=False)

    assert restored.status == "created"
    assert ark.pause_controller.is_paused("scope") is True
    assert ark.pause_controller.is_scope_directly_paused("scope") is False
    ark.pause_controller.resume(None)
    assert ark.pause_controller.is_paused("scope") is False


def test_runtime_snapshot_blocked_scopes_use_final_pending_steps(tmp_path: Path) -> None:
    BlockingSnapshotStep.release_event = Event()
    BlockingSnapshotStep.started_event = Event()
    flow_service, step_service, _, snapshot_service, _ = make_services(tmp_path / ".agent_runtime")
    flow_a = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope-a", params={}), enqueue=False)
    flow_b = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope-b", params={}), enqueue=False)
    active_step = BlockingSnapshotStep(
        step_id="active-step",
        flow_id=flow_a,
        scope_id="scope-a",
        state=SnapshotStepState(),
    )
    stale_step = SnapshotStep(
        step_id="stale-running-step",
        flow_id=flow_b,
        scope_id="scope-b",
        status=StepStatus.RUNNING,
        state=SnapshotStepState(),
    )
    step_service.create_step(active_step, enqueue=False)
    step_service.create_step(stale_step, enqueue=False)
    flow_service.store.update_flow_record(
        flow_a,
        lambda flow: (flow.step_ids.append(active_step.step_id), setattr(flow, "current_step_id", active_step.step_id)),
    )
    flow_service.store.update_flow_record(
        flow_b,
        lambda flow: (flow.step_ids.append(stale_step.step_id), setattr(flow, "current_step_id", stale_step.step_id)),
    )
    step_service.start_step(active_step.step_id)
    assert BlockingSnapshotStep.started_event.wait(timeout=2)
    release_timer = Timer(0.05, BlockingSnapshotStep.release_event.set)
    release_timer.start()

    result = snapshot_service.create_runtime_snapshot_synchronized(timeout_s=1)
    release_timer.cancel()

    assert result.status == "blocked"
    assert result.blocked_scope_ids == ("scope-b",)
    assert result.running_step_ids == ("stale-running-step",)
    assert step_service.wait_step(active_step.step_id, timeout_s=2).status is StepStatus.COMPLETED


def test_selective_runtime_snapshot_restore_rebuilds_queue_and_reuses_other_scope(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    flow_service, step_service, scheduler, snapshot_service, ark = make_services(runtime_root)
    flow_a = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope-a", params={}), enqueue=False)
    flow_b = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope-b", params={}), enqueue=False)
    step_a = flow_service.advance_flow(flow_a)
    assert step_a is not None
    initial_b = snapshot_service.create_scope_snapshot("scope-b")
    assert initial_b.snapshot_id is not None

    runtime_snapshot = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a", "scope-b"],
    )

    assert runtime_snapshot.status == "created"
    assert runtime_snapshot.snapshot_id is not None
    assert runtime_snapshot.scope_snapshot_ids["scope-b"] == initial_b.snapshot_id
    step_service.run_step(step_a)
    flow_service.store.update_flow_record(
        flow_a,
        lambda flow: (setattr(flow, "status", FlowStatus.COMPLETED), setattr(flow, "result", BaseFlowResult(result_type="mutated"))),
    )

    restored = snapshot_service.restore_runtime_snapshot(runtime_snapshot.snapshot_id)

    assert restored.status == "created"
    assert ark.pause_controller is not None
    assert ark.pause_controller.is_paused(None) is True
    assert flow_service.get_flow(flow_a).status is FlowStatus.CREATED
    assert step_service.wait_step(step_a).status is StepStatus.CREATED

    ark.pause_controller.resume(None)
    tick = scheduler.schedule_ready()
    assert tick.started_step_ids == [step_a]
    assert step_service.wait_step(step_a).status is StepStatus.COMPLETED
    assert flow_service.get_flow(flow_b).status is FlowStatus.CREATED


def test_selective_runtime_snapshot_does_not_block_on_unrefreshed_running_step(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    flow_service, step_service, _, snapshot_service, _ = make_services(runtime_root)
    flow_a = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope-a", params={}), enqueue=False)
    flow_b = flow_service.start_flow(FlowRequest(flow_type="snapshot_flow", scope_id="scope-b", params={}), enqueue=False)
    initial_a = snapshot_service.create_scope_snapshot("scope-a")
    initial_b = snapshot_service.create_scope_snapshot("scope-b")
    assert initial_a.snapshot_id is not None
    assert initial_b.snapshot_id is not None
    running_b = SnapshotStep(
        step_id="unrefreshed-running-step",
        flow_id=flow_b,
        scope_id="scope-b",
        status=StepStatus.RUNNING,
        state=SnapshotStepState(),
    )
    step_service.create_step(running_b, enqueue=False)
    flow_service.store.update_flow_record(
        flow_b,
        lambda flow: (flow.step_ids.append(running_b.step_id), setattr(flow, "current_step_id", running_b.step_id)),
    )

    runtime_snapshot = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a", "scope-b"],
        wait=False,
    )

    assert runtime_snapshot.status == "created"
    assert runtime_snapshot.scope_snapshot_ids["scope-a"] != initial_a.snapshot_id
    assert runtime_snapshot.scope_snapshot_ids["scope-b"] == initial_b.snapshot_id


def test_scope_snapshot_serializes_copy_with_agent_step_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    flow_service, step_service, _, snapshot_service, ark = make_services(tmp_path / ".agent_runtime")
    step_id = seed_recoverable_agent_step(flow_service, step_service, ark)
    preview = flow_service.inspect_agent_step_recovery(step_id)
    assert ark.pause_controller is not None
    ark.pause_controller.pause("scope")
    mutation_entered = Event()
    release_mutation = Event()
    recovery_attempt = Barrier(2)
    recovery_done = Event()
    snapshot_results = []
    recovery_results = []
    recovery_errors = []
    original_create = snapshot_service._create_scope_snapshot_unlocked

    def blocked_create(scope_id: str):  # noqa: ANN202
        mutation_entered.set()
        assert release_mutation.wait(timeout=2)
        return original_create(scope_id)

    monkeypatch.setattr(snapshot_service, "_create_scope_snapshot_unlocked", blocked_create)

    def recover() -> None:
        recovery_attempt.wait(timeout=2)
        try:
            recovery_results.append(
                flow_service.recover_agent_step(
                    step_id=step_id,
                    expected_status=StepStatus.SUSPENDED,
                    expected_recovery_token=preview.recovery_token,
                    action="resume_suspended",
                    agent_mode="fresh",
                )
            )
        except BaseException as exc:  # noqa: BLE001 - thread outcome is asserted below.
            recovery_errors.append(exc)
        finally:
            recovery_done.set()

    snapshot_thread = Thread(
        target=lambda: snapshot_results.append(snapshot_service.create_scope_snapshot("scope")),
        daemon=True,
    )
    recovery_thread = Thread(target=recover, daemon=True)
    snapshot_thread.start()
    assert mutation_entered.wait(timeout=2)
    recovery_thread.start()
    recovery_attempt.wait(timeout=2)
    recovery_completed_inside_snapshot = recovery_done.wait(timeout=0.2)
    release_mutation.set()
    snapshot_thread.join(2)
    recovery_thread.join(2)

    assert recovery_completed_inside_snapshot is False
    assert not snapshot_thread.is_alive()
    assert not recovery_thread.is_alive()
    assert recovery_errors == []
    assert snapshot_results[0].status == "created", snapshot_results[0].errors
    assert recovery_results[0].replacement_step_id is not None


def test_scope_restore_serializes_mutation_with_agent_step_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:  # noqa: ANN001
    flow_service, step_service, _, snapshot_service, ark = make_services(tmp_path / ".agent_runtime")
    step_id = seed_recoverable_agent_step(flow_service, step_service, ark)
    snapshot = snapshot_service.create_scope_snapshot("scope")
    assert snapshot.snapshot_id is not None
    preview = flow_service.inspect_agent_step_recovery(step_id)
    mutation_entered = Event()
    release_mutation = Event()
    recovery_attempt = Barrier(2)
    recovery_done = Event()
    restore_results = []
    recovery_results = []
    recovery_errors = []
    original_remove = snapshot_service._remove_report_files_for_agents

    def blocked_remove(agents):  # noqa: ANN001, ANN202
        mutation_entered.set()
        assert release_mutation.wait(timeout=2)
        return original_remove(agents)

    monkeypatch.setattr(snapshot_service, "_remove_report_files_for_agents", blocked_remove)

    def recover() -> None:
        recovery_attempt.wait(timeout=2)
        try:
            recovery_results.append(
                flow_service.recover_agent_step(
                    step_id=step_id,
                    expected_status=StepStatus.SUSPENDED,
                    expected_recovery_token=preview.recovery_token,
                    action="resume_suspended",
                    agent_mode="fresh",
                )
            )
        except BaseException as exc:  # noqa: BLE001 - thread outcome is asserted below.
            recovery_errors.append(exc)
        finally:
            recovery_done.set()

    restore_thread = Thread(
        target=lambda: restore_results.append(snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)),
        daemon=True,
    )
    recovery_thread = Thread(target=recover, daemon=True)
    restore_thread.start()
    assert mutation_entered.wait(timeout=2)
    recovery_thread.start()
    recovery_attempt.wait(timeout=2)
    recovery_completed_inside_restore = recovery_done.wait(timeout=0.2)
    release_mutation.set()
    restore_thread.join(2)
    recovery_thread.join(2)

    assert recovery_completed_inside_restore is False
    assert not restore_thread.is_alive()
    assert not recovery_thread.is_alive()
    assert recovery_errors == []
    assert restore_results[0].status == "created"
    assert recovery_results[0].replacement_step_id is not None


@pytest.mark.parametrize("snapshot_kind", ["synchronized", "selected", "restore"])
def test_runtime_snapshot_mutation_holds_shared_paused_boundary(
    tmp_path: Path,
    monkeypatch,
    snapshot_kind: str,
) -> None:  # noqa: ANN001
    flow_service, _, _, snapshot_service, ark = make_services(tmp_path / ".agent_runtime")
    flow_service.start_flow(
        FlowRequest(flow_type="snapshot_flow", scope_id="scope", params={}),
        enqueue=False,
    )
    flow_service.start_flow(
        FlowRequest(flow_type="snapshot_flow", scope_id="scope-b", params={}),
        enqueue=False,
    )
    restore_snapshot_id = None
    if snapshot_kind == "restore":
        restore_source = snapshot_service.create_runtime_snapshot_synchronized()
        assert restore_source.snapshot_id is not None
        restore_snapshot_id = restore_source.snapshot_id
    mutation_entered = Event()
    release_mutation = Event()
    contender_attempt = Barrier(2)
    contender_entered = Event()
    contender_errors = []
    snapshot_results = []
    original_create = snapshot_service._create_scope_snapshot_unlocked
    original_remove = snapshot_service._remove_report_files_for_agents
    hold_order = []

    assert ark.pause_controller is not None
    original_hold = ark.pause_controller.hold_paused

    @contextmanager
    def observed_hold(scope_id: str | None):
        hold_order.append(scope_id)
        with original_hold(scope_id):
            yield

    monkeypatch.setattr(ark.pause_controller, "hold_paused", observed_hold)
    if snapshot_kind == "synchronized":
        ark.pause_controller.pause(None)
    elif snapshot_kind == "selected":
        ark.pause_controller.pause("scope")
        ark.pause_controller.pause("scope-b")

    def blocked_create(scope_id: str):  # noqa: ANN202
        mutation_entered.set()
        assert release_mutation.wait(timeout=2)
        return original_create(scope_id)

    def blocked_remove(agents):  # noqa: ANN001, ANN202
        mutation_entered.set()
        assert release_mutation.wait(timeout=2)
        return original_remove(agents)

    if snapshot_kind == "restore":
        monkeypatch.setattr(snapshot_service, "_remove_report_files_for_agents", blocked_remove)
    else:
        monkeypatch.setattr(snapshot_service, "_create_scope_snapshot_unlocked", blocked_create)

    def create_snapshot() -> None:
        if snapshot_kind == "synchronized":
            snapshot_results.append(snapshot_service.create_runtime_snapshot_synchronized())
        elif snapshot_kind == "selected":
            snapshot_results.append(
                snapshot_service.create_runtime_snapshot_for_scopes(
                    refresh_scope_ids=["scope-b", "scope"],
                    scope_ids=["scope-b", "scope"],
                    reuse_latest_for_other_scopes=False,
                )
            )
        else:
            assert restore_snapshot_id is not None
            snapshot_results.append(snapshot_service.restore_runtime_snapshot(restore_snapshot_id))

    def contend_for_recovery_boundary() -> None:
        assert ark.pause_controller is not None
        contender_attempt.wait(timeout=2)
        try:
            with ark.pause_controller.hold_paused("scope"):
                contender_entered.set()
        except BaseException as exc:  # noqa: BLE001 - thread outcome is asserted below.
            contender_errors.append(exc)

    snapshot_thread = Thread(target=create_snapshot, daemon=True)
    snapshot_thread.start()
    assert mutation_entered.wait(timeout=2)
    contender = Thread(target=contend_for_recovery_boundary, daemon=True)
    contender.start()
    contender_attempt.wait(timeout=2)
    contender_completed_inside_snapshot = contender_entered.wait(timeout=0.2)
    release_mutation.set()
    snapshot_thread.join(2)
    contender.join(2)

    assert contender_completed_inside_snapshot is False
    assert not snapshot_thread.is_alive()
    assert not contender.is_alive()
    assert contender_errors == []
    assert snapshot_results[0].status == "created", snapshot_results[0].errors
    if snapshot_kind == "selected":
        assert hold_order[:2] == ["scope", "scope-b"]
    else:
        assert hold_order[0] is None
