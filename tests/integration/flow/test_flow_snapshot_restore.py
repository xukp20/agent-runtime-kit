from pathlib import Path
from threading import Event, Timer
from typing import ClassVar

from pydantic import BaseModel

from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.store import AgentStoreService
from agent_runtime_kit.flow import (
    BaseFlow,
    BaseFlowResult,
    BaseFlowState,
    BaseStep,
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
