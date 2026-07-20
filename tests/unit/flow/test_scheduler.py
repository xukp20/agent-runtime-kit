from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import ClassVar

import pytest
from pydantic import BaseModel, ValidationError

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
    FlowStepValidationError,
    RuntimeScheduleService,
    SchedulerRunBudget,
    SchedulerRunDecision,
    SchedulerSemanticRunPolicy,
    StepRunContext,
    StepService,
    StepStatus,
    StepTerminalReceipt,
    FlowTypeRegistry,
    StepTypeRegistry,
)
from agent_runtime_kit.runtime import ARKServices, AppServices


class SchedulerFlowParams(BaseModel):
    mode: str = "step"
    status: str = "created"
    ready: bool = False


class SchedulerFlowState(BaseFlowState):
    state_type: str = "scheduler_flow_state"
    mode: str = "step"
    ready: bool = False
    terminal_seen_count: int = 0


class SchedulerStepState(BaseStepState):
    state_type: str = "scheduler_step_state"


class SchedulerStep(BaseStep):
    step_type: ClassVar[str] = "scheduler_step"
    State: ClassVar[type[BaseStepState]] = SchedulerStepState
    block: ClassVar[bool] = False
    fail: ClassVar[bool] = False
    release_event: ClassVar[Event] = Event()
    started_events: ClassVar[dict[str, Event]] = {}

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        if SchedulerStep.block:
            SchedulerStep.started_events.setdefault(ctx.step_id, Event()).set()
            SchedulerStep.release_event.wait(timeout=5)
        if SchedulerStep.fail:
            return ctx.fail_step(BaseStepError(error_type="scheduler_step_failed", message="failed"))
        return ctx.complete_step(BaseStepResult(result_type="scheduler_step_done", summary="done"))


class SchedulerFlow(BaseFlow):
    flow_type: ClassVar[str] = "scheduler_flow"
    Params: ClassVar[type[BaseModel]] = SchedulerFlowParams
    State: ClassVar[type[BaseFlowState]] = SchedulerFlowState
    stable_seen_flow_ids: ClassVar[set[str]] = set()

    @classmethod
    def build_from_request(cls, ctx: FlowBuildContext) -> "SchedulerFlow":
        return cls(
            flow_id=ctx.flow_id,
            scope_id=ctx.scope_id,
            status=FlowStatus(ctx.params.status),
            state=SchedulerFlowState(mode=ctx.params.mode, ready=ctx.params.ready),
        )

    def can_exit_waiting(self, ctx: FlowContext) -> bool:
        assert isinstance(self.state, SchedulerFlowState)
        return self.state.ready

    def create_next_step(self, ctx: FlowContext) -> str | None:
        assert isinstance(self.state, SchedulerFlowState)
        if self.state.mode == "complete":
            ctx.set_flow_result(BaseFlowResult(result_type="scheduler_flow_done", summary="done"))
            return None
        if self.state.mode == "noop":
            return None
        index = len(self.step_ids) + 1
        step = SchedulerStep(
            step_id=f"{self.flow_id}-step-{index}",
            flow_id=self.flow_id,
            scope_id=self.scope_id,
        )
        return ctx.create_step(step)

    def on_step_terminal(self, ctx) -> None:
        assert isinstance(self.state, SchedulerFlowState)
        self.state.terminal_seen_count += 1
        super().on_step_terminal(ctx)

    def after_step_terminal_stable(self, ctx) -> None:
        assert isinstance(self.state, SchedulerFlowState)
        assert ctx.ark.pause_controller is None or not ctx.ark.pause_controller.is_paused(self.scope_id)
        SchedulerFlow.stable_seen_flow_ids.add(self.flow_id)


class FakePauseController:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused

    def is_paused(self, scope_id: str | None = None) -> bool:
        return self.paused

    def pause(self, scope_id: str | None = None) -> None:
        self.paused = True

    def resume(self, scope_id: str | None = None) -> None:
        self.paused = False


def make_services(
    runtime_root: Path,
    *,
    pause: FakePauseController | None = None,
    max_concurrent_flow_advances: int = 1,
    max_concurrent_steps: int = 1,
) -> tuple[FlowService, StepService, RuntimeScheduleService]:
    flow_registry = FlowTypeRegistry()
    step_registry = StepTypeRegistry()
    flow_registry.register(SchedulerFlow)
    step_registry.register(SchedulerStep)
    ark = ARKServices(pause_controller=pause)
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
    scheduler = RuntimeScheduleService(
        ark_services=ark,
        app_services=AppServices(),
        max_concurrent_flow_advances=max_concurrent_flow_advances,
        max_concurrent_steps=max_concurrent_steps,
    )
    return flow_service, step_service, scheduler


def start_scheduler_flow(
    flow_service: FlowService,
    *,
    scope_id: str = "scope",
    mode: str = "step",
    status: str = "created",
    ready: bool = False,
    enqueue: bool = False,
) -> str:
    return flow_service.start_flow(
        FlowRequest(
            flow_type="scheduler_flow",
            scope_id=scope_id,
            params={"mode": mode, "status": status, "ready": ready},
        ),
        enqueue=enqueue,
    )


def test_enqueue_deduplicates_candidates(tmp_path: Path) -> None:
    _, _, scheduler = make_services(tmp_path / ".agent_runtime")

    scheduler.enqueue_flow("flow-1")
    scheduler.enqueue_flow("flow-1")
    scheduler.enqueue_step("step-1")
    scheduler.enqueue_step("step-1")

    assert list(scheduler.flow_candidate_queue) == ["flow-1"]
    assert list(scheduler.step_candidate_queue) == ["step-1"]
    assert scheduler.queued_flow_ids == {"flow-1"}
    assert scheduler.queued_step_ids == {"step-1"}


def test_scheduler_run_budget_rejects_empty_budget() -> None:
    with pytest.raises(ValidationError, match="at least one action"):
        SchedulerRunBudget(flow_advances=0, step_starts=0)


def test_rebuild_candidate_queues_from_truth(tmp_path: Path) -> None:
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime")
    active_flow_id = start_scheduler_flow(flow_service)
    completed_flow_id = start_scheduler_flow(flow_service)
    flow_service.store.update_flow_record(completed_flow_id, lambda flow: setattr(flow, "status", FlowStatus.COMPLETED))
    flow_service.store.create_step(SchedulerStep(step_id="created-step", flow_id=active_flow_id, scope_id="scope"))

    scheduler.enqueue_flow("stale-flow")
    scheduler.enqueue_step("stale-step")
    scheduler.rebuild_candidate_queues()

    assert list(scheduler.flow_candidate_queue) == [active_flow_id]
    assert list(scheduler.step_candidate_queue) == ["created-step"]
    assert completed_flow_id not in scheduler.queued_flow_ids


def test_scoped_rebuild_candidate_queues_preserves_other_scope_candidates(tmp_path: Path) -> None:
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime")
    scope_a_flow = start_scheduler_flow(flow_service, scope_id="scope-a")
    scope_b_flow = start_scheduler_flow(flow_service, scope_id="scope-b")
    completed_scope_a_flow = start_scheduler_flow(flow_service, scope_id="scope-a")
    flow_service.store.update_flow_record(
        completed_scope_a_flow,
        lambda flow: setattr(flow, "status", FlowStatus.COMPLETED),
    )
    flow_service.store.create_step(SchedulerStep(step_id="scope-a-step", flow_id=scope_a_flow, scope_id="scope-a"))
    flow_service.store.create_step(SchedulerStep(step_id="scope-b-step", flow_id=scope_b_flow, scope_id="scope-b"))
    scheduler.enqueue_flow(scope_b_flow)
    scheduler.enqueue_flow("missing-flow")
    scheduler.enqueue_step("scope-b-step")
    scheduler.enqueue_step("missing-step")

    scheduler.rebuild_candidate_queues(scope_id="scope-a")

    assert list(scheduler.flow_candidate_queue) == [scope_b_flow, scope_a_flow]
    assert list(scheduler.step_candidate_queue) == ["scope-b-step", "scope-a-step"]
    assert completed_scope_a_flow not in scheduler.queued_flow_ids
    assert "missing-flow" not in scheduler.queued_flow_ids
    assert "missing-step" not in scheduler.queued_step_ids


def test_schedule_flow_once_advances_and_enqueues_created_step(tmp_path: Path) -> None:
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime")
    flow_id = start_scheduler_flow(flow_service)
    scheduler.enqueue_flow(flow_id)

    advanced_id = scheduler.schedule_flow_once()

    flow = flow_service.get_flow(flow_id)
    assert advanced_id == flow_id
    assert flow.current_step_id == f"{flow_id}-step-1"
    assert list(scheduler.step_candidate_queue) == [f"{flow_id}-step-1"]


def test_schedule_step_once_runs_step_and_requeues_parent_flow(tmp_path: Path) -> None:
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime")
    flow_id = start_scheduler_flow(flow_service)
    scheduler.enqueue_flow(flow_id)
    scheduler.schedule_flow_once()
    step_id = f"{flow_id}-step-1"

    started_id = scheduler.schedule_step_once()

    step = step_service.wait_step(step_id)
    flow = flow_service.get_flow(flow_id)
    assert started_id == step_id
    assert step.status is StepStatus.COMPLETED
    assert step.result is not None
    assert flow.current_step_id is None
    assert isinstance(flow.state, SchedulerFlowState)
    assert flow.state.terminal_seen_count == 1
    assert flow_id in scheduler.queued_flow_ids


def test_pause_gate_keeps_flow_and_step_candidates(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service)
    scheduler.enqueue_flow(flow_id)
    flow_service.advance_flow(flow_id)
    step_id = f"{flow_id}-step-1"
    scheduler.enqueue_step(step_id)
    pause.paused = True

    tick = scheduler.schedule_ready()

    assert tick.advanced_flow_ids == []
    assert tick.started_step_ids == []
    assert tick.reason == "no_runnable_candidate"
    assert flow_id in scheduler.queued_flow_ids
    assert step_id in scheduler.queued_step_ids


def test_schedule_ready_advances_flow_then_runs_created_step(tmp_path: Path) -> None:
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime")
    flow_id = start_scheduler_flow(flow_service, enqueue=True)

    tick = scheduler.schedule_ready()

    step_id = f"{flow_id}-step-1"
    assert tick.advanced_flow_ids == [flow_id]
    assert tick.started_step_ids == [step_id]
    assert step_service.wait_step(step_id).status is StepStatus.COMPLETED


def test_bounded_flow_only_run_advances_once_and_auto_pauses(tmp_path: Path) -> None:
    pause = FakePauseController(paused=True)
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service, enqueue=True)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=1, step_starts=0))
    pause.resume(None)

    tick = scheduler.schedule_ready()

    step_id = f"{flow_id}-step-1"
    assert tick.advanced_flow_ids == [flow_id]
    assert tick.started_step_ids == []
    assert tick.auto_paused is True
    assert tick.run_control is not None
    assert tick.run_control.mode == "paused"
    assert tick.run_control.pause_reason == "budget_exhausted"
    assert tick.run_control.remaining_flow_advances == 0
    assert tick.run_control.remaining_step_starts == 0
    assert pause.is_paused()
    assert step_service.store.get_step(step_id).status is StepStatus.CREATED


def test_bounded_step_run_drains_until_stable_hook_before_auto_pause(tmp_path: Path) -> None:
    SchedulerStep.block = True
    SchedulerStep.release_event = Event()
    SchedulerStep.started_events = {}
    SchedulerFlow.stable_seen_flow_ids = set()
    pause = FakePauseController(paused=False)
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service)
    step_id = flow_service.advance_flow(flow_id)
    assert step_id is not None
    SchedulerStep.started_events[step_id] = Event()
    pause.pause(None)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=0, step_starts=1))
    pause.resume(None)

    try:
        first_tick = scheduler.schedule_ready()
        assert first_tick.started_step_ids == [step_id]
        assert SchedulerStep.started_events[step_id].wait(timeout=2)
        assert first_tick.auto_paused is False
        assert first_tick.run_control is not None
        assert first_tick.run_control.mode == "draining"
        assert pause.is_paused() is False

        draining_tick = scheduler.schedule_ready()
        assert draining_tick.advanced_flow_ids == []
        assert draining_tick.started_step_ids == []
        assert draining_tick.auto_paused is False
        assert pause.is_paused() is False
    finally:
        SchedulerStep.release_event.set()
        SchedulerStep.block = False

    assert step_service.wait_step(step_id, timeout_s=2).status is StepStatus.COMPLETED
    flow = flow_service.get_flow(flow_id)
    assert isinstance(flow.state, SchedulerFlowState)
    assert flow.state.terminal_seen_count == 1
    assert SchedulerFlow.stable_seen_flow_ids == {flow_id}

    terminal_tick = scheduler.schedule_ready()
    assert terminal_tick.advanced_flow_ids == []
    assert terminal_tick.started_step_ids == []
    assert terminal_tick.auto_paused is True
    assert terminal_tick.run_control is not None
    assert terminal_tick.run_control.pause_reason == "budget_exhausted"
    assert pause.is_paused()


def test_bounded_run_auto_pauses_with_remaining_budget_when_no_candidate(tmp_path: Path) -> None:
    pause = FakePauseController(paused=True)
    _, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=0, step_starts=1))
    pause.resume(None)

    tick = scheduler.schedule_ready()

    assert tick.auto_paused is True
    assert tick.reason == "no_runnable_candidate"
    assert tick.run_control is not None
    assert tick.run_control.pause_reason == "no_runnable_candidate"
    assert tick.run_control.remaining_step_starts == 1
    assert pause.is_paused()


def test_bounded_failed_step_consumes_budget_and_auto_pauses(tmp_path: Path) -> None:
    SchedulerStep.fail = True
    pause = FakePauseController(paused=False)
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service)
    step_id = flow_service.advance_flow(flow_id)
    assert step_id is not None
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=0, step_starts=1))

    try:
        first_tick = scheduler.schedule_ready()
        assert first_tick.started_step_ids == [step_id]
        assert step_service.wait_step(step_id, timeout_s=2).status is StepStatus.FAILED
        terminal_tick = first_tick if first_tick.auto_paused else scheduler.schedule_ready()
    finally:
        SchedulerStep.fail = False

    assert terminal_tick.auto_paused is True
    assert terminal_tick.run_control is not None
    assert terminal_tick.run_control.remaining_step_starts == 0
    assert terminal_tick.run_control.pause_reason == "budget_exhausted"
    assert pause.is_paused()


def test_bounded_step_budget_caps_starts_below_concurrency_limit(tmp_path: Path) -> None:
    SchedulerStep.block = True
    SchedulerStep.release_event = Event()
    SchedulerStep.started_events = {}
    pause = FakePauseController(paused=False)
    flow_service, step_service, scheduler = make_services(
        tmp_path / ".agent_runtime",
        pause=pause,
        max_concurrent_steps=2,
    )
    step_ids: list[str] = []
    for index in range(2):
        flow_id = start_scheduler_flow(flow_service)
        step_id = f"bounded-step-{index}"
        flow_service.store.create_step(SchedulerStep(step_id=step_id, flow_id=flow_id, scope_id="scope"))
        flow_service.store.update_flow_record(flow_id, lambda flow, value=step_id: setattr(flow, "current_step_id", value))
        scheduler.enqueue_step(step_id)
        SchedulerStep.started_events[step_id] = Event()
        step_ids.append(step_id)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=0, step_starts=1))

    try:
        tick = scheduler.schedule_ready()
        assert tick.started_step_ids == [step_ids[0]]
        assert SchedulerStep.started_events[step_ids[0]].wait(timeout=2)
        assert SchedulerStep.started_events[step_ids[1]].is_set() is False
        assert step_service.store.get_step(step_ids[1]).status is StepStatus.CREATED
    finally:
        SchedulerStep.release_event.set()
        SchedulerStep.block = False

    assert step_service.wait_step(step_ids[0], timeout_s=2).status is StepStatus.COMPLETED


def test_bounded_flow_budget_is_not_consumed_by_skipped_candidate(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    waiting_flow_id = start_scheduler_flow(flow_service, status="waiting", ready=False)
    runnable_flow_id = start_scheduler_flow(flow_service)
    scheduler.enqueue_flow(waiting_flow_id)
    scheduler.enqueue_flow(runnable_flow_id)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=1, step_starts=0))

    tick = scheduler.schedule_ready()

    assert tick.skipped_flow_count == 1
    assert tick.advanced_flow_ids == [runnable_flow_id]
    assert tick.run_control is not None
    assert tick.run_control.remaining_flow_advances == 0


def test_bounded_flow_budget_preserves_fifo_across_multiple_runnable_flows(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    first_flow_id = start_scheduler_flow(flow_service)
    second_flow_id = start_scheduler_flow(flow_service)
    scheduler.enqueue_flow(first_flow_id)
    scheduler.enqueue_flow(second_flow_id)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=1, step_starts=0))

    tick = scheduler.schedule_ready()

    assert tick.advanced_flow_ids == [first_flow_id]
    assert second_flow_id in scheduler.queued_flow_ids
    assert flow_service.get_flow(second_flow_id).status is FlowStatus.CREATED
    assert tick.auto_paused is True


def test_bounded_flow_budget_is_reserved_before_concurrent_advance(tmp_path: Path, monkeypatch) -> None:
    pause = FakePauseController(paused=False)
    flow_service, _, scheduler = make_services(
        tmp_path / ".agent_runtime",
        pause=pause,
        max_concurrent_flow_advances=2,
    )
    first_flow_id = start_scheduler_flow(flow_service)
    second_flow_id = start_scheduler_flow(flow_service)
    scheduler.enqueue_flow(first_flow_id)
    scheduler.enqueue_flow(second_flow_id)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=1, step_starts=0))
    advance_entered = Event()
    release_advance = Event()
    original_advance = flow_service.advance_flow

    def blocking_advance(flow_id: str):
        advance_entered.set()
        release_advance.wait(timeout=5)
        return original_advance(flow_id)

    monkeypatch.setattr(flow_service, "advance_flow", blocking_advance)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(scheduler.schedule_flow_once)
        assert advance_entered.wait(timeout=2)
        second = pool.submit(scheduler.schedule_flow_once)
        assert second.result(timeout=2) is None
        release_advance.set()
        assert first.result(timeout=2) == first_flow_id

    view = scheduler.get_run_control_view()
    assert view.remaining_flow_advances == 0
    assert second_flow_id in scheduler.queued_flow_ids


def test_bounded_flow_advance_exception_refunds_reserved_budget(tmp_path: Path, monkeypatch) -> None:
    pause = FakePauseController(paused=False)
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service)
    scheduler.enqueue_flow(flow_id)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=1, step_starts=0))

    def fail_advance(_flow_id: str) -> None:
        raise RuntimeError("advance failed")

    monkeypatch.setattr(flow_service, "advance_flow", fail_advance)

    with pytest.raises(RuntimeError, match="advance failed"):
        scheduler.schedule_flow_once()

    assert scheduler.get_run_control_view().remaining_flow_advances == 1


def test_clear_run_budget_restores_unbounded_mode(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    _, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=2, step_starts=3))

    view = scheduler.clear_run_budget()

    assert view.mode == "unbounded"
    assert view.requested_flow_advances is None
    assert view.remaining_step_starts is None


def test_clear_run_budget_with_reason_retains_cancelled_run_evidence(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    _, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=2, step_starts=3))
    pause.pause(None)

    view = scheduler.clear_run_budget(reason="manual_pause")

    assert view.mode == "paused"
    assert view.requested_flow_advances == 2
    assert view.remaining_step_starts == 3
    assert view.pause_reason == "manual_pause"


def test_semantic_policy_runs_logic_to_a_created_step_and_pauses(tmp_path: Path) -> None:
    pause = FakePauseController(paused=True)
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service, enqueue=True)
    step_id = f"{flow_id}-step-1"
    scheduler.configure_semantic_run(
        SchedulerSemanticRunPolicy(
            name="logic_to_step",
            allow_flow_advance=lambda flow: flow.flow_id == flow_id,
            allow_step_start=lambda step: False,
            decide=lambda service: SchedulerRunDecision(
                action="pause" if step_id in service.queued_step_ids else "continue",
                reason="target_step_created" if step_id in service.queued_step_ids else None,
            ),
            max_flow_advances=10,
            max_step_starts=0,
        )
    )
    pause.resume(None)

    tick = scheduler.schedule_ready()

    assert tick.advanced_flow_ids == [flow_id]
    assert tick.started_step_ids == []
    assert tick.auto_paused is True
    assert step_service.store.get_step(step_id).status is StepStatus.CREATED
    assert tick.run_control is not None
    assert tick.run_control.run_plan == "semantic"
    assert tick.run_control.semantic_policy == "logic_to_step"
    assert tick.run_control.completed_flow_advances == 1
    assert tick.run_control.pause_reason == "target_step_created"

    lease = scheduler.get_run_lease(tick.run_control.lease_id or "")
    assert lease.status == "terminal"
    assert lease.policy_name == "logic_to_step"
    assert lease.completed_flow_advances == 1
    assert lease.advanced_flow_ids == [flow_id]
    assert lease.terminal_reason == "target_step_created"
    assert lease.terminal_at is not None


def test_semantic_lease_wait_times_out_then_wakes_on_terminal(tmp_path: Path) -> None:
    pause = FakePauseController(paused=True)
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service, enqueue=True)
    step_id = f"{flow_id}-step-1"
    control = scheduler.configure_semantic_run(
        SchedulerSemanticRunPolicy(
            name="waitable_logic",
            allow_flow_advance=lambda flow: flow.flow_id == flow_id,
            allow_step_start=lambda step: False,
            decide=lambda service: SchedulerRunDecision(
                action="pause" if step_id in service.queued_step_ids else "continue",
                reason="target_step_created" if step_id in service.queued_step_ids else None,
            ),
            max_flow_advances=10,
            max_step_starts=0,
        )
    )
    lease_id = control.lease_id or ""
    started = scheduler.get_run_lease(lease_id)
    timed_out = scheduler.wait_run_lease(lease_id, after_version=started.version, timeout_s=0)
    assert timed_out.timed_out is True
    assert timed_out.lease.status == "active"

    with ThreadPoolExecutor(max_workers=2) as pool:
        waiter = pool.submit(
            scheduler.wait_run_lease,
            lease_id,
            after_version=started.version,
            timeout_s=2,
        )
        second_waiter = pool.submit(
            scheduler.wait_run_lease,
            lease_id,
            after_version=started.version,
            timeout_s=2,
        )
        pause.resume(None)
        tick = scheduler.schedule_ready()
        waited = waiter.result(timeout=2)
        second_waited = second_waiter.result(timeout=2)

    assert tick.auto_paused is True
    assert waited.timed_out is False
    assert waited.lease.version > started.version
    assert second_waited.timed_out is False
    assert second_waited.lease.version == waited.lease.version
    terminal = scheduler.wait_run_lease(
        lease_id,
        after_version=waited.lease.version,
        timeout_s=0,
    )
    assert terminal.timed_out is False
    assert terminal.lease.status == "terminal"


def test_semantic_lease_records_manual_clear_as_terminal_with_inactive_control(tmp_path: Path) -> None:
    pause = FakePauseController(paused=True)
    _, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    control = scheduler.configure_semantic_run(
        SchedulerSemanticRunPolicy(
            name="manual_clear",
            allow_flow_advance=lambda flow: True,
            allow_step_start=lambda step: True,
            decide=lambda service: SchedulerRunDecision(),
        )
    )
    lease_id = control.lease_id or ""

    scheduler.clear_run_budget(reason="manual_pause")
    lease = scheduler.get_run_lease(lease_id)

    assert lease.status == "terminal"
    assert lease.terminal_reason == "manual_pause"
    assert lease.run_control.mode == "paused"
    assert lease.run_control.pause_reason == "manual_pause"


def test_semantic_lease_lookup_is_process_local(tmp_path: Path) -> None:
    pause = FakePauseController(paused=True)
    _, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)

    with pytest.raises(KeyError, match="process-local scheduler run lease"):
        scheduler.get_run_lease("lease_from_previous_process")
    with pytest.raises(KeyError, match="process-local scheduler run lease"):
        scheduler.wait_run_lease("lease_from_previous_process", timeout_s=0)


def test_semantic_policy_starts_only_target_step_and_stops_after_terminal(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service)
    target_step_id = flow_service.advance_flow(flow_id)
    assert target_step_id is not None
    other_flow_id = start_scheduler_flow(flow_service)
    other_step_id = flow_service.advance_flow(other_flow_id)
    assert other_step_id is not None
    pause.pause(None)

    def decide(_service: RuntimeScheduleService) -> SchedulerRunDecision:
        target = step_service.store.get_step(target_step_id)
        if target.status in {StepStatus.COMPLETED, StepStatus.FAILED}:
            return SchedulerRunDecision(action="pause", reason="target_step_terminal")
        return SchedulerRunDecision()

    scheduler.configure_semantic_run(
        SchedulerSemanticRunPolicy(
            name="one_target_step",
            allow_flow_advance=lambda flow: False,
            allow_step_start=lambda step: step.step_id == target_step_id,
            decide=decide,
            max_flow_advances=0,
            max_step_starts=1,
        )
    )
    pause.resume(None)

    first_tick = scheduler.schedule_ready()
    assert first_tick.started_step_ids == [target_step_id]
    assert step_service.wait_step(target_step_id, timeout_s=2).status is StepStatus.COMPLETED
    terminal_tick = first_tick if first_tick.auto_paused else scheduler.schedule_ready()

    assert terminal_tick.auto_paused is True
    assert terminal_tick.run_control is not None
    assert terminal_tick.run_control.pause_reason in {"target_step_terminal", "semantic_safety_cap_exhausted"}
    assert step_service.store.get_step(other_step_id).status is StepStatus.CREATED
    assert other_step_id in scheduler.queued_step_ids


def test_semantic_policy_and_numeric_budget_are_mutually_exclusive(tmp_path: Path) -> None:
    pause = FakePauseController(paused=True)
    _, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    scheduler.configure_semantic_run(
        SchedulerSemanticRunPolicy(
            name="exclusive",
            allow_flow_advance=lambda flow: True,
            allow_step_start=lambda step: True,
            decide=lambda service: SchedulerRunDecision(),
        )
    )

    with pytest.raises(FlowStepValidationError, match="semantic run lease is active"):
        scheduler.configure_run_budget(SchedulerRunBudget(flow_advances=1, step_starts=0))


def test_semantic_policy_retries_once_for_an_admitted_temporarily_blocked_flow(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service, status="waiting", ready=False, enqueue=True)
    scheduler.configure_semantic_run(
        SchedulerSemanticRunPolicy(
            name="retry_admitted_flow",
            allow_flow_advance=lambda flow: flow.flow_id == flow_id,
            allow_step_start=lambda step: False,
            decide=lambda service: SchedulerRunDecision(),
            max_flow_advances=10,
            max_step_starts=0,
        )
    )

    blocked_tick = scheduler.schedule_ready()

    assert blocked_tick.advanced_flow_ids == []
    assert blocked_tick.auto_paused is False
    assert pause.is_paused() is False

    def mark_ready(flow: BaseFlow) -> None:
        assert isinstance(flow.state, SchedulerFlowState)
        flow.state.ready = True

    flow_service.store.update_flow_record(flow_id, mark_ready)
    resumed_tick = scheduler.schedule_ready()

    assert resumed_tick.advanced_flow_ids == [flow_id]
    assert resumed_tick.auto_paused is False
    assert pause.is_paused() is False


def test_semantic_policy_tolerates_two_idle_settlements_before_flow_becomes_ready(tmp_path: Path) -> None:
    pause = FakePauseController(paused=False)
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime", pause=pause)
    flow_id = start_scheduler_flow(flow_service, status="waiting", ready=False, enqueue=True)
    scheduler.configure_semantic_run(
        SchedulerSemanticRunPolicy(
            name="retry_admitted_flow_twice",
            allow_flow_advance=lambda flow: flow.flow_id == flow_id,
            allow_step_start=lambda step: False,
            decide=lambda service: SchedulerRunDecision(),
            max_flow_advances=10,
            max_step_starts=0,
        )
    )

    first_blocked_tick = scheduler.schedule_ready()
    second_blocked_tick = scheduler.schedule_ready()

    assert first_blocked_tick.advanced_flow_ids == []
    assert first_blocked_tick.auto_paused is False
    assert second_blocked_tick.advanced_flow_ids == []
    assert second_blocked_tick.auto_paused is False
    assert pause.is_paused() is False

    def mark_ready(flow: BaseFlow) -> None:
        assert isinstance(flow.state, SchedulerFlowState)
        flow.state.ready = True

    flow_service.store.update_flow_record(flow_id, mark_ready)
    resumed_tick = scheduler.schedule_ready()

    assert resumed_tick.advanced_flow_ids == [flow_id]
    assert resumed_tick.auto_paused is False
    assert pause.is_paused() is False


def test_schedule_ready_starts_multiple_steps_up_to_concurrency_limit(tmp_path: Path) -> None:
    SchedulerStep.block = True
    SchedulerStep.release_event = Event()
    SchedulerStep.started_events = {}
    flow_service, step_service, scheduler = make_services(tmp_path / ".agent_runtime", max_concurrent_steps=2)
    first_flow_id = start_scheduler_flow(flow_service)
    second_flow_id = start_scheduler_flow(flow_service)
    first_step_id = "blocking-step-1"
    second_step_id = "blocking-step-2"
    flow_service.store.create_step(SchedulerStep(step_id=first_step_id, flow_id=first_flow_id, scope_id="scope"))
    flow_service.store.create_step(SchedulerStep(step_id=second_step_id, flow_id=second_flow_id, scope_id="scope"))
    flow_service.store.update_flow_record(first_flow_id, lambda flow: setattr(flow, "current_step_id", first_step_id))
    flow_service.store.update_flow_record(second_flow_id, lambda flow: setattr(flow, "current_step_id", second_step_id))
    scheduler.enqueue_step(first_step_id)
    scheduler.enqueue_step(second_step_id)
    SchedulerStep.started_events[first_step_id] = Event()
    SchedulerStep.started_events[second_step_id] = Event()

    try:
        tick = scheduler.schedule_ready()

        assert tick.started_step_ids == [first_step_id, second_step_id]
        assert SchedulerStep.started_events[first_step_id].wait(timeout=2)
        assert SchedulerStep.started_events[second_step_id].wait(timeout=2)
        assert {first_step_id, second_step_id}.issubset(set(step_service.list_running_steps("scope")))
    finally:
        SchedulerStep.release_event.set()
        SchedulerStep.block = False

    assert step_service.wait_step(first_step_id, timeout_s=2).status is StepStatus.COMPLETED
    assert step_service.wait_step(second_step_id, timeout_s=2).status is StepStatus.COMPLETED


def test_waiting_flow_exits_waiting_on_later_tick(tmp_path: Path) -> None:
    flow_service, _, scheduler = make_services(tmp_path / ".agent_runtime")
    flow_id = start_scheduler_flow(flow_service, status="waiting", ready=False)
    scheduler.enqueue_flow(flow_id)

    first_tick = scheduler.schedule_ready()
    assert first_tick.advanced_flow_ids == []
    assert flow_id in scheduler.queued_flow_ids

    def mark_ready(flow: BaseFlow) -> None:
        assert isinstance(flow.state, SchedulerFlowState)
        flow.state.ready = True

    flow_service.store.update_flow_record(flow_id, mark_ready)
    second_tick = scheduler.schedule_ready()

    assert second_tick.advanced_flow_ids == [flow_id]
    assert second_tick.started_step_ids == [f"{flow_id}-step-1"]
