from pathlib import Path
from threading import Event
from typing import ClassVar

from pydantic import BaseModel

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
    release_event: ClassVar[Event] = Event()
    started_events: ClassVar[dict[str, Event]] = {}

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        if SchedulerStep.block:
            SchedulerStep.started_events.setdefault(ctx.step_id, Event()).set()
            SchedulerStep.release_event.wait(timeout=5)
        return ctx.complete_step(BaseStepResult(result_type="scheduler_step_done", summary="done"))


class SchedulerFlow(BaseFlow):
    flow_type: ClassVar[str] = "scheduler_flow"
    Params: ClassVar[type[BaseModel]] = SchedulerFlowParams
    State: ClassVar[type[BaseFlowState]] = SchedulerFlowState

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


class FakePauseController:
    def __init__(self, paused: bool = False) -> None:
        self.paused = paused

    def is_paused(self, scope_id: str | None = None) -> bool:
        return self.paused


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
