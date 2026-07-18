from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from agent_runtime_kit.runtime import ARKServices, AppServices

from .models import FlowStatus, FlowStepValidationError, StepStatus
from .store import FlowNotFoundError, StepNotFoundError


class SchedulerRunBudget(BaseModel):
    flow_advances: int = Field(ge=0)
    step_starts: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_non_empty_budget(self) -> "SchedulerRunBudget":
        if self.flow_advances == 0 and self.step_starts == 0:
            raise ValueError("scheduler run budget must allow at least one action")
        return self


class SchedulerRunControlView(BaseModel):
    mode: Literal["unbounded", "bounded", "semantic", "draining", "paused"]
    run_plan: Literal["unbounded", "bounded", "semantic"] = "unbounded"
    lease_id: str | None = None
    semantic_policy: str | None = None
    requested_flow_advances: int | None = None
    requested_step_starts: int | None = None
    remaining_flow_advances: int | None = None
    remaining_step_starts: int | None = None
    completed_flow_advances: int = 0
    completed_step_starts: int = 0
    pause_reason: str | None = None


class SchedulerRunDecision(BaseModel):
    action: Literal["continue", "drain", "pause", "fail"] = "continue"
    reason: str | None = None


@dataclass(frozen=True)
class SchedulerSemanticRunPolicy:
    """Process-local admission and stopping policy without application semantics."""

    name: str
    allow_flow_advance: Callable[[object], bool]
    allow_step_start: Callable[[object], bool]
    decide: Callable[["RuntimeScheduleService"], SchedulerRunDecision]
    max_flow_advances: int = 1000
    max_step_starts: int = 1000


class SchedulerTickResult(BaseModel):
    advanced_flow_ids: list[str] = Field(default_factory=list)
    started_step_ids: list[str] = Field(default_factory=list)
    skipped_flow_count: int = 0
    skipped_step_count: int = 0
    reason: str | None = None
    auto_paused: bool = False
    run_control: SchedulerRunControlView | None = None


class RuntimeScheduleService:
    def __init__(
        self,
        *,
        ark_services: ARKServices | None = None,
        app_services: AppServices | None = None,
        max_concurrent_flow_advances: int = 1,
        max_concurrent_steps: int = 1,
    ) -> None:
        if max_concurrent_flow_advances < 1:
            raise FlowStepValidationError("max_concurrent_flow_advances must be >= 1")
        if max_concurrent_steps < 1:
            raise FlowStepValidationError("max_concurrent_steps must be >= 1")
        self.ark = ark_services or ARKServices()
        self.app = app_services or AppServices()
        self.flow_candidate_queue: deque[str] = deque()
        self.step_candidate_queue: deque[str] = deque()
        self.queued_flow_ids: set[str] = set()
        self.queued_step_ids: set[str] = set()
        self.active_flow_advances: set[str] = set()
        self.max_concurrent_flow_advances = max_concurrent_flow_advances
        self.max_concurrent_steps = max_concurrent_steps
        self.lock = RLock()
        self._requested_run_budget: SchedulerRunBudget | None = None
        self._remaining_flow_advances: int | None = None
        self._remaining_step_starts: int | None = None
        self._bounded_run_active = False
        self._bounded_run_draining = False
        self._bounded_pause_reason: str | None = None
        self._semantic_policy: SchedulerSemanticRunPolicy | None = None
        self._semantic_lease_id: str | None = None
        self._semantic_run_active = False
        self._semantic_run_draining = False
        self._semantic_flow_advances = 0
        self._semantic_step_starts = 0
        self._semantic_idle_retry_pending = False
        self.ark.schedule_service = self

    def configure_run_budget(self, budget: SchedulerRunBudget) -> SchedulerRunControlView:
        with self.lock:
            if self._semantic_run_active:
                raise FlowStepValidationError("cannot configure a numeric budget while a semantic run lease is active")
            self._requested_run_budget = budget.model_copy(deep=True)
            self._remaining_flow_advances = budget.flow_advances
            self._remaining_step_starts = budget.step_starts
            self._bounded_run_active = True
            self._bounded_run_draining = False
            self._bounded_pause_reason = None
            self._semantic_policy = None
            self._semantic_lease_id = None
            self._semantic_flow_advances = 0
            self._semantic_step_starts = 0
            self._semantic_idle_retry_pending = False
        return self.get_run_control_view()

    def configure_semantic_run(self, policy: SchedulerSemanticRunPolicy) -> SchedulerRunControlView:
        if not policy.name.strip():
            raise FlowStepValidationError("semantic scheduler policy name must be non-empty")
        if policy.max_flow_advances < 0 or policy.max_step_starts < 0:
            raise FlowStepValidationError("semantic scheduler safety caps must be non-negative")
        if policy.max_flow_advances == 0 and policy.max_step_starts == 0:
            raise FlowStepValidationError("semantic scheduler safety caps must allow at least one action")
        with self.lock:
            if self._bounded_run_active or self._semantic_run_active:
                raise FlowStepValidationError("a scheduler run plan is already active")
            self._semantic_policy = policy
            self._semantic_lease_id = f"lease_{uuid4().hex}"
            self._semantic_run_active = True
            self._semantic_run_draining = False
            self._semantic_flow_advances = 0
            self._semantic_step_starts = 0
            self._semantic_idle_retry_pending = False
            self._bounded_pause_reason = None
            self._requested_run_budget = None
            self._remaining_flow_advances = None
            self._remaining_step_starts = None
        return self.get_run_control_view()

    def clear_run_budget(self, *, reason: str | None = None) -> SchedulerRunControlView:
        with self.lock:
            self._bounded_run_active = False
            self._bounded_run_draining = False
            if reason is None:
                self._requested_run_budget = None
                self._remaining_flow_advances = None
                self._remaining_step_starts = None
            self._bounded_pause_reason = reason
            self._semantic_run_active = False
            self._semantic_run_draining = False
            self._semantic_idle_retry_pending = False
            if reason is None:
                self._semantic_policy = None
                self._semantic_lease_id = None
                self._semantic_flow_advances = 0
                self._semantic_step_starts = 0
        return self.get_run_control_view()

    def get_run_control_view(self) -> SchedulerRunControlView:
        paused = self._is_globally_paused()
        with self.lock:
            if self._semantic_policy is not None:
                run_plan: Literal["unbounded", "bounded", "semantic"] = "semantic"
            elif self._requested_run_budget is not None:
                run_plan = "bounded"
            else:
                run_plan = "unbounded"
            if paused:
                mode: Literal["unbounded", "bounded", "semantic", "draining", "paused"] = "paused"
            elif (self._bounded_run_active and self._bounded_run_draining) or (
                self._semantic_run_active and self._semantic_run_draining
            ):
                mode = "draining"
            elif self._bounded_run_active:
                mode = "bounded"
            elif self._semantic_run_active:
                mode = "semantic"
            else:
                mode = "unbounded"
            requested = self._requested_run_budget
            semantic = self._semantic_policy
            return SchedulerRunControlView(
                mode=mode,
                run_plan=run_plan,
                lease_id=self._semantic_lease_id,
                semantic_policy=None if semantic is None else semantic.name,
                requested_flow_advances=None if requested is None else requested.flow_advances,
                requested_step_starts=None if requested is None else requested.step_starts,
                remaining_flow_advances=self._remaining_flow_advances,
                remaining_step_starts=self._remaining_step_starts,
                completed_flow_advances=self._semantic_flow_advances,
                completed_step_starts=self._semantic_step_starts,
                pause_reason=self._bounded_pause_reason,
            )

    def rebuild_candidate_queues(self, *, scope_id: str | None = None) -> None:
        flow_service = self._flow_service()
        step_service = self._step_service()
        flows = flow_service.list_non_terminal_flows(scope_id=scope_id)
        step_ids = step_service.list_created_steps(scope_id=scope_id)
        with self.lock:
            if scope_id is None:
                self.flow_candidate_queue.clear()
                self.step_candidate_queue.clear()
                self.queued_flow_ids.clear()
                self.queued_step_ids.clear()
            else:
                self._remove_scope_candidates_locked(scope_id)
            for flow in flows:
                if flow.flow_id not in self.queued_flow_ids:
                    self.flow_candidate_queue.append(flow.flow_id)
                    self.queued_flow_ids.add(flow.flow_id)
            for step_id in step_ids:
                if step_id not in self.queued_step_ids:
                    self.step_candidate_queue.append(step_id)
                    self.queued_step_ids.add(step_id)

    def enqueue_flow(self, flow_id: str) -> None:
        with self.lock:
            if flow_id in self.queued_flow_ids:
                return
            self.flow_candidate_queue.append(flow_id)
            self.queued_flow_ids.add(flow_id)

    def enqueue_step(self, step_id: str) -> None:
        with self.lock:
            if step_id in self.queued_step_ids:
                return
            self.step_candidate_queue.append(step_id)
            self.queued_step_ids.add(step_id)

    def schedule_flow_once(self) -> str | None:
        flow_id, _ = self._schedule_flow_once_with_count()
        return flow_id

    def schedule_step_once(self) -> str | None:
        step_id, _ = self._schedule_step_once_with_count()
        return step_id

    def schedule_ready(self) -> SchedulerTickResult:
        result = SchedulerTickResult()
        while True:
            flow_id, skipped = self._schedule_flow_once_with_count()
            result.skipped_flow_count += skipped
            if flow_id is None:
                break
            result.advanced_flow_ids.append(flow_id)

        while True:
            step_id, skipped = self._schedule_step_once_with_count()
            result.skipped_step_count += skipped
            if step_id is None:
                break
            result.started_step_ids.append(step_id)

        if not result.advanced_flow_ids and not result.started_step_ids:
            result.reason = "no_runnable_candidate"
        result.auto_paused = self._settle_run_control(
            made_progress=bool(result.advanced_flow_ids or result.started_step_ids)
        )
        result.run_control = self.get_run_control_view()
        return result

    def _schedule_flow_once_with_count(self) -> tuple[str | None, int]:
        flow_service = self._flow_service()
        skipped = 0
        with self.lock:
            if not self._flow_budget_available_locked():
                return None, skipped
            if len(self.active_flow_advances) >= self.max_concurrent_flow_advances:
                return None, skipped
            max_scan = len(self.flow_candidate_queue)

        scanned = 0
        while scanned < max_scan:
            with self.lock:
                if not self.flow_candidate_queue:
                    return None, skipped
                flow_id = self.flow_candidate_queue.popleft()
                self.queued_flow_ids.discard(flow_id)
            scanned += 1

            with self.lock:
                if flow_id in self.active_flow_advances:
                    skipped += 1
                    self._requeue_flow_if_non_terminal(flow_id)
                    continue

            if not flow_service.can_advance_flow(flow_id):
                skipped += 1
                self._requeue_flow_if_non_terminal(flow_id)
                continue

            flow = flow_service.get_flow(flow_id)
            if not self._semantic_flow_allowed(flow):
                skipped += 1
                self._requeue_flow_if_non_terminal(flow_id)
                continue

            with self.lock:
                if not self._flow_budget_available_locked():
                    self._requeue_flow_if_non_terminal(flow_id)
                    return None, skipped
                if len(self.active_flow_advances) >= self.max_concurrent_flow_advances:
                    self._requeue_flow_if_non_terminal(flow_id)
                    return None, skipped
                if flow_id in self.active_flow_advances:
                    skipped += 1
                    self._requeue_flow_if_non_terminal(flow_id)
                    continue
                self.active_flow_advances.add(flow_id)
                self._reserve_flow_budget_locked()
            try:
                flow_service.advance_flow(flow_id)
            except Exception:
                with self.lock:
                    self._refund_flow_budget_locked()
                raise
            finally:
                with self.lock:
                    self.active_flow_advances.discard(flow_id)
            return flow_id, skipped

        return None, skipped

    def _schedule_step_once_with_count(self) -> tuple[str | None, int]:
        step_service = self._step_service()
        skipped = 0
        with self.lock:
            if not self._step_budget_available_locked():
                return None, skipped
        if len(step_service.list_running_steps()) >= self.max_concurrent_steps:
            return None, skipped
        with self.lock:
            max_scan = len(self.step_candidate_queue)

        scanned = 0
        while scanned < max_scan:
            with self.lock:
                if not self.step_candidate_queue:
                    return None, skipped
                step_id = self.step_candidate_queue.popleft()
                self.queued_step_ids.discard(step_id)
            scanned += 1

            with self.lock:
                if not self._step_budget_available_locked():
                    self._requeue_step_if_created(step_id)
                    return None, skipped

            if len(step_service.list_running_steps()) >= self.max_concurrent_steps:
                self._requeue_step_if_created(step_id)
                return None, skipped

            if not step_service.can_run_step(step_id):
                skipped += 1
                self._requeue_step_if_created(step_id)
                continue

            step = step_service.store.get_step(step_id)
            if not self._semantic_step_allowed(step):
                skipped += 1
                self._requeue_step_if_created(step_id)
                continue

            with self.lock:
                if not self._step_budget_available_locked():
                    self._requeue_step_if_created(step_id)
                    return None, skipped
                self._reserve_step_budget_locked()
            try:
                step_service.start_step(step_id)
            except Exception:
                with self.lock:
                    self._refund_step_budget_locked()
                raise
            return step_id, skipped

        return None, skipped

    def _requeue_flow_if_non_terminal(self, flow_id: str) -> None:
        try:
            flow = self._flow_service().get_flow(flow_id)
        except FlowNotFoundError:
            return
        if flow.status not in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
            self.enqueue_flow(flow_id)

    def _requeue_step_if_created(self, step_id: str) -> None:
        try:
            step = self._step_service().store.get_step(step_id)
        except StepNotFoundError:
            return
        if step.status is StepStatus.CREATED:
            self.enqueue_step(step_id)

    def _remove_scope_candidates_locked(self, scope_id: str) -> None:
        flow_service = self._flow_service()
        step_service = self._step_service()
        kept_flows: deque[str] = deque()
        kept_flow_ids: set[str] = set()
        while self.flow_candidate_queue:
            flow_id = self.flow_candidate_queue.popleft()
            try:
                flow = flow_service.get_flow(flow_id)
            except FlowNotFoundError:
                continue
            if flow.scope_id == scope_id:
                continue
            if flow_id not in kept_flow_ids:
                kept_flows.append(flow_id)
                kept_flow_ids.add(flow_id)

        kept_steps: deque[str] = deque()
        kept_step_ids: set[str] = set()
        while self.step_candidate_queue:
            step_id = self.step_candidate_queue.popleft()
            try:
                step = step_service.store.get_step(step_id)
            except StepNotFoundError:
                continue
            if step.scope_id == scope_id:
                continue
            if step_id not in kept_step_ids:
                kept_steps.append(step_id)
                kept_step_ids.add(step_id)

        self.flow_candidate_queue = kept_flows
        self.queued_flow_ids = kept_flow_ids
        self.step_candidate_queue = kept_steps
        self.queued_step_ids = kept_step_ids

    def _flow_budget_available_locked(self) -> bool:
        if self._semantic_run_active:
            if self._semantic_run_draining:
                return False
            policy = self._semantic_policy
            return policy is not None and self._semantic_flow_advances < policy.max_flow_advances
        if not self._bounded_run_active:
            return True
        if self._bounded_run_draining:
            return False
        return bool(self._remaining_flow_advances and self._remaining_flow_advances > 0)

    def _step_budget_available_locked(self) -> bool:
        if self._semantic_run_active:
            if self._semantic_run_draining:
                return False
            policy = self._semantic_policy
            return policy is not None and self._semantic_step_starts < policy.max_step_starts
        if not self._bounded_run_active:
            return True
        if self._bounded_run_draining:
            return False
        return bool(self._remaining_step_starts and self._remaining_step_starts > 0)

    def _reserve_flow_budget_locked(self) -> None:
        if self._semantic_run_active:
            self._semantic_flow_advances += 1
        if self._bounded_run_active and self._remaining_flow_advances is not None:
            self._remaining_flow_advances -= 1

    def _refund_flow_budget_locked(self) -> None:
        if self._semantic_run_active and self._semantic_flow_advances > 0:
            self._semantic_flow_advances -= 1
        if self._bounded_run_active and self._remaining_flow_advances is not None:
            self._remaining_flow_advances += 1

    def _reserve_step_budget_locked(self) -> None:
        if self._semantic_run_active:
            self._semantic_step_starts += 1
        if self._bounded_run_active and self._remaining_step_starts is not None:
            self._remaining_step_starts -= 1

    def _refund_step_budget_locked(self) -> None:
        if self._semantic_run_active and self._semantic_step_starts > 0:
            self._semantic_step_starts -= 1
        if self._bounded_run_active and self._remaining_step_starts is not None:
            self._remaining_step_starts += 1

    def _settle_run_control(self, *, made_progress: bool) -> bool:
        if self._semantic_run_active:
            return self._settle_semantic_run(made_progress=made_progress)
        return self._settle_bounded_run()

    def _settle_bounded_run(self) -> bool:
        with self.lock:
            if not self._bounded_run_active:
                return False
            exhausted = self._remaining_flow_advances == 0 and self._remaining_step_starts == 0

        if self._has_active_work():
            if exhausted:
                with self.lock:
                    if self._bounded_run_active:
                        self._bounded_run_draining = True
            return False

        reason = "budget_exhausted" if exhausted else "no_runnable_candidate"
        pause_controller = self.ark.pause_controller
        if pause_controller is None or not hasattr(pause_controller, "pause"):
            raise FlowStepValidationError("bounded scheduler run requires a pause controller")
        pause_controller.pause(None)
        with self.lock:
            self._bounded_run_active = False
            self._bounded_run_draining = False
            self._bounded_pause_reason = reason
        return True

    def _settle_semantic_run(self, *, made_progress: bool) -> bool:
        with self.lock:
            if not self._semantic_run_active:
                return False
            policy = self._semantic_policy
            exhausted = bool(
                policy is not None
                and (
                    (
                        policy.max_flow_advances > 0
                        and self._semantic_flow_advances >= policy.max_flow_advances
                    )
                    or (
                        policy.max_step_starts > 0
                        and self._semantic_step_starts >= policy.max_step_starts
                    )
                )
            )
            draining = self._semantic_run_draining
            if made_progress:
                self._semantic_idle_retry_pending = False
        if policy is None:
            raise FlowStepValidationError("semantic scheduler run is active without a policy")

        decision = policy.decide(self)
        should_stop = draining or exhausted or decision.action in {"drain", "pause", "fail"}
        reason = (
            "semantic_safety_cap_exhausted"
            if exhausted
            else self._bounded_pause_reason or decision.reason or ("semantic_boundary_reached" if should_stop else None)
        )

        if self._has_active_work():
            if should_stop:
                with self.lock:
                    if self._semantic_run_active:
                        self._semantic_run_draining = True
                        self._bounded_pause_reason = reason
            return False

        if not should_stop and made_progress:
            return False
        if not should_stop:
            with self.lock:
                retry_pending = self._semantic_idle_retry_pending
            if not retry_pending and self._has_semantic_admitted_candidate(policy):
                with self.lock:
                    if self._semantic_run_active:
                        self._semantic_idle_retry_pending = True
                return False
            reason = decision.reason or "no_runnable_candidate"

        pause_controller = self.ark.pause_controller
        if pause_controller is None or not hasattr(pause_controller, "pause"):
            raise FlowStepValidationError("semantic scheduler run requires a pause controller")
        pause_controller.pause(None)
        with self.lock:
            self._semantic_run_active = False
            self._semantic_run_draining = False
            self._semantic_idle_retry_pending = False
            self._bounded_pause_reason = reason
        return True

    def _has_semantic_admitted_candidate(self, policy: SchedulerSemanticRunPolicy) -> bool:
        with self.lock:
            flow_ids = tuple(self.flow_candidate_queue)
            step_ids = tuple(self.step_candidate_queue)
        flow_service = self._flow_service()
        step_service = self._step_service()
        for flow_id in flow_ids:
            try:
                flow = flow_service.get_flow(flow_id)
            except FlowNotFoundError:
                continue
            if flow.status not in {FlowStatus.COMPLETED, FlowStatus.FAILED} and policy.allow_flow_advance(flow):
                return True
        for step_id in step_ids:
            try:
                step = step_service.store.get_step(step_id)
            except StepNotFoundError:
                continue
            if step.status is StepStatus.CREATED and policy.allow_step_start(step):
                return True
        return False

    def _semantic_flow_allowed(self, flow: object) -> bool:
        with self.lock:
            policy = self._semantic_policy if self._semantic_run_active else None
        return True if policy is None else bool(policy.allow_flow_advance(flow))

    def _semantic_step_allowed(self, step: object) -> bool:
        with self.lock:
            policy = self._semantic_policy if self._semantic_run_active else None
        return True if policy is None else bool(policy.allow_step_start(step))

    def _has_active_work(self) -> bool:
        with self.lock:
            if self.active_flow_advances:
                return True
        return bool(self._step_service().list_running_steps())

    def _is_globally_paused(self) -> bool:
        pause_controller = self.ark.pause_controller
        if pause_controller is None or not hasattr(pause_controller, "is_paused"):
            return False
        return bool(pause_controller.is_paused(None))

    def _flow_service(self):
        flow_service = self.ark.flow_service
        if flow_service is None:
            raise FlowStepValidationError("ark.flow_service is not registered")
        return flow_service

    def _step_service(self):
        step_service = self.ark.step_service
        if step_service is None:
            raise FlowStepValidationError("ark.step_service is not registered")
        return step_service
