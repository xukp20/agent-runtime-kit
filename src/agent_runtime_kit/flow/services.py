from __future__ import annotations

import uuid
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from time import monotonic, sleep
from typing import Any

from pydantic import BaseModel, ConfigDict
from agent_runtime_kit.runtime import ARKServices, AppServices

from .contexts import FlowBuildContext, FlowContext, FlowReadContext, FlowStepContext, StableStepTerminalContext, StepRunContext
from .models import (
    BaseFlow,
    BaseFlowError,
    BaseStep,
    BaseStepError,
    FlowRequest,
    FlowStatus,
    FlowStepValidationError,
    StepStatus,
    StepTerminalReceipt,
    StepTerminalWaitResult,
    utc_now_iso,
)
from .registry import FlowTypeRegistry, StepTypeRegistry
from .store import FlowStepStore


class ActiveStepRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_id: str
    flow_id: str
    scope_id: str
    started_at: str
    worker_ref: Any | None = None
    done_event: Event | None = None
    exception: Any | None = None
    bypass_pause: bool = False


class FlowService:
    def __init__(
        self,
        runtime_root: Path,
        *,
        flow_registry: FlowTypeRegistry,
        step_registry: StepTypeRegistry,
        ark_services: ARKServices | None = None,
        app_services: AppServices | None = None,
        store: FlowStepStore | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.flow_registry = flow_registry
        self.step_registry = step_registry
        self.ark = ark_services or ARKServices()
        self.app = app_services or AppServices()
        self.store = store or FlowStepStore(
            self.runtime_root,
            flow_registry=self.flow_registry,
            step_registry=self.step_registry,
        )
        self.lock = RLock()
        self.stable_hook_errors: list[dict[str, str]] = []
        self.ark.flow_service = self

    def start_flow(
        self,
        request: FlowRequest,
        *,
        parent_flow_id: str | None = None,
        parent_dispatch_step_id: str | None = None,
        enqueue: bool = True,
    ) -> str:
        with self.lock:
            flow_id = f"f_{uuid.uuid4().hex}"
            flow = self._build_flow_from_request(
                request=request,
                flow_id=flow_id,
                parent_flow_id=parent_flow_id,
                parent_dispatch_step_id=parent_dispatch_step_id,
            )
            self.store.create_flow(flow)
            schedule_service = self.ark.schedule_service
            if enqueue and schedule_service is not None and flow.status not in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                schedule_service.enqueue_flow(flow.flow_id)
            return flow.flow_id

    def start_flows_batch(
        self,
        requests: list[FlowRequest],
        *,
        parent_flow_id: str | None = None,
        parent_dispatch_step_id: str | None = None,
        enqueue: bool = True,
    ) -> list[str]:
        with self.lock:
            built_flows: list[BaseFlow] = []
            for request in requests:
                flow_id = f"f_{uuid.uuid4().hex}"
                built_flows.append(
                    self._build_flow_from_request(
                        request=request,
                        flow_id=flow_id,
                        parent_flow_id=parent_flow_id,
                        parent_dispatch_step_id=parent_dispatch_step_id,
                    )
                )
            with self.store.edit_session(None) as tx:
                for flow in built_flows:
                    tx.add_flow(flow)
            child_ids = [flow.flow_id for flow in built_flows]
            schedule_service = self.ark.schedule_service
            if enqueue and schedule_service is not None:
                for flow in built_flows:
                    if flow.status not in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                        schedule_service.enqueue_flow(flow.flow_id)
            return child_ids

    def get_flow(self, flow_id: str) -> BaseFlow:
        return self.store.get_flow(flow_id)

    def get_step(self, step_id: str):
        return self.store.get_step(step_id)

    def list_flows(
        self,
        *,
        scope_id: str | None = None,
        status: str | FlowStatus | None = None,
        flow_type: str | None = None,
    ) -> list[BaseFlow]:
        return self.store.list_flows(scope_id=scope_id, status=status, flow_type=flow_type)

    def list_steps(
        self,
        *,
        scope_id: str | None = None,
        flow_id: str | None = None,
        status: str | StepStatus | None = None,
        step_type: str | None = None,
    ):
        return self.store.list_steps(scope_id=scope_id, flow_id=flow_id, status=status, step_type=step_type)

    def list_non_terminal_flows(self, *, scope_id: str | None = None) -> list[BaseFlow]:
        return self.store.list_non_terminal_flows(scope_id=scope_id)

    def can_advance_flow(self, flow_id: str) -> bool:
        with self.lock:
            flow = self.store.get_flow(flow_id)
            pause_controller = self.ark.pause_controller
            if pause_controller is not None and pause_controller.is_paused(flow.scope_id):
                return False
            if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                return False
            if flow.manual_pause.active:
                return False
            if flow.current_step_id is not None:
                return False
            if flow.status is FlowStatus.WAITING:
                ctx = FlowReadContext(ark=self.ark, app=self.app, flow=flow)
                return flow.can_exit_waiting(ctx)
            return flow.status in {FlowStatus.CREATED, FlowStatus.RUNNING}

    def prepare_flow_for_advance(self, flow_id: str) -> bool:
        with self.lock:
            flow = self.store.get_flow(flow_id)
            pause_controller = self.ark.pause_controller
            if pause_controller is not None and pause_controller.is_paused(flow.scope_id):
                return False
            if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                return False
            if flow.manual_pause.active:
                return False
            if flow.current_step_id is not None:
                return False
            if flow.status is FlowStatus.WAITING:
                with self.store.edit_session(flow.scope_id) as tx:
                    working = tx.load_flow_for_update(flow_id)
                    ctx = FlowContext(ark=self.ark, app=self.app, flow=working, tx=tx)
                    if not working.can_exit_waiting(ctx):
                        return False
                    working.on_exit_waiting(ctx)
                return True
            return flow.status in {FlowStatus.CREATED, FlowStatus.RUNNING}

    def advance_flow(self, flow_id: str) -> str | None:
        if not self.prepare_flow_for_advance(flow_id):
            raise FlowStepValidationError(f"flow cannot advance: {flow_id}")
        with self.lock:
            flow = self.store.get_flow(flow_id)
            with self.store.edit_session(flow.scope_id) as tx:
                working = tx.load_flow_for_update(flow_id)
                if working.current_step_id is not None:
                    raise FlowStepValidationError(f"flow {flow_id} already has current step {working.current_step_id}")
                ctx = FlowContext(ark=self.ark, app=self.app, flow=working, tx=tx)
                step_id = working.create_next_step(ctx)
                step: BaseStep | None = None
                if step_id is not None:
                    step = tx.new_steps.get(step_id) or tx.working_steps.get(step_id)
                    if step is None:
                        step = tx.load_step_for_update(step_id)
                    if step.flow_id != working.flow_id:
                        raise FlowStepValidationError(
                            f"step {step.step_id} belongs to flow {step.flow_id}, expected {working.flow_id}"
                        )
                    if step.scope_id != working.scope_id:
                        raise FlowStepValidationError(
                            f"step {step.step_id} scope {step.scope_id}, expected {working.scope_id}"
                        )
                elif (
                    working.result is None
                    and working.error is None
                    and working.status not in {FlowStatus.WAITING, FlowStatus.COMPLETED, FlowStatus.FAILED}
                ):
                    self._mark_flow_no_progress(working)
            if step_id is not None:
                schedule_service = self.ark.schedule_service
                if schedule_service is not None:
                    schedule_service.enqueue_step(step_id)
            persisted = self.store.get_flow(flow_id)
            if persisted.error is not None and persisted.error.error_type == "flow_no_progress":
                raise FlowStepValidationError(f"flow {flow_id} made no progress while advancing")
            return step_id

    def handle_step_terminal(self, step_id: str) -> None:
        step = self.store.get_step(step_id)
        if step.status not in {StepStatus.COMPLETED, StepStatus.FAILED}:
            raise FlowStepValidationError(f"step is not terminal: {step_id}")
        flow = self.store.get_flow(step.flow_id)
        if flow.current_step_id != step_id:
            if flow.current_step_id is None and step_id in flow.step_ids:
                return
            raise FlowStepValidationError(
                f"flow {flow.flow_id} current_step_id is {flow.current_step_id}, expected {step_id}"
            )

        try:
            with self.store.edit_session(flow.scope_id) as tx:
                working_flow = tx.load_flow_for_update(flow.flow_id)
                working_step = tx.load_step_for_update(step_id)
                ctx = FlowStepContext(ark=self.ark, app=self.app, flow=working_flow, step=working_step, tx=tx)
                working_flow.on_step_terminal(ctx)
        except Exception as exc:
            self.store.update_flow_record(flow.flow_id, lambda failed_flow: self._mark_flow_terminal_handler_failed(failed_flow, exc))
            raise

        flow_for_hook = self.store.get_flow(flow.flow_id)
        step_for_hook = self.store.get_step(step_id)
        stable_ctx = StableStepTerminalContext(ark=self.ark, app=self.app, flow=flow_for_hook, step=step_for_hook)
        try:
            flow_for_hook.after_step_terminal_stable(stable_ctx)
        except Exception as exc:
            self.stable_hook_errors.append(
                {
                    "flow_id": flow.flow_id,
                    "step_id": step_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc) or type(exc).__name__,
                }
            )
        persisted = self.store.get_flow(flow.flow_id)
        if persisted.status not in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
            schedule_service = self.ark.schedule_service
            if schedule_service is not None:
                schedule_service.enqueue_flow(persisted.flow_id)

    def assert_restorable_flows(self, *, scope_id: str | None = None) -> None:
        try:
            self.store.assert_restorable_truth(scope_id=scope_id)
        except Exception as exc:
            raise FlowStepValidationError(str(exc)) from exc

    def _prepare_and_validate_new_flow(
        self,
        flow: BaseFlow,
        *,
        request: FlowRequest,
        flow_id: str,
        parent_flow_id: str | None,
        parent_dispatch_step_id: str | None,
    ) -> None:
        actual_flow_type = str(getattr(flow, "flow_type", ""))
        if flow.flow_id != flow_id:
            raise FlowStepValidationError(f"build_from_request returned flow_id {flow.flow_id}, expected {flow_id}")
        if actual_flow_type != request.flow_type:
            raise FlowStepValidationError(
                f"build_from_request returned flow_type {actual_flow_type}, expected {request.flow_type}"
            )
        if flow.scope_id != request.scope_id:
            raise FlowStepValidationError(f"build_from_request returned scope {flow.scope_id}, expected {request.scope_id}")
        if flow.parent_flow_id not in {None, parent_flow_id}:
            raise FlowStepValidationError(
                f"build_from_request returned parent_flow_id {flow.parent_flow_id}, expected {parent_flow_id}"
            )
        if flow.parent_dispatch_step_id not in {None, parent_dispatch_step_id}:
            raise FlowStepValidationError(
                "build_from_request returned parent_dispatch_step_id "
                f"{flow.parent_dispatch_step_id}, expected {parent_dispatch_step_id}"
            )
        flow.parent_flow_id = parent_flow_id
        flow.parent_dispatch_step_id = parent_dispatch_step_id
        if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
            raise FlowStepValidationError(f"new flow {flow.flow_id} must not start terminal")
        if not self.flow_registry.can_parse_state(request.flow_type, flow.state.state_type):
            raise FlowStepValidationError(
                f"flow {flow.flow_id} state {flow.state.state_type} cannot be parsed by {request.flow_type}"
            )
        if getattr(type(flow), "requires_callback_input", False) and flow.input is None:
            raise FlowStepValidationError(f"flow type {request.flow_type} requires input for callback rendering")

    def _build_flow_from_request(
        self,
        *,
        request: FlowRequest,
        flow_id: str,
        parent_flow_id: str | None,
        parent_dispatch_step_id: str | None,
    ) -> BaseFlow:
        flow_cls = self.flow_registry.get(request.flow_type)
        params = self.flow_registry.validate_request_params(request)
        ctx = FlowBuildContext(
            ark=self.ark,
            app=self.app,
            request=request,
            params=params,
            flow_id=flow_id,
            scope_id=request.scope_id,
            parent_flow_id=parent_flow_id,
            parent_dispatch_step_id=parent_dispatch_step_id,
        )
        flow = flow_cls.build_from_request(ctx)
        self._prepare_and_validate_new_flow(
            flow,
            request=request,
            flow_id=flow_id,
            parent_flow_id=parent_flow_id,
            parent_dispatch_step_id=parent_dispatch_step_id,
        )
        return flow

    def _mark_flow_no_progress(self, flow: BaseFlow) -> None:
        now = utc_now_iso()
        flow.error = BaseFlowError(
            error_type="flow_no_progress",
            message=f"flow {flow.flow_id} did not create a step, complete, fail, or wait",
        )
        flow.status = FlowStatus.FAILED
        flow.finished_at = now
        flow.updated_at = now

    def _mark_flow_terminal_handler_failed(self, flow: BaseFlow, exc: Exception) -> None:
        now = utc_now_iso()
        flow.error = BaseFlowError(
            error_type="flow_terminal_handler_error",
            message=str(exc) or type(exc).__name__,
            details={"exception_type": type(exc).__name__},
        )
        flow.status = FlowStatus.FAILED
        flow.finished_at = now
        flow.updated_at = now


class StepService:
    def __init__(
        self,
        runtime_root: Path,
        *,
        step_registry: StepTypeRegistry,
        ark_services: ARKServices | None = None,
        app_services: AppServices | None = None,
        store: FlowStepStore | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.step_registry = step_registry
        self.ark = ark_services or ARKServices()
        self.app = app_services or AppServices()
        flow_service = self.ark.flow_service
        flow_registry = getattr(flow_service, "flow_registry", None)
        if store is None:
            if flow_service is not None and isinstance(getattr(flow_service, "store", None), FlowStepStore):
                store = flow_service.store
            elif isinstance(flow_registry, FlowTypeRegistry):
                store = FlowStepStore(self.runtime_root, flow_registry=flow_registry, step_registry=self.step_registry)
            else:
                raise FlowStepValidationError("StepService requires a FlowStepStore or registered FlowService")
        self.store = store
        self.active_steps: dict[str, ActiveStepRun] = {}
        self.lock = RLock()
        self._step_condition = Condition(self.lock)
        self.ark.step_service = self

    def create_step(self, step: BaseStep, *, enqueue: bool = True) -> str:
        with self._step_condition:
            self.store.create_step(step)
            if enqueue and self.ark.schedule_service is not None:
                self.ark.schedule_service.enqueue_step(step.step_id)
            self._step_condition.notify_all()
            return step.step_id

    def can_run_step(self, step_id: str) -> bool:
        with self.lock:
            step = self.store.get_step(step_id)
            if step.status is not StepStatus.CREATED:
                return False
            if step_id in self.active_steps:
                return False
            flow = self.store.get_flow(step.flow_id)
            if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                return False
            pause_controller = self.ark.pause_controller
            if pause_controller is not None and pause_controller.is_paused(step.scope_id):
                return False
            return flow.current_step_id == step_id

    def start_step(self, step_id: str, *, bypass_pause: bool = False) -> ActiveStepRun:
        pause_controller = self.ark.pause_controller
        bypass_context = (
            pause_controller.bypass_current_thread()
            if bypass_pause and pause_controller is not None and hasattr(pause_controller, "bypass_current_thread")
            else None
        )
        if bypass_context is None:
            can_run = self.can_run_step(step_id)
        else:
            with bypass_context:
                can_run = self.can_run_step(step_id)
        if not can_run:
            raise FlowStepValidationError(f"step cannot run: {step_id}")
        step = self.store.get_step(step_id)
        flow = self.store.get_flow(step.flow_id)
        done_event = Event()
        active = ActiveStepRun(
            step_id=step.step_id,
            flow_id=step.flow_id,
            scope_id=step.scope_id,
            started_at=utc_now_iso(),
            done_event=done_event,
            bypass_pause=bypass_pause,
        )
        with self._step_condition:
            if step_id in self.active_steps:
                raise FlowStepValidationError(f"step already active: {step_id}")
            self.active_steps[step_id] = active
            self._step_condition.notify_all()

        worker = Thread(target=self._run_step_body, args=(step_id, active), daemon=True)
        active.worker_ref = worker
        worker.start()
        return active

    def run_step(self, step_id: str, *, bypass_pause: bool = False) -> None:
        active = self.start_step(step_id, bypass_pause=bypass_pause)
        if active.done_event is not None:
            active.done_event.wait()
        if active.exception is not None:
            raise active.exception

    def _run_step_body(self, step_id: str, active: ActiveStepRun) -> None:
        step = self.store.get_step(step_id)
        flow = self.store.get_flow(step.flow_id)
        ctx = StepRunContext(ark=self.ark, app=self.app, step_id=step.step_id, flow_id=flow.flow_id, scope_id=step.scope_id)
        try:
            pause_controller = self.ark.pause_controller
            bypass_context = (
                pause_controller.bypass_current_thread()
                if active.bypass_pause and pause_controller is not None and hasattr(pause_controller, "bypass_current_thread")
                else None
            )
            if bypass_context is None:
                self._run_step_body_with_context(step_id, step, ctx)
            else:
                with bypass_context:
                    self._run_step_body_with_context(step_id, step, ctx)
        except Exception as exc:
            active.exception = exc
        finally:
            with self._step_condition:
                self.active_steps.pop(step_id, None)
                if active.done_event is not None:
                    active.done_event.set()
                self._step_condition.notify_all()

    def _run_step_body_with_context(self, step_id: str, step: BaseStep, ctx: StepRunContext) -> None:
        self._mark_step_running(step_id)
        step_impl = self.store.get_step(step_id)
        try:
            receipt = step_impl.run(ctx)
        except Exception as exc:
            receipt = ctx.fail_step(
                BaseStepError(
                    error_type="step_run_exception",
                    message=str(exc) or type(exc).__name__,
                    details={"exception_type": type(exc).__name__},
                )
            )
        else:
            if not isinstance(receipt, StepTerminalReceipt):
                receipt = self._force_fail_step(
                    ctx,
                    BaseStepError(
                        error_type="step_not_terminal",
                        message=f"step {step_id} did not return StepTerminalReceipt",
                    )
                )
            elif not self._receipt_matches(receipt, step):
                receipt = self._force_fail_step(
                    ctx,
                    BaseStepError(
                        error_type="invalid_terminal_receipt",
                        message=f"step {step_id} returned invalid terminal receipt",
                    )
                )
        self._validate_terminal_receipt(receipt, step)
        flow_service = self.ark.flow_service
        if flow_service is None or not hasattr(flow_service, "handle_step_terminal"):
            raise FlowStepValidationError("ctx.ark.flow_service.handle_step_terminal is not available")
        flow_service.handle_step_terminal(step_id)

    def wait_step(self, step_id: str, *, timeout_s: float | None = None) -> BaseStep:
        active = self.active_steps.get(step_id)
        if active is not None and active.done_event is not None:
            if not active.done_event.wait(timeout_s):
                raise TimeoutError(f"timed out waiting for step: {step_id}")
        else:
            deadline = None if timeout_s is None else monotonic() + timeout_s
            while step_id in self.active_steps:
                if deadline is not None and monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for step: {step_id}")
                sleep(0.01)
        return self.store.get_step(step_id)

    def wait_step_terminal(
        self,
        step_id: str,
        *,
        timeout_s: float | None = None,
    ) -> StepTerminalWaitResult:
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        deadline = None if timeout_s is None else monotonic() + timeout_s
        with self._step_condition:
            while True:
                step = self.store.get_step(step_id)
                active = step_id in self.active_steps
                status_terminal = step.status in {StepStatus.COMPLETED, StepStatus.FAILED}
                if status_terminal and not active:
                    return StepTerminalWaitResult(
                        step=step,
                        terminal=True,
                        timed_out=False,
                        runner_state="settled",
                    )
                if step.status is StepStatus.RUNNING and not active:
                    return StepTerminalWaitResult(
                        step=step,
                        terminal=False,
                        timed_out=False,
                        runner_state="lost",
                        warning="persisted running step has no active runner in this process",
                    )

                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    return StepTerminalWaitResult(
                        step=step,
                        terminal=False,
                        timed_out=True,
                        runner_state="active" if active else "not_started",
                        warning=(
                            "step reached terminal status but terminal handling is still active"
                            if status_terminal
                            else None
                        ),
                    )
                self._step_condition.wait(remaining)

    def list_running_steps(self, scope_id: str | None = None) -> list[str]:
        active_ids = [
            run.step_id
            for run in self.active_steps.values()
            if scope_id is None or run.scope_id == scope_id
        ]
        persisted_ids = [step.step_id for step in self.store.list_steps(scope_id=scope_id, status=StepStatus.RUNNING)]
        return sorted(set(active_ids + persisted_ids))

    def has_running_steps(self, scope_id: str | None = None) -> bool:
        return bool(self.list_running_steps(scope_id=scope_id))

    def list_created_steps(self, scope_id: str | None = None) -> list[str]:
        return [step.step_id for step in self.store.list_created_steps(scope_id=scope_id)]

    def _mark_step_running(self, step_id: str) -> None:
        started_at = utc_now_iso()

        def mark(step: BaseStep) -> None:
            if step.status is not StepStatus.CREATED:
                raise FlowStepValidationError(f"step {step.step_id} is not created")
            step.status = StepStatus.RUNNING
            step.started_at = started_at

        self.store.update_step_record(step_id, mark)
        with self._step_condition:
            self._step_condition.notify_all()

    def _receipt_matches(self, receipt: StepTerminalReceipt, step: BaseStep) -> bool:
        return (
            receipt.step_id == step.step_id
            and receipt.flow_id == step.flow_id
            and receipt.scope_id == step.scope_id
            and receipt.status in {"completed", "failed"}
        )

    def _validate_terminal_receipt(self, receipt: StepTerminalReceipt, step: BaseStep) -> None:
        if not self._receipt_matches(receipt, step):
            raise FlowStepValidationError(f"invalid terminal receipt for step {step.step_id}")
        latest = self.store.get_step(step.step_id)
        if latest.status not in {StepStatus.COMPLETED, StepStatus.FAILED}:
            raise FlowStepValidationError(f"step {step.step_id} did not reach terminal status")

    def _force_fail_step(self, ctx: StepRunContext, error: BaseStepError) -> StepTerminalReceipt:
        finished_at = utc_now_iso()

        def force_fail(step: BaseStep) -> None:
            step.result = None
            step.error = error
            step.status = StepStatus.FAILED
            step.finished_at = finished_at

        failed = self.store.update_step_record(ctx.step_id, force_fail)
        return StepTerminalReceipt(
            step_id=failed.step_id,
            flow_id=failed.flow_id,
            scope_id=failed.scope_id,
            status="failed",
            error_type=error.error_type,
            finished_at=failed.finished_at or finished_at,
        )
