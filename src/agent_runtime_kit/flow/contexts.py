from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel

from agent_runtime_kit.runtime import RuntimeContext

from .models import (
    BaseFlow,
    BaseFlowError,
    BaseFlowResult,
    BaseStep,
    BaseStepError,
    BaseStepResult,
    BaseSubmission,
    FlowStatus,
    FlowRequest,
    FlowStepValidationError,
    StepStatus,
    StepTerminalReceipt,
    utc_now_iso,
)
from .store import FlowStepMutationSession, FlowStepStore


@dataclass(frozen=True)
class FlowBuildContext(RuntimeContext):
    request: FlowRequest
    params: BaseModel
    flow_id: str
    scope_id: str
    parent_flow_id: str | None
    parent_dispatch_step_id: str | None


@dataclass(frozen=True)
class FlowReadContext(RuntimeContext):
    flow: BaseFlow


@dataclass(frozen=True)
class FlowContext(RuntimeContext):
    flow: BaseFlow
    tx: FlowStepMutationSession

    def create_step(self, step: BaseStep) -> str:
        if step.flow_id != self.flow.flow_id:
            raise FlowStepValidationError(
                f"step {step.step_id} belongs to flow {step.flow_id}, expected {self.flow.flow_id}"
            )
        if step.scope_id != self.flow.scope_id:
            raise FlowStepValidationError(
                f"step {step.step_id} scope {step.scope_id}, expected {self.flow.scope_id}"
            )
        step_id = self.tx.add_step(step)
        if step_id not in self.flow.step_ids:
            self.flow.step_ids.append(step_id)
        self.flow.current_step_id = step_id
        return step_id

    def set_flow_result(self, result: BaseFlowResult) -> None:
        now = utc_now_iso()
        self.flow.result = result
        self.flow.error = None
        self.flow.status = FlowStatus.COMPLETED
        self.flow.finished_at = now
        self.flow.updated_at = now

    def set_flow_waiting(self) -> None:
        self.flow.status = FlowStatus.WAITING
        self.flow.updated_at = utc_now_iso()


@dataclass(frozen=True)
class FlowStepContext(RuntimeContext):
    flow: BaseFlow
    step: BaseStep
    tx: FlowStepMutationSession


@dataclass(frozen=True)
class StableStepTerminalContext(RuntimeContext):
    flow: BaseFlow
    step: BaseStep


@dataclass(frozen=True)
class StepRunContext(RuntimeContext):
    step_id: str
    flow_id: str
    scope_id: str

    def load_step(self) -> BaseStep:
        step = self._store().get_step(self.step_id)
        self._validate_step_identity(step)
        return step

    def update_step(self, mutator: Callable[[BaseStep], None]) -> BaseStep:
        def checked_mutator(step: BaseStep) -> None:
            self._validate_step_identity(step)
            mutator(step)

        return self._store().update_step_record(self.step_id, checked_mutator)

    def update_flow(self, mutator: Callable[[BaseFlow], None]) -> BaseFlow:
        def checked_mutator(flow: BaseFlow) -> None:
            if flow.flow_id != self.flow_id:
                raise FlowStepValidationError(f"loaded flow {flow.flow_id}, expected {self.flow_id}")
            if flow.scope_id != self.scope_id:
                raise FlowStepValidationError(f"loaded flow scope {flow.scope_id}, expected {self.scope_id}")
            mutator(flow)

        return self._store().update_flow_record(self.flow_id, checked_mutator)

    def accept_step_submission(
        self,
        submission: BaseSubmission,
        *,
        expected_agent_id: str | None = None,
    ) -> BaseStep:
        def write_submission(step: BaseStep) -> None:
            self._validate_step_identity(step)
            if step.status is not StepStatus.RUNNING:
                raise FlowStepValidationError(f"step {step.step_id} is not running")
            if step.submission is not None:
                raise FlowStepValidationError(f"step {step.step_id} already has accepted submission")
            if expected_agent_id is not None and submission.submitted_by_agent_id != expected_agent_id:
                raise FlowStepValidationError(
                    f"submission agent {submission.submitted_by_agent_id!r} does not match expected "
                    f"agent {expected_agent_id!r}"
                )
            step.validate_submission(self, submission)
            step.submission = submission

        return self._store().update_step_record(self.step_id, write_submission)

    def complete_step(self, result: BaseStepResult) -> StepTerminalReceipt:
        finished_at = utc_now_iso()

        def write_result(step: BaseStep) -> None:
            self._validate_step_identity(step)
            if step.status is not StepStatus.RUNNING:
                raise FlowStepValidationError(f"step {step.step_id} is not running")
            if step.result is not None or step.error is not None:
                raise FlowStepValidationError(f"step {step.step_id} is already terminal")
            step.result = result
            step.error = None
            step.status = StepStatus.COMPLETED
            step.finished_at = finished_at

        updated = self._store().update_step_record(self.step_id, write_result)
        return StepTerminalReceipt(
            step_id=updated.step_id,
            flow_id=updated.flow_id,
            scope_id=updated.scope_id,
            status="completed",
            result_type=result.result_type,
            finished_at=updated.finished_at or finished_at,
        )

    def fail_step(self, error: BaseStepError) -> StepTerminalReceipt:
        finished_at = utc_now_iso()

        def write_error(step: BaseStep) -> None:
            self._validate_step_identity(step)
            if step.status in {StepStatus.COMPLETED, StepStatus.FAILED}:
                raise FlowStepValidationError(f"step {step.step_id} is already terminal")
            if step.result is not None or step.error is not None:
                raise FlowStepValidationError(f"step {step.step_id} is already terminal")
            step.error = error
            step.result = None
            step.status = StepStatus.FAILED
            step.finished_at = finished_at

        updated = self._store().update_step_record(self.step_id, write_error)
        return StepTerminalReceipt(
            step_id=updated.step_id,
            flow_id=updated.flow_id,
            scope_id=updated.scope_id,
            status="failed",
            error_type=error.error_type,
            finished_at=updated.finished_at or finished_at,
        )

    def _store(self) -> FlowStepStore:
        flow_service = self.ark.flow_service
        store = getattr(flow_service, "store", None)
        if not isinstance(store, FlowStepStore):
            raise FlowStepValidationError("ctx.ark.flow_service.store is not available")
        return store

    def _validate_step_identity(self, step: BaseStep) -> None:
        if step.step_id != self.step_id:
            raise FlowStepValidationError(f"loaded step {step.step_id}, expected {self.step_id}")
        if step.flow_id != self.flow_id:
            raise FlowStepValidationError(f"loaded step flow {step.flow_id}, expected {self.flow_id}")
        if step.scope_id != self.scope_id:
            raise FlowStepValidationError(f"loaded step scope {step.scope_id}, expected {self.scope_id}")
