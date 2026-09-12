from __future__ import annotations

import uuid
import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from time import monotonic, sleep
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict

from agent_runtime_kit.agent.models import (
    AgentCompletionCheckError,
    AgentContextCompactionTimeout,
    AgentContextMaintenanceBlocked,
    AgentContextMaintenanceError,
    AgentProviderFailure,
    AgentProviderTurnFailed,
    AgentProviderUnavailable,
)
from agent_runtime_kit.agent.context import AgentContextMaintenanceView
from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePausedError

from .contexts import FlowBuildContext, FlowContext, FlowReadContext, FlowStepContext, StableStepTerminalContext, StepRunContext
from .models import (
    AgentStepRecoveryReceipt,
    AgentStepRecoveryView,
    BaseFlow,
    BaseFlowError,
    BaseStep,
    BaseStepError,
    BaseStepResult,
    BoundAgentReplacementReceipt,
    FlowRequest,
    FlowStatus,
    FlowStepValidationError,
    LostStepSubmissionFinalizeUnavailableError,
    SetAgentStepOperatorInstructionReceipt,
    StepStatus,
    StepSuspensionReceipt,
    StepTerminalReceipt,
    StepTerminalWaitResult,
    utc_now_iso,
)
from .registry import FlowTypeRegistry, StepTypeRegistry
from .standard_steps.agent_step import AgentStep, AgentStepState
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


@dataclass(frozen=True)
class _AgentStepRecoveryAssessment:
    view: AgentStepRecoveryView
    step: AgentStep
    flow: BaseFlow
    agent: object | None
    provider_turn: object | None
    agent_audit_classification: str | None


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

    def set_agent_step_operator_instruction(
        self,
        *,
        step_id: str,
        expected_step_updated_at: str,
        expected_flow_updated_at: str,
        instruction: str | None,
    ) -> SetAgentStepOperatorInstructionReceipt:
        if instruction is not None and not instruction.strip():
            raise FlowStepValidationError("AgentStep operator instruction must not be blank")
        initial = self.store.get_step(step_id)
        pause_controller = self.ark.pause_controller
        agent_service = self.ark.agent_service
        if pause_controller is None or agent_service is None:
            raise FlowStepValidationError(
                "AgentStep operator instruction requires PauseController and AgentService"
            )
        agent_guard = getattr(agent_service, "hold_agent_boundary", nullcontext)
        with pause_controller.hold_paused(initial.scope_id), agent_guard(), self.lock:
            with self.store.edit_session(initial.scope_id) as tx:
                step = tx.load_step_for_update(step_id)
                flow = self.store.get_flow(step.flow_id)
                if not isinstance(step, AgentStep) or not isinstance(step.state, AgentStepState):
                    raise FlowStepValidationError(f"step is not an AgentStep: {step_id}")
                if step.scope_id != flow.scope_id:
                    raise FlowStepValidationError("AgentStep and owning Flow scope do not match")
                if step.updated_at != expected_step_updated_at:
                    raise FlowStepValidationError("AgentStep updated_at changed")
                if flow.updated_at != expected_flow_updated_at:
                    raise FlowStepValidationError("owning Flow updated_at changed")
                if flow.current_step_id != step_id:
                    raise FlowStepValidationError("AgentStep is not the owning Flow current step")
                if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                    raise FlowStepValidationError("terminal Flow AgentStep instruction cannot be changed")
                if step.status is not StepStatus.CREATED:
                    raise FlowStepValidationError("AgentStep instruction can only be changed before start")
                if any(
                    value is not None
                    for value in (
                        step.submission,
                        step.result,
                        step.error,
                        step.started_at,
                        step.finished_at,
                    )
                ):
                    raise FlowStepValidationError("AgentStep has already started or produced execution truth")
                self._assert_scope_quiescent(step.scope_id)
                if not pause_controller.is_paused(step.scope_id):
                    raise FlowStepValidationError("runtime pause changed during AgentStep instruction update")
                instruction_before = step.state.operator_instruction
                step_updated_at_before = step.updated_at
                flow_updated_at = flow.updated_at
                step.state.operator_instruction = instruction
            return SetAgentStepOperatorInstructionReceipt(
                step_id=step.step_id,
                flow_id=step.flow_id,
                scope_id=step.scope_id,
                instruction_before=instruction_before,
                instruction_after=instruction,
                instruction_present=instruction is not None,
                step_updated_at_before=step_updated_at_before,
                step_updated_at_after=step.updated_at,
                flow_updated_at_before=flow_updated_at,
                flow_updated_at_after=flow_updated_at,
                summary=(
                    "Set AgentStep operator instruction."
                    if instruction is not None
                    else "Cleared AgentStep operator instruction."
                ),
            )

    def replace_bound_agent(
        self,
        *,
        flow_id: str,
        role: str,
        expected_agent_id: str,
        replacement_mode: Literal["fresh", "fork_current"],
        created_step_id: str | None = None,
        boundary_mutator: Callable[[BaseFlow, AgentStep | None], None] | None = None,
    ) -> BoundAgentReplacementReceipt:
        flow = self.store.get_flow(flow_id)
        pause_controller = self.ark.pause_controller
        agent_service = self.ark.agent_service
        if pause_controller is None or agent_service is None:
            raise FlowStepValidationError("binding replacement requires PauseController and AgentService")
        if flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
            raise FlowStepValidationError("terminal Flow binding cannot be replaced")
        agent_guard = getattr(agent_service, "hold_agent_boundary", nullcontext)
        with pause_controller.hold_paused(flow.scope_id), agent_guard(), self.lock:
            self._assert_scope_quiescent(flow.scope_id)
            current = self.store.get_flow(flow_id)
            if current.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                raise FlowStepValidationError("terminal Flow binding cannot be replaced")
            if current.agent_bindings.get(role) != expected_agent_id:
                raise FlowStepValidationError(f"flow Agent binding changed for role {role!r}")
            source_agent = agent_service.get_agent(expected_agent_id)
            if source_agent.scope_id != current.scope_id or source_agent.status == "running":
                raise FlowStepValidationError("bound Agent is not a quiescent member of the Flow scope")
            if replacement_mode == "fresh":
                replacement_agent = agent_service.create_agent(
                    current.scope_id,
                    source_agent.agent_type,
                    provider_type=source_agent.provider_type,
                    home_id=source_agent.home_id,
                )
            else:
                replacement_agent = agent_service.fork_agent_for_recovery(
                    expected_agent_id,
                    target_scope_id=current.scope_id,
                )

            with self.store.edit_session(current.scope_id) as tx:
                working_flow = tx.load_flow_for_update(flow_id)
                if working_flow.status in {FlowStatus.COMPLETED, FlowStatus.FAILED}:
                    raise FlowStepValidationError("terminal Flow binding cannot be replaced")
                if working_flow.agent_bindings.get(role) != expected_agent_id:
                    raise FlowStepValidationError(f"flow Agent binding changed for role {role!r}")
                working_step: AgentStep | None = None
                if created_step_id is not None:
                    candidate = tx.load_step_for_update(created_step_id)
                    if (
                        not isinstance(candidate, AgentStep)
                        or candidate.flow_id != working_flow.flow_id
                        or working_flow.current_step_id != candidate.step_id
                        or candidate.status is not StepStatus.CREATED
                    ):
                        raise FlowStepValidationError("created AgentStep replacement boundary changed")
                    working_step = candidate
                if boundary_mutator is not None:
                    boundary_mutator(working_flow, working_step)
                self._assert_scope_quiescent(current.scope_id)
                if not pause_controller.is_paused(current.scope_id):
                    raise FlowStepValidationError("runtime pause changed during binding replacement")
                working_flow.agent_bindings.by_role[role] = replacement_agent.agent_id
                if working_step is not None:
                    working_step.agent_bindings.by_role[role] = replacement_agent.agent_id

            return BoundAgentReplacementReceipt(
                flow_id=current.flow_id,
                role=role,
                previous_agent_id=expected_agent_id,
                replacement_agent_id=replacement_agent.agent_id,
                replacement_mode=replacement_mode,
                step_id=created_step_id,
            )

    def inspect_agent_step_recovery(self, step_id: str) -> AgentStepRecoveryView:
        return self._assess_agent_step_recovery(step_id).view

    def inspect_agent_step_context_maintenance(
        self,
        step_id: str,
    ) -> AgentContextMaintenanceView | None:
        step = self.store.get_step(step_id)
        if not isinstance(step, AgentStep) or not isinstance(step.state, AgentStepState):
            raise FlowStepValidationError(f"step is not an AgentStep: {step_id}")
        flow = self.store.get_flow(step.flow_id)
        role = step.state.agent_role
        agent_id = step.agent_bindings.get(role) or flow.agent_bindings.get(role)
        if agent_id is None:
            return None
        agent_service = self.ark.agent_service
        inspect_maintenance = getattr(agent_service, "inspect_agent_context_maintenance", None)
        if not callable(inspect_maintenance):
            return None
        return inspect_maintenance(agent_id)

    def reconcile_agent_step_context_maintenance(
        self,
        *,
        step_id: str,
        expected_reconciliation_token: str,
    ) -> AgentContextMaintenanceView:
        step, _, agent_id = self._agent_step_context_target(
            step_id,
            require_suspended_current=True,
        )
        pause_controller = self.ark.pause_controller
        agent_service = self.ark.agent_service
        if pause_controller is None or agent_service is None:
            raise FlowStepValidationError(
                "AgentStep context reconciliation requires PauseController and AgentService"
            )
        agent_guard = getattr(agent_service, "hold_agent_boundary", nullcontext)
        with pause_controller.hold_paused(step.scope_id), agent_guard(), self.lock:
            if not pause_controller.is_paused(None):
                raise FlowStepValidationError(
                    "AgentStep context reconciliation requires global runtime pause"
                )
            current_step, _, current_agent_id = self._agent_step_context_target(
                step_id,
                require_suspended_current=True,
            )
            if current_agent_id != agent_id:
                raise FlowStepValidationError(
                    "AgentStep bound Agent changed during context reconciliation"
                )
            self._assert_scope_quiescent(current_step.scope_id)
            ctx = StepRunContext(
                ark=self.ark,
                app=self.app,
                step_id=current_step.step_id,
                flow_id=current_step.flow_id,
                scope_id=current_step.scope_id,
            )
            reconcile = getattr(agent_service, "reconcile_agent_context_maintenance", None)
            if not callable(reconcile):
                raise FlowStepValidationError("AgentService context reconciliation is unavailable")
            reconcile(
                current_agent_id,
                expected_reconciliation_token=expected_reconciliation_token,
                env=current_step.build_agent_env(ctx, current_agent_id),
                workdir=current_step.resolve_workdir(ctx, current_agent_id),
            )
            inspected = agent_service.inspect_agent_context_maintenance(current_agent_id)
            if inspected is None:
                raise FlowStepValidationError(
                    "Agent context maintenance journal disappeared during reconciliation"
                )
            return inspected

    def _agent_step_context_target(
        self,
        step_id: str,
        *,
        require_suspended_current: bool = False,
    ) -> tuple[AgentStep, BaseFlow, str]:
        step = self.store.get_step(step_id)
        if not isinstance(step, AgentStep) or not isinstance(step.state, AgentStepState):
            raise FlowStepValidationError(f"step is not an AgentStep: {step_id}")
        flow = self.store.get_flow(step.flow_id)
        if require_suspended_current and (
            step.status is not StepStatus.SUSPENDED
            or step.submission is not None
            or flow.status is not FlowStatus.RUNNING
            or flow.current_step_id != step.step_id
        ):
            raise FlowStepValidationError(
                "AgentStep is not the current resumable suspended Step"
            )
        role = step.state.agent_role
        agent_id = step.agent_bindings.get(role) or flow.agent_bindings.get(role)
        if agent_id is None:
            raise FlowStepValidationError("AgentStep has no bound Agent for context maintenance")
        return step, flow, agent_id

    def recover_agent_step(
        self,
        *,
        step_id: str,
        expected_status: StepStatus,
        expected_recovery_token: str,
        action: Literal[
            "restart",
            "resume_suspended",
            "finalize_submission",
            "settle_runner_lost",
        ],
        agent_mode: Literal["auto", "reuse", "fresh", "fork_current"] = "auto",
        boundary_mutator: Callable[[BaseFlow, AgentStep, AgentStep], None] | None = None,
        boundary_compensator: Callable[[], None] | None = None,
    ) -> AgentStepRecoveryReceipt:
        initial = self._assess_agent_step_recovery(step_id)
        if initial.view.step_status is not expected_status:
            raise FlowStepValidationError("AgentStep recovery status changed")
        if initial.view.recovery_token != expected_recovery_token:
            raise FlowStepValidationError("AgentStep recovery token changed")
        if action not in initial.view.available_actions:
            if action == "finalize_submission" and initial.step.submission is not None:
                raise LostStepSubmissionFinalizeUnavailableError(step_id)
            raise FlowStepValidationError(f"AgentStep recovery action is not available: {action}")
        if action in {"finalize_submission", "settle_runner_lost"} and agent_mode != "auto":
            raise FlowStepValidationError(f"{action} requires agent_mode=auto")

        pause_controller = self.ark.pause_controller
        if pause_controller is None:
            raise FlowStepValidationError("AgentStep recovery requires PauseController")
        boundary_applied = False
        ark_committed = False
        try:
            agent_service = self.ark.agent_service
            if agent_service is None:
                raise FlowStepValidationError("AgentStep recovery requires AgentService")
            agent_guard = getattr(agent_service, "hold_agent_boundary", nullcontext)
            with pause_controller.hold_paused(initial.step.scope_id), agent_guard(), self.lock:
                current = self._assess_agent_step_recovery(step_id)
                if current.view.recovery_token != expected_recovery_token:
                    raise FlowStepValidationError("AgentStep recovery identity changed")
                self._assert_scope_quiescent(
                    current.step.scope_id,
                    allowed_lost_step_id=(step_id if expected_status is StepStatus.RUNNING else None),
                    allowed_agent_id=(
                        getattr(current.agent, "agent_id", None)
                        if expected_status is StepStatus.RUNNING
                        else None
                    ),
                )
                if action == "finalize_submission":
                    current = self._repair_lost_agent_if_needed(current)
                    return self._finalize_lost_submission(current)
                if action == "settle_runner_lost":
                    current = self._repair_lost_agent_if_needed(current)
                    return self._settle_lost_runner(current)

                context_maintenance_unresolved = (
                    self._recovery_context_maintenance_unresolved(
                        current,
                        agent_mode=agent_mode,
                    )
                )
                if context_maintenance_unresolved and agent_mode in {
                    "reuse",
                    "fork_current",
                }:
                    raise FlowStepValidationError(
                        "bound Agent context maintenance is unresolved"
                    )
                current = self._repair_lost_agent_if_needed(current)

                replacement_agent_id, agent_reused = self._select_recovery_agent(
                    current,
                    agent_mode=agent_mode,
                    context_maintenance_unresolved=context_maintenance_unresolved,
                )
                replacement = self._build_replacement_step(
                    current.step,
                    replacement_agent_id=replacement_agent_id,
                )
                before = current.flow
                with self.store.edit_session(current.step.scope_id) as tx:
                    source = (
                        tx.load_step_for_update(step_id)
                        if expected_status is StepStatus.RUNNING
                        else current.step.model_copy(deep=True)
                    )
                    flow = tx.load_flow_for_update(before.flow_id)
                    if source.model_dump(mode="json") != current.step.model_dump(mode="json"):
                        raise FlowStepValidationError("source AgentStep changed during recovery")
                    if flow.model_dump(mode="json") != current.flow.model_dump(mode="json"):
                        raise FlowStepValidationError("owning Flow changed during recovery")
                    if expected_status is StepStatus.RUNNING:
                        source.error = BaseStepError(
                            error_type="runner_lost",
                            message="persisted running AgentStep has no active runner",
                        )
                        source.status = StepStatus.FAILED
                        source.finished_at = utc_now_iso()
                    if boundary_mutator is not None:
                        boundary_applied = True
                        boundary_mutator(flow, source, replacement)
                    rechecked = self._assess_agent_step_recovery(step_id)
                    expected_after_repair = current.view.recovery_token
                    if rechecked.view.recovery_token != expected_after_repair:
                        raise FlowStepValidationError("AgentStep recovery identity changed before commit")
                    self._assert_scope_quiescent(
                        current.step.scope_id,
                        allowed_lost_step_id=(step_id if expected_status is StepStatus.RUNNING else None),
                        allowed_agent_id=(
                            getattr(current.agent, "agent_id", None)
                            if expected_status is StepStatus.RUNNING
                            else None
                        ),
                    )
                    tx.add_step(replacement)
                    flow.step_ids.append(replacement.step_id)
                    flow.current_step_id = replacement.step_id
                    flow.status = FlowStatus.RUNNING
                    flow.error = None
                    flow.result = None
                    flow.finished_at = None
                    role = current.step.state.agent_role
                    if current.step.state.bind_created_agent_to == "flow" or flow.agent_bindings.get(role):
                        flow.agent_bindings.by_role[role] = replacement_agent_id

                ark_committed = True
                after = self.store.get_flow(before.flow_id)
                schedule_service = self.ark.schedule_service
                enqueued = schedule_service is not None
                if schedule_service is not None:
                    schedule_service.enqueue_step(replacement.step_id)
                return self._recovery_receipt(
                    current,
                    action=action,
                    agent_mode=agent_mode,
                    replacement_step_id=replacement.step_id,
                    replacement_agent_id=replacement_agent_id,
                    agent_reused=agent_reused,
                    submission_disposition="not_present",
                    after=after,
                    enqueued=enqueued,
                )
        except Exception:
            if boundary_applied and not ark_committed and boundary_compensator is not None:
                boundary_compensator()
            raise

    def _assess_agent_step_recovery(self, step_id: str) -> _AgentStepRecoveryAssessment:
        step = self.store.get_step(step_id)
        if not isinstance(step, AgentStep) or not isinstance(step.state, AgentStepState):
            raise FlowStepValidationError(f"step is not an AgentStep: {step_id}")
        flow = self.store.get_flow(step.flow_id)
        step_service = self.ark.step_service
        active = bool(step_service is not None and step_id in step_service.active_steps)
        if active:
            runner_state = "active"
        elif step.status is StepStatus.CREATED:
            runner_state = "not_started"
        elif step.status is StepStatus.RUNNING:
            runner_state = "lost"
        else:
            runner_state = "settled"

        role = step.state.agent_role
        agent_id = step.agent_bindings.get(role) or flow.agent_bindings.get(role)
        agent_service = self.ark.agent_service
        agent = None
        audit_classification = None
        provider_turn = None
        if agent_id is not None and agent_service is not None:
            try:
                agent = agent_service.get_agent(agent_id)
            except (KeyError, ValueError):
                agent = None
            if agent is not None and agent.status == "running":
                audit = next(
                    (item for item in agent_service.audit_running_agents(step.scope_id) if item.agent_id == agent_id),
                    None,
                )
                audit_classification = None if audit is None else audit.classification
            elif agent is not None:
                audit_classification = "not_running"
            if agent is not None:
                try:
                    provider_turn = agent_service.query_turn(
                        agent_id,
                        latest=True,
                        prepare_session_access=False,
                    )
                except Exception:  # Provider query absence is part of the read-only assessment.
                    provider_turn = None

        latest = bool(flow.step_ids and flow.step_ids[-1] == step_id)
        actions: list[str] = []
        if not active and latest:
            if (
                step.status is StepStatus.FAILED
                and step.submission is None
                and flow.status is FlowStatus.FAILED
                and flow.current_step_id is None
            ):
                actions.append("restart")
            elif (
                step.status is StepStatus.SUSPENDED
                and step.submission is None
                and flow.status is FlowStatus.RUNNING
                and flow.current_step_id == step_id
            ):
                actions.append("resume_suspended")
            elif (
                step.status is StepStatus.RUNNING
                and flow.status is FlowStatus.RUNNING
                and flow.current_step_id == step_id
                and audit_classification != "healthy_running"
            ):
                if step.submission is None:
                    actions.extend(["restart", "settle_runner_lost"])
                elif self._can_finalize_submission(step, provider_turn, agent):
                    actions.append("finalize_submission")

        token_payload = {
            "step": step.model_dump(mode="json"),
            "flow": flow.model_dump(mode="json"),
            "current_latest": latest,
            "runner_state": runner_state,
            "agent": _jsonable(agent),
            "agent_audit_classification": audit_classification,
            "provider_turn": _safe_provider_turn_identity(provider_turn),
        }
        token = hashlib.sha256(
            json.dumps(token_payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        view = AgentStepRecoveryView(
            step_id=step.step_id,
            flow_id=flow.flow_id,
            step_status=step.status,
            runner_state=runner_state,
            available_actions=actions,
            recovery_token=token,
        )
        return _AgentStepRecoveryAssessment(
            view=view,
            step=step,
            flow=flow,
            agent=agent,
            provider_turn=provider_turn,
            agent_audit_classification=audit_classification,
        )

    def _repair_lost_agent_if_needed(
        self,
        assessment: _AgentStepRecoveryAssessment,
    ) -> _AgentStepRecoveryAssessment:
        agent = assessment.agent
        if assessment.step.status is not StepStatus.RUNNING or getattr(agent, "status", None) != "running":
            return assessment
        if assessment.agent_audit_classification == "healthy_running":
            raise FlowStepValidationError("healthy running Agent cannot be recovered")
        agent_service = self.ark.agent_service
        before_agent = _jsonable(agent)
        before_provider = _safe_provider_turn_identity(assessment.provider_turn)
        agent_service.repair_running_agent(
            agent.agent_id,
            expected_scope_id=agent.scope_id,
            expected_session_id=(agent.session_locator.session_id if agent.session_locator is not None else None),
            expected_artifact_ref=(
                agent.artifact_locator.native_primary_ref if agent.artifact_locator is not None else None
            ),
            action="mark_idle",
            dry_run=False,
        )
        repaired = self._assess_agent_step_recovery(assessment.step.step_id)
        if getattr(repaired.agent, "status", None) != "idle" or repaired.agent_audit_classification != "not_running":
            raise FlowStepValidationError("Agent repair did not produce a not-running idle Agent")
        after_agent = _jsonable(repaired.agent)
        if isinstance(before_agent, dict) and isinstance(after_agent, dict):
            for key in ("status", "updated_at"):
                before_agent.pop(key, None)
                after_agent.pop(key, None)
        if before_agent != after_agent or before_provider != _safe_provider_turn_identity(repaired.provider_turn):
            raise FlowStepValidationError("provider identity changed during Agent repair")
        return repaired

    def _select_recovery_agent(
        self,
        assessment: _AgentStepRecoveryAssessment,
        *,
        agent_mode: Literal["auto", "reuse", "fresh", "fork_current"],
        context_maintenance_unresolved: bool,
    ) -> tuple[str, bool]:
        agent_service = self.ark.agent_service
        if agent_service is None:
            raise FlowStepValidationError("AgentStep recovery requires AgentService")
        source_agent = assessment.agent
        reusable = (
            source_agent is not None
            and source_agent.status == "idle"
            and not context_maintenance_unresolved
        )
        resolved_mode = "reuse" if agent_mode == "auto" and reusable else agent_mode
        if resolved_mode == "auto":
            resolved_mode = "fresh"
        if resolved_mode == "reuse":
            if not reusable:
                raise FlowStepValidationError("bound Agent is not safely reusable")
            return source_agent.agent_id, True
        if resolved_mode == "fork_current":
            if source_agent is None:
                raise FlowStepValidationError("bound Agent is unavailable for recovery fork")
            forked = agent_service.fork_agent_for_recovery(
                source_agent.agent_id,
                target_scope_id=assessment.step.scope_id,
            )
            return forked.agent_id, False
        if source_agent is not None:
            agent_type = source_agent.agent_type
            provider_type = source_agent.provider_type
            home_id = source_agent.home_id
        else:
            state = assessment.step.state
            if state.agent_type is None:
                raise FlowStepValidationError("AgentStep cannot create a fresh Agent without agent_type")
            agent_type = state.agent_type
            provider_type = state.provider_type
            home_id = state.home_id
        created = agent_service.create_agent(
            assessment.step.scope_id,
            agent_type,
            provider_type=provider_type,
            home_id=home_id,
        )
        return created.agent_id, False

    def _recovery_context_maintenance_unresolved(
        self,
        assessment: _AgentStepRecoveryAssessment,
        *,
        agent_mode: Literal["auto", "reuse", "fresh", "fork_current"],
    ) -> bool:
        if agent_mode == "fresh" or assessment.agent is None:
            return False
        agent_service = self.ark.agent_service
        inspect_maintenance = getattr(
            agent_service,
            "inspect_agent_context_maintenance",
            None,
        )
        if not callable(inspect_maintenance):
            return False
        maintenance = inspect_maintenance(assessment.agent.agent_id)
        return bool(maintenance is not None and maintenance.unresolved)

    def _build_replacement_step(
        self,
        source: AgentStep,
        *,
        replacement_agent_id: str,
    ) -> AgentStep:
        replacement = source.model_copy(deep=True)
        now = utc_now_iso()
        replacement.step_id = f"agent_recovery_{uuid.uuid4().hex}"
        replacement.status = StepStatus.CREATED
        replacement.submission = None
        replacement.result = None
        replacement.error = None
        replacement.created_at = now
        replacement.updated_at = now
        replacement.started_at = None
        replacement.finished_at = None
        replacement.state.restart_of_step_id = source.step_id
        replacement.agent_bindings.by_role[source.state.agent_role] = replacement_agent_id
        return replacement

    def _can_finalize_submission(
        self,
        step: AgentStep,
        provider_turn: object | None,
        agent: object | None,
    ) -> bool:
        if not type(step).offline_submission_finalize_supported or provider_turn is None:
            return False
        result = getattr(provider_turn, "result", None)
        status = getattr(result, "status", None)
        status_value = getattr(status, "value", status)
        if status_value != "completed" or agent is None:
            return False
        agent_session = getattr(agent, "session_locator", None)
        agent_turn = getattr(agent, "latest_turn_locator", None)
        result_session = getattr(result, "session_locator", None)
        result_turn = getattr(result, "turn_locator", None)
        view_turn = getattr(provider_turn, "locator", None)
        if agent_session is not None and result_session != agent_session:
            return False
        if agent_turn is not None and (result_turn != agent_turn or view_turn != agent_turn):
            return False
        try:
            built = step.build_result_from_submission(
                StepRunContext(
                    ark=self.ark,
                    app=self.app,
                    step_id=step.step_id,
                    flow_id=step.flow_id,
                    scope_id=step.scope_id,
                ),
                agent.agent_id,
                result,
            )
        except Exception:
            return False
        return isinstance(built, BaseStepResult)

    def _finalize_lost_submission(
        self,
        assessment: _AgentStepRecoveryAssessment,
    ) -> AgentStepRecoveryReceipt:
        if not self._can_finalize_submission(
            assessment.step,
            assessment.provider_turn,
            assessment.agent,
        ):
            raise LostStepSubmissionFinalizeUnavailableError(assessment.step.step_id)
        agent_id = getattr(assessment.agent, "agent_id", None)
        if agent_id is None:
            raise LostStepSubmissionFinalizeUnavailableError(assessment.step.step_id)
        ctx = StepRunContext(
            ark=self.ark,
            app=self.app,
            step_id=assessment.step.step_id,
            flow_id=assessment.flow.flow_id,
            scope_id=assessment.step.scope_id,
        )
        provider_result = getattr(assessment.provider_turn, "result", None)
        result = assessment.step.build_result_from_submission(ctx, agent_id, provider_result)
        rechecked = self._assess_agent_step_recovery(assessment.step.step_id)
        if rechecked.view.recovery_token != assessment.view.recovery_token:
            raise FlowStepValidationError("AgentStep recovery identity changed before completion")
        ctx.complete_step(result)
        self.handle_step_terminal(assessment.step.step_id)
        after = self.store.get_flow(assessment.flow.flow_id)
        return self._recovery_receipt(
            assessment,
            action="finalize_submission",
            agent_mode="auto",
            replacement_step_id=None,
            replacement_agent_id=None,
            agent_reused=False,
            submission_disposition="accepted_finalized",
            after=after,
            enqueued=False,
        )

    def _settle_lost_runner(
        self,
        assessment: _AgentStepRecoveryAssessment,
    ) -> AgentStepRecoveryReceipt:
        ctx = StepRunContext(
            ark=self.ark,
            app=self.app,
            step_id=assessment.step.step_id,
            flow_id=assessment.flow.flow_id,
            scope_id=assessment.step.scope_id,
        )
        ctx.fail_step(
            BaseStepError(
                error_type="runner_lost",
                message="persisted running AgentStep has no active runner",
            )
        )
        self.handle_step_terminal(assessment.step.step_id)
        after = self.store.get_flow(assessment.flow.flow_id)
        return self._recovery_receipt(
            assessment,
            action="settle_runner_lost",
            agent_mode="auto",
            replacement_step_id=None,
            replacement_agent_id=None,
            agent_reused=False,
            submission_disposition="not_present",
            after=after,
            enqueued=False,
        )

    def _recovery_receipt(
        self,
        assessment: _AgentStepRecoveryAssessment,
        *,
        action: str,
        agent_mode: str,
        replacement_step_id: str | None,
        replacement_agent_id: str | None,
        agent_reused: bool,
        submission_disposition: str,
        after: BaseFlow,
        enqueued: bool,
    ) -> AgentStepRecoveryReceipt:
        return AgentStepRecoveryReceipt(
            source_step_id=assessment.step.step_id,
            replacement_step_id=replacement_step_id,
            flow_id=assessment.flow.flow_id,
            scope_id=assessment.step.scope_id,
            action=action,
            previous_status=assessment.step.status.value,
            previous_agent_id=getattr(assessment.agent, "agent_id", None),
            replacement_agent_id=replacement_agent_id,
            agent_mode=agent_mode,
            agent_reused=agent_reused,
            submission_disposition=submission_disposition,
            flow_status_before=assessment.flow.status.value,
            flow_current_step_id_before=assessment.flow.current_step_id,
            flow_updated_at_before=assessment.flow.updated_at,
            flow_status_after=after.status.value,
            flow_current_step_id_after=after.current_step_id,
            flow_updated_at_after=after.updated_at,
            enqueued=enqueued,
        )

    def _assert_scope_quiescent(
        self,
        scope_id: str,
        *,
        allowed_lost_step_id: str | None = None,
        allowed_agent_id: str | None = None,
    ) -> None:
        pause_controller = self.ark.pause_controller
        if pause_controller is None or not pause_controller.is_paused(scope_id):
            raise FlowStepValidationError(f"runtime scope is not paused: {scope_id}")
        step_service = self.ark.step_service
        if step_service is not None:
            running_steps = set(step_service.list_running_steps(scope_id))
            if allowed_lost_step_id is not None:
                running_steps.discard(allowed_lost_step_id)
            if running_steps:
                raise FlowStepValidationError("scope has running Steps: " + ",".join(sorted(running_steps)))
        agent_service = self.ark.agent_service
        if agent_service is not None:
            if hasattr(agent_service, "list_running_agents"):
                running_agents = {agent.agent_id for agent in agent_service.list_running_agents(scope_id)}
            elif hasattr(agent_service, "has_running_agents"):
                if agent_service.has_running_agents(scope_id):
                    raise FlowStepValidationError("scope has running Agents")
                running_agents = set()
            else:
                running_agents = set()
            if allowed_agent_id is not None:
                running_agents.discard(allowed_agent_id)
            if running_agents:
                raise FlowStepValidationError("scope has running Agents: " + ",".join(sorted(running_agents)))
        schedule_service = self.ark.schedule_service
        if schedule_service is not None:
            active = [
                flow_id
                for flow_id in set(getattr(schedule_service, "active_flow_advances", set()))
                if self.store.get_flow(flow_id).scope_id == scope_id
            ]
            if active:
                raise FlowStepValidationError("scope has active Flow advances: " + ",".join(sorted(active)))

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
        initial = self.store.get_flow(flow_id)
        pause_controller = self.ark.pause_controller
        pause_guard = (
            pause_controller.hold_unpaused(initial.scope_id)
            if pause_controller is not None and hasattr(pause_controller, "hold_unpaused")
            else nullcontext()
        )
        with pause_guard, self.lock:
            if not self.prepare_flow_for_advance(flow_id):
                raise FlowStepValidationError(f"flow cannot advance: {flow_id}")
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
        step = self.store.get_step(step_id)
        pause_guard = (
            pause_controller.hold_unpaused(step.scope_id)
            if pause_controller is not None and hasattr(pause_controller, "hold_unpaused")
            else nullcontext()
        )
        outer = bypass_context if bypass_context is not None else nullcontext()
        try:
            with outer, pause_guard, self._step_condition:
                if not self.can_run_step(step_id):
                    raise FlowStepValidationError(f"step cannot run: {step_id}")
                step = self.store.get_step(step_id)
                done_event = Event()
                active = ActiveStepRun(
                    step_id=step.step_id,
                    flow_id=step.flow_id,
                    scope_id=step.scope_id,
                    started_at=utc_now_iso(),
                    done_event=done_event,
                    bypass_pause=bypass_pause,
                )
                self.active_steps[step_id] = active
                self._step_condition.notify_all()
                worker = Thread(target=self._run_step_body, args=(step_id, active), daemon=True)
                active.worker_ref = worker
                worker.start()
                return active
        except RuntimePausedError as exc:
            raise FlowStepValidationError(f"step cannot run: {step_id}") from exc

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
            if isinstance(step_impl, AgentStep):
                receipt = ctx.suspend_step(
                    _agent_step_exception_error(exc),
                    require_no_submission=True,
                )
                if not self._suspension_receipt_matches(receipt, step):
                    raise FlowStepValidationError(
                        f"AgentStep {step_id} did not persist matching suspension evidence"
                    )
                return
            receipt = ctx.fail_step(
                BaseStepError(
                    error_type="step_run_exception",
                    message=str(exc) or type(exc).__name__,
                    details={"exception_type": type(exc).__name__},
                )
            )
        else:
            if isinstance(receipt, StepSuspensionReceipt):
                if not self._suspension_receipt_matches(receipt, step):
                    receipt = self._force_fail_step(
                        ctx,
                        BaseStepError(
                            error_type="invalid_suspension_receipt",
                            message=f"step {step_id} returned invalid suspension receipt",
                        ),
                    )
                else:
                    latest = self.store.get_step(step_id)
                    if latest.status is StepStatus.SUSPENDED:
                        return
                    receipt = self._force_fail_step(
                        ctx,
                        BaseStepError(
                            error_type="invalid_suspension_receipt",
                            message=f"step {step_id} did not persist suspended state",
                        ),
                    )
            elif not isinstance(receipt, StepTerminalReceipt):
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
                if step.status is StepStatus.SUSPENDED and not active:
                    return StepTerminalWaitResult(
                        step=step,
                        terminal=False,
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

    def _suspension_receipt_matches(self, receipt: StepSuspensionReceipt, step: BaseStep) -> bool:
        latest = self.store.get_step(step.step_id)
        return (
            receipt.step_id == step.step_id
            and receipt.flow_id == step.flow_id
            and receipt.scope_id == step.scope_id
            and receipt.status == "suspended"
            and latest.status is StepStatus.SUSPENDED
            and latest.result is None
            and latest.error is not None
            and latest.error.error_type == receipt.error_type
            and latest.finished_at == receipt.finished_at
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

def _agent_step_exception_error(exc: Exception) -> BaseStepError:
    if isinstance(exc, AgentProviderFailure):
        if isinstance(exc, AgentProviderTurnFailed):
            error_type = "agent_provider_turn_failed"
        elif isinstance(exc, AgentProviderUnavailable):
            error_type = "agent_provider_unavailable"
        else:
            error_type = "agent_provider_failure"
        return BaseStepError(
            error_type=error_type,
            message="Agent provider execution suspended before business completion.",
            code=exc.code,
            details={
                "provider_type": exc.provider_type,
                "provider_error_type": exc.provider_error_type,
                "retryable": exc.retryable,
                "operator_action_required": exc.retryable is not True,
                "run_id": exc.run_id,
                "session_id": exc.session_id,
                "turn_id": exc.turn_id,
            },
        )
    if isinstance(exc, AgentContextCompactionTimeout):
        error_type = "agent_context_compaction_timeout"
        message = "Agent context compaction did not reach confirmed completion."
        category = "context_maintenance"
    elif isinstance(exc, AgentContextMaintenanceBlocked):
        error_type = "agent_context_maintenance_blocked"
        message = "Agent context maintenance remains unresolved."
        category = "context_maintenance"
    elif isinstance(exc, AgentContextMaintenanceError):
        error_type = "agent_context_maintenance_error"
        message = "Agent context maintenance interrupted execution."
        category = "context_maintenance"
    elif isinstance(exc, AgentCompletionCheckError):
        error_type = "agent_completion_check_error"
        message = "Agent completion checking interrupted execution."
        category = "completion_check"
    else:
        error_type = "agent_step_unexpected_exception"
        message = "AgentStep execution stopped on an unexpected internal exception."
        category = "unexpected_exception"
    return BaseStepError(
        error_type=error_type,
        message=message,
        details={
            "exception_type": type(exc).__name__,
            "category": category,
            "retryable": None,
            "operator_action_required": True,
        },
    )


def _jsonable(value: object) -> object:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    return str(value)


def _safe_provider_turn_identity(view: object | None) -> object:
    if view is None:
        return None
    result = getattr(view, "result", None)
    locator = getattr(view, "locator", None)
    if result is None:
        return {
            "view_turn_id": getattr(locator, "turn_id", None),
            "result": None,
        }
    error = getattr(result, "error", None)
    session = getattr(result, "session_locator", None)
    turn = getattr(result, "turn_locator", None)
    status = getattr(result, "status", None)
    return {
        "view_turn_id": getattr(locator, "turn_id", None),
        "provider_type": getattr(result, "provider_type", None),
        "status": getattr(status, "value", status),
        "error_type": getattr(error, "error_type", None),
        "error_code": getattr(error, "code", None),
        "error_retryable": getattr(error, "retryable", None),
        "run_id": getattr(result, "run_id", None),
        "started_at": getattr(result, "started_at", None),
        "completed_at": getattr(result, "completed_at", None),
        "session_id": getattr(session, "session_id", None),
        "turn_id": getattr(turn, "turn_id", None),
    }
