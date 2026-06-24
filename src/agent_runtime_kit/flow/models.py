from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

if TYPE_CHECKING:
    from .contexts import FlowContext, FlowReadContext, FlowStepContext, StableStepTerminalContext
    from .rendering import RenderContext


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class FlowStepError(Exception):
    """Base exception for Flow / Step orchestration."""


class FlowStepValidationError(FlowStepError):
    """Raised when Flow / Step truth violates runtime invariants."""


class FlowStepTypeError(FlowStepError):
    """Raised when Flow / Step type lookup or parsing fails."""


class StepStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FlowStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BaseStepState(StrictModel):
    state_type: str = "__base_step_state__"


class BaseStepResult(StrictModel):
    result_type: str
    summary: str | None = None

    def render_for_agent(self, ctx: RenderContext) -> str:
        raise NotImplementedError


class BaseStepError(StrictModel):
    error_type: str
    message: str
    code: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class BaseFlowInput(StrictModel):
    input_type: str
    summary: str | None = None

    def render_for_agent(self, ctx: RenderContext) -> str:
        raise NotImplementedError


class FlowPosition(StrictModel):
    phase: str = "initial"
    round_index: int = 0


class BaseFlowState(StrictModel):
    state_type: str = "__base_flow_state__"
    position: FlowPosition = Field(default_factory=FlowPosition)


class BaseFlowResult(StrictModel):
    result_type: str
    summary: str | None = None

    def render_for_agent(self, ctx: RenderContext) -> str:
        raise NotImplementedError


class BaseFlowError(StrictModel):
    error_type: str
    message: str
    code: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ManualPauseState(StrictModel):
    active: bool = False
    reason: str | None = None
    paused_at: str | None = None


class AgentRoleBindings(StrictModel):
    by_role: dict[str, str] = Field(default_factory=dict)

    def get(self, role: str) -> str | None:
        return self.by_role.get(role)


class FlowRequest(StrictModel):
    flow_type: str
    scope_id: str
    params: dict[str, Any] = Field(default_factory=dict)


DispatchContinuation = Literal["wait_for_callback", "terminal_handoff"]


class BaseSubmission(StrictModel):
    submission_id: str
    submission_type: str
    tool_name: str
    submitted_by_agent_id: str | None = None
    submitted_at: str = Field(default_factory=utc_now_iso)
    summary: str | None = None


class ChildFlowDispatchSubmission(BaseSubmission):
    submission_type: Literal["child_flow_dispatch"] = "child_flow_dispatch"
    requests: list[FlowRequest]
    continuation: DispatchContinuation = "wait_for_callback"


class CreatedChildFlow(StrictModel):
    request_index: int
    child_flow_id: str


class DispatchRequestFailure(StrictModel):
    request_index: int
    error_type: str
    message: str


class StepTerminalReceipt(StrictModel):
    step_id: str
    flow_id: str
    scope_id: str
    status: Literal["completed", "failed"]
    result_type: str | None = None
    error_type: str | None = None
    finished_at: str


class BaseStep(StrictModel):
    Result: ClassVar[type[BaseStepResult]] = BaseStepResult
    Results: ClassVar[dict[str, type[BaseStepResult]]] = {}
    Submission: ClassVar[type[BaseSubmission] | None] = BaseSubmission
    Submissions: ClassVar[dict[str, type[BaseSubmission]]] = {}
    SubmitTools: ClassVar[set[str] | None] = None

    step_id: str
    flow_id: str
    step_type: str
    scope_id: str
    status: StepStatus = StepStatus.CREATED
    state: SerializeAsAny[BaseStepState] = Field(default_factory=BaseStepState)
    submission: SerializeAsAny[BaseSubmission] | None = None
    result: SerializeAsAny[BaseStepResult] | None = None
    error: SerializeAsAny[BaseStepError] | None = None
    agent_bindings: AgentRoleBindings = Field(default_factory=AgentRoleBindings)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None

    def run(self, ctx: Any) -> StepTerminalReceipt:
        raise NotImplementedError

    def validate_submission(self, ctx: Any, submission: BaseSubmission) -> None:
        if type(self).SubmitTools is not None and submission.tool_name not in type(self).SubmitTools:
            raise FlowStepValidationError(
                f"step {self.step_id} does not accept submit tool {submission.tool_name!r}"
            )
        submission_cls = type(self).Submissions.get(submission.submission_type)
        if submission_cls is None:
            submission_cls = type(self).Submission
        if submission_cls is None:
            raise FlowStepValidationError(
                f"step {self.step_id} does not accept submission type {submission.submission_type!r}"
            )
        if not isinstance(submission, submission_cls):
            raise FlowStepValidationError(
                f"step {self.step_id} expected submission {submission_cls.__name__}, "
                f"got {type(submission).__name__}"
            )


class BaseFlow(StrictModel):
    Input: ClassVar[type[BaseFlowInput]] = BaseFlowInput
    Result: ClassVar[type[BaseFlowResult]] = BaseFlowResult
    Results: ClassVar[dict[str, type[BaseFlowResult]]] = {}

    flow_id: str
    flow_type: str
    scope_id: str
    status: FlowStatus = FlowStatus.CREATED
    input: SerializeAsAny[BaseFlowInput] | None = None
    state: SerializeAsAny[BaseFlowState] = Field(default_factory=BaseFlowState)
    result: SerializeAsAny[BaseFlowResult] | None = None
    error: SerializeAsAny[BaseFlowError] | None = None
    step_ids: list[str] = Field(default_factory=list)
    current_step_id: str | None = None
    parent_flow_id: str | None = None
    parent_dispatch_step_id: str | None = None
    agent_bindings: AgentRoleBindings = Field(default_factory=AgentRoleBindings)
    manual_pause: ManualPauseState = Field(default_factory=ManualPauseState)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
    started_at: str | None = None
    finished_at: str | None = None

    def can_exit_waiting(self, ctx: FlowReadContext) -> bool:
        return False

    def on_exit_waiting(self, ctx: FlowContext) -> None:
        self.status = FlowStatus.RUNNING
        self.updated_at = utc_now_iso()

    def create_next_step(self, ctx: FlowContext) -> str | None:
        return None

    def on_step_terminal(self, ctx: FlowStepContext) -> None:
        if self.current_step_id == ctx.step.step_id:
            self.current_step_id = None
        if self.result is not None:
            self.status = FlowStatus.COMPLETED
            if self.finished_at is None:
                self.finished_at = utc_now_iso()
        elif self.error is not None:
            self.status = FlowStatus.FAILED
            if self.finished_at is None:
                self.finished_at = utc_now_iso()
        elif self.manual_pause.active:
            self.status = FlowStatus.WAITING
        else:
            self.status = FlowStatus.RUNNING
        self.updated_at = utc_now_iso()

    def after_step_terminal_stable(self, ctx: StableStepTerminalContext) -> None:
        return None
