from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from agent_runtime_kit.agent.context import AgentContextMaintenancePolicy
from agent_runtime_kit.flow.contexts import StepRunContext
from agent_runtime_kit.flow.models import (
    BaseStep,
    BaseStepResult,
    BaseStepState,
    BaseSubmission,
    ChildFlowDispatchSubmission,
    FlowStatus,
    FlowStepValidationError,
    StepTerminalReceipt,
)
from agent_runtime_kit.flow.rendering import RenderContext


class AgentStepState(BaseStepState):
    state_type: str = "agent_step"
    agent_role: str
    agent_type: str | None = None
    cli_type: str | None = None
    home_id: str | None = None
    create_agent_if_missing: bool = False
    bind_created_agent_to: Literal["step", "flow"] = "step"
    variables: dict[str, Any] = Field(default_factory=dict)
    prompt_mode: Literal["initial", "callback"] = "initial"
    prompt_override: str | None = None
    continue_prompt_override: str | None = None
    followup_of_step_id: str | None = None
    callback_dispatch_step_id: str | None = None
    max_auto_continue_turns: int = 2
    require_submission: bool = True
    env_overrides: dict[str, str] = Field(default_factory=dict)
    workdir_override: str | None = None
    agent_wait_timeout_s: float | None = None


class AgentStepCompletionDecision(BaseModel):
    complete: bool
    reason: str | None = None
    continue_prompt: str | None = None


class AgentStepResult(BaseStepResult):
    result_type: str = "agent_step"
    outcome: Literal["submitted", "incomplete"]
    agent_id: str | None = None
    submission_id: str | None = None
    submission_type: str | None = None
    attempts: int = 0
    reason: str | None = None


class AgentStepSubmissionResult(AgentStepResult):
    result_type: str = "agent_step_submission"
    outcome: Literal["submitted"] = "submitted"


class AgentStepIncompleteResult(AgentStepResult):
    result_type: str = "agent_step_incomplete"
    outcome: Literal["incomplete"] = "incomplete"


class AgentStep(BaseStep):
    step_type: ClassVar[str] = "agent_step"
    State: ClassVar[type[BaseStepState]] = AgentStepState
    Result: ClassVar[type[BaseStepResult]] = AgentStepResult
    Results: ClassVar[dict[str, type[BaseStepResult]]] = {
        "agent_step": AgentStepResult,
        "agent_step_submission": AgentStepSubmissionResult,
        "agent_step_incomplete": AgentStepIncompleteResult,
    }
    Submission: ClassVar[type[BaseSubmission] | None] = BaseSubmission
    Submissions: ClassVar[dict[str, type[BaseSubmission]]] = {
        "child_flow_dispatch": ChildFlowDispatchSubmission,
    }

    def validate_submission(self, ctx: StepRunContext, submission: BaseSubmission) -> None:
        super().validate_submission(ctx, submission)
        agent_id = self._bound_agent_id(ctx)
        if agent_id is not None and submission.submitted_by_agent_id != agent_id:
            raise FlowStepValidationError(
                f"submission agent {submission.submitted_by_agent_id!r} does not match AgentStep agent {agent_id!r}"
            )

    def prepare_agent(self, ctx: StepRunContext) -> str:
        latest = self._latest_agent_step(ctx)
        state = self._agent_step_state(latest)
        role = state.agent_role
        agent_id = latest.resolve_bound_agent_id(ctx, role)
        created = False
        if agent_id is None:
            agent_id = latest.create_agent_for_step(ctx, state)
            created = True

        if created and state.bind_created_agent_to == "flow":
            latest.bind_agent_to_flow(ctx, role, agent_id)
        latest.bind_agent_to_step(ctx, role, agent_id)
        return agent_id

    def resolve_bound_agent_id(self, ctx: StepRunContext, role: str) -> str | None:
        agent_id = self.resolve_step_bound_agent_id(ctx, role)
        if agent_id is not None:
            return agent_id
        return self.resolve_flow_bound_agent_id(ctx, role)

    def resolve_step_bound_agent_id(self, ctx: StepRunContext, role: str) -> str | None:
        latest = self._latest_agent_step(ctx)
        return latest.agent_bindings.get(role)

    def resolve_flow_bound_agent_id(self, ctx: StepRunContext, role: str) -> str | None:
        flow = self._flow_service(ctx).get_flow(ctx.flow_id)
        return flow.agent_bindings.get(role)

    def create_agent_for_step(self, ctx: StepRunContext, state: AgentStepState) -> str:
        role = state.agent_role
        if not state.create_agent_if_missing:
            raise FlowStepValidationError(f"agent role {role!r} is not bound for step {ctx.step_id}")
        if state.agent_type is None:
            raise FlowStepValidationError(
                f"agent_type is required when create_agent_if_missing=True for step {ctx.step_id}"
            )
        agent_service = self._agent_service(ctx)
        agent = agent_service.create_agent(
            ctx.scope_id,
            state.agent_type,
            cli_type=state.cli_type,
            home_id=state.home_id,
        )
        return str(agent.agent_id)

    def bind_agent_to_step(self, ctx: StepRunContext, role: str, agent_id: str) -> None:
        ctx.update_step(lambda step: step.agent_bindings.by_role.__setitem__(role, agent_id))

    def bind_agent_to_flow(self, ctx: StepRunContext, role: str, agent_id: str) -> None:
        ctx.update_flow(lambda flow: flow.agent_bindings.by_role.__setitem__(role, agent_id))

    def _bound_agent_id(self, ctx: StepRunContext) -> str | None:
        latest = self._latest_agent_step(ctx)
        state = self._agent_step_state(latest)
        agent_id = latest.agent_bindings.get(state.agent_role)
        if agent_id is not None:
            return agent_id
        flow = self._flow_service(ctx).get_flow(ctx.flow_id)
        return flow.agent_bindings.get(state.agent_role)

    def build_start_prompt(self, ctx: StepRunContext, agent_id: str) -> str | None:
        latest = self._latest_agent_step(ctx)
        state = self._agent_step_state(latest)
        if state.prompt_mode == "callback":
            return latest.build_callback_prompt(ctx, agent_id)
        return state.prompt_override

    def prepare_agent_context_before_first_turn(
        self,
        ctx: StepRunContext,
        agent_id: str,
    ) -> AgentContextMaintenancePolicy | None:
        del ctx, agent_id
        return None

    def build_callback_prompt(self, ctx: StepRunContext, agent_id: str) -> str:
        latest = self._latest_agent_step(ctx)
        state = self._agent_step_state(latest)
        if state.callback_dispatch_step_id is None:
            raise FlowStepValidationError(f"callback AgentStep {ctx.step_id} has no callback_dispatch_step_id")
        flow_service = self._flow_service(ctx)
        dispatch_step = flow_service.get_step(state.callback_dispatch_step_id)
        created_children = getattr(dispatch_step.state, "created_children", None)
        if not created_children:
            raise FlowStepValidationError(
                f"dispatch step {state.callback_dispatch_step_id} has no created child flows"
            )
        render_ctx = RenderContext(ark=ctx.ark, app=ctx.app, scope_id=ctx.scope_id, viewer="agent")
        sections: list[str] = []
        for child in created_children:
            child_flow_id = getattr(child, "child_flow_id", None)
            if not isinstance(child_flow_id, str) or not child_flow_id:
                raise FlowStepValidationError(
                    f"dispatch step {state.callback_dispatch_step_id} has invalid child flow reference"
                )
            child_flow = flow_service.get_flow(child_flow_id)
            if child_flow.input is None:
                raise FlowStepValidationError(f"child flow for callback has no renderable input")
            input_text = child_flow.input.render_for_agent(render_ctx)
            if child_flow.status is FlowStatus.FAILED:
                message = child_flow.error.message if child_flow.error is not None else "Unknown runtime error"
                result_text = f"Runtime error:\n{message}"
            else:
                if child_flow.result is None:
                    raise FlowStepValidationError(f"completed child flow for callback has no renderable result")
                result_text = child_flow.result.render_for_agent(render_ctx)
            sections.append(f"---\n{input_text}\n\n{result_text}\n---")
        rendered_sections = "\n\n".join(sections)
        return (
            "The child workflows you requested have finished.\n\n"
            "Results:\n\n"
            f"{rendered_sections}\n\n"
            "Continue the current task using these results."
        )

    def build_continue_prompt(
        self,
        ctx: StepRunContext,
        agent_id: str,
        turn_result: object,
        decision: AgentStepCompletionDecision,
    ) -> str:
        latest = self._latest_agent_step(ctx)
        state = self._agent_step_state(latest)
        if decision.continue_prompt is not None:
            return decision.continue_prompt
        if state.continue_prompt_override is not None:
            return state.continue_prompt_override
        reason = decision.reason or "the required submission was not accepted"
        return (
            "Continue the current task. The previous turn did not complete the Step: "
            f"{reason}. Submit a valid result when the task is ready."
        )

    def build_agent_env(self, ctx: StepRunContext, agent_id: str) -> dict[str, str]:
        latest = self._latest_agent_step(ctx)
        state = self._agent_step_state(latest)
        env = dict(state.env_overrides)
        env.update(
            {
                "ARK_STEP_ID": ctx.step_id,
                "ARK_FLOW_ID": ctx.flow_id,
                "ARK_AGENT_ID": agent_id,
            }
        )
        return env

    def resolve_workdir(self, ctx: StepRunContext, agent_id: str) -> str | None:
        latest = self._latest_agent_step(ctx)
        return self._agent_step_state(latest).workdir_override

    def build_developer_instructions_override(self, ctx: StepRunContext, agent_id: str) -> str | None:
        return None

    def check_completion(
        self,
        ctx: StepRunContext,
        agent_id: str,
        turn_result: object,
        auto_continue_count: int,
    ) -> AgentStepCompletionDecision:
        latest = self._latest_agent_step(ctx)
        state = self._agent_step_state(latest)
        if latest.submission is not None:
            return AgentStepCompletionDecision(complete=True)
        reason = "step has no accepted submission"
        if not state.require_submission:
            reason = "default AgentStep completion requires a subclass when require_submission=False"
        return AgentStepCompletionDecision(complete=False, reason=reason)

    def build_result_from_submission(
        self,
        ctx: StepRunContext,
        agent_id: str,
        turn_result: object | None,
    ) -> BaseStepResult:
        latest = self._latest_agent_step(ctx)
        if latest.submission is None:
            raise FlowStepValidationError(f"step {ctx.step_id} has no accepted submission")
        return AgentStepSubmissionResult(
            summary=latest.submission.summary,
            agent_id=agent_id,
            submission_id=latest.submission.submission_id,
            submission_type=latest.submission.submission_type,
        )

    def build_incomplete_result(
        self,
        ctx: StepRunContext,
        agent_id: str | None,
        reason: str,
        turn_result: object | None,
        attempt_count: int,
    ) -> BaseStepResult:
        return AgentStepIncompleteResult(
            summary=reason,
            agent_id=agent_id,
            attempts=attempt_count,
            reason=reason,
        )

    def run(self, ctx: StepRunContext) -> StepTerminalReceipt:
        agent_id = self.prepare_agent(ctx)
        prompt = self.build_start_prompt(ctx, agent_id)
        latest = self._latest_agent_step(ctx)
        context_maintenance_policy = latest.prepare_agent_context_before_first_turn(ctx, agent_id)
        turn_result: object | None = None
        auto_continue_count = 0

        while True:
            latest = self._latest_agent_step(ctx)
            state = self._agent_step_state(latest)
            env = latest.build_agent_env(ctx, agent_id)
            workdir = latest.resolve_workdir(ctx, agent_id)
            developer_instructions_override = latest.build_developer_instructions_override(ctx, agent_id)
            agent_service = self._agent_service(ctx)
            start_kwargs = {
                "variables": state.variables,
                "prompt": prompt,
                "developer_instructions_template_override": developer_instructions_override,
                "env": env,
                "workdir": workdir,
            }
            if auto_continue_count == 0 and context_maintenance_policy is not None:
                start_kwargs["context_maintenance_policy"] = context_maintenance_policy
            agent_service.start_agent(agent_id, **start_kwargs)
            if state.agent_wait_timeout_s is None:
                turn_result = agent_service.wait_agent(agent_id)
            else:
                turn_result = agent_service.wait_agent(agent_id, timeout_s=state.agent_wait_timeout_s)

            latest = self._latest_agent_step(ctx)
            decision = latest.check_completion(
                ctx=ctx,
                agent_id=agent_id,
                turn_result=turn_result,
                auto_continue_count=auto_continue_count,
            )
            if decision.complete:
                result = latest.build_result_from_submission(
                    ctx=ctx,
                    agent_id=agent_id,
                    turn_result=turn_result,
                )
                return ctx.complete_step(result)

            if auto_continue_count >= self._agent_step_state(latest).max_auto_continue_turns:
                result = latest.build_incomplete_result(
                    ctx=ctx,
                    agent_id=agent_id,
                    reason=decision.reason or "agent did not reach required submission",
                    turn_result=turn_result,
                    attempt_count=auto_continue_count,
                )
                return ctx.complete_step(result)

            auto_continue_count += 1
            prompt = latest.build_continue_prompt(
                ctx=ctx,
                agent_id=agent_id,
                turn_result=turn_result,
                decision=decision,
            )

    def _latest_agent_step(self, ctx: StepRunContext) -> "AgentStep":
        latest = ctx.load_step()
        if not isinstance(latest, AgentStep):
            raise FlowStepValidationError(f"step {ctx.step_id} is not an AgentStep")
        return latest

    def _agent_step_state(self, step: "AgentStep") -> AgentStepState:
        if not isinstance(step.state, AgentStepState):
            raise FlowStepValidationError(f"step {step.step_id} does not have AgentStepState")
        return step.state

    def _flow_service(self, ctx: StepRunContext):
        flow_service = ctx.ark.flow_service
        if flow_service is None:
            raise FlowStepValidationError("ark.flow_service is not registered")
        return flow_service

    def _agent_service(self, ctx: StepRunContext):
        agent_service = ctx.ark.agent_service
        if agent_service is None:
            raise FlowStepValidationError("ark.agent_service is not registered")
        return agent_service


def build_followup_agent_step_from_dispatch(
    ctx,
    *,
    step_id: str,
    source_agent_step_id: str,
    dispatch_step_id: str,
) -> AgentStep:
    flow_service = ctx.ark.flow_service
    if flow_service is None:
        raise FlowStepValidationError("ark.flow_service is not registered")
    source_step = flow_service.get_step(source_agent_step_id)
    if not isinstance(source_step, AgentStep):
        raise FlowStepValidationError(f"source step {source_agent_step_id} is not an AgentStep")
    if not isinstance(source_step.state, AgentStepState):
        raise FlowStepValidationError(f"source step {source_agent_step_id} does not have AgentStepState")
    role = source_step.state.agent_role
    source_agent_id = source_step.agent_bindings.get(role)
    if source_agent_id is None:
        source_agent_id = ctx.flow.agent_bindings.get(role)
    if source_agent_id is None:
        raise FlowStepValidationError(f"source AgentStep {source_agent_step_id} has no bound agent for role {role!r}")

    source_state = source_step.state
    followup_state = AgentStepState(
        agent_role=source_state.agent_role,
        agent_type=source_state.agent_type,
        cli_type=source_state.cli_type,
        home_id=source_state.home_id,
        create_agent_if_missing=False,
        bind_created_agent_to="step",
        variables=dict(source_state.variables),
        prompt_mode="callback",
        followup_of_step_id=source_agent_step_id,
        callback_dispatch_step_id=dispatch_step_id,
        max_auto_continue_turns=source_state.max_auto_continue_turns,
        require_submission=source_state.require_submission,
        env_overrides=dict(source_state.env_overrides),
        workdir_override=source_state.workdir_override,
        agent_wait_timeout_s=source_state.agent_wait_timeout_s,
    )
    followup = AgentStep(
        step_id=step_id,
        flow_id=ctx.flow.flow_id,
        scope_id=ctx.flow.scope_id,
        state=followup_state,
    )
    followup.agent_bindings.by_role[role] = source_agent_id
    return followup
