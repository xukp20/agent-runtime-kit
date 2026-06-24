from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from agent_runtime_kit.agent.models import Agent
from agent_runtime_kit.flow.contexts import StepRunContext
from agent_runtime_kit.flow.models import BaseFlow, BaseStep, BaseSubmission, FlowStepValidationError, StepStatus

from .services import ARKServices, AppServices


@dataclass(frozen=True)
class RuntimeToolIdentity:
    step_id: str
    flow_id: str
    agent_id: str


@dataclass(frozen=True)
class RuntimeToolContext:
    identity: RuntimeToolIdentity
    scope_id: str
    step: BaseStep
    flow: BaseFlow
    agent: Agent


class RuntimeToolContextError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


class RuntimeMcpContextResolver:
    def __init__(self, *, ark_services: ARKServices) -> None:
        self.ark = ark_services

    def identity_from_http_headers(self, headers: Mapping[str, str]) -> RuntimeToolIdentity:
        normalized = {str(key).lower(): str(value) for key, value in headers.items()}
        return self._identity_from_mapping(
            {
                "step_id": normalized.get("x-ark-step-id"),
                "flow_id": normalized.get("x-ark-flow-id"),
                "agent_id": normalized.get("x-ark-agent-id"),
            },
            source="headers",
        )

    def identity_from_env(self, env: Mapping[str, str] | None = None) -> RuntimeToolIdentity:
        source = os.environ if env is None else env
        return self._identity_from_mapping(
            {
                "step_id": source.get("ARK_STEP_ID"),
                "flow_id": source.get("ARK_FLOW_ID"),
                "agent_id": source.get("ARK_AGENT_ID"),
            },
            source="env",
        )

    def resolve_from_identity(
        self,
        identity: RuntimeToolIdentity,
        *,
        require_running_step: bool = False,
        allowed_submit_tool_name: str | None = None,
    ) -> RuntimeToolContext:
        step = self._load_step(identity.step_id)
        flow = self._load_flow(identity.flow_id)
        agent = self._load_agent(identity.agent_id)

        if step.flow_id != identity.flow_id:
            raise RuntimeToolContextError(
                "step_flow_mismatch",
                f"step {step.step_id} belongs to flow {step.flow_id}, expected {identity.flow_id}",
                details={"step_id": step.step_id, "step_flow_id": step.flow_id, "flow_id": identity.flow_id},
            )
        if step.scope_id != flow.scope_id:
            raise RuntimeToolContextError(
                "step_scope_mismatch",
                f"step {step.step_id} scope {step.scope_id}, expected flow scope {flow.scope_id}",
                details={"step_id": step.step_id, "step_scope_id": step.scope_id, "flow_scope_id": flow.scope_id},
            )
        if not self._agent_is_bound_to_step_or_flow(identity.agent_id, step, flow):
            raise RuntimeToolContextError(
                "agent_not_bound_to_step",
                f"agent {identity.agent_id} is not bound to step {step.step_id}",
                details={"agent_id": identity.agent_id, "step_id": step.step_id, "flow_id": flow.flow_id},
            )
        if require_running_step and step.status is not StepStatus.RUNNING:
            raise RuntimeToolContextError(
                "step_not_running",
                f"step {step.step_id} is not running",
                details={"step_id": step.step_id, "status": str(step.status)},
            )
        if allowed_submit_tool_name is not None:
            self._assert_submit_tool_allowed(step, allowed_submit_tool_name)

        return RuntimeToolContext(identity=identity, scope_id=step.scope_id, step=step, flow=flow, agent=agent)

    def resolve_from_http_headers(
        self,
        headers: Mapping[str, str],
        *,
        require_running_step: bool = False,
        allowed_submit_tool_name: str | None = None,
    ) -> RuntimeToolContext:
        return self.resolve_from_identity(
            self.identity_from_http_headers(headers),
            require_running_step=require_running_step,
            allowed_submit_tool_name=allowed_submit_tool_name,
        )

    def resolve_from_env(
        self,
        env: Mapping[str, str] | None = None,
        *,
        require_running_step: bool = False,
        allowed_submit_tool_name: str | None = None,
    ) -> RuntimeToolContext:
        return self.resolve_from_identity(
            self.identity_from_env(env),
            require_running_step=require_running_step,
            allowed_submit_tool_name=allowed_submit_tool_name,
        )

    def _identity_from_mapping(self, values: Mapping[str, str | None], *, source: str) -> RuntimeToolIdentity:
        missing = [key for key, value in values.items() if value is None or not str(value).strip()]
        if missing:
            raise RuntimeToolContextError(
                "missing_identity",
                f"missing ARK runtime identity in {source}: {', '.join(missing)}",
                details={"source": source, "missing": ",".join(missing)},
            )
        return RuntimeToolIdentity(
            step_id=str(values["step_id"]).strip(),
            flow_id=str(values["flow_id"]).strip(),
            agent_id=str(values["agent_id"]).strip(),
        )

    def _load_step(self, step_id: str) -> BaseStep:
        flow_service = self.ark.flow_service
        get_step = getattr(flow_service, "get_step", None)
        if not callable(get_step):
            raise RuntimeToolContextError("step_not_found", "ARK flow_service.get_step is not available")
        try:
            step = get_step(step_id)
        except Exception as exc:
            raise RuntimeToolContextError(
                "step_not_found",
                f"step not found: {step_id}",
                details={"step_id": step_id, "exception_type": type(exc).__name__},
            ) from exc
        if not isinstance(step, BaseStep):
            raise RuntimeToolContextError(
                "step_not_found",
                f"loaded object is not a BaseStep: {step_id}",
                details={"step_id": step_id, "loaded_type": type(step).__name__},
            )
        return step

    def _load_flow(self, flow_id: str) -> BaseFlow:
        flow_service = self.ark.flow_service
        get_flow = getattr(flow_service, "get_flow", None)
        if not callable(get_flow):
            raise RuntimeToolContextError("flow_not_found", "ARK flow_service.get_flow is not available")
        try:
            flow = get_flow(flow_id)
        except Exception as exc:
            raise RuntimeToolContextError(
                "flow_not_found",
                f"flow not found: {flow_id}",
                details={"flow_id": flow_id, "exception_type": type(exc).__name__},
            ) from exc
        if not isinstance(flow, BaseFlow):
            raise RuntimeToolContextError(
                "flow_not_found",
                f"loaded object is not a BaseFlow: {flow_id}",
                details={"flow_id": flow_id, "loaded_type": type(flow).__name__},
            )
        return flow

    def _load_agent(self, agent_id: str) -> Agent:
        agent_service = self.ark.agent_service
        get_agent = getattr(agent_service, "get_agent", None)
        if not callable(get_agent):
            store = getattr(agent_service, "store", None)
            get_agent = getattr(store, "get_agent", None)
        if not callable(get_agent):
            raise RuntimeToolContextError("agent_not_found", "ARK agent service get_agent is not available")
        try:
            agent = get_agent(agent_id)
        except Exception as exc:
            raise RuntimeToolContextError(
                "agent_not_found",
                f"agent not found: {agent_id}",
                details={"agent_id": agent_id, "exception_type": type(exc).__name__},
            ) from exc
        if not isinstance(agent, Agent):
            raise RuntimeToolContextError(
                "agent_not_found",
                f"loaded object is not an Agent: {agent_id}",
                details={"agent_id": agent_id, "loaded_type": type(agent).__name__},
            )
        return agent

    def _agent_is_bound_to_step_or_flow(self, agent_id: str, step: BaseStep, flow: BaseFlow) -> bool:
        return agent_id in set(step.agent_bindings.by_role.values()) | set(flow.agent_bindings.by_role.values())

    def _assert_submit_tool_allowed(self, step: BaseStep, tool_name: str) -> None:
        if type(step).SubmitTools is not None and tool_name not in type(step).SubmitTools:
            raise RuntimeToolContextError(
                "tool_not_allowed_for_step",
                f"step {step.step_id} does not accept submit tool {tool_name!r}",
                details={"step_id": step.step_id, "tool_name": tool_name},
            )


class RuntimeMcpToolGateway:
    def __init__(
        self,
        *,
        ark_services: ARKServices,
        app_services: AppServices | None = None,
    ) -> None:
        self.ark = ark_services
        self.app = app_services or AppServices()
        self.resolver = RuntimeMcpContextResolver(ark_services=ark_services)

    def resolve_context_from_http_headers(
        self,
        headers: Mapping[str, str],
        *,
        require_running_step: bool = False,
        allowed_submit_tool_name: str | None = None,
    ) -> RuntimeToolContext:
        return self.resolver.resolve_from_http_headers(
            headers,
            require_running_step=require_running_step,
            allowed_submit_tool_name=allowed_submit_tool_name,
        )

    def resolve_context_from_env(
        self,
        env: Mapping[str, str] | None = None,
        *,
        require_running_step: bool = False,
        allowed_submit_tool_name: str | None = None,
    ) -> RuntimeToolContext:
        return self.resolver.resolve_from_env(
            env,
            require_running_step=require_running_step,
            allowed_submit_tool_name=allowed_submit_tool_name,
        )

    def accept_step_submission(
        self,
        ctx: RuntimeToolContext,
        submission: BaseSubmission,
    ) -> BaseStep:
        if submission.submitted_by_agent_id != ctx.identity.agent_id:
            raise RuntimeToolContextError(
                "submission_agent_mismatch",
                "submission submitted_by_agent_id does not match runtime agent identity",
                details={
                    "runtime_agent_id": ctx.identity.agent_id,
                    "submitted_by_agent_id": str(submission.submitted_by_agent_id),
                },
            )
        step_ctx = StepRunContext(
            ark=self.ark,
            app=self.app,
            step_id=ctx.identity.step_id,
            flow_id=ctx.identity.flow_id,
            scope_id=ctx.scope_id,
        )
        try:
            return step_ctx.accept_step_submission(submission, expected_agent_id=ctx.identity.agent_id)
        except FlowStepValidationError as exc:
            raise self._submission_error(ctx, submission, exc) from exc

    def _submission_error(
        self,
        ctx: RuntimeToolContext,
        submission: BaseSubmission,
        exc: FlowStepValidationError,
    ) -> RuntimeToolContextError:
        latest_step: BaseStep | None = None
        try:
            latest_step = self.resolver._load_step(ctx.identity.step_id)
        except RuntimeToolContextError:
            return RuntimeToolContextError(
                "step_not_found",
                str(exc) or "step not found while accepting submission",
                details={"step_id": ctx.identity.step_id},
            )
        if latest_step.status is not StepStatus.RUNNING:
            return RuntimeToolContextError(
                "step_not_running",
                str(exc) or f"step {latest_step.step_id} is not running",
                details={"step_id": latest_step.step_id, "status": str(latest_step.status)},
            )
        if submission.submitted_by_agent_id != ctx.identity.agent_id:
            return RuntimeToolContextError(
                "submission_agent_mismatch",
                str(exc) or "submission agent mismatch",
                details={
                    "runtime_agent_id": ctx.identity.agent_id,
                    "submitted_by_agent_id": str(submission.submitted_by_agent_id),
                },
            )
        return RuntimeToolContextError(
            "tool_not_allowed_for_step",
            str(exc) or "submission was rejected by step",
            details={"step_id": latest_step.step_id, "tool_name": submission.tool_name},
        )


def runtime_context_from_fastmcp_context(ctx: object, gateway: RuntimeMcpToolGateway) -> RuntimeToolContext:
    headers = _headers_from_fastmcp_context(ctx)
    return gateway.resolve_context_from_http_headers(headers)


def _headers_from_fastmcp_context(ctx: object) -> dict[str, str]:
    request_context = getattr(ctx, "request_context", None)
    request = getattr(request_context, "request", None)
    headers = getattr(request, "headers", None)
    if headers is None:
        return {}
    items = headers.items() if hasattr(headers, "items") else []
    return {str(key).lower(): str(value) for key, value in items}

