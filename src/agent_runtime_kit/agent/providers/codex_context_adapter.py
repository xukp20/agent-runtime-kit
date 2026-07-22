from __future__ import annotations

from dataclasses import replace

from ..provider_contracts import (
    ProviderContextCompactionRequest,
    ProviderContextCompactionResult,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderContextUsage,
)
from .codex import CodexProvider


class CodexContextAdapter:
    provider_type = "codex"

    def __init__(self, provider: CodexProvider) -> None:
        self.provider = provider

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage:
        ctx = _execution_context(request.execution_context)
        usage = self.provider.inspect_thread_context(
            home_id=ctx.home_id,
            home_root=ctx.home_root,
            env=ctx.process_environment,
            thread_id=request.session.session_id,
            workdir=ctx.workdir,
            agent_id=request.agent_id or "",
        )
        return replace(usage, model_identity=ctx.resolved_defaults)

    def compact(self, request: ProviderContextCompactionRequest) -> ProviderContextCompactionResult:
        ctx = _execution_context(request.execution_context)
        if request.timeout_s is None:
            raise ValueError("Codex context compaction requires timeout_s")
        result = self.provider.compact_thread(
            home_id=ctx.home_id,
            home_root=ctx.home_root,
            env=ctx.process_environment,
            thread_id=request.session.session_id,
            workdir=ctx.workdir,
            agent_id=request.agent_id or "",
            timeout_s=request.timeout_s,
            on_compaction_started=request.on_started,
        )
        return _with_model(result, ctx.resolved_defaults)

    def reconcile(
        self,
        request: ProviderContextReconcileRequest,
    ) -> ProviderContextCompactionResult | None:
        ctx = _execution_context(request.execution_context)
        if not isinstance(request.baseline, dict):
            raise ValueError("Codex context reconciliation requires a baseline mapping")
        result = self.provider.reconcile_thread_compaction(
            home_id=ctx.home_id,
            home_root=ctx.home_root,
            env=ctx.process_environment,
            thread_id=request.session.session_id,
            workdir=ctx.workdir,
            agent_id=request.agent_id or "",
            baseline=request.baseline,
            provider_operation_id=request.operation_id,
        )
        return None if result is None else _with_model(result, ctx.resolved_defaults)


def _execution_context(value):  # noqa: ANN001, ANN202
    if value is None or value.provider_type != "codex":
        raise ValueError("Codex context adapter requires a Codex ProviderExecutionContext")
    return value


def _with_model(result: ProviderContextCompactionResult, model_identity):  # noqa: ANN001, ANN202
    usage = result.usage_after
    return replace(
        result,
        usage_after=(replace(usage, model_identity=model_identity) if usage is not None else None),
    )
