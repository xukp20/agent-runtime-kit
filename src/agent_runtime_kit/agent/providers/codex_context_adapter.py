from __future__ import annotations

from ..context import (
    ProviderContextCompactionResult as LegacyCompactionResult,
    ProviderContextUsage as LegacyContextUsage,
)
from ..provider_contracts import (
    ModelBackendIdentity,
    ProviderContextCompactionRequest,
    ProviderContextCompactionResult,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderContextUsage,
    build_provider_payload,
)
from .codex import CodexProvider


class CodexContextAdapter:
    """Project Codex context evidence and controls onto the provider-neutral SPI."""

    provider_type = "codex"
    adapter_version = "1"

    def __init__(self, provider: CodexProvider) -> None:
        self.provider = provider

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage:
        ctx = _require_execution_context(request.execution_context)
        result = self.provider.inspect_thread_context(
            home_id=ctx.home_id,
            home_root=ctx.home_root,
            env=ctx.process_environment,
            thread_id=request.session.session_id,
            workdir=ctx.workdir,
            agent_id=request.agent_id or "",
        )
        return _context_usage(result, model_identity=ctx.resolved_defaults)

    def compact(self, request: ProviderContextCompactionRequest) -> ProviderContextCompactionResult:
        ctx = _require_execution_context(request.execution_context)
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
        return _compaction_result(result, model_identity=ctx.resolved_defaults)

    def reconcile(
        self,
        request: ProviderContextReconcileRequest,
    ) -> ProviderContextCompactionResult | None:
        ctx = _require_execution_context(request.execution_context)
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
        if result is None:
            return None
        return _compaction_result(result, model_identity=ctx.resolved_defaults)


def _require_execution_context(value):  # noqa: ANN001, ANN202
    if value is None:
        raise ValueError("Codex Context adapter requires ProviderExecutionContext")
    if value.provider_type != "codex":
        raise ValueError(f"Codex Context adapter received provider: {value.provider_type}")
    return value


def _context_usage(
    usage: LegacyContextUsage,
    *,
    model_identity: ModelBackendIdentity | None,
) -> ProviderContextUsage:
    if not isinstance(usage, LegacyContextUsage):
        raise TypeError("Codex provider returned invalid legacy context usage")
    window = usage.context_window
    remaining = None
    if usage.total_tokens is not None and window is not None:
        remaining = max(window - usage.total_tokens, 0)
    return ProviderContextUsage(
        session_id=usage.session_id,
        observed_at=usage.observed_at,
        source=usage.source,
        available=usage.available,
        used_tokens=usage.total_tokens,
        context_window_tokens=window,
        effective_context_window_tokens=window,
        remaining_tokens=remaining,
        measurement="provider_artifact",
        reason=usage.reason,
        model_identity=model_identity,
        provider_payload=build_provider_payload(
            provider_type="codex",
            payload_type="context_usage",
            data={
                "source": usage.source,
                "available": usage.available,
                "reason": usage.reason,
            },
            adapter_version="1",
        ),
    )


def _compaction_result(
    result: LegacyCompactionResult,
    *,
    model_identity: ModelBackendIdentity | None,
) -> ProviderContextCompactionResult:
    if not isinstance(result, LegacyCompactionResult):
        raise TypeError("Codex provider returned invalid legacy compaction result")
    return ProviderContextCompactionResult(
        session_id=result.session_id,
        status="compacted",
        reason="provider_confirmed",
        started_at=result.started_at,
        completed_at=result.completed_at,
        usage_after=(
            _context_usage(result.usage_after, model_identity=model_identity)
            if result.usage_after is not None
            else None
        ),
        provider_operation_id=result.provider_operation_id,
        provider_payload=build_provider_payload(
            provider_type="codex",
            payload_type="context_compaction_result",
            data={"provider_operation_id": result.provider_operation_id},
            adapter_version="1",
        ),
    )
