from __future__ import annotations

import uuid

from ..provider_contracts import (
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    ProviderContextCompactionRequest,
    ProviderContextCompactionResult,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderContextUsage,
)
from ..store_utils import utc_now_iso
from .grok_session import session_directory, native_session, completed_compactions


class GrokContextAdapter:
    provider_type = "grok"

    def __init__(self, runtime) -> None:  # noqa: ANN001
        self.runtime = runtime

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage:
        if request.execution_context is None:
            raise ValueError("Grok context requires an execution context")
        session_directory(request.execution_context, request.session)
        return ProviderContextUsage(
            session_id=request.session.session_id,
            observed_at=utc_now_iso(),
            source="grok_native_session",
            available=False,
            measurement="unavailable",
            stale=True,
            reason="Grok adapter has no verified current-context measurement; manual compact remains available",
            compact_capability=CapabilitySupport(
                capability=CapabilityKey.CONTROL_COMPACT,
                status=CapabilityStatus.NATIVE,
                available=True,
                evidence_version="grok-1.0.30",
            ),
        )

    def compact(
        self, request: ProviderContextCompactionRequest
    ) -> ProviderContextCompactionResult:
        if request.execution_context is None:
            raise ValueError("Grok compaction requires an execution context")
        started = utc_now_iso()
        options = request.provider_options or {}
        if not isinstance(options, dict) or set(options) - {"user_context"}:
            raise ValueError("Grok compact options accept only user_context")
        user_context = options.get(
            "user_context",
            "Preserve the user's requirements, important facts, and pending work.",
        )
        if not isinstance(user_context, str):
            raise ValueError("Grok compact user_context must be text")
        with native_session(
            self.runtime, request.execution_context, request.session
        ) as (process, path):
            baseline = {
                "completed_checkpoints": completed_compactions(
                    path, request.session.session_id
                )
            }
            operation_id = "grok-compact-" + uuid.uuid4().hex
            if request.on_started is not None:
                request.on_started(baseline, operation_id)
            process.request(
                "_x.ai/compact_conversation",
                {
                    "session_id": request.session.session_id,
                    "user_context": user_context,
                },
                timeout_s=request.timeout_s or 120,
            )
        fresh = [
            item
            for item in completed_compactions(path, request.session.session_id)
            if item not in baseline["completed_checkpoints"]
        ]
        if not fresh:
            raise RuntimeError(
                "Grok compact response has no new persisted completion boundary"
            )
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="compacted",
            reason="grok_native_compaction_confirmed",
            started_at=started,
            completed_at=utc_now_iso(),
            provider_operation_id=fresh[-1],
        )

    def reconcile(
        self, request: ProviderContextReconcileRequest
    ) -> ProviderContextCompactionResult | None:
        if request.execution_context is None or not isinstance(request.baseline, dict):
            raise ValueError("Grok reconciliation requires context and baseline")
        if not self.runtime.is_session_stable(request.session.session_id):
            return None
        _, path = session_directory(request.execution_context, request.session)
        if "completed_checkpoints" not in request.baseline:
            raise ValueError(
                "Grok reconciliation baseline is missing checkpoint identities"
            )
        fresh = [
            item
            for item in completed_compactions(path, request.session.session_id)
            if item not in request.baseline["completed_checkpoints"]
        ]
        if not fresh:
            return None
        now = utc_now_iso()
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="compacted",
            reason="grok_compaction_reconciled",
            started_at=now,
            completed_at=now,
            provider_operation_id=fresh[-1],
        )
