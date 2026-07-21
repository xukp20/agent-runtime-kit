from __future__ import annotations

import time
import uuid
from typing import Mapping

from ..provider_contracts import (
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    ProviderContextCompactionRequest,
    ProviderContextCompactionResult,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderContextUsage,
    ProviderUsageQuery,
    TokenUsage,
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .opencode_models import ADAPTER_VERSION, PROVIDER_TYPE
from .opencode_query import OpenCodeQueryAdapter
from .opencode_runtime import OpenCodeRuntimeRegistry


class OpenCodeContextAdapter:
    provider_type = PROVIDER_TYPE

    def __init__(
        self,
        *,
        registry: OpenCodeRuntimeRegistry,
        query: OpenCodeQueryAdapter,
    ) -> None:
        self.registry = registry
        self.query = query

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage:
        usage = self.query.read_usage(
            ProviderUsageQuery(session=request.session, latest=True)
        )
        tokens = usage.token_usage
        used = _observed_input(tokens)
        identity = request.session.backend_identity
        return ProviderContextUsage(
            session_id=request.session.session_id,
            observed_at=utc_now_iso(),
            source="opencode.latest_assistant_usage",
            available=used is not None,
            used_tokens=used,
            context_window_tokens=None,
            remaining_tokens=None,
            auto_compact_enabled=False,
            compact_capability=CapabilitySupport(
                capability=CapabilityKey.CONTROL_COMPACT,
                status=CapabilityStatus.NATIVE,
                available=True,
                limitations=("OpenCode summarize performs a normal model-backed summary call",),
                evidence_version="opencode-1.18.4",
            ),
            measurement="provider_reported_latest_request" if used is not None else "unavailable",
            model_identity=identity,
            reason=None if used is not None else "OpenCode did not report latest input usage",
        )

    def compact(self, request: ProviderContextCompactionRequest) -> ProviderContextCompactionResult:
        started = utc_now_iso()
        client = self.registry.client_for_locator(request.session)
        baseline = client.list_messages(request.session.session_id)
        operation_id = f"compact-{uuid.uuid4().hex}"
        if request.on_started is not None:
            request.on_started(
                {"message_ids": sorted(_message_ids(baseline)), "message_count": len(baseline)},
                operation_id,
            )
        identity = request.session.backend_identity
        if identity is None or not identity.api_provider or not identity.effective_model:
            raise ValueError("OpenCode compact requires provider/model identity")
        client.summarize(
            request.session.session_id,
            {
                "providerID": identity.api_provider,
                "modelID": identity.effective_model,
                "auto": False,
            },
        )
        deadline = time.monotonic() + (request.timeout_s or 300)
        latest: list[object] = baseline
        while time.monotonic() < deadline:
            latest = client.list_messages(request.session.session_id)
            if _has_completed_summary(latest, baseline) and _idle(
                client.session_status(), request.session.session_id
            ):
                return ProviderContextCompactionResult(
                    session_id=request.session.session_id,
                    status="completed",
                    reason="OpenCode persisted a compaction part and completed summary message",
                    started_at=started,
                    completed_at=utc_now_iso(),
                    usage_after=self.inspect(ProviderContextQuery(session=request.session)),
                    provider_operation_id=operation_id,
                    provider_payload=build_provider_payload(
                        provider_type=PROVIDER_TYPE,
                        payload_type="compact_evidence",
                        data={"message_count_before": len(baseline), "message_count_after": len(latest)},
                        adapter_version=ADAPTER_VERSION,
                    ),
                )
            time.sleep(0.25)
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="ambiguous",
            reason="OpenCode summarize did not produce complete persisted evidence before timeout",
            started_at=started,
            completed_at=utc_now_iso(),
            provider_operation_id=operation_id,
        )

    def reconcile(
        self, request: ProviderContextReconcileRequest
    ) -> ProviderContextCompactionResult | None:
        if not isinstance(request.baseline, Mapping):
            return None
        before = set(str(value) for value in request.baseline.get("message_ids", []))
        client = self.registry.client_for_locator(request.session)
        messages = client.list_messages(request.session.session_id)
        changed = bool(_message_ids(messages) - before)
        completed = changed and _has_completed_summary(messages, [])
        now = utc_now_iso()
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="completed" if completed else "not_started" if not changed else "ambiguous",
            reason=(
                "persisted OpenCode summary evidence found"
                if completed
                else "no messages were added after compact baseline"
                if not changed
                else "messages changed without complete summary evidence"
            ),
            started_at=now,
            completed_at=now,
            provider_operation_id=request.operation_id,
        )


def _observed_input(tokens: TokenUsage) -> int | None:
    if tokens.input_tokens is None:
        return None
    return tokens.input_tokens + (tokens.cache_read_input_tokens or 0)


def _message_ids(messages: list[object]) -> set[str]:
    values: set[str] = set()
    for value in messages:
        if not isinstance(value, Mapping):
            continue
        info = value.get("info") if isinstance(value.get("info"), Mapping) else value
        if info.get("id") is not None:
            values.add(str(info["id"]))
    return values


def _has_completed_summary(messages: list[object], baseline: list[object]) -> bool:
    old = _message_ids(baseline)
    compaction = False
    summary = False
    for value in messages:
        if not isinstance(value, Mapping):
            continue
        info = value.get("info") if isinstance(value.get("info"), Mapping) else value
        if str(info.get("id")) in old:
            continue
        parts = value.get("parts") if isinstance(value.get("parts"), list) else []
        compaction = compaction or any(
            isinstance(part, Mapping) and part.get("type") == "compaction" for part in parts
        )
        summary = summary or (
            info.get("role") == "assistant"
            and bool(info.get("summary"))
            and info.get("error") is None
        )
    return compaction and summary


def _idle(statuses: Mapping[str, object], session_id: str) -> bool:
    value = statuses.get(session_id)
    return value is None or (
        isinstance(value, Mapping) and str(value.get("type") or value.get("status")) == "idle"
    )
