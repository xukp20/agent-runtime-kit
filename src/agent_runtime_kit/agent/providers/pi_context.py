from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path

from ..provider_contracts import (
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    ModelBackendIdentity,
    ProviderContextCompactionRequest,
    ProviderContextCompactionResult,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderContextUsage,
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .pi_rpc import PiRpcProcess
from .pi_runtime import build_pi_command
from .pi_session import PI_ADAPTER_VERSION, PI_CLI_VERSION, PiSessionTranscript, find_pi_session


class PiContextAdapter:
    provider_type = "pi"

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage:
        context, path = _context_and_path(request.execution_context, request.session.session_id)
        rpc = PiRpcProcess(
            build_pi_command(context, session_path=path, model=request.session.backend_identity),
            cwd=Path(context.workdir or context.home_root),
            env=context.process_environment,
        )
        try:
            state = _data(rpc.command("get_state", timeout_s=15))
            stats = _data(rpc.command("get_session_stats", timeout_s=15))
        finally:
            rpc.close()
        return _context_usage(
            session_id=request.session.session_id,
            state=state,
            stats=stats,
            fallback_identity=request.session.backend_identity or context.resolved_defaults,
        )

    def compact(self, request: ProviderContextCompactionRequest) -> ProviderContextCompactionResult:
        context, path = _context_and_path(request.execution_context, request.session.session_id)
        transcript = PiSessionTranscript.read(path)
        baseline = {
            "leaf_id": transcript.leaf_id,
            "compaction_entry_ids": _compaction_ids(transcript),
        }
        operation_id = f"pi-compact-{uuid.uuid4().hex}"
        if request.on_started is not None:
            request.on_started(baseline, operation_id)
        started_at = utc_now_iso()
        rpc = PiRpcProcess(
            build_pi_command(context, session_path=path, model=request.session.backend_identity),
            cwd=Path(context.workdir or context.home_root),
            env=context.process_environment,
        )
        try:
            state = _data(rpc.command("get_state", timeout_s=15))
            if bool(state.get("isStreaming")) or bool(state.get("isCompacting")):
                raise RuntimeError("Pi session is not idle for compaction")
            payload: dict[str, object] = {}
            options = request.provider_options
            if isinstance(options, Mapping) and isinstance(options.get("custom_instructions"), str):
                payload["customInstructions"] = options["custom_instructions"]
            response = rpc.command("compact", payload, timeout_s=request.timeout_s or 120)
            event, _ = rpc.wait_for(
                lambda item: item.get("type") == "compaction_end",
                timeout_s=request.timeout_s or 120,
            )
            if event.get("aborted") is True or event.get("errorMessage"):
                raise RuntimeError(str(event.get("errorMessage") or "Pi compaction was aborted"))
            state = _data(rpc.command("get_state", timeout_s=15))
            if bool(state.get("isStreaming")) or bool(state.get("isCompacting")):
                raise RuntimeError("Pi compaction ended without an idle terminal barrier")
            stats = _data(rpc.command("get_session_stats", timeout_s=15))
        finally:
            rpc.close()
        updated = PiSessionTranscript.read(path)
        new_ids = [item for item in _compaction_ids(updated) if item not in baseline["compaction_entry_ids"]]
        persisted_operation = new_ids[-1] if new_ids else operation_id
        result_data = response.get("data") if isinstance(response.get("data"), Mapping) else {}
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="compacted",
            reason="pi_agent_owned_compaction_confirmed",
            started_at=started_at,
            completed_at=utc_now_iso(),
            usage_after=_context_usage(
                session_id=request.session.session_id,
                state=state,
                stats=stats,
                fallback_identity=request.session.backend_identity or context.resolved_defaults,
            ),
            provider_operation_id=persisted_operation,
            provider_payload=build_provider_payload(
                provider_type="pi",
                payload_type="agent_owned_compaction",
                data={
                    "operation_kind": "pi_agent_owned_history_summary",
                    "backend_responses_compact": False,
                    "response": result_data,
                    "event": event,
                    "baseline": baseline,
                },
                adapter_version=PI_ADAPTER_VERSION,
                sdk_or_cli_version=PI_CLI_VERSION,
            ),
        )

    def reconcile(
        self,
        request: ProviderContextReconcileRequest,
    ) -> ProviderContextCompactionResult | None:
        _, path = _context_and_path(request.execution_context, request.session.session_id)
        if not isinstance(request.baseline, Mapping):
            raise ValueError("Pi compaction reconciliation requires a baseline mapping")
        baseline_ids = {
            str(item) for item in request.baseline.get("compaction_entry_ids", ())
        }
        transcript = PiSessionTranscript.read(path)
        new_ids = [item for item in _compaction_ids(transcript) if item not in baseline_ids]
        if not new_ids:
            return None
        now = utc_now_iso()
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="compacted",
            reason="pi_compaction_artifact_reconciled",
            started_at=now,
            completed_at=now,
            provider_operation_id=new_ids[-1],
            provider_payload=build_provider_payload(
                provider_type="pi",
                payload_type="agent_owned_compaction_reconciliation",
                data={"new_compaction_entry_ids": new_ids},
                adapter_version=PI_ADAPTER_VERSION,
                sdk_or_cli_version=PI_CLI_VERSION,
            ),
        )


def _context_and_path(execution_context, session_id: str):  # noqa: ANN001, ANN202
    if execution_context is None or execution_context.provider_type != "pi":
        raise ValueError("Pi Context adapter requires a Pi ProviderExecutionContext")
    path = find_pi_session(execution_context.home_root / ".pi" / "sessions", session_id)
    if path is None:
        raise KeyError(f"unknown Pi session: {session_id}")
    return execution_context, path


def _data(response: Mapping[str, object]) -> dict[str, object]:
    value = response.get("data")
    return dict(value) if isinstance(value, Mapping) else {}


def _context_usage(
    *,
    session_id: str,
    state: Mapping[str, object],
    stats: Mapping[str, object],
    fallback_identity: ModelBackendIdentity | None,
) -> ProviderContextUsage:
    context_raw = stats.get("contextUsage")
    context = context_raw if isinstance(context_raw, Mapping) else {}
    used = _int(context.get("tokens"))
    window = _positive_int(context.get("contextWindow"))
    model_raw = state.get("model")
    model = model_raw if isinstance(model_raw, Mapping) else {}
    identity = _identity(model) or fallback_identity
    window = window or _positive_int(model.get("contextWindow"))
    max_output = _int(model.get("maxTokens"))
    available = used is not None
    remaining = max(window - used, 0) if window is not None and used is not None else None
    return ProviderContextUsage(
        session_id=session_id,
        observed_at=utc_now_iso(),
        source="pi_session_stats",
        available=available,
        used_tokens=used,
        context_window_tokens=window,
        effective_context_window_tokens=window,
        max_output_tokens=max_output,
        remaining_tokens=remaining,
        auto_compact_enabled=(
            bool(state.get("autoCompactionEnabled"))
            if state.get("autoCompactionEnabled") is not None
            else None
        ),
        compact_capability=CapabilitySupport(
            capability=CapabilityKey.CONTROL_COMPACT,
            status=CapabilityStatus.NATIVE,
            available=True,
            limitations=("Pi agent-owned history summary; not backend responses.compact",),
            evidence_version="pi-rpc-0.80.10",
        ),
        measurement="estimated" if available else "unavailable",
        stale=not available,
        reason=None if available else "Pi context usage is unavailable after compaction until a new valid assistant response",
        model_identity=identity,
        provider_payload=build_provider_payload(
            provider_type="pi",
            payload_type="session_stats",
            data={"state": state, "stats": stats},
            adapter_version=PI_ADAPTER_VERSION,
            sdk_or_cli_version=PI_CLI_VERSION,
        ),
    )


def _identity(model: Mapping[str, object]) -> ModelBackendIdentity | None:
    provider = model.get("provider")
    api = model.get("api")
    if not isinstance(provider, str) or not provider or not isinstance(api, str) or not api:
        return None
    mode = {
        "openai-completions": "chat_completions",
        "openai-responses": "responses",
        "openai-codex-responses": "responses",
        "anthropic-messages": "messages",
    }.get(api, api)
    return ModelBackendIdentity(
        api_provider=provider,
        api_mode=mode,
        endpoint_id=str(model.get("baseUrl")) if model.get("baseUrl") is not None else None,
        requested_model=str(model.get("id")) if model.get("id") is not None else None,
    )


def _compaction_ids(transcript: PiSessionTranscript) -> tuple[str, ...]:
    return tuple(
        str(entry["id"])
        for entry in transcript.entries
        if entry.get("type") == "compaction" and isinstance(entry.get("id"), str)
    )


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: object) -> int | None:
    parsed = _int(value)
    return parsed if parsed is not None and parsed > 0 else None

