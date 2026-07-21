from __future__ import annotations

import asyncio
import threading
import uuid
from dataclasses import replace
from typing import Callable, Mapping, TypeVar

from ..models import (
    AgentContextCompactionEvidenceError,
    AgentContextCompactionRequestUnknown,
    AgentContextMaintenanceUnsupported,
)
from ..provider_contracts import (
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    ContextUsageCategory,
    ProviderContextCompactionRequest,
    ProviderContextCompactionResult,
    ProviderContextQuery,
    ProviderContextReconcileRequest,
    ProviderContextUsage,
    ProviderExecutionContext,
    ProviderRunRequest,
    build_provider_payload,
)
from ..store_utils import read_json, utc_now_iso
from .claude_code import ClaudeCodeProvider
from .claude_code_normalization import (
    compact_boundary_snapshot,
    find_new_compact_boundary,
)
from .claude_code_runtime import _build_options


T = TypeVar("T")


class ClaudeCodeContextAdapter:
    provider_type = "claude_code"

    def __init__(self, provider: ClaudeCodeProvider) -> None:
        self.provider = provider

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage:
        context = _context(request.execution_context)
        try:
            raw = _run_sync(
                lambda: self._inspect_async(
                    context,
                    request.session.session_id,
                    request.agent_id or "context-inspect",
                )
            )
        except BaseException as exc:
            return ProviderContextUsage(
                session_id=request.session.session_id,
                observed_at=utc_now_iso(),
                source="claude_context_control",
                available=False,
                measurement="unavailable",
                reason=f"{type(exc).__name__}: {str(exc)}",
                compact_capability=_compact_support(context, available=False),
            )
        return _context_usage(request.session.session_id, raw, context)

    def compact(self, request: ProviderContextCompactionRequest) -> ProviderContextCompactionResult:
        context = _context(request.execution_context)
        if not _context_cli_supported(context):
            raise AgentContextMaintenanceUnsupported(
                "Claude compact requires the verified CLI version declared by the Home"
            )
        baseline = compact_boundary_snapshot(context.home_root, request.session.session_id)
        operation_id = f"claude-compact-{uuid.uuid4().hex}"
        if request.on_started is not None:
            request.on_started(dict(baseline), operation_id)
        started_at = utc_now_iso()
        try:
            terminal, raw_usage = _run_sync(
                lambda: self._compact_async(
                    context,
                    request.session.session_id,
                    request.agent_id or "context-compact",
                    request.timeout_s,
                )
            )
        except TimeoutError as exc:
            raise AgentContextCompactionRequestUnknown(
                "Claude compact request was sent but terminal Result was not observed"
            ) from exc
        if bool(getattr(terminal, "is_error", False)):
            raise AgentContextCompactionEvidenceError(
                f"Claude compact returned terminal error: {getattr(terminal, 'subtype', 'unknown')}"
            )
        boundary = find_new_compact_boundary(
            context.home_root,
            request.session.session_id,
            baseline,
        )
        if boundary is None:
            raise AgentContextCompactionEvidenceError(
                "Claude compact terminal Result had no new compact_boundary transcript evidence"
            )
        completed_at = utc_now_iso()
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="compacted",
            reason="provider_compact_boundary_confirmed",
            started_at=started_at,
            completed_at=completed_at,
            usage_after=(
                _context_usage(request.session.session_id, raw_usage, context)
                if raw_usage is not None
                else None
            ),
            provider_operation_id=str(boundary.get("uuid") or operation_id),
            provider_payload=build_provider_payload(
                provider_type=self.provider_type,
                payload_type="compact_evidence",
                adapter_version="1",
                data={
                    "operation_id": operation_id,
                    "terminal_subtype": getattr(terminal, "subtype", None),
                    "boundary_uuid": boundary.get("uuid"),
                    "baseline_boundary_count": baseline.get("boundary_count"),
                },
            ),
        )

    def reconcile(
        self,
        request: ProviderContextReconcileRequest,
    ) -> ProviderContextCompactionResult | None:
        context = _context(request.execution_context)
        if not isinstance(request.baseline, Mapping):
            return None
        boundary = find_new_compact_boundary(
            context.home_root,
            request.session.session_id,
            request.baseline,
        )
        if boundary is None:
            return None
        completed = str(boundary.get("timestamp") or utc_now_iso())
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="compacted",
            reason="provider_compact_boundary_reconciled",
            started_at=completed,
            completed_at=completed,
            provider_operation_id=str(boundary.get("uuid") or request.operation_id or ""),
        )

    async def _inspect_async(
        self,
        context: ProviderExecutionContext,
        session_id: str,
        agent_id: str,
    ) -> Mapping[str, object]:
        sdk = self.provider.sdk()
        request = _session_request(context, session_id, agent_id)
        client = sdk.ClaudeSDKClient(
            options=_build_options(sdk, request, session_id=session_id, resume=True)
        )
        try:
            await client.connect()
            value = await client.get_context_usage()
            if not isinstance(value, Mapping):
                raise TypeError("Claude get_context_usage returned a non-mapping")
            return value
        finally:
            await client.disconnect()

    async def _compact_async(
        self,
        context: ProviderExecutionContext,
        session_id: str,
        agent_id: str,
        timeout_s: float | None,
    ) -> tuple[object, Mapping[str, object] | None]:
        sdk = self.provider.sdk()
        request = _session_request(context, session_id, agent_id)
        client = sdk.ClaudeSDKClient(
            options=_build_options(sdk, request, session_id=session_id, resume=True)
        )
        try:
            await client.connect()
            await client.query("/compact Preserve the current task state for continuation.")

            async def receive_terminal() -> object:
                terminal = None
                async for message in client.receive_response():
                    if type(message).__name__ == "ResultMessage":
                        terminal = message
                if terminal is None:
                    raise RuntimeError("Claude compact response ended without ResultMessage")
                return terminal

            try:
                terminal = await asyncio.wait_for(receive_terminal(), timeout=timeout_s)
            except asyncio.TimeoutError as exc:
                raise TimeoutError("Claude compact terminal timeout") from exc
            try:
                usage = await client.get_context_usage()
            except Exception:
                usage = None
            return terminal, usage if isinstance(usage, Mapping) else None
        finally:
            await client.disconnect()


def _session_request(
    context: ProviderExecutionContext,
    session_id: str,
    agent_id: str,
) -> ProviderRunRequest:
    return ProviderRunRequest(
        agent_id=agent_id,
        scope_id="context-maintenance",
        agent_type="ContextMaintenance",
        provider_type="claude_code",
        home_id=context.home_id,
        session_locator=None,
        prompt="",
        workdir=context.workdir,
        environment=context.process_environment,
        model_overrides=context.resolved_defaults,
        execution_context=context,
    )


def _context_usage(
    session_id: str,
    raw: Mapping[str, object],
    context: ProviderExecutionContext,
) -> ProviderContextUsage:
    categories = tuple(
        ContextUsageCategory(
            kind=_category_kind(str(item.get("name") or "other")),
            name=str(item.get("name") or "other"),
            tokens=_int(item.get("tokens")),
            deferred=bool(item.get("isDeferred")) if "isDeferred" in item else None,
            measurement="provider_reported",
        )
        for item in raw.get("categories") or []
        if isinstance(item, Mapping)
    )
    used = _int(raw.get("totalTokens"))
    effective = _positive_int(raw.get("maxTokens"))
    window = _positive_int(raw.get("rawMaxTokens"))
    model = str(raw.get("model")) if raw.get("model") is not None else None
    base = context.resolved_defaults
    identity = replace(base, resolved_model=model) if base is not None and model else base
    return ProviderContextUsage(
        session_id=session_id,
        observed_at=utc_now_iso(),
        source="claude_context_control",
        available=used is not None,
        used_tokens=used,
        context_window_tokens=window,
        effective_context_window_tokens=effective,
        remaining_tokens=(max(0, effective - used) if used is not None and effective is not None else None),
        categories=categories,
        auto_compact_enabled=(
            bool(raw.get("isAutoCompactEnabled"))
            if "isAutoCompactEnabled" in raw
            else None
        ),
        auto_compact_threshold_tokens=_int(raw.get("autoCompactThreshold")),
        compact_capability=_compact_support(context, available=_context_cli_supported(context)),
        measurement="provider_reported",
        model_identity=identity,
        provider_payload=build_provider_payload(
            provider_type="claude_code",
            payload_type="context_usage",
            adapter_version="1",
            data={
                "percentage": raw.get("percentage"),
                "memoryFiles": raw.get("memoryFiles"),
                "mcpTools": raw.get("mcpTools"),
                "agents": raw.get("agents"),
            },
        ),
    )


def _compact_support(
    context: ProviderExecutionContext,
    *,
    available: bool,
) -> CapabilitySupport:
    return CapabilitySupport(
        capability=CapabilityKey.CONTROL_COMPACT,
        status=CapabilityStatus.ADAPTABLE if available else CapabilityStatus.UNVERIFIED,
        available=available,
        reason=None if available else "Claude CLI context/compact control version is unverified",
        resolved_for_home_id=context.home_id,
        evidence_version="claude-cli-2.1.216",
    )


def _context_cli_supported(context: ProviderExecutionContext) -> bool:
    marker = context.home_root / ".ark" / "claude_home_initialized.json"
    if not marker.is_file():
        return False
    payload = read_json(marker)
    actual = _version_tuple(payload.get("cli_version"))
    config = context.runtime_payload if isinstance(context.runtime_payload, Mapping) else {}
    required = _version_tuple(config.get("minimum_context_cli_version"))
    return actual is not None and required is not None and actual >= required


def _version_tuple(value: object) -> tuple[int, ...] | None:
    if value is None:
        return None
    text = str(value)
    for token in text.replace("(", " ").split():
        parts = token.strip("v,)").split(".")
        if len(parts) >= 2 and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)
    return None


def _run_sync(factory: Callable[[], "asyncio.Future[T] | object"]) -> T:
    result: list[T] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            result.append(asyncio.run(factory()))  # type: ignore[arg-type]
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()
    if errors:
        raise errors[0]
    if not result:
        raise RuntimeError("Claude async helper produced no result")
    return result[0]


def _context(value: ProviderExecutionContext | None) -> ProviderExecutionContext:
    if value is None or value.provider_type != "claude_code":
        raise ValueError("Claude context operation requires ProviderExecutionContext")
    return value


def _category_kind(name: str) -> str:
    lowered = name.lower()
    for key in ("system", "instruction", "message", "tool", "skill", "memory", "agent"):
        if key in lowered:
            return "tools" if key == "tool" else key + "s" if key in {"instruction", "message", "agent"} else key
    return "other"


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
