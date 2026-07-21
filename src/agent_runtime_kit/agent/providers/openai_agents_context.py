from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from pathlib import Path
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
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .openai_agents_storage import OpenAIAgentsSessionStore


class OpenAIAgentsContextAdapter:
    provider_type = "openai_agents"

    def inspect(self, request: ProviderContextQuery) -> ProviderContextUsage:
        ctx = _ctx(request.execution_context)
        config = _config(ctx)
        store = _store(ctx.home_root, request.session.session_id, request.session.home_id)
        used = None
        turn_id = None
        rows = store.turn_rows()
        if rows:
            payload = json.loads(rows[-1]["result_json"]) if rows[-1]["result_json"] else {}
            turn_id = rows[-1]["turn_id"]
            usages = payload.get("request_usages") if isinstance(payload, dict) else None
            if isinstance(usages, list) and usages:
                tokens = usages[-1].get("token_usage") if isinstance(usages[-1], dict) else None
                if isinstance(tokens, dict) and tokens.get("input_tokens") is not None:
                    used = int(tokens["input_tokens"])
        window = _positive_int(config.get("context_window_tokens"))
        max_output = _positive_int(config.get("max_output_tokens"))
        remaining = max(window - used, 0) if window is not None and used is not None else None
        compact = _compact_support(config, request.session.home_id, request.session.backend_identity.backend_key if request.session.backend_identity else None)
        return ProviderContextUsage(
            session_id=request.session.session_id,
            observed_at=utc_now_iso(),
            source="agents_sdk_request_usage",
            available=used is not None,
            used_tokens=used,
            context_window_tokens=window,
            effective_context_window_tokens=window,
            max_output_tokens=max_output,
            remaining_tokens=remaining,
            compact_capability=compact,
            measurement="request_usage" if used is not None else "unavailable",
            as_of_turn_id=str(turn_id) if turn_id else None,
            reason=None if used is not None else "no provider-reported request usage is stored",
            model_identity=request.session.backend_identity,
        )

    def compact(self, request: ProviderContextCompactionRequest) -> ProviderContextCompactionResult:
        ctx = _ctx(request.execution_context)
        config = _config(ctx)
        identity = request.session.backend_identity or ctx.resolved_defaults
        support = _compact_support(config, request.session.home_id, identity.backend_key if identity else None)
        if not support.available:
            raise RuntimeError(support.reason or "OpenAI Agents compact is unsupported")
        if identity is None or identity.api_mode != "responses" or not identity.effective_model:
            raise RuntimeError("OpenAI Agents compact requires a Responses model identity")
        store = _store(ctx.home_root, request.session.session_id, request.session.home_id)
        if not store.is_quiescent():
            raise RuntimeError("cannot compact an active OpenAI Agents session")
        operation_id = f"oai-compact-{uuid.uuid4().hex}"
        started_at = utc_now_iso()
        baseline = {"item_count": asyncio.run(_session_item_count(store.path, request.session.session_id))}
        previous_status = str(store.session_row()["status"])
        store.set_session_status("compacting")
        try:
            if request.on_started is not None:
                request.on_started(baseline, operation_id)
            _maintenance(store.path, operation_id, "running", baseline, None, started_at, None)
            raw_types = asyncio.run(
                _compact(
                    store.path,
                    request.session.session_id,
                    identity.effective_model,
                    config,
                    ctx.process_environment,
                )
            )
            completed_at = utc_now_iso()
            result_payload = {"raw_item_types": raw_types, "normalized_type": "compaction_summary"}
            _maintenance(store.path, operation_id, "succeeded", baseline, result_payload, started_at, completed_at)
            return ProviderContextCompactionResult(
                session_id=request.session.session_id,
                status="compacted",
                reason="provider_confirmed_input_history_compaction",
                started_at=started_at,
                completed_at=completed_at,
                usage_after=self.inspect(ProviderContextQuery(session=request.session, agent_id=request.agent_id, execution_context=ctx)),
                provider_operation_id=operation_id,
                provider_payload=build_provider_payload(
                    provider_type=self.provider_type,
                    payload_type="compaction_result",
                    data=result_payload,
                    adapter_version="1",
                    sdk_or_cli_version="0.18.3",
                ),
            )
        except BaseException as exc:
            completed_at = utc_now_iso()
            _maintenance(store.path, operation_id, "failed", baseline, {"error_type": type(exc).__name__}, started_at, completed_at)
            raise
        finally:
            store.set_session_status(previous_status)

    def reconcile(self, request: ProviderContextReconcileRequest) -> ProviderContextCompactionResult | None:
        if request.operation_id is None:
            return None
        ctx = _ctx(request.execution_context)
        path = OpenAIAgentsSessionStore.path_for(ctx.home_root, request.session.session_id)
        with sqlite3.connect(path) as conn:
            row = conn.execute("select * from ark_maintenance where operation_id=?", (request.operation_id,)).fetchone()
        if row is None:
            return None
        status = str(row[1])
        return ProviderContextCompactionResult(
            session_id=request.session.session_id,
            status="compacted" if status == "succeeded" else status,
            reason=f"maintenance_journal_{status}",
            started_at=str(row[4]),
            completed_at=str(row[5] or utc_now_iso()),
            provider_operation_id=request.operation_id,
        )


async def _session_item_count(path: Path, session_id: str) -> int:
    from agents.memory import SQLiteSession

    session = SQLiteSession(session_id, db_path=path)
    try:
        return len(await session.get_items(limit=10**9))
    finally:
        session.close()


async def _compact(path: Path, session_id: str, model: str, config: Mapping[str, object], env: Mapping[str, str]) -> list[str]:
    from agents.memory import OpenAIResponsesCompactionSession, SQLiteSession
    from openai import AsyncOpenAI

    api_key = env.get(str(config["api_key_env"]))
    if not api_key:
        raise RuntimeError("missing OpenAI Agents compact API key")
    base_url = config.get("base_url")
    base_url_env = config.get("base_url_env")
    if base_url_env:
        base_url = env.get(str(base_url_env))
    client = AsyncOpenAI(api_key=api_key, base_url=str(base_url) if base_url else None)
    session = SQLiteSession(session_id, db_path=path)
    wrapper = OpenAIResponsesCompactionSession(
        session_id,
        session,
        client=client,
        model=model,
        compaction_mode="input",
    )
    try:
        await wrapper.run_compaction({"force": True, "compaction_mode": "input", "store": False})
        items = await session.get_items(limit=10**9)
        return [str(item.get("type")) for item in items if isinstance(item, dict) and item.get("type") in {"compaction", "compaction_summary"}]
    finally:
        session.close()
        await client.close()


def _maintenance(path: Path, operation_id: str, status: str, baseline: object, result: object, started_at: str, completed_at: str | None) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """insert into ark_maintenance(operation_id,status,baseline_json,result_json,started_at,completed_at)
               values(?,?,?,?,?,?) on conflict(operation_id) do update set
               status=excluded.status,result_json=excluded.result_json,completed_at=excluded.completed_at""",
            (operation_id, status, json.dumps(baseline), json.dumps(result) if result is not None else None, started_at, completed_at),
        )


def _compact_support(config: Mapping[str, object], home_id: str | None, backend: str | None) -> CapabilitySupport:
    api_mode = None
    identity = config.get("model_identity")
    if isinstance(identity, Mapping):
        api_mode = identity.get("api_mode")
    available = api_mode == "responses" and config.get("compaction_mode") == "input_history"
    return CapabilitySupport(
        capability=CapabilityKey.CONTROL_COMPACT,
        status=CapabilityStatus.ADAPTABLE if available else CapabilityStatus.UNSUPPORTED,
        available=available,
        reason=None if available else "compact requires a verified Responses input-history endpoint",
        limitations=("previous_response_id compaction is not used",) if available else (),
        resolved_for_home_id=home_id,
        resolved_for_backend=backend,
        evidence_version="openai-agents-0.18.3-input-history-v1",
    )


def _ctx(value):  # noqa: ANN001, ANN202
    if value is None or value.provider_type != "openai_agents":
        raise ValueError("OpenAI Agents context adapter requires ProviderExecutionContext")
    return value


def _config(ctx) -> Mapping[str, object]:  # noqa: ANN001
    if not isinstance(ctx.runtime_payload, Mapping):
        raise RuntimeError("OpenAI Agents execution context lacks provider config")
    return ctx.runtime_payload


def _store(home_root: Path, session_id: str, home_id: str) -> OpenAIAgentsSessionStore:
    return OpenAIAgentsSessionStore(OpenAIAgentsSessionStore.path_for(home_root, session_id), session_id=session_id, home_id=home_id)


def _positive_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and value > 0 else None
