from __future__ import annotations

from pathlib import Path
from typing import Mapping

from ..provider_contracts import (
    AgentContentBlock,
    AgentError,
    AgentEvent,
    AgentSessionUsage,
    AgentSessionView,
    AgentToolCall,
    AgentTurnUsage,
    AgentTurnView,
    ModelBackendIdentity,
    ModelRequestUsage,
    Page,
    ProviderEventQuery,
    ProviderRunState,
    ProviderSessionListQuery,
    ProviderSessionLocator,
    ProviderSessionQuery,
    ProviderToolQuery,
    ProviderTurnLocator,
    ProviderTurnQuery,
    ProviderTurnResult,
    ProviderUsageQuery,
    TokenUsage,
)
from .openai_agents_storage import OpenAIAgentsSessionStore, load_json


class OpenAIAgentsQueryAdapter:
    provider_type = "openai_agents"

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)

    def list_sessions(self, query: ProviderSessionListQuery) -> Page:
        if query.home_id is None:
            raise ValueError("OpenAI Agents session listing requires home_id")
        root = self._home_root(query.home_id) / "sessions"
        paths = sorted(root.glob("*.sqlite3")) if root.exists() else []
        views: list[AgentSessionView] = []
        for path in paths:
            session_id = path.stem
            try:
                store = OpenAIAgentsSessionStore(path, session_id=session_id, home_id=query.home_id)
                views.append(self._session_view(store, include_turns=False))
            except Exception:
                continue
        start, end = _page(query.cursor, query.limit, len(views))
        return Page(items=tuple(views[start:end]), next_cursor=str(end) if end < len(views) else None)

    def read_session(self, query: ProviderSessionQuery) -> AgentSessionView:
        return self._session_view(self._store(query.locator), include_turns=query.include_turns)

    def list_turns(self, query: ProviderTurnQuery) -> Page:
        store = self._store(query.session)
        rows = store.turn_rows()
        if query.turn is not None:
            rows = [row for row in rows if row["turn_id"] == query.turn.turn_id]
        if query.latest:
            rows = rows[-1:]
        start, end = _page(query.cursor, query.limit, len(rows))
        return Page(
            items=tuple(self._turn_view(query.session, row) for row in rows[start:end]),
            next_cursor=str(end) if end < len(rows) else None,
        )

    def read_turn(self, query: ProviderTurnQuery) -> AgentTurnView | None:
        page = self.list_turns(
            ProviderTurnQuery(
                session=query.session,
                turn=query.turn,
                limit=1,
                latest=query.latest or query.turn is None,
            )
        )
        return page.items[0] if page.items else None  # type: ignore[return-value]

    def list_events(self, query: ProviderEventQuery) -> Page:
        rows = self._store(query.session).event_rows()
        if query.turn is not None:
            rows = [row for row in rows if row["turn_id"] == query.turn.turn_id]
        if query.kind is not None:
            rows = [row for row in rows if row["kind"] == query.kind]
        if query.latest:
            rows = rows[-1:]
        start, end = _page(query.cursor, query.limit, len(rows))
        return Page(
            items=tuple(_event(load_json(row["event_json"])) for row in rows[start:end]),
            next_cursor=str(end) if end < len(rows) else None,
        )

    def list_tool_calls(self, query: ProviderToolQuery) -> Page:
        turns = self._selected_turns(query)
        calls = [call for turn in turns for call in (turn.result.tool_calls if turn.result else ())]
        if query.call_id is not None:
            calls = [call for call in calls if call.call_id == query.call_id]
        start, end = _page(query.cursor, query.limit, len(calls))
        return Page(items=tuple(calls[start:end]), next_cursor=str(end) if end < len(calls) else None)

    def read_usage(self, query: ProviderUsageQuery) -> AgentTurnUsage | AgentSessionUsage:
        turns = self._selected_turns(query)
        usages = tuple(turn.usage for turn in turns if turn.usage is not None)
        if query.include_session_aggregate:
            aggregate = TokenUsage.aggregate_complete(tuple(item.token_usage for item in usages))
            return AgentSessionUsage(
                turn_count=len(turns),
                request_count=sum(item.request_count or 0 for item in usages) if usages else 0,
                token_usage=aggregate,
                turns=usages,
                aggregate_complete=bool(usages) and all(item.aggregate_complete for item in usages),
            )
        return usages[-1] if usages else AgentTurnUsage(
            request_count=None,
            requests=(),
            token_usage=TokenUsage(),
            aggregate_complete=False,
        )

    def _selected_turns(self, query: ProviderTurnQuery) -> list[AgentTurnView]:
        page = self.list_turns(
            ProviderTurnQuery(
                session=query.session,
                turn=query.turn,
                cursor=query.cursor,
                limit=query.limit,
                latest=query.latest,
            )
        )
        return [item for item in page.items if isinstance(item, AgentTurnView)]

    def _session_view(self, store: OpenAIAgentsSessionStore, *, include_turns: bool) -> AgentSessionView:
        row = store.session_row()
        identity = _model(load_json(row["backend_json"]))
        locator = ProviderSessionLocator(
            provider_type=self.provider_type,
            session_id=str(row["session_id"]),
            home_id=str(row["home_id"]),
            created_at=str(row["created_at"]),
            backend_identity=identity,
            native_locator={"sqlite_relpath": f"sessions/{row['session_id']}.sqlite3"},
        )
        turns = tuple(self._turn_view(locator, item) for item in store.turn_rows()) if include_turns else ()
        usages = tuple(turn.usage for turn in turns if turn.usage is not None)
        usage = None
        if include_turns:
            usage = AgentSessionUsage(
                turn_count=len(turns),
                request_count=sum(item.request_count or 0 for item in usages),
                token_usage=TokenUsage.aggregate_complete(tuple(item.token_usage for item in usages)),
                turns=usages,
                aggregate_complete=bool(usages) and all(item.aggregate_complete for item in usages),
            )
        return AgentSessionView(locator=locator, status=str(row["status"]), turns=turns, usage=usage)

    def _turn_view(self, session: ProviderSessionLocator, row: Mapping[str, object]) -> AgentTurnView:
        locator = ProviderTurnLocator(
            session=session,
            turn_id=str(row["turn_id"]),
            sequence=int(row["sequence"]),
        )
        payload = load_json(row["result_json"] if isinstance(row["result_json"], str) else None)
        result = _result(payload, session=session, fallback_locator=locator) if isinstance(payload, Mapping) else None
        return AgentTurnView(
            locator=locator,
            result=result,
            tool_calls=result.tool_calls if result else (),
            usage=result.turn_usage if result else None,
        )

    def _store(self, locator: ProviderSessionLocator) -> OpenAIAgentsSessionStore:
        if locator.provider_type != self.provider_type:
            raise ValueError(f"OpenAI Agents query received {locator.provider_type}")
        return OpenAIAgentsSessionStore(
            OpenAIAgentsSessionStore.path_for(self._home_root(locator.home_id), locator.session_id),
            session_id=locator.session_id,
            home_id=locator.home_id,
        )

    def _home_root(self, home_id: str) -> Path:
        return self.runtime_root / "homes" / self.provider_type / home_id


def _result(value: Mapping[str, object], *, session: ProviderSessionLocator, fallback_locator: ProviderTurnLocator) -> ProviderTurnResult:
    turn_payload = value.get("turn_locator")
    turn = fallback_locator
    if isinstance(turn_payload, Mapping):
        turn = ProviderTurnLocator(
            session=session,
            turn_id=str(turn_payload["turn_id"]),
            request_ids=tuple(str(item) for item in turn_payload.get("request_ids", [])),
            sequence=int(turn_payload["sequence"]) if turn_payload.get("sequence") is not None else None,
        )
    requests = tuple(_request_usage(item) for item in value.get("request_usages", []) if isinstance(item, Mapping))
    turn_usage = _turn_usage(value.get("turn_usage"), requests)
    error = _error(value.get("error"))
    return ProviderTurnResult(
        provider_type=str(value.get("provider_type") or "openai_agents"),
        run_id=str(value["run_id"]),
        session_locator=session,
        turn_locator=turn,
        status=ProviderRunState(str(value["status"])),
        started_at=str(value["started_at"]),
        completed_at=str(value["completed_at"]),
        duration_ms=float(value["duration_ms"]) if value.get("duration_ms") is not None else None,
        final_text=_optional_str(value.get("final_text")),
        structured_output=value.get("structured_output"),
        content_blocks=tuple(_block(item) for item in value.get("content_blocks", []) if isinstance(item, Mapping)),
        tool_calls=tuple(_tool(item) for item in value.get("tool_calls", []) if isinstance(item, Mapping)),
        request_usages=requests,
        turn_usage=turn_usage,
        error=error,
        event_cursor=_optional_str(value.get("event_cursor")),
    )


def _request_usage(value: Mapping[str, object]) -> ModelRequestUsage:
    return ModelRequestUsage(
        request_index=int(value["request_index"]),
        model_identity=_model(value["model_identity"]) or ModelBackendIdentity("unknown", "unknown"),
        token_usage=_tokens(value.get("token_usage")),
        request_id=_optional_str(value.get("request_id")),
        response_id=_optional_str(value.get("response_id")),
        session_id=_optional_str(value.get("session_id")),
        turn_id=_optional_str(value.get("turn_id")),
        status=_optional_str(value.get("status")),
        stop_reason=_optional_str(value.get("stop_reason")),
        reported_fields=tuple(str(item) for item in value.get("reported_fields", [])),
        derived_fields=tuple(str(item) for item in value.get("derived_fields", [])),
        estimated_fields=tuple(str(item) for item in value.get("estimated_fields", [])),
        unavailable_fields=dict(value.get("unavailable_fields") or {}),
    )


def _turn_usage(value: object, requests: tuple[ModelRequestUsage, ...]) -> AgentTurnUsage | None:
    if not isinstance(value, Mapping):
        return AgentTurnUsage.from_requests(requests) if requests else None
    return AgentTurnUsage(
        request_count=int(value["request_count"]) if value.get("request_count") is not None else None,
        requests=requests,
        token_usage=_tokens(value.get("token_usage")),
        models_used=tuple(_model(item) for item in value.get("models_used", []) if _model(item) is not None),  # type: ignore[arg-type]
        aggregate_complete=bool(value.get("aggregate_complete", False)),
    )


def _tokens(value: object) -> TokenUsage:
    raw = value if isinstance(value, Mapping) else {}
    names = (
        "input_tokens", "output_tokens", "total_tokens", "uncached_input_tokens",
        "cached_input_tokens", "cache_read_input_tokens", "cache_write_input_tokens",
        "cache_creation_input_tokens", "cache_creation_5m_input_tokens",
        "cache_creation_1h_input_tokens", "reasoning_output_tokens", "visible_output_tokens",
    )
    return TokenUsage(
        **{name: int(raw[name]) if raw.get(name) is not None else None for name in names},
        input_tokens_by_modality=dict(raw.get("input_tokens_by_modality") or {}),
        output_tokens_by_modality=dict(raw.get("output_tokens_by_modality") or {}),
        other_token_details=dict(raw.get("other_token_details") or {}),
        semantics=dict(raw.get("semantics") or {}),
    )


def _model(value: object) -> ModelBackendIdentity | None:
    if not isinstance(value, Mapping):
        return None
    return ModelBackendIdentity(
        api_provider=str(value["api_provider"]),
        api_mode=str(value["api_mode"]),
        endpoint_id=_optional_str(value.get("endpoint_id")),
        requested_model=_optional_str(value.get("requested_model")),
        resolved_model=_optional_str(value.get("resolved_model")),
        model_version=_optional_str(value.get("model_version")),
        service_tier=_optional_str(value.get("service_tier")),
        reasoning_effort=_optional_str(value.get("reasoning_effort")),
        tokenizer_id=_optional_str(value.get("tokenizer_id")),
        model_config_hash=_optional_str(value.get("model_config_hash")),
    )


def _block(value: Mapping[str, object]) -> AgentContentBlock:
    return AgentContentBlock(
        kind=str(value["kind"]), data=value.get("data"), block_id=_optional_str(value.get("block_id")),
        sequence=int(value["sequence"]) if value.get("sequence") is not None else None,
        parent_id=_optional_str(value.get("parent_id")), call_id=_optional_str(value.get("call_id")),
        model_request_id=_optional_str(value.get("model_request_id")),
    )


def _tool(value: Mapping[str, object]) -> AgentToolCall:
    return AgentToolCall(
        call_id=str(value["call_id"]), tool_name=str(value["tool_name"]),
        tool_kind=str(value["tool_kind"]), status=str(value["status"]),
        turn_id=_optional_str(value.get("turn_id")), request_id=_optional_str(value.get("request_id")),
        display_name=_optional_str(value.get("display_name")), server_name=_optional_str(value.get("server_name")),
        arguments=value.get("arguments"), result=value.get("result"), approval=value.get("approval"),
        error=_error(value.get("error")),
    )


def _error(value: object) -> AgentError | None:
    if not isinstance(value, Mapping):
        return None
    return AgentError(
        error_type=str(value["error_type"]), message=str(value["message"]),
        code=_optional_str(value.get("code")), retryable=value.get("retryable") if isinstance(value.get("retryable"), bool) else None,
    )


def _event(value: object) -> AgentEvent:
    if not isinstance(value, Mapping):
        raise ValueError("stored OpenAI Agents event must be an object")
    return AgentEvent(
        provider_type=str(value["provider_type"]), sequence=int(value["sequence"]),
        timestamp=str(value["timestamp"]), kind=str(value["kind"]),
        session_id=_optional_str(value.get("session_id")), turn_id=_optional_str(value.get("turn_id")),
        request_id=_optional_str(value.get("request_id")), phase=_optional_str(value.get("phase")),
        block_id=_optional_str(value.get("block_id")), call_id=_optional_str(value.get("call_id")),
        parent_id=_optional_str(value.get("parent_id")), terminal=bool(value.get("terminal", False)),
        data=value.get("data"),
    )


def _page(cursor: str | None, limit: int, length: int) -> tuple[int, int]:
    if limit <= 0:
        raise ValueError("page limit must be positive")
    start = int(cursor) if cursor is not None else 0
    if start < 0:
        raise ValueError("page cursor must not be negative")
    return start, min(start + limit, length)


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None
