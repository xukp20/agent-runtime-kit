from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Callable, Mapping, Sequence

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
    ReportedCost,
    TokenUsage,
    build_provider_payload,
)
from .opencode_client import OpenCodeClient
from .opencode_models import ADAPTER_VERSION, PROVIDER_TYPE


ClientResolver = Callable[[ProviderSessionLocator], OpenCodeClient]


class OpenCodeQueryAdapter:
    provider_type = PROVIDER_TYPE

    def __init__(self, client_resolver: ClientResolver) -> None:
        self._client_resolver = client_resolver

    def list_sessions(self, query: ProviderSessionListQuery) -> Page:
        del query
        # Session enumeration across per-Agent databases needs an Agent/native locator.
        return Page(items=())

    def read_session(self, query: ProviderSessionQuery) -> AgentSessionView:
        client = self._client_resolver(query.locator)
        session = client.get_session(query.locator.session_id)
        turns = self._turns(query.locator, client.list_messages(query.locator.session_id))
        status = _session_status(client.session_status(), query.locator.session_id)
        usage = _session_usage(turns)
        return AgentSessionView(
            locator=query.locator,
            status=status,
            turns=turns if query.include_turns else (),
            usage=usage,
        )

    def list_turns(self, query: ProviderTurnQuery) -> Page:
        turns = list(self._turns(query.session, self._messages(query.session)))
        if query.latest:
            turns = turns[-1:]
        start = int(query.cursor or 0)
        items = turns[start : start + query.limit]
        next_cursor = str(start + query.limit) if start + query.limit < len(turns) else None
        return Page(items=tuple(items), next_cursor=next_cursor)

    def read_turn(self, query: ProviderTurnQuery) -> AgentTurnView | None:
        turns = self._turns(query.session, self._messages(query.session))
        if query.turn is not None:
            return next((turn for turn in turns if turn.locator.turn_id == query.turn.turn_id), None)
        return turns[-1] if turns else None

    def list_events(self, query: ProviderEventQuery) -> Page:
        turn = self.read_turn(query)
        events = list(turn.events if turn else ())
        if query.kind is not None:
            events = [event for event in events if event.kind == query.kind]
        start = int(query.cursor or 0)
        items = events[start : start + query.limit]
        next_cursor = str(start + query.limit) if start + query.limit < len(events) else None
        return Page(items=tuple(items), next_cursor=next_cursor)

    def list_tool_calls(self, query: ProviderToolQuery) -> Page:
        turn = self.read_turn(query)
        tools = list(turn.tool_calls if turn else ())
        if query.call_id is not None:
            tools = [tool for tool in tools if tool.call_id == query.call_id]
        start = int(query.cursor or 0)
        items = tools[start : start + query.limit]
        next_cursor = str(start + query.limit) if start + query.limit < len(tools) else None
        return Page(items=tuple(items), next_cursor=next_cursor)

    def read_usage(self, query: ProviderUsageQuery) -> AgentTurnUsage | AgentSessionUsage:
        turns = self._turns(query.session, self._messages(query.session))
        if query.include_session_aggregate:
            return _session_usage(turns)
        turn = None
        if query.turn is not None:
            turn = next((item for item in turns if item.locator.turn_id == query.turn.turn_id), None)
        elif turns:
            turn = turns[-1]
        return turn.usage if turn and turn.usage is not None else AgentTurnUsage.from_requests(())

    def _messages(self, locator: ProviderSessionLocator) -> list[object]:
        return self._client_resolver(locator).list_messages(locator.session_id)

    def _turns(
        self, locator: ProviderSessionLocator, messages: Sequence[object]
    ) -> tuple[AgentTurnView, ...]:
        return project_turns(locator, messages)


def project_turns(
    session: ProviderSessionLocator, messages: Sequence[object]
) -> tuple[AgentTurnView, ...]:
    normalized = [_message(value) for value in messages]
    users = [value for value in normalized if value[0].get("role") == "user"]
    assistants = [value for value in normalized if value[0].get("role") == "assistant"]
    turns: list[AgentTurnView] = []
    for sequence, (user_info, user_parts, raw_user) in enumerate(users):
        turn_id = str(user_info.get("id") or f"turn-{sequence}")
        related = [item for item in assistants if str(item[0].get("parentID") or "") == turn_id]
        turns.append(_project_turn(session, turn_id, sequence, user_info, user_parts, raw_user, related))
    return tuple(turns)


def completed_turn_result(
    *,
    session: ProviderSessionLocator,
    messages: Sequence[object],
    turn_id: str,
    run_id: str,
    started_at: str,
    status: ProviderRunState = ProviderRunState.COMPLETED,
    error: AgentError | None = None,
    artifact_locator=None,
) -> ProviderTurnResult:
    turn = next((item for item in project_turns(session, messages) if item.locator.turn_id == turn_id), None)
    if turn is None:
        return ProviderTurnResult(
            provider_type=PROVIDER_TYPE,
            run_id=run_id,
            session_locator=session,
            turn_locator=ProviderTurnLocator(session=session, turn_id=turn_id),
            status=status,
            started_at=started_at,
            completed_at=_now(),
            error=error,
            artifact_locator=artifact_locator,
        )
    result = turn.result
    assert result is not None
    return ProviderTurnResult(
        provider_type=PROVIDER_TYPE,
        run_id=run_id,
        session_locator=session,
        turn_locator=turn.locator,
        status=status,
        started_at=started_at,
        completed_at=result.completed_at,
        final_text=result.final_text,
        content_blocks=result.content_blocks,
        tool_calls=result.tool_calls,
        request_usages=result.request_usages,
        turn_usage=result.turn_usage,
        error=error or result.error,
        artifact_locator=artifact_locator,
        provider_payload=result.provider_payload,
    )


def _project_turn(
    session: ProviderSessionLocator,
    turn_id: str,
    sequence: int,
    user_info: Mapping[str, object],
    user_parts: Sequence[object],
    raw_user: object,
    assistants: Sequence[tuple[Mapping[str, object], Sequence[object], object]],
) -> AgentTurnView:
    locator = ProviderTurnLocator(session=session, turn_id=turn_id, sequence=sequence)
    blocks: list[AgentContentBlock] = []
    tools: list[AgentToolCall] = []
    events: list[AgentEvent] = []
    requests: list[ModelRequestUsage] = []
    final_texts: list[str] = []
    error: AgentError | None = None
    raw_messages: list[object] = [raw_user]
    for info, parts, raw in assistants:
        raw_messages.append(raw)
        request_index = len(requests)
        requests.append(_request_usage(info, turn_id, request_index, session.backend_identity))
        if info.get("error") is not None:
            error = AgentError(
                error_type="opencode_session_error",
                message=str(info.get("error")),
                provider_payload=build_provider_payload(
                    provider_type=PROVIDER_TYPE,
                    payload_type="assistant_error",
                    data=info.get("error"),
                    adapter_version=ADAPTER_VERSION,
                ),
            )
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            block, tool = _part_projection(part, turn_id, len(blocks))
            blocks.append(block)
            if tool is not None:
                tools.append(tool)
            if block.kind == "text" and isinstance(block.data, Mapping):
                text = block.data.get("text")
                if isinstance(text, str) and text:
                    final_texts.append(text)
            events.append(
                AgentEvent(
                    provider_type=PROVIDER_TYPE,
                    sequence=len(events),
                    timestamp=_part_timestamp(part) or _message_time(info, "created") or _now(),
                    kind=f"message.part.{block.kind}",
                    session_id=session.session_id,
                    turn_id=turn_id,
                    block_id=block.block_id,
                    call_id=block.call_id,
                    data=block.data,
                    provider_payload=block.provider_payload,
                )
            )
    usage = AgentTurnUsage.from_requests(tuple(requests))
    started = _message_time(user_info, "created") or _now()
    completed = next(
        (_message_time(info, "completed") for info, _, _ in reversed(assistants) if _message_time(info, "completed")),
        _now(),
    )
    status = ProviderRunState.FAILED if error is not None else ProviderRunState.COMPLETED
    result = ProviderTurnResult(
        provider_type=PROVIDER_TYPE,
        run_id=f"query-{turn_id}",
        session_locator=session,
        turn_locator=locator,
        status=status,
        started_at=started,
        completed_at=completed,
        final_text="\n".join(final_texts) or None,
        content_blocks=tuple(blocks),
        tool_calls=tuple(tools),
        request_usages=tuple(requests),
        turn_usage=usage,
        error=error,
        provider_payload=build_provider_payload(
            provider_type=PROVIDER_TYPE,
            payload_type="turn_messages",
            data=raw_messages,
            adapter_version=ADAPTER_VERSION,
        ),
    )
    return AgentTurnView(locator=locator, result=result, events=tuple(events), tool_calls=tuple(tools), usage=usage)


def _part_projection(
    part: Mapping[str, object], turn_id: str, sequence: int
) -> tuple[AgentContentBlock, AgentToolCall | None]:
    kind = str(part.get("type") or "other")
    payload = build_provider_payload(
        provider_type=PROVIDER_TYPE,
        payload_type="message_part",
        data=part,
        adapter_version=ADAPTER_VERSION,
    )
    block_id = _optional(part.get("id"))
    call_id = _optional(part.get("callID") or part.get("callId"))
    tool: AgentToolCall | None = None
    data: object
    if kind == "text":
        data = {"text": part.get("text")}
    elif kind == "reasoning":
        data = {"text": part.get("text"), "metadata": part.get("metadata")}
    elif kind == "tool":
        state = part.get("state") if isinstance(part.get("state"), Mapping) else {}
        call_id = call_id or str(part.get("id") or f"tool-{sequence}")
        status = str(state.get("status") or "unknown")
        tool = AgentToolCall(
            call_id=call_id,
            tool_name=str(part.get("tool") or "unknown"),
            tool_kind="tool",
            status=status,
            turn_id=turn_id,
            arguments=state.get("input"),
            result=state.get("output"),
            started_at=_epoch_iso(_nested(state, "time", "start")),
            completed_at=_epoch_iso(_nested(state, "time", "end")),
            error=(
                AgentError(error_type="tool_error", message=str(state.get("error")))
                if state.get("error") is not None
                else None
            ),
            provider_payload=payload,
        )
        data = {"tool": part.get("tool"), "state": status}
    elif kind in {"step-start", "step-finish", "file", "retry", "compaction", "snapshot", "patch"}:
        data = dict(part)
    else:
        data = {"provider_part_type": kind}
    return (
        AgentContentBlock(
            kind=kind,
            data=data,
            block_id=block_id,
            sequence=sequence,
            parent_id=_optional(part.get("messageID") or part.get("messageId")),
            call_id=call_id,
            provider_payload=payload,
        ),
        tool,
    )


def _request_usage(
    info: Mapping[str, object],
    turn_id: str,
    index: int,
    session_backend: ModelBackendIdentity | None,
) -> ModelRequestUsage:
    tokens = info.get("tokens") if isinstance(info.get("tokens"), Mapping) else {}
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), Mapping) else {}
    provider_id = str(info.get("providerID") or "unknown")
    model_id = str(info.get("modelID") or "unknown")
    identity = ModelBackendIdentity(
        api_provider=provider_id,
        api_mode=(
            session_backend.api_mode
            if session_backend is not None and session_backend.api_provider == provider_id
            else _api_mode(info, provider_id)
        ),
        requested_model=model_id,
        resolved_model=model_id,
    )
    reported = tuple(
        name
        for name, value in {
            "input_tokens": tokens.get("input"),
            "output_tokens": tokens.get("output"),
            "reasoning_output_tokens": tokens.get("reasoning"),
            "cache_read_input_tokens": cache.get("read"),
            "cache_write_input_tokens": cache.get("write"),
        }.items()
        if isinstance(value, int)
    )
    cost = info.get("cost")
    return ModelRequestUsage(
        request_index=index,
        request_id=_optional(info.get("id")),
        session_id=_optional(info.get("sessionID")),
        turn_id=turn_id,
        model_identity=identity,
        token_usage=TokenUsage(
            input_tokens=_integer(tokens.get("input")),
            output_tokens=_integer(tokens.get("output")),
            reasoning_output_tokens=_integer(tokens.get("reasoning")),
            cache_read_input_tokens=_integer(cache.get("read")),
            cache_write_input_tokens=_integer(cache.get("write")),
        ),
        reported_cost=(
            ReportedCost(currency="USD", total_cost=str(Decimal(str(cost))), provider_payload=build_provider_payload(
                provider_type=PROVIDER_TYPE,
                payload_type="opencode_cost",
                data={"cost": cost, "origin": "opencode_model_catalog"},
                adapter_version=ADAPTER_VERSION,
            ))
            if isinstance(cost, (int, float, str))
            else None
        ),
        status="failed" if info.get("error") is not None else "completed",
        stop_reason=_optional(info.get("finish")),
        started_at=_message_time(info, "created"),
        completed_at=_message_time(info, "completed"),
        reported_fields=reported,
        unavailable_fields={"total_tokens": "OpenCode did not report an authoritative total"},
        provider_payload=build_provider_payload(
            provider_type=PROVIDER_TYPE,
            payload_type="assistant_info",
            data=info,
            adapter_version=ADAPTER_VERSION,
        ),
    )


def _session_usage(turns: Sequence[AgentTurnView]) -> AgentSessionUsage:
    usages = tuple(turn.usage for turn in turns if turn.usage is not None)
    request_count = sum(item.request_count or 0 for item in usages)
    return AgentSessionUsage(
        turn_count=len(turns),
        request_count=request_count,
        token_usage=TokenUsage.aggregate_complete(tuple(item.token_usage for item in usages)),
        turns=usages,
        aggregate_complete=bool(usages) and all(item.aggregate_complete for item in usages),
    )


def _message(value: object) -> tuple[Mapping[str, object], Sequence[object], object]:
    if not isinstance(value, Mapping):
        return {}, (), value
    info = value.get("info") if isinstance(value.get("info"), Mapping) else value
    parts = value.get("parts") if isinstance(value.get("parts"), list) else ()
    return info, parts, value


def _api_mode(info: Mapping[str, object], provider_id: str) -> str:
    mode = info.get("apiMode") or info.get("api_mode")
    if isinstance(mode, str) and mode:
        return mode
    return "chat_completions" if provider_id == "deepseek" else "other"


def _session_status(statuses: Mapping[str, object], session_id: str) -> str:
    value = statuses.get(session_id)
    if not isinstance(value, Mapping):
        return "idle"
    return str(value.get("type") or value.get("status") or "unknown")


def _message_time(info: Mapping[str, object], key: str) -> str | None:
    time = info.get("time")
    return _epoch_iso(time.get(key)) if isinstance(time, Mapping) else None


def _part_timestamp(part: Mapping[str, object]) -> str | None:
    time = part.get("time")
    if isinstance(time, Mapping):
        return _epoch_iso(time.get("end") or time.get("start"))
    return None


def _epoch_iso(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()


def _nested(value: Mapping[str, object], first: str, second: str) -> object:
    child = value.get(first)
    return child.get(second) if isinstance(child, Mapping) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _optional(value: object) -> str | None:
    return str(value) if value is not None and str(value) else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
