from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..provider_contracts import (
    AgentError,
    AgentEvent,
    AgentSessionUsage,
    AgentSessionView,
    AgentToolCall,
    AgentTurnUsage,
    AgentTurnView,
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
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .codex import CodexProvider
from .codex_trace import AgentTraceReader


class CodexQueryAdapter:
    provider_type = "codex"
    adapter_version = "1"

    def __init__(self, *, runtime_root: Path, provider: CodexProvider) -> None:
        self.runtime_root = Path(runtime_root)
        self.provider = provider

    def list_sessions(self, query: ProviderSessionListQuery) -> Page:
        if query.home_id is None:
            raise ValueError("Codex session listing requires home_id")
        sessions_root = self.runtime_root / "homes" / "codex" / query.home_id / ".codex" / "sessions"
        paths = sorted(sessions_root.rglob("*.jsonl")) if sessions_root.exists() else []
        sessions = tuple(
            view
            for path in paths
            if (view := self._session_view(path=path, home_id=query.home_id)) is not None
        )
        start, end = _page_bounds(query.cursor, query.limit, len(sessions))
        return Page(
            items=sessions[start:end],
            next_cursor=str(end) if end < len(sessions) else None,
        )

    def _session_view(self, *, path: Path, home_id: str) -> AgentSessionView | None:
        metadata = _read_session_metadata(path)
        session_id = metadata.get("id") or metadata.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            return None
        created_at = str(
            metadata.get("timestamp")
            or metadata.get("created_at")
            or utc_now_iso()
        )
        home_root = self.runtime_root / "homes" / "codex" / home_id
        return AgentSessionView(
            locator=ProviderSessionLocator(
                provider_type="codex",
                session_id=session_id,
                home_id=home_id,
                created_at=created_at,
                native_locator={
                    "rollout_relpath": str(path.relative_to(home_root / ".codex")),
                },
            )
        )

    def read_session(self, query: ProviderSessionQuery) -> AgentSessionView:
        reader = self._reader(query.locator)
        turns = tuple(self._turn_view(query.locator, item) for item in reader.list_turns())
        if not query.include_turns:
            turns = ()
        return AgentSessionView(locator=query.locator, turns=turns)

    def list_turns(self, query: ProviderTurnQuery) -> Page:
        turns = self._reader(query.session).list_turns()
        if query.latest:
            turns = turns[-1:]
        start, end = _page_bounds(query.cursor, query.limit, len(turns))
        return Page(
            items=tuple(self._turn_view(query.session, item) for item in turns[start:end]),
            next_cursor=str(end) if end < len(turns) else None,
        )

    def read_turn(self, query: ProviderTurnQuery) -> AgentTurnView | None:
        reader = self._reader(query.session)
        summary = reader.get_turn(
            turn_id=query.turn.turn_id if query.turn is not None else None,
            latest=query.latest,
        )
        return self._turn_view(query.session, summary) if summary is not None else None

    def list_events(self, query: ProviderEventQuery) -> Page:
        reader = self._reader(query.session)
        events = reader.tail_events(limit=10**9)
        if query.turn is not None:
            events = [event for event in events if event.turn_id == query.turn.turn_id]
        if query.kind is not None:
            events = [
                event
                for event in events
                if query.kind in {event.event_type, event.payload_type}
            ]
        start, end = _page_bounds(query.cursor, query.limit, len(events))
        projected = tuple(
            AgentEvent(
                provider_type="codex",
                session_id=query.session.session_id,
                turn_id=event.turn_id,
                sequence=event.index,
                timestamp=str(event.timestamp or ""),
                kind=event.payload_type or event.event_type or "unknown",
                terminal=(event.payload_type or event.event_type) in {
                    "task_complete",
                    "turn_completed",
                    "task_completed",
                    "turn_aborted",
                    "task_failed",
                },
                data={
                    "event_type": event.event_type,
                    "payload_type": event.payload_type,
                },
                provider_payload=build_provider_payload(
                    provider_type="codex",
                    payload_type="rollout_event",
                    data=event.raw_event,
                    adapter_version=self.adapter_version,
                ),
            )
            for event in events[start:end]
        )
        return Page(items=projected, next_cursor=str(end) if end < len(events) else None)

    def list_tool_calls(self, query: ProviderToolQuery) -> Page:
        calls = self._reader(query.session).list_tool_calls(
            turn_id=query.turn.turn_id if query.turn is not None else None,
            latest=query.latest,
        )
        if query.call_id is not None:
            calls = [call for call in calls if call.call_id == query.call_id]
        start, end = _page_bounds(query.cursor, query.limit, len(calls))
        items = tuple(
            AgentToolCall(
                call_id=call.call_id,
                turn_id=call.turn_id,
                tool_name=call.tool_name or "unknown",
                display_name=call.display_name,
                tool_kind="other",
                arguments=call.arguments,
                result=call.output,
                status="completed" if call.ok is not False else "failed",
                started_at=str(call.started_at) if call.started_at is not None else None,
                completed_at=str(call.completed_at) if call.completed_at is not None else None,
                duration_ms=call.duration_ms,
                error=(
                    AgentError(error_type="tool_error", message="Codex tool call reported failure")
                    if call.ok is False
                    else None
                ),
                provider_payload=build_provider_payload(
                    provider_type="codex",
                    payload_type="tool_call",
                    data={
                        "call": call.raw_call_event,
                        "output": call.raw_output_event,
                    },
                    adapter_version=self.adapter_version,
                ),
            )
            for call in calls[start:end]
        )
        return Page(items=items, next_cursor=str(end) if end < len(calls) else None)

    def read_usage(self, query: ProviderUsageQuery) -> AgentTurnUsage | AgentSessionUsage:
        turns = self._reader(query.session).list_turns()
        if query.turn is not None:
            turns = [turn for turn in turns if turn.turn_id == query.turn.turn_id]
        elif query.latest:
            turns = turns[-1:]
        usages = tuple(
            _turn_usage(turn.usage, model_identity=query.session.backend_identity)
            for turn in turns
        )
        if query.include_session_aggregate:
            tokens = TokenUsage.aggregate_complete(tuple(item.token_usage for item in usages))
            return AgentSessionUsage(
                turn_count=len(turns),
                request_count=None,
                token_usage=tokens,
                turns=usages,
                aggregate_complete=all(item.aggregate_complete for item in usages),
            )
        return usages[-1] if usages else AgentTurnUsage(
            request_count=None,
            requests=(),
            token_usage=TokenUsage(),
            aggregate_complete=False,
        )

    def _reader(self, locator) -> AgentTraceReader:  # noqa: ANN001
        rollout_path, _ = self._rollout_path(locator)
        events: list[dict] = []
        if rollout_path is not None and rollout_path.exists():
            for line in rollout_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        return AgentTraceReader(events=events)

    def _rollout_path(self, locator) -> tuple[Path | None, str | None]:  # noqa: ANN001
        native = locator.native_locator if isinstance(locator.native_locator, dict) else {}
        rollout_relpath = native.get("rollout_relpath")
        home_root = self.runtime_root / "homes" / "codex" / locator.home_id
        if not rollout_relpath:
            rollout_relpath = self.provider.find_rollout_relpath(
                home_root=home_root,
                thread_id=locator.session_id,
            )
        if not rollout_relpath:
            return None, None
        return home_root / ".codex" / str(rollout_relpath), str(rollout_relpath)

    def _turn_view(self, session, summary) -> AgentTurnView:  # noqa: ANN001
        locator = ProviderTurnLocator(session=session, turn_id=summary.turn_id)
        usage = _turn_usage(summary.usage, model_identity=session.backend_identity)
        status = _run_state(summary.status)
        if status is ProviderRunState.RUNNING:
            return AgentTurnView(locator=locator, result=None, usage=usage)
        error = None
        if status is ProviderRunState.FAILED:
            error = AgentError(
                error_type="codex_turn_failed",
                message="Codex rollout reports a failed turn",
            )
        result = ProviderTurnResult(
            provider_type="codex",
            run_id=f"offline:{session.session_id}:{summary.turn_id}",
            session_locator=session,
            turn_locator=locator,
            status=status,
            started_at=str(summary.started_at or ""),
            completed_at=str(summary.completed_at or ""),
            duration_ms=summary.duration_ms,
            final_text=summary.final_response,
            turn_usage=usage,
            error=error,
            provider_payload=build_provider_payload(
                provider_type="codex",
                payload_type="turn_summary",
                data={
                    "status": summary.status,
                    "usage": summary.usage,
                    "event_start_index": summary.event_start_index,
                    "event_end_index": summary.event_end_index,
                },
                adapter_version=self.adapter_version,
            ),
        )
        return AgentTurnView(locator=locator, result=result, usage=usage)


def _read_session_metadata(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if index >= 100:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("type") != "session_meta":
                    continue
                payload = event.get("payload")
                if isinstance(payload, dict):
                    return payload
    except OSError:
        pass
    try:
        modified_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        modified_at = utc_now_iso()
    return {"timestamp": modified_at}


def _page_bounds(cursor: str | None, limit: int, length: int) -> tuple[int, int]:
    if limit <= 0:
        raise ValueError("page limit must be positive")
    start = int(cursor) if cursor is not None else 0
    if start < 0:
        raise ValueError("page cursor must not be negative")
    return start, min(start + limit, length)


def _run_state(value: object) -> ProviderRunState:
    status = str(value or "completed")
    return {
        "interrupted": ProviderRunState.INTERRUPTED,
        "cancelled": ProviderRunState.CANCELLED,
        "failed": ProviderRunState.FAILED,
        "inProgress": ProviderRunState.RUNNING,
    }.get(status, ProviderRunState.COMPLETED)


def _turn_usage(raw: object, *, model_identity) -> AgentTurnUsage:  # noqa: ANN001
    payload = raw if isinstance(raw, dict) else {}
    total = payload.get("total") if isinstance(payload.get("total"), dict) else payload
    token_usage = TokenUsage(
        input_tokens=_int(total.get("input_tokens") or total.get("inputTokens")),
        output_tokens=_int(total.get("output_tokens") or total.get("outputTokens")),
        total_tokens=_int(total.get("total_tokens") or total.get("totalTokens")),
        cached_input_tokens=_int(
            total.get("cached_input_tokens") or total.get("cachedInputTokens")
        ),
        reasoning_output_tokens=_int(
            total.get("reasoning_output_tokens") or total.get("reasoningOutputTokens")
        ),
        semantics={
            "cached_input_tokens": "subset_of_input_tokens",
            "reasoning_output_tokens": "subset_of_output_tokens",
        },
    )
    return AgentTurnUsage(
        request_count=None,
        requests=(),
        token_usage=token_usage,
        models_used=(model_identity,) if model_identity is not None else (),
        aggregate_complete=all(
            getattr(token_usage, name) is not None
            for name in ("input_tokens", "output_tokens", "total_tokens")
        ),
    )


def _int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
