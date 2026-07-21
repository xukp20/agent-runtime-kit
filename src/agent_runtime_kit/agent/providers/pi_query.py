from __future__ import annotations

from pathlib import Path

from ..provider_contracts import (
    AgentSessionView,
    AgentTurnUsage,
    AgentTurnView,
    Page,
    ProviderEventQuery,
    ProviderSessionListQuery,
    ProviderSessionQuery,
    ProviderToolQuery,
    ProviderTurnQuery,
    ProviderUsageQuery,
    TokenUsage,
)
from .pi_session import PiSessionTranscript, find_pi_session, list_pi_sessions


class PiQueryAdapter:
    provider_type = "pi"

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)

    def list_sessions(self, query: ProviderSessionListQuery) -> Page:
        if query.home_id is None:
            raise ValueError("Pi session listing requires home_id")
        sessions = tuple(
            AgentSessionView(locator=item.locator(query.home_id))
            for item in list_pi_sessions(self._sessions_root(query.home_id))
        )
        start, end = _page_bounds(query.cursor, query.limit, len(sessions))
        return Page(items=sessions[start:end], next_cursor=str(end) if end < len(sessions) else None)

    def read_session(self, query: ProviderSessionQuery) -> AgentSessionView:
        transcript = self._transcript(query.locator.home_id, query.locator.session_id)
        turns = ()
        if query.include_turns:
            turns = tuple(self._turn_view(transcript, item, query.locator.home_id) for item in transcript.turns())
        return AgentSessionView(
            locator=transcript.locator(query.locator.home_id),
            status="idle",
            turns=turns,
            usage=transcript.session_usage(),
        )

    def list_turns(self, query: ProviderTurnQuery) -> Page:
        transcript = self._transcript(query.session.home_id, query.session.session_id)
        turns = list(transcript.turns())
        if query.latest:
            turns = turns[-1:]
        start, end = _page_bounds(query.cursor, query.limit, len(turns))
        return Page(
            items=tuple(self._turn_view(transcript, item, query.session.home_id) for item in turns[start:end]),
            next_cursor=str(end) if end < len(turns) else None,
        )

    def read_turn(self, query: ProviderTurnQuery) -> AgentTurnView | None:
        transcript = self._transcript(query.session.home_id, query.session.session_id)
        turns = list(transcript.turns())
        if query.turn is not None:
            turns = [item for item in turns if item.turn_id == query.turn.turn_id]
        elif query.latest:
            turns = turns[-1:]
        return self._turn_view(transcript, turns[-1], query.session.home_id) if turns else None

    def list_events(self, query: ProviderEventQuery) -> Page:
        transcript = self._transcript(query.session.home_id, query.session.session_id)
        events = [
            event
            for turn in transcript.turns()
            if query.turn is None or turn.turn_id == query.turn.turn_id
            for event in transcript.events(turn)
            if query.kind is None or event.kind == query.kind
        ]
        start, end = _page_bounds(query.cursor, query.limit, len(events))
        return Page(items=tuple(events[start:end]), next_cursor=str(end) if end < len(events) else None)

    def list_tool_calls(self, query: ProviderToolQuery) -> Page:
        transcript = self._transcript(query.session.home_id, query.session.session_id)
        calls = [
            call
            for turn in transcript.turns()
            if query.turn is None or turn.turn_id == query.turn.turn_id
            for call in transcript.tool_calls(turn)
            if query.call_id is None or call.call_id == query.call_id
        ]
        start, end = _page_bounds(query.cursor, query.limit, len(calls))
        return Page(items=tuple(calls[start:end]), next_cursor=str(end) if end < len(calls) else None)

    def read_usage(self, query: ProviderUsageQuery):  # noqa: ANN201
        transcript = self._transcript(query.session.home_id, query.session.session_id)
        if query.include_session_aggregate:
            return transcript.session_usage()
        turns = list(transcript.turns())
        if query.turn is not None:
            turns = [item for item in turns if item.turn_id == query.turn.turn_id]
        elif query.latest:
            turns = turns[-1:]
        return transcript.turn_usage(turns[-1]) if turns else AgentTurnUsage(
            request_count=0,
            requests=(),
            token_usage=TokenUsage(),
            aggregate_complete=False,
        )

    def _turn_view(self, transcript: PiSessionTranscript, turn, home_id: str) -> AgentTurnView:  # noqa: ANN001
        result = transcript.turn_result(turn, home_id=home_id)
        assert result.turn_locator is not None
        return AgentTurnView(
            locator=result.turn_locator,
            result=result,
            events=transcript.events(turn),
            tool_calls=result.tool_calls,
            usage=result.turn_usage,
        )

    def _transcript(self, home_id: str, session_id: str) -> PiSessionTranscript:
        path = find_pi_session(self._sessions_root(home_id), session_id)
        if path is None:
            raise KeyError(f"unknown Pi session: {session_id}")
        return PiSessionTranscript.read(path)

    def _sessions_root(self, home_id: str) -> Path:
        return self.runtime_root / "homes" / "pi" / home_id / ".pi" / "sessions"


def _page_bounds(cursor: str | None, limit: int, length: int) -> tuple[int, int]:
    if limit <= 0:
        raise ValueError("page limit must be positive")
    start = int(cursor) if cursor is not None else 0
    if start < 0:
        raise ValueError("page cursor must not be negative")
    return start, min(start + limit, length)
