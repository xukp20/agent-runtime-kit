from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

from ..provider_contracts import (
    AgentArtifactLocator,
    AgentSessionUsage,
    AgentSessionView,
    AgentTurnUsage,
    AgentTurnView,
    Page,
    ProviderEventQuery,
    ProviderSessionListQuery,
    ProviderSessionLocator,
    ProviderSessionQuery,
    ProviderToolQuery,
    ProviderTurnQuery,
    ProviderUsageQuery,
    TokenUsage,
)
from ..store_utils import utc_now_iso
from .claude_code import ClaudeCodeProvider
from .claude_code_normalization import (
    find_session_file,
    group_transcript_turns,
    project_transcript_turn,
    read_transcript,
    transcript_events,
)


class ClaudeCodeQueryAdapter:
    provider_type = "claude_code"

    def __init__(self, *, runtime_root: Path, provider: ClaudeCodeProvider) -> None:
        self.runtime_root = Path(runtime_root)
        self.provider = provider

    def list_sessions(self, query: ProviderSessionListQuery) -> Page:
        roots = (
            [self._home_root(query.home_id)]
            if query.home_id is not None
            else sorted((self.runtime_root / "homes" / self.provider_type).glob("*"))
        )
        sessions: list[AgentSessionView] = []
        for home_root in roots:
            if not home_root.is_dir():
                continue
            projects = home_root / ".claude" / "projects"
            if not projects.exists():
                continue
            for path in projects.rglob("*.jsonl"):
                try:
                    session_id = str(uuid.UUID(path.stem))
                except ValueError:
                    continue
                locator = self._locator(home_root.name, session_id, path)
                sessions.append(self._session_view(locator, include_turns=False))
        sessions.sort(key=lambda item: item.locator.created_at, reverse=True)
        start, end = _page(query.cursor, query.limit, len(sessions))
        return Page(items=tuple(sessions[start:end]), next_cursor=str(end) if end < len(sessions) else None)

    def read_session(self, query: ProviderSessionQuery) -> AgentSessionView:
        return self._session_view(query.locator, include_turns=query.include_turns)

    def list_turns(self, query: ProviderTurnQuery) -> Page:
        turns = list(self._turn_views(query.session))
        if query.turn is not None:
            turns = [item for item in turns if item.locator.turn_id == query.turn.turn_id]
        if query.latest:
            turns = turns[-1:]
        start, end = _page(query.cursor, query.limit, len(turns))
        return Page(items=tuple(turns[start:end]), next_cursor=str(end) if end < len(turns) else None)

    def read_turn(self, query: ProviderTurnQuery) -> AgentTurnView | None:
        items = self.list_turns(replace(query, limit=max(query.limit, 1))).items
        if query.latest:
            return items[-1] if items else None
        return items[0] if items else None

    def list_events(self, query: ProviderEventQuery) -> Page:
        path = self._session_path(query.session)
        events = list(transcript_events(session=query.session, records=read_transcript(path)))
        if query.turn is not None:
            events = [event for event in events if event.turn_id == query.turn.turn_id]
        if query.kind is not None:
            events = [event for event in events if event.kind == query.kind]
        start, end = _page(query.cursor, query.limit, len(events))
        return Page(items=tuple(events[start:end]), next_cursor=str(end) if end < len(events) else None)

    def list_tool_calls(self, query: ProviderToolQuery) -> Page:
        turns = self._turn_views(query.session)
        calls = [call for turn in turns for call in turn.tool_calls]
        if query.turn is not None:
            calls = [call for call in calls if call.turn_id == query.turn.turn_id]
        if query.call_id is not None:
            calls = [call for call in calls if call.call_id == query.call_id]
        start, end = _page(query.cursor, query.limit, len(calls))
        return Page(items=tuple(calls[start:end]), next_cursor=str(end) if end < len(calls) else None)

    def read_usage(self, query: ProviderUsageQuery) -> AgentTurnUsage | AgentSessionUsage:
        turns = list(self._turn_views(query.session))
        if query.turn is not None:
            turns = [item for item in turns if item.locator.turn_id == query.turn.turn_id]
        if query.latest:
            turns = turns[-1:]
        usages = tuple(item.usage for item in turns if item.usage is not None)
        if query.include_session_aggregate:
            token_usage = TokenUsage.aggregate_complete(tuple(item.token_usage for item in usages))
            complete = bool(usages) and all(
                getattr(token_usage, field) is not None
                for field in ("input_tokens", "output_tokens", "total_tokens")
            )
            request_counts = [usage.request_count for usage in usages]
            return AgentSessionUsage(
                turn_count=len(turns),
                request_count=(
                    sum(value for value in request_counts if value is not None)
                    if all(value is not None for value in request_counts)
                    else None
                ),
                token_usage=token_usage,
                turns=usages,
                aggregate_complete=complete,
            )
        if not usages:
            return AgentTurnUsage(request_count=0, requests=(), token_usage=TokenUsage())
        return usages[-1]

    def _session_view(
        self,
        locator: ProviderSessionLocator,
        *,
        include_turns: bool,
    ) -> AgentSessionView:
        turns = self._turn_views(locator) if include_turns else ()
        usage = None
        if include_turns:
            turn_usages = tuple(item.usage for item in turns if item.usage is not None)
            usage = AgentSessionUsage(
                turn_count=len(turns),
                request_count=(
                    sum(item.request_count or 0 for item in turn_usages)
                    if all(item.request_count is not None for item in turn_usages)
                    else None
                ),
                token_usage=TokenUsage.aggregate_complete(
                    tuple(item.token_usage for item in turn_usages)
                ),
                turns=turn_usages,
                aggregate_complete=False,
            )
        status = "completed" if turns and turns[-1].result is not None else "unknown"
        return AgentSessionView(locator=locator, status=status, turns=turns, usage=usage)

    def _turn_views(self, session: ProviderSessionLocator) -> tuple[AgentTurnView, ...]:
        path = self._session_path(session)
        records = read_transcript(path)
        events = transcript_events(session=session, records=records)
        views: list[AgentTurnView] = []
        for turn in group_transcript_turns(records):
            result = project_transcript_turn(session=session, turn=turn)
            if result is not None:
                result = replace(result, artifact_locator=self._artifact_locator(session, path))
                calls = tuple(replace(call, turn_id=turn.turn_id) for call in result.tool_calls)
                result = replace(result, tool_calls=calls)
            else:
                calls = ()
            views.append(
                AgentTurnView(
                    locator=(
                        result.turn_locator
                        if result is not None and result.turn_locator is not None
                        else _turn_locator(session, turn.turn_id)
                    ),
                    result=result,
                    events=tuple(event for event in events if event.turn_id == turn.turn_id),
                    tool_calls=calls,
                    usage=result.turn_usage if result is not None else None,
                )
            )
        return tuple(views)

    def _session_path(self, locator: ProviderSessionLocator) -> Path:
        path = find_session_file(self._home_root(locator.home_id), locator.session_id)
        if path is None:
            raise FileNotFoundError(f"Claude transcript is missing: {locator.session_id}")
        return path

    def _locator(self, home_id: str, session_id: str, path: Path) -> ProviderSessionLocator:
        records = read_transcript(path)
        created_at = next(
            (str(record["timestamp"]) for record in records if record.get("timestamp")),
            utc_now_iso(),
        )
        return ProviderSessionLocator(
            provider_type=self.provider_type,
            session_id=session_id,
            home_id=home_id,
            created_at=created_at,
            native_locator={"transcript_relpath": str(path.relative_to(self._home_root(home_id)))},
        )

    def _artifact_locator(
        self,
        session: ProviderSessionLocator,
        path: Path,
    ) -> AgentArtifactLocator:
        return AgentArtifactLocator(
            provider_type=self.provider_type,
            home_id=session.home_id,
            session_id=session.session_id,
            adapter_version="1",
            native_primary_ref=str(path.relative_to(self._home_root(session.home_id) / ".claude")),
        )

    def _home_root(self, home_id: str) -> Path:
        return self.runtime_root / "homes" / self.provider_type / home_id


def _turn_locator(session: ProviderSessionLocator, turn_id: str):  # noqa: ANN202
    from ..provider_contracts import ProviderTurnLocator

    return ProviderTurnLocator(session=session, turn_id=turn_id)


def _page(cursor: str | None, limit: int, length: int) -> tuple[int, int]:
    if limit <= 0:
        raise ValueError("query limit must be positive")
    try:
        start = int(cursor) if cursor is not None else 0
    except ValueError as exc:
        raise ValueError("invalid Claude query cursor") from exc
    if start < 0:
        raise ValueError("invalid Claude query cursor")
    return start, min(length, start + limit)
