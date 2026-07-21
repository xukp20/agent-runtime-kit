from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Mapping

from ..provider_contracts import (
    AgentContentBlock,
    AgentError,
    AgentEvent,
    AgentSessionUsage,
    AgentToolCall,
    AgentTurnUsage,
    ModelBackendIdentity,
    ModelRequestUsage,
    ProviderRunState,
    ProviderSessionLocator,
    ProviderTurnLocator,
    ProviderTurnResult,
    ReportedCost,
    TokenUsage,
    build_provider_payload,
)


PI_ADAPTER_VERSION = "1"
PI_CLI_VERSION = "0.80.10"


@dataclass(frozen=True)
class PiTurn:
    user_entry: Mapping[str, object]
    entries: tuple[Mapping[str, object], ...]
    sequence: int

    @property
    def turn_id(self) -> str:
        return str(self.user_entry["id"])


class PiSessionTranscript:
    """Read and project one Pi v3 session JSONL transcript."""

    def __init__(
        self,
        *,
        path: Path,
        header: Mapping[str, object],
        entries: tuple[Mapping[str, object], ...],
    ) -> None:
        self.path = Path(path)
        self.header = header
        self.entries = entries
        self._by_id = {
            str(entry["id"]): entry
            for entry in entries
            if isinstance(entry.get("id"), str) and str(entry["id"])
        }

    @classmethod
    def read(cls, path: Path) -> "PiSessionTranscript":
        resolved = Path(path)
        records: list[Mapping[str, object]] = []
        with resolved.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid Pi session JSONL at {resolved}:{line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"Pi session entry must be an object at {resolved}:{line_number}"
                    )
                records.append(record)
        if not records:
            raise RuntimeError(f"empty Pi session transcript: {resolved}")
        header = records[0]
        if header.get("type") != "session" or int(header.get("version", 0)) != 3:
            raise RuntimeError(f"unsupported Pi session header: {resolved}")
        if not isinstance(header.get("id"), str) or not str(header["id"]).strip():
            raise RuntimeError(f"Pi session header has no id: {resolved}")
        return cls(path=resolved, header=header, entries=tuple(records[1:]))

    @property
    def session_id(self) -> str:
        return str(self.header["id"])

    @property
    def created_at(self) -> str:
        return str(self.header.get("timestamp") or _file_timestamp(self.path))

    @property
    def leaf_id(self) -> str | None:
        for entry in reversed(self.entries):
            value = entry.get("id")
            if isinstance(value, str) and value:
                return value
        return None

    def active_entries(self) -> tuple[Mapping[str, object], ...]:
        leaf = self.leaf_id
        if leaf is None:
            return ()
        branch: list[Mapping[str, object]] = []
        seen: set[str] = set()
        current: str | None = leaf
        while current is not None:
            if current in seen:
                raise RuntimeError(f"cycle in Pi session entry tree: {self.path}")
            seen.add(current)
            entry = self._by_id.get(current)
            if entry is None:
                raise RuntimeError(f"missing Pi session parent entry {current}: {self.path}")
            branch.append(entry)
            parent = entry.get("parentId")
            current = str(parent) if isinstance(parent, str) and parent else None
        branch.reverse()
        return tuple(branch)

    def turns(self) -> tuple[PiTurn, ...]:
        turns: list[PiTurn] = []
        current: list[Mapping[str, object]] = []
        user: Mapping[str, object] | None = None
        for entry in self.active_entries():
            message = _message(entry)
            if message is not None and message.get("role") == "user":
                if user is not None:
                    turns.append(PiTurn(user_entry=user, entries=tuple(current), sequence=len(turns)))
                user = entry
                current = [entry]
            elif user is not None:
                current.append(entry)
        if user is not None:
            turns.append(PiTurn(user_entry=user, entries=tuple(current), sequence=len(turns)))
        return tuple(turns)

    def session_usage(self) -> AgentSessionUsage:
        usages = tuple(self.turn_usage(turn) for turn in self.turns())
        return AgentSessionUsage(
            turn_count=len(usages),
            request_count=sum(item.request_count or 0 for item in usages),
            token_usage=TokenUsage.aggregate_complete(tuple(item.token_usage for item in usages)),
            turns=usages,
            aggregate_complete=bool(usages) and all(item.aggregate_complete for item in usages),
        )

    def turn_usage(self, turn: PiTurn) -> AgentTurnUsage:
        requests: list[ModelRequestUsage] = []
        for entry in turn.entries:
            message = _message(entry)
            if message is not None and message.get("role") == "assistant":
                usage = _request_usage(
                    message,
                    request_index=len(requests),
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    entry=entry,
                )
                if usage is not None:
                    requests.append(usage)
            elif entry.get("type") in {"compaction", "branch_summary"}:
                usage_raw = entry.get("usage")
                if isinstance(usage_raw, Mapping):
                    synthetic = {
                        "usage": usage_raw,
                        "provider": entry.get("provider") or "pi",
                        "api": entry.get("api") or "other",
                        "model": entry.get("model") or entry.get("modelId"),
                        "stopReason": "stop",
                    }
                    usage = _request_usage(
                        synthetic,
                        request_index=len(requests),
                        session_id=self.session_id,
                        turn_id=turn.turn_id,
                        entry=entry,
                    )
                    if usage is not None:
                        requests.append(usage)
        return AgentTurnUsage.from_requests(tuple(requests)) if requests else AgentTurnUsage(
            request_count=0,
            requests=(),
            token_usage=TokenUsage(),
            aggregate_complete=False,
        )

    def tool_calls(self, turn: PiTurn) -> tuple[AgentToolCall, ...]:
        calls: dict[str, dict[str, object]] = {}
        order: list[str] = []
        for entry in turn.entries:
            message = _message(entry)
            if message is None:
                continue
            role = message.get("role")
            if role == "assistant":
                for block in _content(message):
                    if block.get("type") != "toolCall":
                        continue
                    call_id = str(block.get("id") or "")
                    if not call_id:
                        continue
                    calls[call_id] = {"call": block, "call_entry": entry}
                    order.append(call_id)
            elif role == "toolResult":
                call_id = str(message.get("toolCallId") or "")
                if call_id:
                    calls.setdefault(call_id, {})["result"] = message
                    calls[call_id]["result_entry"] = entry
                    if call_id not in order:
                        order.append(call_id)
        projected: list[AgentToolCall] = []
        for call_id in order:
            pair = calls[call_id]
            call = pair.get("call") if isinstance(pair.get("call"), Mapping) else {}
            result = pair.get("result") if isinstance(pair.get("result"), Mapping) else None
            failed = bool(result.get("isError")) if result is not None else False
            projected.append(
                AgentToolCall(
                    call_id=call_id,
                    turn_id=turn.turn_id,
                    tool_name=str(call.get("name") or (result or {}).get("toolName") or "unknown"),
                    tool_kind="mcp" if str(call.get("name") or "").startswith("mcp__") else "other",
                    status="failed" if failed else ("completed" if result is not None else "started"),
                    arguments=call.get("arguments"),
                    result=_content(result) if result is not None else None,
                    started_at=_entry_timestamp(pair.get("call_entry")),
                    completed_at=_entry_timestamp(pair.get("result_entry")),
                    error=(
                        AgentError(error_type="tool_error", message=_content_text(result) or "Pi tool failed")
                        if failed and result is not None
                        else None
                    ),
                    provider_payload=build_provider_payload(
                        provider_type="pi",
                        payload_type="tool_call",
                        data=pair,
                        adapter_version=PI_ADAPTER_VERSION,
                        sdk_or_cli_version=PI_CLI_VERSION,
                    ),
                )
            )
        return tuple(projected)

    def events(self, turn: PiTurn) -> tuple[AgentEvent, ...]:
        events: list[AgentEvent] = []
        for index, entry in enumerate(turn.entries):
            kind = str(entry.get("type") or "unknown")
            message = _message(entry)
            if message is not None:
                role = str(message.get("role") or "unknown")
                kind = f"message.{role}"
            events.append(
                AgentEvent(
                    provider_type="pi",
                    session_id=self.session_id,
                    turn_id=turn.turn_id,
                    sequence=index,
                    timestamp=_entry_timestamp(entry) or "",
                    kind=kind,
                    terminal=(
                        message is not None
                        and message.get("role") == "assistant"
                        and message.get("stopReason") in {"stop", "length", "error", "aborted"}
                    ),
                    data={"entry_id": entry.get("id")},
                    provider_payload=build_provider_payload(
                        provider_type="pi",
                        payload_type="session_entry",
                        data=entry,
                        adapter_version=PI_ADAPTER_VERSION,
                        sdk_or_cli_version=PI_CLI_VERSION,
                    ),
                )
            )
        return tuple(events)

    def turn_result(
        self,
        turn: PiTurn,
        *,
        home_id: str,
        run_id: str | None = None,
    ) -> ProviderTurnResult:
        session = self.locator(home_id)
        usages = self.turn_usage(turn)
        tools = self.tool_calls(turn)
        blocks: list[AgentContentBlock] = []
        assistant_messages: list[Mapping[str, object]] = []
        for entry in turn.entries:
            message = _message(entry)
            if message is None or message.get("role") != "assistant":
                continue
            assistant_messages.append(message)
            for block in _content(message):
                kind = str(block.get("type") or "unknown")
                data: object
                if kind == "text":
                    data = block.get("text") or ""
                elif kind == "thinking":
                    data = block.get("thinking") or block.get("text") or ""
                else:
                    data = dict(block)
                blocks.append(
                    AgentContentBlock(
                        kind=kind,
                        data=data,
                        block_id=str(block.get("id")) if block.get("id") is not None else None,
                        sequence=len(blocks),
                        call_id=str(block.get("id")) if kind == "toolCall" and block.get("id") else None,
                        provider_payload=build_provider_payload(
                            provider_type="pi",
                            payload_type="content_block",
                            data=block,
                            adapter_version=PI_ADAPTER_VERSION,
                            sdk_or_cli_version=PI_CLI_VERSION,
                        ),
                    )
                )
        last = assistant_messages[-1] if assistant_messages else None
        status = _run_state(last)
        error = None
        if status is ProviderRunState.FAILED:
            error = AgentError(
                error_type="pi_turn_failed",
                message=str((last or {}).get("errorMessage") or "Pi turn failed"),
            )
        locator = ProviderTurnLocator(
            session=session,
            turn_id=turn.turn_id,
            request_ids=tuple(
                request.response_id for request in usages.requests if request.response_id is not None
            ),
            sequence=turn.sequence,
        )
        started_at = _entry_timestamp(turn.user_entry) or self.created_at
        completed_at = _entry_timestamp(turn.entries[-1]) if turn.entries else started_at
        return ProviderTurnResult(
            provider_type="pi",
            run_id=run_id or f"offline:{self.session_id}:{turn.turn_id}",
            session_locator=session,
            turn_locator=locator,
            status=status,
            started_at=started_at,
            completed_at=completed_at or started_at,
            final_text=_content_text(last),
            content_blocks=tuple(blocks),
            tool_calls=tools,
            request_usages=usages.requests,
            turn_usage=usages,
            error=error,
            provider_payload=build_provider_payload(
                provider_type="pi",
                payload_type="turn",
                data={"entry_ids": [entry.get("id") for entry in turn.entries]},
                adapter_version=PI_ADAPTER_VERSION,
                sdk_or_cli_version=PI_CLI_VERSION,
            ),
        )

    def locator(self, home_id: str) -> ProviderSessionLocator:
        identity = None
        for entry in reversed(self.active_entries()):
            message = _message(entry)
            if message is not None and message.get("role") == "assistant":
                identity = _model_identity(message)
                break
        return ProviderSessionLocator(
            provider_type="pi",
            session_id=self.session_id,
            home_id=home_id,
            created_at=self.created_at,
            backend_identity=identity,
            native_locator={"session_relpath": self.path.name},
        )


def find_pi_session(session_root: Path, session_id: str) -> Path | None:
    root = Path(session_root)
    matches: list[Path] = []
    if not root.exists():
        return None
    for path in sorted(root.glob("*.jsonl")):
        try:
            with path.open(encoding="utf-8") as stream:
                first = next((line for line in stream if line.strip()), "")
            header = json.loads(first) if first else None
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(header, dict) and header.get("type") == "session" and header.get("id") == session_id:
            matches.append(path)
    if len(matches) > 1:
        raise RuntimeError(f"multiple Pi session files have id {session_id}")
    return matches[0] if matches else None


def list_pi_sessions(session_root: Path) -> tuple[PiSessionTranscript, ...]:
    sessions: list[PiSessionTranscript] = []
    root = Path(session_root)
    if not root.exists():
        return ()
    for path in sorted(root.glob("*.jsonl")):
        try:
            sessions.append(PiSessionTranscript.read(path))
        except RuntimeError:
            continue
    return tuple(sessions)


def _message(entry: Mapping[str, object]) -> Mapping[str, object] | None:
    value = entry.get("message")
    return value if entry.get("type") == "message" and isinstance(value, Mapping) else None


def _content(message: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if message is None:
        return []
    value = message.get("content")
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _content_text(message: Mapping[str, object] | None) -> str | None:
    texts = [str(item.get("text")) for item in _content(message) if item.get("type") == "text" and item.get("text") is not None]
    return "\n".join(texts) if texts else None


def _entry_timestamp(entry: object) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get("timestamp")
    return str(value) if value is not None else None


def _file_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _model_identity(message: Mapping[str, object]) -> ModelBackendIdentity:
    api = str(message.get("api") or "other")
    mode = {
        "openai-completions": "chat_completions",
        "openai-responses": "responses",
        "openai-codex-responses": "responses",
        "anthropic-messages": "messages",
    }.get(api, api)
    return ModelBackendIdentity(
        api_provider=str(message.get("provider") or "unknown"),
        api_mode=mode,
        requested_model=str(message.get("model")) if message.get("model") is not None else None,
        resolved_model=(
            str(message.get("responseModel")) if message.get("responseModel") is not None else None
        ),
        provider_payload=build_provider_payload(
            provider_type="pi",
            payload_type="model_identity",
            data={
                "api": message.get("api"),
                "provider": message.get("provider"),
                "model": message.get("model"),
                "responseModel": message.get("responseModel"),
            },
            adapter_version=PI_ADAPTER_VERSION,
            sdk_or_cli_version=PI_CLI_VERSION,
        ),
    )


def _request_usage(
    message: Mapping[str, object],
    *,
    request_index: int,
    session_id: str,
    turn_id: str,
    entry: Mapping[str, object],
) -> ModelRequestUsage | None:
    raw = message.get("usage")
    if not isinstance(raw, Mapping):
        return None
    token_usage = TokenUsage(
        input_tokens=_int(raw.get("input")),
        output_tokens=_int(raw.get("output")),
        total_tokens=_int(raw.get("totalTokens")),
        cache_read_input_tokens=_int(raw.get("cacheRead")),
        cache_write_input_tokens=_int(raw.get("cacheWrite")),
        cache_creation_1h_input_tokens=_int(raw.get("cacheWrite1h")),
        reasoning_output_tokens=_int(raw.get("reasoning")),
        semantics={
            "input_tokens": "Pi input excludes cacheRead and cacheWrite tokens",
            "reasoning_output_tokens": "subset_of_output_tokens",
            "total_tokens": "Pi provider-reported totalTokens",
        },
    )
    reported = tuple(
        name
        for source, name in (
            ("input", "input_tokens"),
            ("output", "output_tokens"),
            ("totalTokens", "total_tokens"),
            ("cacheRead", "cache_read_input_tokens"),
            ("cacheWrite", "cache_write_input_tokens"),
            ("cacheWrite1h", "cache_creation_1h_input_tokens"),
            ("reasoning", "reasoning_output_tokens"),
        )
        if source in raw
    )
    cost = _reported_cost(raw.get("cost"))
    return ModelRequestUsage(
        request_index=request_index,
        model_identity=_model_identity(message),
        token_usage=token_usage,
        response_id=(str(message["responseId"]) if message.get("responseId") is not None else None),
        session_id=session_id,
        turn_id=turn_id,
        reported_cost=cost,
        status="failed" if message.get("stopReason") == "error" else "completed",
        stop_reason=str(message.get("stopReason")) if message.get("stopReason") is not None else None,
        completed_at=_entry_timestamp(entry),
        reported_fields=reported,
        provider_payload=build_provider_payload(
            provider_type="pi",
            payload_type="model_request_usage",
            data={"entry_id": entry.get("id"), "message": message},
            adapter_version=PI_ADAPTER_VERSION,
            sdk_or_cli_version=PI_CLI_VERSION,
        ),
    )


def _reported_cost(raw: object) -> ReportedCost | None:
    if not isinstance(raw, Mapping) or not any(value is not None for value in raw.values()):
        return None
    def decimal(name: str) -> str | None:
        value = raw.get(name)
        if value is None:
            return None
        return str(Decimal(str(value)))
    return ReportedCost(
        currency="USD",
        input_cost=decimal("input"),
        output_cost=decimal("output"),
        cache_read_cost=decimal("cacheRead"),
        cache_write_cost=decimal("cacheWrite"),
        total_cost=decimal("total"),
        provider_payload=build_provider_payload(
            provider_type="pi",
            payload_type="reported_cost",
            data=raw,
            adapter_version=PI_ADAPTER_VERSION,
            sdk_or_cli_version=PI_CLI_VERSION,
        ),
    )


def _run_state(message: Mapping[str, object] | None) -> ProviderRunState:
    if message is None:
        return ProviderRunState.RUNNING
    return {
        "aborted": ProviderRunState.INTERRUPTED,
        "error": ProviderRunState.FAILED,
    }.get(str(message.get("stopReason")), ProviderRunState.COMPLETED)


def _int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
