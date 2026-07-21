from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from typing import Mapping, Sequence

from ..provider_contracts import (
    AgentContentBlock,
    AgentError,
    AgentEvent,
    AgentToolCall,
    AgentTurnUsage,
    BillableUnit,
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


PROVIDER_TYPE = "claude_code"
ADAPTER_VERSION = "1"


@dataclass(frozen=True)
class ClaudeTranscriptTurn:
    turn_id: str
    records: tuple[dict[str, object], ...]
    started_at: str | None
    completed_at: str | None


@dataclass(frozen=True)
class ClaudeNormalizedContent:
    blocks: tuple[AgentContentBlock, ...]
    tools: tuple[AgentToolCall, ...]
    requests: tuple[ModelRequestUsage, ...]
    final_text: str | None
    error: AgentError | None


def find_session_file(home_root: Path, session_id: str) -> Path | None:
    try:
        normalized = str(uuid.UUID(session_id))
    except ValueError as exc:
        raise ValueError(f"Claude session id must be a UUID: {session_id}") from exc
    projects_root = (Path(home_root) / ".claude" / "projects").resolve()
    if not projects_root.exists():
        return None
    matches = sorted(path.resolve() for path in projects_root.rglob(f"{normalized}.jsonl"))
    safe = [path for path in matches if projects_root in path.parents]
    if len(safe) > 1:
        raise RuntimeError(f"multiple Claude transcripts found for session {session_id}")
    return safe[0] if safe else None


def read_transcript(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid Claude transcript JSON at line {index}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"invalid Claude transcript entry at line {index}")
            records.append(value)
    return records


def group_transcript_turns(records: Sequence[dict[str, object]]) -> tuple[ClaudeTranscriptTurn, ...]:
    turns: list[ClaudeTranscriptTurn] = []
    current: list[dict[str, object]] = []
    turn_id: str | None = None
    for record in records:
        if _is_external_user_record(record):
            if turn_id is not None:
                turns.append(_build_transcript_turn(turn_id, current))
            raw_id = record.get("uuid")
            turn_id = str(raw_id) if raw_id else f"turn-{len(turns)}"
            current = [record]
        elif turn_id is not None:
            current.append(record)
    if turn_id is not None:
        turns.append(_build_transcript_turn(turn_id, current))
    return tuple(turns)


def latest_turn_id(home_root: Path, session_id: str) -> str | None:
    path = find_session_file(home_root, session_id)
    if path is None:
        return None
    turns = group_transcript_turns(read_transcript(path))
    return turns[-1].turn_id if turns else None


def compact_boundary_snapshot(home_root: Path, session_id: str) -> dict[str, object]:
    path = find_session_file(home_root, session_id)
    if path is None:
        raise FileNotFoundError(f"Claude transcript is missing: {session_id}")
    records = read_transcript(path)
    boundaries = [
        record
        for record in records
        if record.get("type") == "system" and record.get("subtype") == "compact_boundary"
    ]
    return {
        "session_id": session_id,
        "transcript_relpath": str(path.relative_to(Path(home_root))),
        "line_count": len(records),
        "boundary_count": len(boundaries),
        "latest_boundary_uuid": str(boundaries[-1].get("uuid")) if boundaries else None,
        "size_bytes": path.stat().st_size,
    }


def find_new_compact_boundary(
    home_root: Path,
    session_id: str,
    baseline: Mapping[str, object],
) -> dict[str, object] | None:
    path = find_session_file(home_root, session_id)
    if path is None:
        return None
    boundary_count = int(baseline.get("boundary_count") or 0)
    boundaries = [
        record
        for record in read_transcript(path)
        if record.get("type") == "system" and record.get("subtype") == "compact_boundary"
    ]
    if len(boundaries) <= boundary_count:
        return None
    return boundaries[boundary_count]


def normalize_stream_result(
    *,
    run_id: str,
    session: ProviderSessionLocator,
    turn_id: str,
    messages: Sequence[object],
    terminal: object,
    started_at: str,
    completed_at: str,
    duration_ms: float,
    interrupted: bool,
) -> ProviderTurnResult:
    normalized = _normalize_sdk_messages(messages, session.backend_identity)
    status, error = _terminal_state(terminal, interrupted=interrupted, fallback=normalized.error)
    turn = ProviderTurnLocator(
        session=session,
        turn_id=turn_id,
        request_ids=tuple(
            request.request_id for request in normalized.requests if request.request_id is not None
        ),
    )
    turn_usage = AgentTurnUsage.from_requests(normalized.requests) if normalized.requests else None
    cost = _reported_turn_cost(getattr(terminal, "total_cost_usd", None))
    if turn_usage is not None and cost is not None:
        turn_usage = replace(turn_usage, reported_costs=(cost,))
    final_text = getattr(terminal, "result", None) or normalized.final_text
    return ProviderTurnResult(
        provider_type=PROVIDER_TYPE,
        run_id=run_id,
        session_locator=session,
        turn_locator=turn,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        final_text=str(final_text) if final_text is not None else None,
        structured_output=getattr(terminal, "structured_output", None),
        content_blocks=normalized.blocks,
        tool_calls=normalized.tools,
        request_usages=normalized.requests,
        turn_usage=turn_usage,
        error=error,
        provider_payload=build_provider_payload(
            provider_type=PROVIDER_TYPE,
            payload_type="result",
            adapter_version=ADAPTER_VERSION,
            data={
                "subtype": getattr(terminal, "subtype", None),
                "is_error": getattr(terminal, "is_error", None),
                "num_turns": getattr(terminal, "num_turns", None),
                "stop_reason": getattr(terminal, "stop_reason", None),
                "api_error_status": getattr(terminal, "api_error_status", None),
                "usage": getattr(terminal, "usage", None),
                "model_usage": getattr(terminal, "model_usage", None),
            },
        ),
    )


def project_transcript_turn(
    *,
    session: ProviderSessionLocator,
    turn: ClaudeTranscriptTurn,
) -> ProviderTurnResult | None:
    normalized = _normalize_transcript_records(turn.records, session.backend_identity)
    status, error = _transcript_terminal(turn.records, normalized.error)
    if status is None:
        return None
    locator = ProviderTurnLocator(
        session=session,
        turn_id=turn.turn_id,
        request_ids=tuple(
            request.request_id for request in normalized.requests if request.request_id is not None
        ),
    )
    turn_usage = AgentTurnUsage.from_requests(normalized.requests) if normalized.requests else None
    started_at = turn.started_at or session.created_at
    completed_at = turn.completed_at or started_at
    return ProviderTurnResult(
        provider_type=PROVIDER_TYPE,
        run_id=f"offline-{turn.turn_id}",
        session_locator=session,
        turn_locator=locator,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        final_text=normalized.final_text,
        content_blocks=normalized.blocks,
        tool_calls=normalized.tools,
        request_usages=normalized.requests,
        turn_usage=turn_usage,
        error=error,
        provider_payload=build_provider_payload(
            provider_type=PROVIDER_TYPE,
            payload_type="offline_turn",
            adapter_version=ADAPTER_VERSION,
            data={"record_count": len(turn.records)},
        ),
    )


def transcript_events(
    *,
    session: ProviderSessionLocator,
    records: Sequence[dict[str, object]],
) -> tuple[AgentEvent, ...]:
    turn_by_record: dict[int, str | None] = {}
    current_turn: str | None = None
    for index, record in enumerate(records):
        if _is_external_user_record(record):
            current_turn = str(record.get("uuid") or f"turn-{index}")
        turn_by_record[index] = current_turn
    events: list[AgentEvent] = []
    for index, record in enumerate(records):
        event_type = str(record.get("type") or "unknown")
        subtype = record.get("subtype")
        kind = (
            "compact.completed"
            if event_type == "system" and subtype == "compact_boundary"
            else f"transcript.{event_type}"
        )
        events.append(
            AgentEvent(
                provider_type=PROVIDER_TYPE,
                session_id=session.session_id,
                turn_id=turn_by_record[index],
                sequence=index,
                timestamp=str(record.get("timestamp") or session.created_at),
                kind=kind,
                data={"subtype": subtype} if subtype is not None else None,
                provider_payload=build_provider_payload(
                    provider_type=PROVIDER_TYPE,
                    payload_type="transcript_event",
                    adapter_version=ADAPTER_VERSION,
                    data={
                        "type": record.get("type"),
                        "subtype": subtype,
                        "uuid": record.get("uuid"),
                        "parentUuid": record.get("parentUuid"),
                    },
                ),
            )
        )
    return tuple(events)


def _normalize_sdk_messages(
    messages: Sequence[object],
    backend: ModelBackendIdentity | None,
) -> ClaudeNormalizedContent:
    blocks: list[AgentContentBlock] = []
    tool_parts: list[tuple[str, object]] = []
    usage_by_id: dict[str, tuple[object, Mapping[str, object]]] = {}
    final_texts: list[str] = []
    assistant_error: AgentError | None = None
    sequence = 0
    for message in messages:
        name = type(message).__name__
        if name == "AssistantMessage":
            message_id = str(getattr(message, "message_id", "") or f"request-{len(usage_by_id)}")
            raw_usage = getattr(message, "usage", None)
            if isinstance(raw_usage, Mapping):
                usage_by_id[message_id] = (message, raw_usage)
            raw_error = getattr(message, "error", None)
            if raw_error:
                assistant_error = _agent_error(str(raw_error), str(raw_error))
            for native in getattr(message, "content", ()) or ():
                block, tool_part = _sdk_block(native, sequence, message_id)
                sequence += 1
                if block is not None:
                    blocks.append(block)
                    if block.kind == "text" and isinstance(block.data, str):
                        final_texts.append(block.data)
                if tool_part is not None:
                    tool_parts.append(tool_part)
        elif name == "UserMessage":
            for native in getattr(message, "content", ()) or () if isinstance(
                getattr(message, "content", None), list
            ) else ():
                block, tool_part = _sdk_block(native, sequence, None)
                sequence += 1
                if block is not None:
                    blocks.append(block)
                if tool_part is not None:
                    tool_parts.append(tool_part)
    requests = tuple(
        _request_usage(index, request_id, message, raw, backend)
        for index, (request_id, (message, raw)) in enumerate(usage_by_id.items())
    )
    return ClaudeNormalizedContent(
        blocks=tuple(blocks),
        tools=_assemble_tools(tool_parts),
        requests=requests,
        final_text=final_texts[-1] if final_texts else None,
        error=assistant_error,
    )


def _normalize_transcript_records(
    records: Sequence[dict[str, object]],
    backend: ModelBackendIdentity | None,
) -> ClaudeNormalizedContent:
    blocks: list[AgentContentBlock] = []
    tool_parts: list[tuple[str, object]] = []
    usage_by_id: dict[str, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    final_texts: list[str] = []
    error: AgentError | None = None
    sequence = 0
    for record in records:
        record_type = record.get("type")
        if record_type == "system" and record.get("subtype") == "compact_boundary":
            blocks.append(
                AgentContentBlock(
                    kind="compaction_boundary",
                    sequence=sequence,
                    block_id=_optional_str(record.get("uuid")),
                    data={"content": record.get("content")},
                )
            )
            sequence += 1
            continue
        message = record.get("message")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if record_type == "assistant":
            message_id = str(message.get("id") or record.get("uuid") or f"request-{len(usage_by_id)}")
            raw_usage = message.get("usage")
            if isinstance(raw_usage, Mapping):
                usage_by_id[message_id] = (message, raw_usage)
            if record.get("isApiErrorMessage"):
                error = _agent_error("provider", "Claude API error message")
        else:
            message_id = None
        for native in content:
            if not isinstance(native, Mapping):
                continue
            block, tool_part = _mapping_block(native, sequence, message_id)
            sequence += 1
            if block is not None:
                blocks.append(block)
                if block.kind == "text" and record_type == "assistant" and isinstance(block.data, str):
                    final_texts.append(block.data)
            if tool_part is not None:
                tool_parts.append(tool_part)
    requests = tuple(
        _request_usage(index, request_id, message, raw, backend)
        for index, (request_id, (message, raw)) in enumerate(usage_by_id.items())
    )
    return ClaudeNormalizedContent(
        blocks=tuple(blocks),
        tools=_assemble_tools(tool_parts),
        requests=requests,
        final_text=final_texts[-1] if final_texts else None,
        error=error,
    )


def _sdk_block(
    native: object,
    sequence: int,
    request_id: str | None,
) -> tuple[AgentContentBlock | None, tuple[str, object] | None]:
    name = type(native).__name__
    if name == "TextBlock":
        return AgentContentBlock(
            kind="text",
            data=str(getattr(native, "text", "")),
            sequence=sequence,
            model_request_id=request_id,
        ), None
    if name == "ThinkingBlock":
        return AgentContentBlock(
            kind="reasoning",
            data=str(getattr(native, "thinking", "")),
            sequence=sequence,
            model_request_id=request_id,
            provider_payload=build_provider_payload(
                provider_type=PROVIDER_TYPE,
                payload_type="thinking",
                adapter_version=ADAPTER_VERSION,
                data={"signature_present": bool(getattr(native, "signature", None))},
            ),
        ), None
    if name in {"ToolUseBlock", "ServerToolUseBlock"}:
        call_id = str(getattr(native, "id", ""))
        kind = "server_tool_call" if name.startswith("Server") else "tool_call"
        data = {
            "name": getattr(native, "name", None),
            "arguments": getattr(native, "input", None),
        }
        return AgentContentBlock(
            kind=kind,
            data=data,
            sequence=sequence,
            call_id=call_id,
            block_id=call_id,
            model_request_id=request_id,
        ), ("use", native)
    if name in {"ToolResultBlock", "ServerToolResultBlock"}:
        call_id = str(getattr(native, "tool_use_id", ""))
        kind = "server_tool_result" if name.startswith("Server") else "tool_result"
        data = {
            "result": getattr(native, "content", None),
            "is_error": getattr(native, "is_error", None),
        }
        return AgentContentBlock(
            kind=kind,
            data=data,
            sequence=sequence,
            call_id=call_id,
            model_request_id=request_id,
        ), ("result", native)
    return AgentContentBlock(
        kind="other",
        data={"native_type": name},
        sequence=sequence,
        model_request_id=request_id,
    ), None


def _mapping_block(
    native: Mapping[str, object],
    sequence: int,
    request_id: str | None,
) -> tuple[AgentContentBlock | None, tuple[str, object] | None]:
    kind = str(native.get("type") or "other")
    if kind == "text":
        return AgentContentBlock(
            kind="text",
            data=str(native.get("text") or ""),
            sequence=sequence,
            model_request_id=request_id,
        ), None
    if kind == "thinking":
        return AgentContentBlock(
            kind="reasoning",
            data=str(native.get("thinking") or ""),
            sequence=sequence,
            model_request_id=request_id,
        ), None
    if kind in {"tool_use", "server_tool_use"}:
        call_id = str(native.get("id") or "")
        standard_kind = "server_tool_call" if kind == "server_tool_use" else "tool_call"
        return AgentContentBlock(
            kind=standard_kind,
            data={"name": native.get("name"), "arguments": native.get("input")},
            sequence=sequence,
            call_id=call_id,
            block_id=call_id,
            model_request_id=request_id,
        ), ("use", native)
    if kind in {"tool_result", "advisor_tool_result", "server_tool_result"}:
        call_id = str(native.get("tool_use_id") or "")
        standard_kind = "tool_result" if kind == "tool_result" else "server_tool_result"
        return AgentContentBlock(
            kind=standard_kind,
            data={"result": native.get("content"), "is_error": native.get("is_error")},
            sequence=sequence,
            call_id=call_id,
            model_request_id=request_id,
        ), ("result", native)
    return AgentContentBlock(kind="other", data=dict(native), sequence=sequence, model_request_id=request_id), None


def _assemble_tools(parts: Sequence[tuple[str, object]]) -> tuple[AgentToolCall, ...]:
    uses: dict[str, object] = {}
    results: dict[str, object] = {}
    order: list[str] = []
    for kind, native in parts:
        call_id = _native_value(native, "id" if kind == "use" else "tool_use_id")
        if not call_id:
            continue
        if kind == "use":
            if call_id not in uses:
                order.append(call_id)
            uses[call_id] = native
        else:
            results[call_id] = native
    calls: list[AgentToolCall] = []
    for call_id in order:
        native = uses[call_id]
        name = _native_value(native, "name") or "unknown"
        tool_kind, server_name, tool_name = _tool_identity(name, type(native).__name__)
        result = results.get(call_id)
        is_error = _native_raw(result, "is_error") if result is not None else None
        calls.append(
            AgentToolCall(
                call_id=call_id,
                tool_name=tool_name,
                display_name=name,
                tool_kind=tool_kind,
                server_name=server_name,
                status="failed" if is_error else "completed" if result is not None else "started",
                arguments=_native_raw(native, "input"),
                result=_native_raw(result, "content") if result is not None else None,
                error=_agent_error("tool", "Claude tool result reported an error") if is_error else None,
            )
        )
    return tuple(calls)


def _request_usage(
    index: int,
    request_id: str,
    message: object,
    raw: Mapping[str, object],
    backend: ModelBackendIdentity | None,
) -> ModelRequestUsage:
    model = _native_value(message, "model")
    service_tier = _optional_str(raw.get("service_tier"))
    base = backend or ModelBackendIdentity(api_provider="anthropic", api_mode="anthropic_messages")
    identity = replace(
        base,
        resolved_model=model or base.resolved_model,
        service_tier=service_tier or base.service_tier,
    )
    cache = raw.get("cache_creation") if isinstance(raw.get("cache_creation"), Mapping) else {}
    tokens = TokenUsage(
        input_tokens=_optional_int(raw.get("input_tokens")),
        output_tokens=_optional_int(raw.get("output_tokens")),
        total_tokens=_optional_int(raw.get("total_tokens")),
        cache_read_input_tokens=_optional_int(raw.get("cache_read_input_tokens")),
        cache_creation_input_tokens=_optional_int(raw.get("cache_creation_input_tokens")),
        cache_creation_5m_input_tokens=_optional_int(cache.get("ephemeral_5m_input_tokens")),
        cache_creation_1h_input_tokens=_optional_int(cache.get("ephemeral_1h_input_tokens")),
        semantics={
            "input_tokens": "provider_reported_non_cached_input",
            "cache_read_input_tokens": "provider_reported_separate_input_category",
            "cache_creation_input_tokens": "provider_reported_separate_input_category",
        },
    )
    billable: list[BillableUnit] = []
    server_tools = raw.get("server_tool_use")
    if isinstance(server_tools, Mapping):
        for unit, quantity in server_tools.items():
            parsed = _optional_int(quantity)
            if parsed is not None:
                billable.append(
                    BillableUnit(category="server_tool", unit=str(unit), quantity=str(parsed))
                )
    reported_fields = tuple(
        field
        for field, value in {
            "input_tokens": tokens.input_tokens,
            "output_tokens": tokens.output_tokens,
            "total_tokens": tokens.total_tokens,
            "cache_read_input_tokens": tokens.cache_read_input_tokens,
            "cache_creation_input_tokens": tokens.cache_creation_input_tokens,
            "cache_creation_5m_input_tokens": tokens.cache_creation_5m_input_tokens,
            "cache_creation_1h_input_tokens": tokens.cache_creation_1h_input_tokens,
        }.items()
        if value is not None
    )
    unavailable = {
        field: "Claude did not report this field"
        for field in ("total_tokens", "reasoning_output_tokens")
        if getattr(tokens, field) is None
    }
    return ModelRequestUsage(
        request_index=index,
        request_id=request_id,
        response_id=request_id,
        model_identity=identity,
        token_usage=tokens,
        billable_units=tuple(billable),
        status=_native_value(message, "stop_reason"),
        stop_reason=_native_value(message, "stop_reason"),
        reported_fields=reported_fields,
        unavailable_fields=unavailable,
        provider_payload=build_provider_payload(
            provider_type=PROVIDER_TYPE,
            payload_type="request_usage",
            adapter_version=ADAPTER_VERSION,
            data=dict(raw),
        ),
    )


def _terminal_state(
    terminal: object,
    *,
    interrupted: bool,
    fallback: AgentError | None,
) -> tuple[ProviderRunState, AgentError | None]:
    deferred = getattr(terminal, "deferred_tool_use", None)
    if deferred is not None:
        return ProviderRunState.NEEDS_INPUT, None
    if interrupted:
        return ProviderRunState.INTERRUPTED, None
    if not bool(getattr(terminal, "is_error", False)):
        return ProviderRunState.COMPLETED, None
    subtype = str(getattr(terminal, "subtype", "provider_error"))
    errors = getattr(terminal, "errors", None)
    message = "; ".join(str(item) for item in errors) if errors else subtype
    return ProviderRunState.FAILED, fallback or _agent_error(subtype, message)


def _transcript_terminal(
    records: Sequence[dict[str, object]],
    fallback: AgentError | None,
) -> tuple[ProviderRunState | None, AgentError | None]:
    assistants = [record for record in records if record.get("type") == "assistant"]
    if not assistants:
        return None, None
    if any(record.get("isApiErrorMessage") for record in assistants):
        return ProviderRunState.FAILED, fallback or _agent_error("provider", "Claude API error")
    last = assistants[-1].get("message")
    stop_reason = last.get("stop_reason") if isinstance(last, Mapping) else None
    if stop_reason in {"end_turn", "stop_sequence", "max_tokens", "stop"}:
        return ProviderRunState.COMPLETED, None
    return None, None


def _agent_error(error_type: str, message: str) -> AgentError:
    lowered = error_type.lower()
    category = "provider"
    if "auth" in lowered:
        category = "authentication"
    elif "rate" in lowered:
        category = "rate_limit"
    elif "context" in lowered or "max_tokens" in lowered:
        category = "context_limit"
    elif "cancel" in lowered or "interrupt" in lowered:
        category = "cancelled"
    elif "tool" in lowered:
        category = "tool"
    return AgentError(
        error_type=category,
        code=error_type,
        message=message,
        retryable=category in {"rate_limit", "provider"},
    )


def _reported_turn_cost(value: object) -> ReportedCost | None:
    if value is None:
        return None
    try:
        decimal = Decimal(str(value))
    except Exception:
        return None
    if decimal < 0:
        return None
    return ReportedCost(currency="USD", total_cost=format(decimal, "f"))


def _is_external_user_record(record: Mapping[str, object]) -> bool:
    if record.get("type") != "user":
        return False
    if record.get("isMeta") or record.get("isCompactSummary") or record.get("isSidechain"):
        return False
    message = record.get("message")
    if not isinstance(message, Mapping):
        return False
    content = message.get("content")
    if isinstance(content, str) and content.lstrip().startswith(
        ("<command-name>/", "<local-command-")
    ):
        # Claude persists slash-command control traffic as user-shaped records.
        # It is session maintenance metadata, not a new ARK Agent turn.
        return False
    if isinstance(content, list) and content and all(
        isinstance(item, Mapping) and item.get("type") == "tool_result" for item in content
    ):
        return False
    return True


def _build_transcript_turn(
    turn_id: str,
    records: list[dict[str, object]],
) -> ClaudeTranscriptTurn:
    timestamps = [str(record["timestamp"]) for record in records if record.get("timestamp")]
    return ClaudeTranscriptTurn(
        turn_id=turn_id,
        records=tuple(records),
        started_at=timestamps[0] if timestamps else None,
        completed_at=timestamps[-1] if timestamps else None,
    )


def _tool_identity(name: str, native_type: str) -> tuple[str, str | None, str]:
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return "mcp", parts[1], parts[2]
    if native_type.startswith("Server"):
        return "server", None, name
    lowered = name.lower()
    if lowered in {"bash", "shell"}:
        return "shell", None, name
    if lowered in {"read", "write", "edit", "multiedit", "glob", "grep"}:
        return "file", None, name
    if lowered in {"agent", "task"}:
        return "agent", None, name
    return "function", None, name


def _native_raw(value: object | None, key: str) -> object | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _native_value(value: object | None, key: str) -> str | None:
    raw = _native_raw(value, key)
    return str(raw) if raw is not None else None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
