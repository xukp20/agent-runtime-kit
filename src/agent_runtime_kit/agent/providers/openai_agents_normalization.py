from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Mapping

from ..models import to_jsonable
from ..provider_contracts import (
    AgentContentBlock,
    AgentError,
    AgentToolCall,
    AgentTurnUsage,
    ModelBackendIdentity,
    ModelRequestUsage,
    TokenUsage,
    build_provider_payload,
)


PROVIDER_TYPE = "openai_agents"
ADAPTER_VERSION = "1"


def normalize_items(items: list[object], *, turn_id: str) -> tuple[tuple[AgentContentBlock, ...], tuple[AgentToolCall, ...]]:
    blocks: list[AgentContentBlock] = []
    calls: dict[str, AgentToolCall] = {}
    for sequence, item in enumerate(items):
        item_type = str(getattr(item, "type", type(item).__name__))
        raw = _plain(getattr(item, "raw_item", item))
        raw_type = _field(raw, "type")
        tool_kind, server_name, tool_origin = _tool_origin(item, raw_type)
        payload = build_provider_payload(
            provider_type=PROVIDER_TYPE,
            payload_type="run_item",
            data={
                "item_type": item_type,
                "raw_type": raw_type,
                "raw_item": raw,
                "tool_origin": tool_origin,
            },
            adapter_version=ADAPTER_VERSION,
            sdk_or_cli_version="0.18.3",
        )
        if item_type == "message_output_item":
            text = _message_text(raw)
            blocks.append(
                AgentContentBlock(
                    kind="text",
                    data=text,
                    sequence=sequence,
                    provider_payload=payload,
                )
            )
            continue
        if item_type == "reasoning_item":
            blocks.append(
                AgentContentBlock(
                    kind="reasoning",
                    data=raw,
                    sequence=sequence,
                    provider_payload=payload,
                )
            )
            continue
        if item_type == "compaction_item" or raw_type in {"compaction", "compaction_summary"}:
            blocks.append(
                AgentContentBlock(
                    kind="compaction_summary",
                    data=raw,
                    sequence=sequence,
                    provider_payload=payload,
                )
            )
            continue
        if item_type in {"handoff_call_item", "handoff_output_item"}:
            blocks.append(
                AgentContentBlock(
                    kind="handoff",
                    data=raw,
                    sequence=sequence,
                    provider_payload=payload,
                )
            )
            continue
        if item_type in {"tool_call_item", "tool_approval_item", "mcp_approval_request_item"}:
            call_id = _call_id(item, raw, sequence)
            tool_name = str(
                getattr(item, "tool_name", None)
                or _field(raw, "name")
                or _field(raw, "tool_name")
                or "unknown"
            )
            approval = raw if "approval" in item_type else None
            calls[call_id] = AgentToolCall(
                call_id=call_id,
                turn_id=turn_id,
                tool_name=tool_name,
                tool_kind=tool_kind,
                server_name=server_name,
                status="needs_approval" if approval is not None else "called",
                arguments=_field(raw, "arguments") or _field(raw, "input"),
                approval=approval,
                provider_payload=payload,
            )
            continue
        if item_type in {"tool_call_output_item", "mcp_approval_response_item"}:
            call_id = _call_id(item, raw, sequence)
            previous = calls.get(call_id)
            output = getattr(item, "output", None)
            if output is None:
                output = _field(raw, "output")
            calls[call_id] = AgentToolCall(
                call_id=call_id,
                turn_id=turn_id,
                tool_name=previous.tool_name if previous else "unknown",
                tool_kind=previous.tool_kind if previous else tool_kind,
                server_name=previous.server_name if previous else server_name,
                status="completed",
                arguments=previous.arguments if previous else None,
                result=_plain(output),
                provider_payload=payload,
            )
    return tuple(blocks), tuple(calls.values())


def _tool_origin(item: object, raw_type: object) -> tuple[str, str | None, object | None]:
    origin = getattr(item, "tool_origin", None)
    plain = _plain(origin) if origin is not None else None
    origin_type = getattr(origin, "type", None)
    origin_value = getattr(origin_type, "value", origin_type)
    kind = str(origin_value) if origin_value in {"function", "mcp", "agent_as_tool"} else "function"
    if str(raw_type).startswith("mcp_"):
        kind = "mcp"
    server_name = getattr(origin, "mcp_server_name", None)
    return kind, str(server_name) if server_name is not None else None, plain


def normalize_usage(
    raw_responses: list[object],
    *,
    model_identity: ModelBackendIdentity,
    session_id: str,
    turn_id: str,
) -> tuple[tuple[ModelRequestUsage, ...], AgentTurnUsage]:
    requests: list[ModelRequestUsage] = []
    for index, response in enumerate(raw_responses):
        usage = getattr(response, "usage", None)
        input_tokens = _int_attr(usage, "input_tokens")
        output_tokens = _int_attr(usage, "output_tokens")
        total_tokens = _int_attr(usage, "total_tokens")
        input_details = getattr(usage, "input_tokens_details", None)
        output_details = getattr(usage, "output_tokens_details", None)
        cached = _int_attr(input_details, "cached_tokens")
        cache_write = _int_attr(input_details, "cache_write_tokens")
        reasoning = _int_attr(output_details, "reasoning_tokens")
        visible = None
        if output_tokens is not None and reasoning is not None and output_tokens >= reasoning:
            visible = output_tokens - reasoning
        token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached,
            cache_read_input_tokens=cached,
            cache_write_input_tokens=cache_write,
            reasoning_output_tokens=reasoning,
            visible_output_tokens=visible,
            semantics={
                "cached_input_tokens": "subset_of_input_tokens",
                "reasoning_output_tokens": "subset_of_output_tokens",
                "visible_output_tokens": "derived_output_minus_reasoning",
            },
        )
        reported = tuple(
            name
            for name, value in (
                ("input_tokens", input_tokens),
                ("output_tokens", output_tokens),
                ("total_tokens", total_tokens),
                ("cached_input_tokens", cached),
                ("cache_read_input_tokens", cached),
                ("cache_write_input_tokens", cache_write),
                ("reasoning_output_tokens", reasoning),
            )
            if value is not None
        )
        requests.append(
            ModelRequestUsage(
                request_index=index,
                model_identity=model_identity,
                token_usage=token_usage,
                request_id=_optional_str(getattr(response, "request_id", None)),
                response_id=_optional_str(getattr(response, "response_id", None)),
                session_id=session_id,
                turn_id=turn_id,
                status="completed",
                reported_fields=reported,
                derived_fields=("visible_output_tokens",) if visible is not None else (),
                unavailable_fields={
                    "reported_cost": "Agents SDK model response did not report price",
                    "resolved_model": "Agents SDK model response did not expose a resolved model",
                },
                provider_payload=build_provider_payload(
                    provider_type=PROVIDER_TYPE,
                    payload_type="model_response_usage",
                    data={
                        "usage": _plain(usage),
                        "request_id": getattr(response, "request_id", None),
                        "response_id": getattr(response, "response_id", None),
                    },
                    adapter_version=ADAPTER_VERSION,
                    sdk_or_cli_version="0.18.3",
                ),
            )
        )
    values = tuple(requests)
    return values, AgentTurnUsage.from_requests(values)


def normalize_error(exc: BaseException) -> AgentError:
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None) or (str(status) if status is not None else None)
    retryable = bool(status in {408, 409, 429} or (isinstance(status, int) and status >= 500)) if status is not None else None
    return AgentError(
        error_type=_error_type(exc),
        message=str(exc),
        code=_optional_str(code),
        retryable=retryable,
        provider_payload=build_provider_payload(
            provider_type=PROVIDER_TYPE,
            payload_type="error",
            data={
                "class": type(exc).__name__,
                "status_code": status,
                "code": code,
                "request_id": getattr(exc, "request_id", None),
            },
            adapter_version=ADAPTER_VERSION,
            sdk_or_cli_version="0.18.3",
        ),
    )


def _error_type(exc: BaseException) -> str:
    name = type(exc).__name__
    normalized = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            normalized.append("_")
        normalized.append(char.lower())
    return "openai_agents_" + "".join(normalized)


def _message_text(raw: object) -> str:
    content = _field(raw, "content")
    if not isinstance(content, list):
        return str(_field(raw, "text") or "")
    parts: list[str] = []
    for item in content:
        text = _field(item, "text") or _field(item, "refusal")
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


def _call_id(item: object, raw: object, sequence: int) -> str:
    value = getattr(item, "call_id", None) or _field(raw, "call_id") or _field(raw, "id")
    return str(value or f"unknown-{sequence}")


def _plain(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _plain(dump(mode="json", exclude_none=True))
        except TypeError:
            return _plain(dump())
    if is_dataclass(value):
        return _plain(asdict(value))
    try:
        return to_jsonable(value)
    except (TypeError, ValueError):
        return repr(value)


def _field(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _int_attr(value: object, name: str) -> int | None:
    raw = getattr(value, name, None) if value is not None else None
    return int(raw) if isinstance(raw, (int, float)) else None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None
