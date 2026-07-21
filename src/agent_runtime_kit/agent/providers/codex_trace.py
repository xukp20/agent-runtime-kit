from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class AgentRolloutInfo:
    agent_id: str
    agent_type: str
    home_id: str
    thread_id: str | None
    rollout_relpath: str | None
    rollout_path: str | None
    exists: bool
    event_count: int


@dataclass(frozen=True)
class AgentTraceEventView:
    index: int
    event_type: str | None
    payload_type: str | None
    turn_id: str | None
    timestamp: object | None
    raw_event: dict[str, Any]


@dataclass
class AgentTurnSummary:
    turn_id: str
    status: str | None = None
    started_at: object | None = None
    completed_at: object | None = None
    duration_ms: int | None = None
    final_response: str | None = None
    usage: object | None = None
    event_start_index: int | None = None
    event_end_index: int | None = None


@dataclass(frozen=True)
class AgentResponseTextView:
    turn_id: str | None
    event_index: int
    phase: str
    text: str


@dataclass
class AgentToolCallView:
    turn_id: str | None
    call_id: str
    call_index: int
    event_index: int
    tool_name: str | None
    display_name: str | None = None
    arguments: object | None = None
    output: object | None = None
    ok: bool | None = None
    started_at: object | None = None
    completed_at: object | None = None
    duration_ms: int | None = None
    raw_call_event: dict[str, Any] | None = None
    raw_output_event: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentArtifactView:
    path: str | None
    exists: bool
    json: object | None = None
    summary: dict[str, Any] | None = None
    read_error: str | None = None


@dataclass(frozen=True)
class AgentTraceReport:
    rollout: AgentRolloutInfo
    turns: list[AgentTurnSummary]
    latest_turn: AgentTurnSummary | None
    response_texts: list[AgentResponseTextView]
    tool_search_count: int
    tool_calls: list[AgentToolCallView]
    slow_tool_calls: list[AgentToolCallView]
    artifact: AgentArtifactView | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AgentTraceReportPaths:
    agent_id: str
    reports_root: str
    latest_json_path: str
    latest_markdown_path: str
    turn_json_path: str | None = None
    turn_markdown_path: str | None = None
    written_paths: list[str] = field(default_factory=list)


class AgentTraceReader:
    """Read-only parser for provider rollout JSONL events."""

    def __init__(
        self,
        *,
        agent: object,
        rollout_path: Path | None,
        events: list[dict[str, Any]],
    ) -> None:
        self.agent = agent
        self.rollout_path = Path(rollout_path) if rollout_path is not None else None
        self.events = list(events)
        self._parsed: _ParsedTrace | None = None

    def get_rollout_info(self) -> AgentRolloutInfo:
        return AgentRolloutInfo(
            agent_id=str(getattr(self.agent, "agent_id", "")),
            agent_type=str(getattr(self.agent, "agent_type", "")),
            home_id=str(getattr(self.agent, "home_id", "")),
            thread_id=_optional_str(getattr(self.agent, "thread_id", None)),
            rollout_relpath=_optional_str(getattr(self.agent, "rollout_relpath", None)),
            rollout_path=str(self.rollout_path) if self.rollout_path is not None else None,
            exists=bool(self.rollout_path is not None and self.rollout_path.exists()),
            event_count=len(self.events),
        )

    def list_turns(self) -> list[AgentTurnSummary]:
        return list(self._parse().turns)

    def get_turn(
        self,
        *,
        turn_id: str | None = None,
        index: int | None = None,
        latest: bool = False,
    ) -> AgentTurnSummary | None:
        turns = self.list_turns()
        if latest:
            return turns[-1] if turns else None
        if turn_id is not None:
            return next((turn for turn in turns if turn.turn_id == str(turn_id)), None)
        if index is not None:
            return _get_by_index(turns, index)
        return None

    def get_event(
        self,
        *,
        index: int | None = None,
        last: bool = False,
    ) -> AgentTraceEventView | None:
        events = self._parse().event_views
        if last:
            return events[-1] if events else None
        if index is not None:
            return _get_by_index(events, index)
        return None

    def tail_events(
        self,
        *,
        limit: int = 20,
        event_type: str | None = None,
        payload_type: str | None = None,
    ) -> list[AgentTraceEventView]:
        events = self._parse().event_views
        filtered = [
            event
            for event in events
            if (event_type is None or event.event_type == event_type)
            and (payload_type is None or event.payload_type == payload_type)
        ]
        return filtered[-max(limit, 0) :]

    def list_response_texts(
        self,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ) -> list[AgentResponseTextView]:
        responses = self._parse().responses
        if latest:
            latest_turn = self.get_turn(latest=True)
            turn_id = latest_turn.turn_id if latest_turn is not None else None
        if turn_id is None:
            return list(responses)
        return [response for response in responses if response.turn_id == str(turn_id)]

    def get_latest_response_text(self) -> str | None:
        responses = self.list_response_texts(latest=True)
        if responses:
            return responses[-1].text
        latest_turn = self.get_turn(latest=True)
        return latest_turn.final_response if latest_turn is not None else None

    def list_tool_calls(
        self,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ) -> list[AgentToolCallView]:
        calls = self._parse().tool_calls
        if latest:
            latest_turn = self.get_turn(latest=True)
            turn_id = latest_turn.turn_id if latest_turn is not None else None
        if turn_id is None:
            return list(calls)
        return [call for call in calls if call.turn_id == str(turn_id)]

    def get_tool_call(
        self,
        *,
        call_id: str | None = None,
        index: int | None = None,
        last: bool = False,
    ) -> AgentToolCallView | None:
        calls = self.list_tool_calls()
        if last:
            return calls[-1] if calls else None
        if call_id is not None:
            return next((call for call in calls if call.call_id == str(call_id)), None)
        if index is not None:
            return _get_by_index(calls, index)
        return None

    def build_trace_report(
        self,
        *,
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
    ) -> AgentTraceReport:
        parsed = self._parse()
        warnings = list(parsed.warnings)
        rollout = self.get_rollout_info()
        if not rollout.exists:
            warnings.append("rollout file is missing")
        if rollout.rollout_relpath is None:
            warnings.append("agent has no rollout_relpath")
        slow_calls = sorted(
            [call for call in parsed.tool_calls if call.duration_ms is not None],
            key=lambda call: call.duration_ms or 0,
            reverse=True,
        )[: max(slow_call_limit, 0)]
        return AgentTraceReport(
            rollout=rollout,
            turns=list(parsed.turns),
            latest_turn=parsed.turns[-1] if parsed.turns else None,
            response_texts=list(parsed.responses),
            tool_search_count=parsed.tool_search_count,
            tool_calls=list(parsed.tool_calls),
            slow_tool_calls=slow_calls,
            artifact=_read_artifact(artifact_path) if artifact_path is not None else None,
            warnings=warnings,
        )

    def export_trace_report(
        self,
        *,
        output_path: str | Path,
        format: Literal["json", "markdown"] = "json",
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
    ) -> AgentTraceReport:
        report = self.build_trace_report(
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            from ..models import to_jsonable

            path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif format == "markdown":
            path.write_text(_trace_report_markdown(report), encoding="utf-8")
        else:
            raise ValueError(f"unsupported trace report format: {format}")
        return report

    def _parse(self) -> "_ParsedTrace":
        if self._parsed is None:
            self._parsed = _parse_trace_events(self.events)
        return self._parsed


@dataclass
class _ParsedTrace:
    event_views: list[AgentTraceEventView]
    turns: list[AgentTurnSummary]
    responses: list[AgentResponseTextView]
    tool_calls: list[AgentToolCallView]
    tool_search_count: int
    warnings: list[str]


def _parse_trace_events(events: list[dict[str, Any]]) -> _ParsedTrace:
    turns: dict[str, AgentTurnSummary] = {}
    turn_order: list[str] = []
    event_views: list[AgentTraceEventView] = []
    responses: list[AgentResponseTextView] = []
    tool_calls: list[AgentToolCallView] = []
    calls_by_id: dict[str, AgentToolCallView] = {}
    pending_outputs: dict[str, tuple[int, dict[str, Any]]] = {}
    current_turn_id: str | None = None
    tool_search_count = 0
    warnings: list[str] = []

    for index, raw_event in enumerate(events):
        event = raw_event if isinstance(raw_event, dict) else {"raw_event": raw_event}
        payload = _payload(event)
        event_type = _event_type(event)
        payload_type = _payload_type(event)
        timestamp = _timestamp_value(event, payload)
        turn_id = _turn_id(event, payload) or current_turn_id
        if event_type == "turn_context" and _turn_id(event, payload):
            turn_id = _turn_id(event, payload)
            current_turn_id = turn_id
        elif turn_id is not None:
            current_turn_id = turn_id
        event_views.append(
            AgentTraceEventView(
                index=index,
                event_type=event_type,
                payload_type=payload_type,
                turn_id=turn_id,
                timestamp=timestamp,
                raw_event=event,
            )
        )
        if turn_id is not None:
            turn = _ensure_turn(turns, turn_order, turn_id)
            if turn.event_start_index is None:
                turn.event_start_index = index
            turn.event_end_index = index
            _update_turn_from_event(turn, event, payload, event_type, payload_type)
        if _is_tool_search_event(event, payload, event_type, payload_type):
            tool_search_count += 1
        response = _response_text_from_event(event, payload, event_type, payload_type)
        if response is not None:
            response_turn_id = turn_id
            responses.append(
                AgentResponseTextView(
                    turn_id=response_turn_id,
                    event_index=index,
                    phase=payload_type or event_type or "response",
                    text=response,
                )
            )
            if response_turn_id is not None:
                _ensure_turn(turns, turn_order, response_turn_id).final_response = response
        if _is_tool_output_event(event, payload, event_type, payload_type):
            call_id = _tool_call_id(event, payload)
            if call_id is None:
                warnings.append(f"tool output at event {index} has no call_id")
                continue
            call = calls_by_id.get(call_id)
            if call is None:
                pending_outputs[call_id] = (index, event)
                continue
            _attach_tool_output(call, index, event)
        elif _is_tool_call_event(event, payload, event_type, payload_type):
            call_id = _tool_call_id(event, payload) or f"call_{index}"
            call = AgentToolCallView(
                turn_id=turn_id,
                call_id=call_id,
                call_index=len(tool_calls),
                event_index=index,
                tool_name=_tool_name(event, payload),
                display_name=_display_name(event, payload),
                arguments=_tool_arguments(event, payload),
                started_at=timestamp,
                raw_call_event=event,
            )
            tool_calls.append(call)
            calls_by_id[call_id] = call
            if call_id in pending_outputs:
                output_index, output_event = pending_outputs.pop(call_id)
                _attach_tool_output(call, output_index, output_event)

    for call_id, (index, _event) in pending_outputs.items():
        warnings.append(f"tool output at event {index} has no matching call event: {call_id}")
    parsed_turns = [turns[turn_id] for turn_id in turn_order]
    return _ParsedTrace(
        event_views=event_views,
        turns=parsed_turns,
        responses=responses,
        tool_calls=tool_calls,
        tool_search_count=tool_search_count,
        warnings=warnings,
    )


def _ensure_turn(
    turns: dict[str, AgentTurnSummary],
    order: list[str],
    turn_id: str,
) -> AgentTurnSummary:
    if turn_id not in turns:
        turns[turn_id] = AgentTurnSummary(turn_id=turn_id)
        order.append(turn_id)
    return turns[turn_id]


def _update_turn_from_event(
    turn: AgentTurnSummary,
    event: dict[str, Any],
    payload: dict[str, Any],
    event_type: str | None,
    payload_type: str | None,
) -> None:
    if event_type == "turn_context":
        turn.status = turn.status or "inProgress"
        turn.started_at = turn.started_at or _timestamp_value(event, payload)
        usage = payload.get("usage") or event.get("usage")
        if usage is not None:
            turn.usage = usage
    if payload_type in {"task_started", "turn_started"}:
        turn.status = "inProgress"
        turn.started_at = _first_present(payload, "started_at", "timestamp") or _timestamp_value(event, payload)
    if event_type == "turn_result":
        turn.status = _optional_str(event.get("status")) or "completed"
        turn.started_at = _first_present(event, "started_at", "timestamp") or turn.started_at
        turn.completed_at = _first_present(event, "completed_at", "finished_at") or turn.completed_at
        turn.duration_ms = _int_or_none(event.get("duration_ms")) or turn.duration_ms
        response = event.get("final_response")
        if isinstance(response, str):
            turn.final_response = response
        usage = event.get("usage")
        if usage is not None:
            turn.usage = usage
    if payload_type in {"task_complete", "turn_completed", "task_completed"}:
        turn.status = "completed"
        turn.completed_at = _first_present(payload, "completed_at", "finished_at", "timestamp") or _timestamp_value(event, payload)
        turn.duration_ms = _int_or_none(payload.get("duration_ms")) or turn.duration_ms
        usage = payload.get("usage") or event.get("usage")
        if usage is not None:
            turn.usage = usage
        response = payload.get("last_agent_message") or payload.get("final_response")
        if isinstance(response, str):
            turn.final_response = response
    if payload_type in {"turn_aborted", "task_aborted", "task_failed"}:
        turn.status = "interrupted" if payload_type == "turn_aborted" else "failed"
        turn.completed_at = _first_present(payload, "completed_at", "finished_at", "timestamp") or _timestamp_value(event, payload)
    if turn.duration_ms is None:
        turn.duration_ms = _duration_ms(turn.started_at, turn.completed_at)


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _event_type(event: dict[str, Any]) -> str | None:
    value = event.get("type")
    return str(value) if value is not None else None


def _payload_type(event: dict[str, Any]) -> str | None:
    payload = _payload(event)
    value = payload.get("type") or event.get("payload_type")
    return str(value) if value is not None else None


def _turn_id(event: dict[str, Any], payload: dict[str, Any]) -> str | None:
    value = payload.get("turn_id") or event.get("turn_id")
    return str(value) if value is not None and str(value) else None


def _timestamp_value(event: dict[str, Any], payload: dict[str, Any]) -> object | None:
    return _first_present(event, "timestamp", "created_at", "started_at", "completed_at") or _first_present(
        payload,
        "timestamp",
        "created_at",
        "started_at",
        "completed_at",
    )


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _response_text_from_event(
    event: dict[str, Any],
    payload: dict[str, Any],
    event_type: str | None,
    payload_type: str | None,
) -> str | None:
    if payload_type == "agent_message":
        return _string_from(payload, "message", "text")
    if payload_type in {"task_complete", "turn_completed", "task_completed"}:
        return _string_from(payload, "last_agent_message", "final_response", "message", "text")
    if event_type == "turn_result":
        return _string_from(event, "final_response")
    if payload_type == "message" and payload.get("role") == "assistant":
        return _message_content_to_text(payload.get("content"))
    return None


def _message_content_to_text(content: object) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = []
    for item in content:
        if isinstance(item, dict):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts) if parts else None


def _is_tool_search_event(
    event: dict[str, Any],
    payload: dict[str, Any],
    event_type: str | None,
    payload_type: str | None,
) -> bool:
    name = _tool_name(event, payload)
    type_text = " ".join(filter(None, [event_type, payload_type, name]))
    return "tool_search" in type_text or "tool-search" in type_text


def _is_tool_call_event(
    event: dict[str, Any],
    payload: dict[str, Any],
    event_type: str | None,
    payload_type: str | None,
) -> bool:
    if _is_tool_output_event(event, payload, event_type, payload_type):
        return False
    if payload_type == "function_call":
        return True
    type_text = " ".join(filter(None, [event_type, payload_type]))
    if "tool_call" in type_text and not any(token in type_text for token in ("_end", "_output", "_result")):
        return True
    return _tool_name(event, payload) is not None and _tool_arguments(event, payload) is not None


def _is_tool_output_event(
    event: dict[str, Any],
    payload: dict[str, Any],
    event_type: str | None,
    payload_type: str | None,
) -> bool:
    if payload_type in {
        "function_call_output",
        "custom_tool_call_output",
        "tool_result",
        "tool_output",
        "mcp_tool_output",
        "mcp_tool_call_end",
    }:
        return True
    type_text = " ".join(filter(None, [event_type, payload_type]))
    if "tool_output" in type_text or "tool_result" in type_text or "function_call_output" in type_text:
        return True
    return _tool_call_id(event, payload) is not None and any(
        key in payload or key in event
        for key in ("output", "result", "content", "response")
    )


def _tool_call_id(event: dict[str, Any], payload: dict[str, Any]) -> str | None:
    value = (
        payload.get("call_id")
        or event.get("call_id")
        or payload.get("tool_call_id")
        or event.get("tool_call_id")
        or payload.get("id")
    )
    return str(value) if value is not None and str(value) else None


def _tool_name(event: dict[str, Any], payload: dict[str, Any]) -> str | None:
    value = payload.get("tool_name") or event.get("tool_name") or payload.get("name") or event.get("name")
    if value is None and isinstance(payload.get("function"), dict):
        value = payload["function"].get("name")
    if value is None and isinstance(payload.get("call"), dict):
        value = payload["call"].get("name")
    if value is None and isinstance(payload.get("invocation"), dict):
        value = payload["invocation"].get("tool")
    return str(value) if value is not None and str(value) else None


def _display_name(event: dict[str, Any], payload: dict[str, Any]) -> str | None:
    value = payload.get("display_name") or event.get("display_name")
    return str(value) if value is not None and str(value) else None


def _tool_arguments(event: dict[str, Any], payload: dict[str, Any]) -> object | None:
    value = (
        payload.get("arguments")
        if "arguments" in payload
        else payload.get("args")
        if "args" in payload
        else payload.get("input")
        if "input" in payload
        else payload.get("parameters")
        if "parameters" in payload
        else event.get("arguments")
    )
    if value is None and isinstance(payload.get("function"), dict):
        value = payload["function"].get("arguments")
    if value is None and isinstance(payload.get("invocation"), dict):
        value = payload["invocation"].get("arguments")
    return _maybe_parse_json(value)


def _attach_tool_output(call: AgentToolCallView, output_index: int, output_event: dict[str, Any]) -> None:
    payload = _payload(output_event)
    payload_type = _payload_type(output_event)
    current_payload_type = _payload_type(call.raw_output_event) if call.raw_output_event is not None else None
    if current_payload_type == "mcp_tool_call_end" and payload_type != "mcp_tool_call_end":
        return
    output_timestamp = _timestamp_value(output_event, payload)
    call.raw_output_event = output_event
    call.completed_at = _first_present(payload, "completed_at", "finished_at") or output_timestamp
    call.output = _tool_output(payload, output_event)
    call.ok = _tool_ok(payload, output_event)
    output_tool_name = _tool_name(output_event, payload)
    if output_tool_name is not None and call.tool_name != output_tool_name:
        call.display_name = call.tool_name
        call.tool_name = output_tool_name
    call.duration_ms = (
        _duration_object_ms(payload.get("duration"))
        or _int_or_none(payload.get("duration_ms") or output_event.get("duration_ms"))
        or _duration_ms(call.started_at, call.completed_at)
    )


def _tool_output(payload: dict[str, Any], event: dict[str, Any]) -> object | None:
    if "output" in payload:
        return _maybe_parse_json(payload.get("output"))
    if "result" in payload:
        return payload.get("result")
    if "content" in payload:
        return payload.get("content")
    if "response" in payload:
        return payload.get("response")
    if "output" in event:
        return _maybe_parse_json(event.get("output"))
    if "result" in event:
        return event.get("result")
    return None


def _tool_ok(payload: dict[str, Any], event: dict[str, Any]) -> bool | None:
    for mapping in (payload, event):
        for key in ("ok", "success"):
            value = mapping.get(key)
            if isinstance(value, bool):
                return value
        result = mapping.get("result")
        if isinstance(result, dict):
            if "Ok" in result:
                return True
            if "Err" in result:
                return False
            for key in ("ok", "success"):
                value = result.get(key)
                if isinstance(value, bool):
                    return value
    return None


def _maybe_parse_json(value: object) -> object:
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _read_artifact(path: str | Path | None) -> AgentArtifactView:
    if path is None:
        return AgentArtifactView(path=None, exists=False)
    artifact_path = Path(path)
    if not artifact_path.exists():
        return AgentArtifactView(path=str(artifact_path), exists=False)
    try:
        data = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - report should preserve artifact read failures.
        return AgentArtifactView(path=str(artifact_path), exists=True, read_error=str(exc))
    return AgentArtifactView(
        path=str(artifact_path),
        exists=True,
        json=data,
        summary=_artifact_summary(data),
    )


def _artifact_summary(data: object) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {"type": type(data).__name__}
    summary: dict[str, Any] = {"keys": sorted(str(key) for key in data)}
    for key in (
        "prompt_marker_seen",
        "developer_marker_seen",
        "skill_keys_seen",
        "application_tools_called",
        "submit_tool_called",
        "tool_results",
    ):
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, list):
            summary[key] = {"count": len(value), "items": value[:20]}
        elif isinstance(value, dict):
            summary[key] = {"keys": sorted(str(item_key) for item_key in value)}
        else:
            summary[key] = value
    return summary


def _trace_report_markdown(report: AgentTraceReport) -> str:
    latest = report.latest_turn
    lines = [
        "# Agent Trace Report",
        "",
        "## Rollout",
        "",
        f"- Agent: `{report.rollout.agent_id}` (`{report.rollout.agent_type}`)",
        f"- Thread: `{report.rollout.thread_id}`",
        f"- Rollout: `{report.rollout.rollout_path}`",
        f"- Events: {report.rollout.event_count}",
        "",
        "## Latest Turn",
        "",
    ]
    if latest is None:
        lines.append("- No parsed turns.")
    else:
        lines.extend(
            [
                f"- Turn: `{latest.turn_id}`",
                f"- Status: `{latest.status}`",
                f"- Duration ms: `{latest.duration_ms}`",
                f"- Final response: {latest.final_response or ''}",
            ]
        )
    lines.extend(["", "## Tool Calls", "", "| # | Tool | Duration ms | OK |", "|---:|---|---:|---|"])
    for call in report.tool_calls:
        lines.append(
            f"| {call.call_index} | `{call.tool_name or ''}` | "
            f"{'' if call.duration_ms is None else call.duration_ms} | "
            f"{'' if call.ok is None else call.ok} |"
        )
    lines.extend(["", "## Slow Tool Calls", "", "| # | Tool | Duration ms |", "|---:|---|---:|"])
    for call in report.slow_tool_calls:
        lines.append(f"| {call.call_index} | `{call.tool_name or ''}` | {call.duration_ms} |")
    if report.artifact is not None:
        lines.extend(
            [
                "",
                "## Artifact",
                "",
                f"- Path: `{report.artifact.path}`",
                f"- Exists: `{report.artifact.exists}`",
                f"- Summary: `{json.dumps(report.artifact.summary or {}, sort_keys=True)}`",
            ]
        )
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    return "\n".join(lines) + "\n"


def _duration_ms(started_at: object | None, completed_at: object | None) -> int | None:
    start = _timestamp_ms(started_at)
    end = _timestamp_ms(completed_at)
    if start is None or end is None:
        return None
    return max(int(end - start), 0)


def _duration_object_ms(value: object | None) -> int | None:
    if not isinstance(value, dict):
        return None
    secs = _int_or_none(value.get("secs")) or 0
    nanos = _int_or_none(value.get("nanos")) or 0
    return int(secs * 1000 + nanos / 1_000_000)


def _timestamp_ms(value: object | None) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp() * 1000.0


def _int_or_none(value: object | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _string_from(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str):
            return value
    return None


def _optional_str(value: object | None) -> str | None:
    return str(value) if value is not None else None


def _get_by_index(values: list[Any], index: int) -> Any | None:
    try:
        return values[index]
    except IndexError:
        return None
