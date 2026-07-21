from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic

from ..provider_contracts import (
    AgentContentBlock,
    AgentArtifactLocator,
    AgentError,
    AgentEvent,
    AgentToolCall,
    AgentTurnUsage,
    ProviderControlAction,
    ProviderControlRequest,
    ProviderControlResult,
    ProviderEventBatch,
    ProviderForkRequest,
    ProviderForkResult,
    ProviderRunRequest,
    ProviderRunState,
    ProviderSessionLocator,
    ProviderTurnLocator,
    ProviderTurnResult,
    TokenUsage,
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .codex import CodexProvider, CodexTurnResult


class CodexProviderRunHandle:
    def __init__(self, provider: CodexProvider, request: ProviderRunRequest, *, resume: bool) -> None:
        self.provider = provider
        self.request = request
        self.resume = resume
        self._run_id = f"r_{uuid.uuid4().hex}"
        self._started_at = utc_now_iso()
        self._started_monotonic = monotonic()
        self._session: ProviderSessionLocator | None = request.session_locator
        self._turn: ProviderTurnLocator | None = None
        self._state = ProviderRunState.STARTING
        self._native_result: CodexTurnResult | None = None
        self._result: ProviderTurnResult | None = None
        self._error: BaseException | None = None
        self._done = threading.Event()
        self._closed = False
        self._events: list[AgentEvent] = []
        self._lock = threading.RLock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def legacy_turn_result(self) -> object | None:
        native = self._native_result
        return native.turn_result if native is not None else None

    def session_locator(self) -> ProviderSessionLocator | None:
        with self._lock:
            return self._session

    def turn_locator(self) -> ProviderTurnLocator | None:
        with self._lock:
            return self._turn

    def poll_state(self) -> ProviderRunState:
        with self._lock:
            return self._state

    def drain_events(self, after_cursor: str | None = None) -> ProviderEventBatch:
        start = int(after_cursor) if after_cursor else 0
        with self._lock:
            events = tuple(self._events[start:])
            cursor = str(len(self._events))
            terminal = self._state.terminal
        return ProviderEventBatch(events=events, next_cursor=cursor, terminal=terminal)

    def wait_terminal(self, timeout_s: float | None = None) -> ProviderTurnResult:
        if not self._done.wait(timeout_s):
            raise TimeoutError(self.run_id)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def interrupt(self, timeout_s: float | None = None) -> ProviderControlResult:
        requested_at = utc_now_iso()
        accepted = self.provider.interrupt_agent(self.request.agent_id)
        terminal_confirmed = self._done.wait(timeout_s) if accepted else self._done.is_set()
        return ProviderControlResult(
            action=ProviderControlAction.INTERRUPT,
            accepted=accepted,
            terminal_confirmed=terminal_confirmed,
            resulting_state=self.poll_state() if terminal_confirmed else None,
            requested_at=requested_at,
            completed_at=utc_now_iso(),
            session_locator=self.session_locator(),
            turn_locator=self.turn_locator(),
            reason=None if accepted else "no active Codex turn handle",
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.action is ProviderControlAction.INTERRUPT:
            raw_timeout = request.options.get("timeout_s")
            timeout_s = float(raw_timeout) if isinstance(raw_timeout, (int, float)) else None
            return self.interrupt(timeout_s)
        return ProviderControlResult(
            action=request.action,
            accepted=False,
            terminal_confirmed=self._done.is_set(),
            resulting_state=self.poll_state() if self._done.is_set() else None,
            requested_at=request.requested_at,
            completed_at=utc_now_iso(),
            session_locator=self.session_locator(),
            turn_locator=self.turn_locator(),
            reason="Codex runtime adapter does not implement this control action",
        )

    def close(self) -> None:
        self._closed = True

    def _run(self) -> None:
        try:
            with self._lock:
                self._state = ProviderRunState.RUNNING
            context = self.request.execution_context
            if context is None:
                raise ValueError("Codex runtime requires ProviderExecutionContext")
            common = {
                "home_id": self.request.home_id,
                "home_root": context.home_root,
                "env": context.process_environment,
                "workdir": self.request.workdir,
                "prompt": self.request.prompt,
                "developer_instructions": self.request.developer_instructions,
                "agent_id": self.request.agent_id,
                "overwrite_developer_instructions": self.request.replace_developer_instructions,
                "on_turn_started": self._on_turn_started,
            }
            if self.resume:
                if self.request.session_locator is None:
                    raise ValueError("Codex resume requires session_locator")
                native = self.provider.resume_thread(
                    **common,
                    thread_id=self.request.session_locator.session_id,
                )
            else:
                native = self.provider.start_thread(
                    **common,
                    on_thread_started=self._on_thread_started,
                )
            self._native_result = native
            self._on_thread_started(native.thread_id)
            self._result = _normalize_codex_turn_result(
                request=self.request,
                run_id=self.run_id,
                native=native,
                session=self.session_locator(),
                turn=self.turn_locator(),
                started_at=self._started_at,
                duration_ms=(monotonic() - self._started_monotonic) * 1000,
                adapter_version="1",
            )
            with self._lock:
                self._state = self._result.status
                self._append_event("terminal." + self._state.value, terminal=True)
        except BaseException as exc:
            self._error = exc
            with self._lock:
                self._state = ProviderRunState.FAILED
                self._append_event("terminal.failed", terminal=True, data={"error_type": type(exc).__name__})
        finally:
            self._done.set()

    def _on_thread_started(self, thread_id: str) -> None:
        with self._lock:
            if self._session is None:
                if self.request.session_start_home_commit is not None:
                    self.request.session_start_home_commit()
                self._session = ProviderSessionLocator(
                    provider_type="codex",
                    session_id=thread_id,
                    home_id=self.request.home_id,
                    created_at=utc_now_iso(),
                    backend_identity=self.request.model_overrides,
                )
                self._append_event("session.started", data={"session_id": thread_id})

    def _on_turn_started(self, thread_id: str, turn_id: str) -> None:
        self._on_thread_started(thread_id)
        with self._lock:
            assert self._session is not None
            self._turn = ProviderTurnLocator(session=self._session, turn_id=turn_id)
            self._append_event("turn.started", data={"turn_id": turn_id})

    def _append_event(self, kind: str, *, terminal: bool = False, data: object | None = None) -> None:
        event = AgentEvent(
            provider_type="codex",
            session_id=self._session.session_id if self._session else None,
            turn_id=self._turn.turn_id if self._turn else None,
            sequence=len(self._events),
            timestamp=utc_now_iso(),
            kind=kind,
            terminal=terminal,
            data=data,
        )
        self._events.append(event)
        if self.request.event_sink is not None:
            self.request.event_sink(event)


class CodexRuntimeAdapter:
    provider_type = "codex"

    def __init__(self, provider: CodexProvider) -> None:
        self.provider = provider
        self._handles: dict[str, CodexProviderRunHandle] = {}
        self._lock = threading.RLock()

    def start(self, request: ProviderRunRequest) -> CodexProviderRunHandle:
        return self._start(request, resume=False)

    def resume(self, request: ProviderRunRequest) -> CodexProviderRunHandle:
        return self._start(request, resume=True)

    def _start(self, request: ProviderRunRequest, *, resume: bool) -> CodexProviderRunHandle:
        if request.provider_type != self.provider_type:
            raise ValueError("CodexRuntimeAdapter received a different provider_type")
        handle = CodexProviderRunHandle(self.provider, request, resume=resume)
        with self._lock:
            self._handles[handle.run_id] = handle
        return handle

    def fork(self, request: ProviderForkRequest) -> ProviderForkResult:
        ctx = request.execution_context
        if ctx is None or ctx.provider_type != "codex":
            raise ValueError("Codex fork requires a Codex ProviderExecutionContext")
        native = self.provider.fork_thread(
            home_id=ctx.home_id,
            home_root=ctx.home_root,
            env=ctx.process_environment,
            thread_id=request.source_session.session_id,
            agent_id=request.target_agent_id,
        )
        target_session = ProviderSessionLocator(
            provider_type="codex",
            session_id=native.thread_id,
            home_id=request.target_home_id,
            created_at=utc_now_iso(),
            backend_identity=request.source_session.backend_identity,
            native_locator={"rollout_relpath": native.rollout_relpath},
        )
        artifact_locator = None
        if native.rollout_relpath:
            artifact_locator = AgentArtifactLocator(
                provider_type="codex",
                home_id=request.target_home_id,
                session_id=native.thread_id,
                adapter_version="1",
                native_primary_ref=native.rollout_relpath,
            )
        return ProviderForkResult(
            source_session=request.source_session,
            target_session=target_session,
            status="created",
            source_turn=request.source_turn,
            fork_mode="session_only",
            workspace_isolated=False,
            artifact_locator=artifact_locator,
            limitations=("workspace files are not isolated or rolled back",),
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.run_id is None:
            return ProviderControlResult(
                action=request.action,
                accepted=False,
                terminal_confirmed=False,
                requested_at=request.requested_at,
                completed_at=utc_now_iso(),
                reason="run_id is required for runtime control",
            )
        with self._lock:
            handle = self._handles.get(request.run_id)
        if handle is None:
            return ProviderControlResult(
                action=request.action,
                accepted=False,
                terminal_confirmed=False,
                requested_at=request.requested_at,
                completed_at=utc_now_iso(),
                reason="unknown or expired run_id",
            )
        return handle.control(request)

    def close_session(self, locator: ProviderSessionLocator) -> ProviderControlResult:
        now = utc_now_iso()
        return ProviderControlResult(
            action=ProviderControlAction.ARCHIVE_SESSION,
            accepted=False,
            terminal_confirmed=True,
            requested_at=now,
            completed_at=now,
            session_locator=locator,
            reason="Codex session close is not implemented by the current SDK adapter",
        )

    def close(self) -> None:
        with self._lock:
            handles = tuple(self._handles.values())
        for handle in handles:
            handle.close()
        self.provider.close()


def _normalize_codex_turn_result(
    *,
    request: ProviderRunRequest,
    run_id: str,
    native: CodexTurnResult,
    session: ProviderSessionLocator | None,
    turn: ProviderTurnLocator | None,
    started_at: str,
    duration_ms: float,
    adapter_version: str,
) -> ProviderTurnResult:
    session = session or ProviderSessionLocator(
        provider_type="codex",
        session_id=native.thread_id,
        home_id=request.home_id,
        created_at=started_at,
    )
    raw_turn = native.turn_result
    turn_id = str(getattr(raw_turn, "id", "")) or f"turn-{run_id}"
    turn = turn or ProviderTurnLocator(session=session, turn_id=turn_id)
    status = _run_state(getattr(raw_turn, "status", "completed"))
    raw_usage = getattr(raw_turn, "usage", None)
    turn_usage, context_after = _normalize_usage(raw_usage, request, native.thread_id)
    raw_items = list(getattr(raw_turn, "items", None) or [])
    content_blocks, tool_calls = _normalize_items(
        raw_items,
        turn_id=turn.turn_id,
        adapter_version=adapter_version,
    )
    error_value = getattr(raw_turn, "error", None)
    error = None
    if error_value is not None:
        error = AgentError(
            error_type=type(error_value).__name__,
            message=str(getattr(error_value, "message", error_value)),
            provider_payload=build_provider_payload(
                provider_type="codex",
                payload_type="turn_error",
                data=error_value,
                adapter_version=adapter_version,
            ),
        )
    payload = build_provider_payload(
        provider_type="codex",
        payload_type="turn_result",
        data={
            "status": _enum_value(getattr(raw_turn, "status", None)),
            "started_at": getattr(raw_turn, "started_at", None),
            "completed_at": getattr(raw_turn, "completed_at", None),
            "duration_ms": getattr(raw_turn, "duration_ms", None),
            "usage": raw_usage,
            "rollout_relpath": native.rollout_relpath,
        },
        adapter_version=adapter_version,
    )
    return ProviderTurnResult(
        provider_type="codex",
        run_id=run_id,
        session_locator=session,
        turn_locator=turn,
        status=status,
        started_at=_timestamp_iso(getattr(raw_turn, "started_at", None)) or started_at,
        completed_at=_timestamp_iso(getattr(raw_turn, "completed_at", None)) or utc_now_iso(),
        duration_ms=getattr(raw_turn, "duration_ms", None) or duration_ms,
        final_text=getattr(raw_turn, "final_response", None),
        content_blocks=content_blocks,
        tool_calls=tool_calls,
        turn_usage=turn_usage,
        context_after=context_after,
        error=error,
        artifact_locator=AgentArtifactLocator(
            provider_type="codex",
            home_id=request.home_id,
            session_id=native.thread_id,
            adapter_version=adapter_version,
            native_primary_ref=native.rollout_relpath,
        ),
        provider_payload=payload,
    )


def _normalize_items(
    raw_items: list[object],
    *,
    turn_id: str,
    adapter_version: str,
) -> tuple[tuple[AgentContentBlock, ...], tuple[AgentToolCall, ...]]:
    blocks: list[AgentContentBlock] = []
    tools: list[AgentToolCall] = []
    for sequence, wrapped in enumerate(raw_items):
        item = getattr(wrapped, "root", wrapped)
        item_type = str(getattr(item, "type", type(item).__name__))
        item_id = _optional_string(getattr(item, "id", None))
        payload = build_provider_payload(
            provider_type="codex",
            payload_type="thread_item",
            data=item,
            adapter_version=adapter_version,
        )
        block_kind, data = _content_projection(item_type, item)
        tool = _tool_projection(
            item_type,
            item,
            turn_id=turn_id,
            provider_payload=payload,
        )
        call_id = tool.call_id if tool is not None else None
        blocks.append(
            AgentContentBlock(
                kind=block_kind,
                data=data,
                block_id=item_id,
                call_id=call_id,
                sequence=sequence,
                provider_payload=payload,
            )
        )
        if tool is not None:
            tools.append(tool)
    return tuple(blocks), tuple(tools)


def _content_projection(item_type: str, item: object) -> tuple[str, object]:
    if item_type == "agentMessage":
        return "text", {
            "text": getattr(item, "text", None),
            "phase": _enum_value(getattr(item, "phase", None)),
        }
    if item_type == "reasoning":
        return "reasoning", {
            "content": list(getattr(item, "content", None) or []),
            "summary": list(getattr(item, "summary", None) or []),
        }
    if item_type == "plan":
        return "text", {"text": getattr(item, "text", None), "phase": "plan"}
    if item_type == "contextCompaction":
        return "compaction_boundary", {"provider_item_type": item_type}
    if item_type == "userMessage":
        return "other", {"provider_item_type": item_type, "role": "user"}
    if item_type in {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
        "webSearch",
        "imageView",
        "imageGeneration",
    }:
        return "tool_call", {"provider_item_type": item_type}
    return "other", {"provider_item_type": item_type}


def _tool_projection(
    item_type: str,
    item: object,
    *,
    turn_id: str,
    provider_payload,
) -> AgentToolCall | None:
    if item_type not in {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "dynamicToolCall",
        "collabAgentToolCall",
        "webSearch",
        "imageView",
        "imageGeneration",
    }:
        return None
    call_id = _optional_string(getattr(item, "id", None)) or f"codex-{item_type}-{uuid.uuid4().hex}"
    tool_name = {
        "commandExecution": "shell",
        "fileChange": "file_change",
        "webSearch": "web_search",
        "imageView": "image_view",
        "imageGeneration": "image_generation",
    }.get(item_type)
    server_name = None
    arguments: object | None = None
    result: object | None = None
    tool_kind = "other"
    if item_type == "commandExecution":
        tool_kind = "shell"
        arguments = {"command": getattr(item, "command", None), "cwd": getattr(item, "cwd", None)}
        result = {
            "output": getattr(item, "aggregated_output", None),
            "exit_code": getattr(item, "exit_code", None),
        }
    elif item_type == "fileChange":
        tool_kind = "file"
        arguments = {"changes": getattr(item, "changes", None)}
    elif item_type == "mcpToolCall":
        tool_kind = "mcp"
        server_name = _optional_string(getattr(item, "server", None))
        tool_name = _optional_string(getattr(item, "tool", None)) or "mcp_tool"
        arguments = getattr(item, "arguments", None)
        result = getattr(item, "result", None)
    elif item_type == "dynamicToolCall":
        tool_kind = "function"
        tool_name = _optional_string(getattr(item, "tool", None)) or "dynamic_tool"
        arguments = getattr(item, "arguments", None)
        result = getattr(item, "content_items", None)
    elif item_type == "collabAgentToolCall":
        tool_kind = "agent"
        tool_name = str(_enum_value(getattr(item, "tool", None)) or "agent")
        arguments = {"prompt": getattr(item, "prompt", None), "model": getattr(item, "model", None)}
        result = {"receiver_thread_ids": getattr(item, "receiver_thread_ids", None)}
    elif item_type == "webSearch":
        tool_kind = "server"
        arguments = {"query": getattr(item, "query", None)}
        result = getattr(item, "action", None)
    elif item_type in {"imageView", "imageGeneration"}:
        tool_kind = "server"
        arguments = {
            "path": getattr(item, "path", None),
            "revised_prompt": getattr(item, "revised_prompt", None),
        }
        result = getattr(item, "result", None)
    status = str(_enum_value(getattr(item, "status", None)) or "completed")
    error_value = getattr(item, "error", None)
    error = None
    if error_value is not None:
        error = AgentError(
            error_type=type(error_value).__name__,
            message=str(getattr(error_value, "message", error_value)),
            provider_payload=provider_payload,
        )
    return AgentToolCall(
        call_id=call_id,
        turn_id=turn_id,
        tool_name=tool_name or item_type,
        tool_kind=tool_kind,
        server_name=server_name,
        arguments=arguments,
        result=result,
        status=status,
        duration_ms=getattr(item, "duration_ms", None),
        error=error,
        provider_payload=provider_payload,
    )


def _normalize_usage(
    raw_usage: object,
    request: ProviderRunRequest,
    session_id: str,
):
    if raw_usage is None:
        return None, None
    total = getattr(raw_usage, "total", None)
    last = getattr(raw_usage, "last", None)
    total_tokens = _token_usage(total)
    model_identity = request.model_overrides
    turn_usage = AgentTurnUsage(
        request_count=None,
        requests=(),
        token_usage=total_tokens,
        models_used=(model_identity,) if model_identity is not None else (),
        aggregate_complete=all(
            getattr(total_tokens, name) is not None
            for name in ("input_tokens", "output_tokens", "total_tokens")
        ),
    )
    context_window = getattr(raw_usage, "model_context_window", None)
    last_tokens = _token_usage(last)
    from ..provider_contracts import ProviderContextUsage

    available = last_tokens.total_tokens is not None and context_window is not None
    context_after = ProviderContextUsage(
        session_id=session_id,
        used_tokens=last_tokens.total_tokens,
        context_window_tokens=context_window,
        observed_at=utc_now_iso(),
        source="codex_turn_usage",
        measurement="request_usage",
        available=available,
        reason=None if available else "Codex turn usage omitted context evidence",
        model_identity=model_identity,
    )
    return turn_usage, context_after


def _token_usage(value: object | None) -> TokenUsage:
    if value is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=getattr(value, "input_tokens", None),
        output_tokens=getattr(value, "output_tokens", None),
        total_tokens=getattr(value, "total_tokens", None),
        cached_input_tokens=getattr(value, "cached_input_tokens", None),
        reasoning_output_tokens=getattr(value, "reasoning_output_tokens", None),
        semantics={
            "cached_input_tokens": "subset_of_input_tokens",
            "reasoning_output_tokens": "subset_of_output_tokens",
        },
    )


def _run_state(value: object) -> ProviderRunState:
    status = str(_enum_value(value) or value)
    return {
        "completed": ProviderRunState.COMPLETED,
        "interrupted": ProviderRunState.INTERRUPTED,
        "cancelled": ProviderRunState.CANCELLED,
        "failed": ProviderRunState.FAILED,
    }.get(status, ProviderRunState.COMPLETED)


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _timestamp_iso(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)
