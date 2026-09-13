from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from time import monotonic, sleep
from urllib.parse import quote

from ..provider_contracts import (
    AgentArtifactLocator,
    AgentContentBlock,
    AgentError,
    AgentEvent,
    AgentToolCall,
    AgentTurnUsage,
    ModelBackendIdentity,
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
from .grok_acp import GrokAcpError, GrokAcpProcess
from .grok_home import (
    GROK_ADAPTER_VERSION,
    GROK_CLI_VERSION,
    validate_grok_workdir,
)


_KNOWN_STOP_REASONS = {"end_turn", "cancelled", "max_tokens", "max_turn_requests", "refusal"}


class GrokProcessCleanupError(RuntimeError):
    pass


class GrokProviderRunHandle:
    def __init__(self, *, request: ProviderRunRequest, resume: bool, on_done) -> None:  # noqa: ANN001
        self.request = request
        self.resume = resume
        self._on_done = on_done
        self._run_id = f"r_{uuid.uuid4().hex}"
        self._started_at = utc_now_iso()
        self._started_monotonic = monotonic()
        self._state = ProviderRunState.STARTING
        self._session = request.session_locator
        self._turn: ProviderTurnLocator | None = None
        self._result: ProviderTurnResult | None = None
        self._error: BaseException | None = None
        self._transport: GrokAcpProcess | None = None
        self._events: list[AgentEvent] = []
        self._text_parts: list[str] = []
        self._tool_calls: dict[str, AgentToolCall] = {}
        self._accept_updates = False
        self._prompt_active = False
        self._prompt_finished = False
        self._requested_stop: ProviderControlAction | None = None
        self._cleanup_confirmed = False
        self._cleanup_uncertain = False
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)

    def begin(self) -> None:
        self._worker.start()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def cleanup_confirmed(self) -> bool:
        return self._cleanup_confirmed

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
        start = int(after_cursor) if after_cursor is not None else 0
        with self._lock:
            return ProviderEventBatch(
                events=tuple(self._events[start:]),
                next_cursor=str(len(self._events)),
                terminal=self._state.terminal,
            )

    def wait_terminal(self, timeout_s: float | None = None) -> ProviderTurnResult:
        effective = timeout_s if timeout_s is not None else self.request.run_options.timeout_s
        if not self._done.wait(effective):
            raise TimeoutError(self.run_id)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def interrupt(self, timeout_s: float | None = None) -> ProviderControlResult:
        return self._stop(ProviderControlAction.INTERRUPT, timeout_s)

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.action is ProviderControlAction.STEER:
            with self._lock:
                ready = self._prompt_active and self._requested_stop is None and not self._done.is_set()
                session = self._session
                transport = self._transport
            if not ready or session is None or transport is None or not isinstance(request.content, str) or not request.content.strip():
                return ProviderControlResult(action=request.action, accepted=False, terminal_confirmed=False,
                    requested_at=request.requested_at, completed_at=utc_now_iso(), reason="Grok steer requires an active prompt and nonempty text")
            message_id = str(uuid.uuid4())
            try:
                response = transport.request("_x.ai/interject", {
                    "sessionId": session.session_id, "text": request.content, "interjectionId": message_id,
                }, timeout_s=10)
            except (GrokAcpError, TimeoutError) as exc:
                return ProviderControlResult(action=request.action, accepted=False, terminal_confirmed=False,
                    requested_at=request.requested_at, completed_at=utc_now_iso(), session_locator=session,
                    reason=f"Grok steer delivery unconfirmed: {exc}")
            with self._lock:
                accepted = response.get("result", {}).get("status") == "queued" and self._prompt_active and self._requested_stop is None
            return ProviderControlResult(action=request.action, accepted=accepted, terminal_confirmed=False,
                requested_at=request.requested_at, completed_at=utc_now_iso(), session_locator=session,
                reason="queued, not proof of consumption" if accepted else "prompt ended or cancelled while steer was in flight; delivery unconfirmed",
                provider_payload=build_provider_payload(provider_type="grok", payload_type="interject_receipt",
                    data={"interjection_id": message_id, "response": response}, adapter_version=GROK_ADAPTER_VERSION))
        if request.action in {ProviderControlAction.INTERRUPT, ProviderControlAction.CANCEL}:
            timeout = request.options.get("timeout_s")
            effective = float(timeout) if isinstance(timeout, (int, float)) else None
            return self._stop(request.action, effective, requested_at=request.requested_at)
        confirmed = self._done.is_set() and self._cleanup_confirmed
        return ProviderControlResult(
            action=request.action,
            accepted=False,
            terminal_confirmed=confirmed,
            resulting_state=self.poll_state() if confirmed else None,
            requested_at=request.requested_at,
            completed_at=utc_now_iso(),
            session_locator=self.session_locator(),
            turn_locator=self.turn_locator(),
            reason=(
                "Grok run ended without confirmed process-group cleanup"
                if self._done.is_set() and not self._cleanup_confirmed
                else "this Grok run control is unsupported; fork and compact use their dedicated SPI methods"
            ),
        )

    def close(self) -> None:
        if not self._done.is_set():
            self._stop(ProviderControlAction.CANCEL, 5.0)
        transport = self._transport
        if transport is not None and not transport.cleanup_confirmed:
            transport.close_process_group()

    def _stop(
        self,
        action: ProviderControlAction,
        timeout_s: float | None,
        *,
        requested_at: str | None = None,
    ) -> ProviderControlResult:
        requested = requested_at or utc_now_iso()
        if self._done.is_set():
            confirmed = self._cleanup_confirmed
            return ProviderControlResult(
                action=action,
                accepted=False,
                terminal_confirmed=confirmed,
                resulting_state=self.poll_state() if confirmed else None,
                requested_at=requested,
                completed_at=utc_now_iso(),
                session_locator=self.session_locator(),
                turn_locator=self.turn_locator(),
                reason=(
                    "Grok run is not active"
                    if confirmed
                    else "Grok run ended without confirmed process-group cleanup"
                ),
            )
        with self._lock:
            self._requested_stop = action
        transport = self._transport
        if transport is None:
            effective = timeout_s if timeout_s is not None else 10.0
            deadline = monotonic() + effective
            while not self._done.is_set() and self._transport is None and monotonic() < deadline:
                self._done.wait(0.02)
            confirmed = self._done.is_set() and self._cleanup_confirmed
            transport = self._transport
            if not confirmed and transport is not None:
                try:
                    session = self.session_locator()
                    if session is not None:
                        transport.notify("session/cancel", {"sessionId": session.session_id})
                    else:
                        transport.terminate_process_group()
                except BaseException:
                    pass
                remaining = max(deadline - monotonic(), 0.0)
                confirmed = self._done.wait(remaining) and self._cleanup_confirmed
                if not confirmed:
                    try:
                        transport.terminate_process_group()
                    except BaseException:
                        pass
                    confirmed = self._done.wait(5.0) and self._cleanup_confirmed
            return ProviderControlResult(
                action=action,
                accepted=True,
                terminal_confirmed=confirmed,
                resulting_state=self.poll_state() if confirmed else None,
                requested_at=requested,
                completed_at=utc_now_iso(),
                session_locator=self.session_locator(),
                turn_locator=self.turn_locator(),
                reason=None if confirmed else "Grok startup cancellation could not be confirmed",
            )
        session = self.session_locator()
        if session is not None:
            try:
                transport.notify("session/cancel", {"sessionId": session.session_id})
            except BaseException:
                pass
        effective = timeout_s if timeout_s is not None else 10.0
        confirmed = self._done.wait(effective) and self._cleanup_confirmed
        if not confirmed:
            try:
                transport.terminate_process_group()
            except BaseException:
                pass
            confirmed = self._done.wait(5.0) and self._cleanup_confirmed
        return ProviderControlResult(
            action=action,
            accepted=True,
            terminal_confirmed=confirmed,
            resulting_state=self.poll_state() if confirmed else None,
            requested_at=requested,
            completed_at=utc_now_iso(),
            session_locator=self.session_locator(),
            turn_locator=self.turn_locator(),
            reason=None if confirmed else "Grok process-group cleanup could not be confirmed",
        )

    def _run(self) -> None:
        transport: GrokAcpProcess | None = None
        try:
            context = self.request.execution_context
            if context is None or context.provider_type != "grok":
                raise ValueError("Grok runtime requires a Grok ProviderExecutionContext")
            runtime = context.runtime_payload
            if not isinstance(runtime, Mapping):
                raise ValueError("Grok execution context has no runtime configuration")
            workdir = Path(self.request.workdir or context.workdir or context.home_root).resolve(strict=True)
            self._raise_if_stop_requested()
            validate_grok_workdir(workdir, managed_skills=bool(runtime.get("skills")))
            _validate_resume_identity(self.request, context.home_root, workdir)
            command = build_grok_command(context, model=self.request.model_overrides)
            required_mcp = tuple(str(item) for item in runtime.get("required_mcp_server_names") or ())
            if required_mcp:
                _verify_required_mcp(
                    command[0],
                    required_mcp,
                    workdir,
                    context.process_environment,
                    cancelled=lambda: self._requested_stop is not None,
                )
            self._raise_if_stop_requested()
            transport = GrokAcpProcess(
                command,
                cwd=workdir,
                env=context.process_environment,
                incoming_request=self._incoming_request,
                on_record=self._native_record,
            )
            self._transport = transport
            with self._lock:
                self._state = ProviderRunState.RUNNING
            init = _mapping(
                transport.request(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {},
                        "clientInfo": {"name": "agent-runtime-kit", "version": "0.3.0"},
                    },
                    timeout_s=20,
                ),
                "initialize result",
            )
            if init.get("protocolVersion") != 1:
                raise GrokAcpError("Grok must negotiate ACP protocol version 1")
            meta = _mapping_or_empty(init.get("_meta"))
            if meta.get("agentVersion") != GROK_CLI_VERSION:
                raise GrokAcpError(f"Grok adapter requires agentVersion {GROK_CLI_VERSION}")
            self._raise_if_stop_requested()
            params: dict[str, object] = {"cwd": str(workdir), "mcpServers": []}
            if self.resume:
                capabilities = _mapping_or_empty(init.get("agentCapabilities"))
                if capabilities.get("loadSession") is not True:
                    raise GrokAcpError("Grok did not advertise session/load")
                assert self.request.session_locator is not None
                session_id = self.request.session_locator.session_id
                params["sessionId"] = session_id
                transport.request("session/load", params, timeout_s=30)
                created_at = self.request.session_locator.created_at
            else:
                created = _mapping(transport.request("session/new", params, timeout_s=30), "session/new result")
                session_id = str(created.get("sessionId") or "")
                created_at = self._started_at
            if not session_id:
                raise GrokAcpError("Grok did not return a session ID")
            native_ref = _session_relpath(context.home_root, workdir, session_id)
            backend = self.request.model_overrides or context.resolved_defaults
            self._session = ProviderSessionLocator(
                provider_type="grok",
                session_id=session_id,
                home_id=self.request.home_id,
                created_at=created_at,
                backend_identity=backend,
                native_locator={
                    "session_relpath": native_ref,
                    "workdir": str(workdir),
                    "grok_home": str(context.home_root / ".grok"),
                },
            )
            if not self.resume and self.request.session_start_home_commit is not None:
                self.request.session_start_home_commit()
            self._raise_if_stop_requested()
            catalog = _mapping(
                transport.request("_x.ai/commands/list", {"sessionId": session_id}, timeout_s=20),
                "commands/list result",
            )
            observed = catalog.get("tools")
            expected = tuple(str(item) for item in runtime.get("tools") or ())
            if (
                not isinstance(observed, list)
                or any(not isinstance(item, str) for item in observed)
                or len(observed) != len(set(observed))
                or set(observed) != set(expected)
            ):
                raise GrokAcpError(f"Grok tool profile mismatch: expected {list(expected)!r}, observed {observed!r}")
            with self._lock:
                self._events.clear()
                self._text_parts.clear()
                self._tool_calls.clear()
                self._accept_updates = True
            self._append_event("session.started", data={"session_id": session_id, "resumed": self.resume})
            self._append_event("tools.verified", data={"source": "_x.ai/commands/list", "tools": observed})
            self._raise_if_stop_requested()
            prompt = _compose_prompt(self.request)
            self._append_event("turn.accepted")
            result = _mapping(
                transport.request(
                    "session/prompt",
                    {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt}]},
                    timeout_s=(self.request.run_options.timeout_s or 3600) + 20,
                ),
                "session/prompt result",
            )
            stop_reason = str(result.get("stopReason") or "")
            if stop_reason not in _KNOWN_STOP_REASONS:
                raise GrokAcpError(f"unknown Grok ACP stopReason: {stop_reason}")
            response_meta = _mapping_or_empty(result.get("_meta"))
            request_id = _string_or_none(response_meta.get("requestId"))
            turn_id = _string_or_none(response_meta.get("promptId")) or request_id or f"turn-{uuid.uuid4().hex}"
            self._turn = ProviderTurnLocator(
                session=self._session,
                turn_id=turn_id,
                request_ids=(request_id,) if request_id else (),
            )
            transport.close_process_group()
            self._cleanup_confirmed = True
            state = _terminal_state(stop_reason, self._requested_stop)
            error = None
            if state is ProviderRunState.FAILED:
                error = AgentError(
                    error_type="grok_stopped",
                    code=stop_reason,
                    message=f"Grok stopped without completing the turn: {stop_reason}",
                )
            final = self._build_result(
                state=state,
                stop_reason=stop_reason,
                response_meta=response_meta,
                error=error,
                forced=False,
            )
            with self._lock:
                self._result = final
                self._state = state
            self._append_event("terminal." + state.value, terminal=True, data={"stop_reason": stop_reason})
        except BaseException as exc:
            if isinstance(exc, GrokProcessCleanupError):
                self._cleanup_uncertain = True
            cleanup_error: BaseException | None = None
            if transport is not None and not transport.cleanup_confirmed:
                try:
                    transport.close_process_group()
                    self._cleanup_confirmed = True
                except BaseException as close_exc:
                    cleanup_error = close_exc
                    self._cleanup_uncertain = True
            requested_stop = self._requested_stop
            if requested_stop is not None and self._session is not None and cleanup_error is None:
                state = (
                    ProviderRunState.INTERRUPTED
                    if requested_stop is ProviderControlAction.INTERRUPT
                    else ProviderRunState.CANCELLED
                )
                if self._turn is None:
                    self._turn = ProviderTurnLocator(session=self._session, turn_id=f"turn-{uuid.uuid4().hex}")
                final = self._build_result(
                    state=state,
                    stop_reason="unknown_after_forced_cleanup",
                    response_meta={},
                    error=None,
                    forced=True,
                )
                with self._lock:
                    self._result = final
                    self._state = state
                self._append_event("terminal." + state.value, terminal=True, data={"forced": True})
            else:
                self._error = cleanup_error or exc
                with self._lock:
                    self._state = ProviderRunState.FAILED
                self._append_event(
                    "terminal.failed",
                    terminal=True,
                    data={"error_type": type(self._error).__name__, "message": str(self._error)},
                )
        finally:
            self._prompt_active = False
            if transport is None and not self._cleanup_uncertain:
                self._cleanup_confirmed = True
            self._done.set()
            self._on_done(self)

    def _build_result(
        self,
        *,
        state: ProviderRunState,
        stop_reason: str,
        response_meta: Mapping[str, object],
        error: AgentError | None,
        forced: bool,
    ) -> ProviderTurnResult:
        assert self._session is not None
        usage, requests, model = _normalize_usage(response_meta, self._session, self._turn, stop_reason)
        if model is not None and self._session.backend_identity != model:
            self._session = replace(self._session, backend_identity=model)
            if self._turn is not None:
                self._turn = replace(self._turn, session=self._session)
        text = "".join(self._text_parts)
        blocks = (AgentContentBlock(kind="text", data=text, sequence=0),) if text else ()
        artifact_ref = None
        native = self._session.native_locator
        if isinstance(native, Mapping):
            artifact_ref = _string_or_none(native.get("session_relpath"))
        return ProviderTurnResult(
            provider_type="grok",
            run_id=self.run_id,
            session_locator=self._session,
            turn_locator=self._turn,
            status=state,
            started_at=self._started_at,
            completed_at=utc_now_iso(),
            duration_ms=(monotonic() - self._started_monotonic) * 1000,
            final_text=text or None,
            content_blocks=blocks,
            tool_calls=tuple(self._tool_calls.values()),
            request_usages=requests,
            turn_usage=usage,
            error=error,
            event_cursor=str(len(self._events)),
            artifact_locator=AgentArtifactLocator(
                provider_type="grok",
                home_id=self._session.home_id,
                session_id=self._session.session_id,
                adapter_version=GROK_ADAPTER_VERSION,
                native_primary_ref=artifact_ref,
            ),
            provider_payload=build_provider_payload(
                provider_type="grok",
                payload_type="prompt_response",
                data={
                    "native_stop_reason": None if forced else stop_reason,
                    "termination": "forced_process_group_cleanup" if forced else "native_terminal",
                    "_meta": dict(response_meta),
                },
                adapter_version=GROK_ADAPTER_VERSION,
                sdk_or_cli_version=GROK_CLI_VERSION,
            ),
        )

    def _raise_if_stop_requested(self) -> None:
        if self._requested_stop is not None:
            raise GrokAcpError("Grok run was cancelled before prompt submission")

    def _native_record(self, record: dict[str, object]) -> None:
        if isinstance(record.get("result"), dict) and "stopReason" in record["result"]:
            with self._lock:
                self._prompt_active = False
                self._prompt_finished = True
        if not self._accept_updates or record.get("method") not in {"session/update", "_x.ai/session/update"}:
            return
        params = _mapping_or_empty(record.get("params"))
        session = self.session_locator()
        if session is None or params.get("sessionId") != session.session_id:
            return
        update = _mapping_or_empty(params.get("update"))
        kind = str(update.get("sessionUpdate") or "unknown")
        if kind in {"user_message_chunk", "agent_message_chunk", "agent_thought_chunk", "tool_call"}:
            with self._lock:
                if not self._prompt_finished:
                    self._prompt_active = True
        if kind == "agent_message_chunk":
            content = _mapping_or_empty(update.get("content"))
            if content.get("type") == "text" and isinstance(content.get("text"), str):
                self._text_parts.append(str(content["text"]))
                self._append_event("text.delta", data={"text": str(content["text"])})
                return
        if kind in {"tool_call", "tool_call_update"}:
            self._record_tool(update, kind)
        self._append_event(kind.replace("_", "."), data=update, provider_record=record)

    def _record_tool(self, update: Mapping[str, object], update_kind: str) -> None:
        call_id = str(update.get("toolCallId") or update.get("id") or f"tool-{uuid.uuid4().hex}")
        existing = self._tool_calls.get(call_id)
        title = str(update.get("title") or (existing.tool_name if existing else "unknown"))
        status = str(update.get("status") or ("running" if update_kind == "tool_call" else "updated"))
        raw_input = update.get("rawInput") if "rawInput" in update else (existing.arguments if existing else None)
        raw_output = update.get("rawOutput") if "rawOutput" in update else (existing.result if existing else None)
        self._tool_calls[call_id] = AgentToolCall(
            call_id=call_id,
            tool_name=title,
            tool_kind=str(_mapping_or_empty(update.get("_meta")).get("x.ai/tool", {}).get("kind", "other"))
            if isinstance(_mapping_or_empty(update.get("_meta")).get("x.ai/tool"), Mapping)
            else "other",
            status=status,
            turn_id=self._turn.turn_id if self._turn is not None else None,
            display_name=title,
            arguments=raw_input,
            result=raw_output,
        )

    def _incoming_request(self, record: dict[str, object]) -> dict[str, object]:
        method = record.get("method")
        if method != "session/request_permission":
            self._append_event("interaction.unsupported", data={"method": method})
            raise GrokAcpError("Grok interactive client requests are unsupported")
        params = _mapping_or_empty(record.get("params"))
        session = self.session_locator()
        tool_call = _mapping_or_empty(params.get("toolCall"))
        kind = str(tool_call.get("kind") or "")
        allowed = bool(session is not None and params.get("sessionId") == session.session_id)
        runtime = self.request.execution_context.runtime_payload if self.request.execution_context else None
        tools = set(str(item) for item in runtime.get("tools", ())) if isinstance(runtime, Mapping) else set()
        mcp_servers = set(str(item) for item in runtime.get("mcp_server_names", ())) if isinstance(runtime, Mapping) else set()
        if kind in {"read", "search"}:
            allowed = allowed and bool(tools & {"read_file", "list_dir", "grep", "search_tool"})
        elif kind == "execute":
            allowed = allowed and "run_terminal_cmd" in tools
        elif kind == "edit":
            allowed = allowed and "search_replace" in tools
        elif kind == "other":
            raw = _mapping_or_empty(tool_call.get("rawInput"))
            target = str(raw.get("tool_name") or "")
            allowed = allowed and "use_tool" in tools and any(
                target.startswith(f"{server}__") for server in mcp_servers
            )
        else:
            allowed = False
        options = params.get("options")
        candidates = options if isinstance(options, list) else []
        wanted = "allow_once" if allowed else "reject_once"
        choice = next(
            (
                item
                for item in candidates
                if isinstance(item, Mapping) and item.get("kind") == wanted and item.get("optionId") is not None
            ),
            None,
        )
        selected = choice is not None
        self._append_event("permission.decision", data={"allowed": allowed and selected, "kind": kind})
        return (
            {"outcome": {"outcome": "selected", "optionId": choice["optionId"]}}
            if selected
            else {"outcome": {"outcome": "cancelled"}}
        )

    def _append_event(
        self,
        kind: str,
        *,
        terminal: bool = False,
        data: object | None = None,
        provider_record: object | None = None,
    ) -> None:
        with self._lock:
            event = AgentEvent(
                provider_type="grok",
                session_id=self._session.session_id if self._session is not None else None,
                turn_id=self._turn.turn_id if self._turn is not None else None,
                sequence=len(self._events),
                timestamp=utc_now_iso(),
                kind=kind,
                terminal=terminal,
                data=data,
                provider_payload=(
                    build_provider_payload(
                        provider_type="grok",
                        payload_type="acp_update",
                        data=provider_record,
                        adapter_version=GROK_ADAPTER_VERSION,
                        sdk_or_cli_version=GROK_CLI_VERSION,
                    )
                    if provider_record is not None
                    else None
                ),
            )
            self._events.append(event)
        if self.request.event_sink is not None:
            self.request.event_sink(event)


class GrokRuntimeAdapter:
    provider_type = "grok"

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)
        self._handles: dict[str, GrokProviderRunHandle] = {}
        self._unstable_sessions: set[str] = set()
        self._maintenance_sessions: set[str] = set()
        self._lock = threading.RLock()

    def start(self, request: ProviderRunRequest) -> GrokProviderRunHandle:
        return self._start(request, resume=False)

    def resume(self, request: ProviderRunRequest) -> GrokProviderRunHandle:
        if request.session_locator is None:
            raise ValueError("Grok resume requires session_locator")
        return self._start(request, resume=True)

    def _start(self, request: ProviderRunRequest, *, resume: bool) -> GrokProviderRunHandle:
        if request.provider_type != self.provider_type:
            raise ValueError("GrokRuntimeAdapter received a different provider_type")
        if request.run_options.max_turns is not None:
            raise ValueError("Grok adapter does not support run_options.max_turns")
        handle = GrokProviderRunHandle(request=request, resume=resume, on_done=self._on_done)
        with self._lock:
            if request.session_locator is not None and not self.is_session_stable(request.session_locator.session_id):
                raise RuntimeError("Grok session is active, under maintenance, or unclean")
            self._handles[handle.run_id] = handle
        handle.begin()
        return handle

    def fork(self, request: ProviderForkRequest) -> ProviderForkResult:
        from .grok_session import native_session, latest_prompt_id

        context = request.execution_context
        if context is None or request.target_home_id != request.source_session.home_id:
            raise ValueError("Grok fork requires the same Home execution context")
        source = request.source_session
        target_id = str(uuid.uuid4())
        with native_session(self, context, source) as (process, path):
            if request.source_turn is not None:
                if request.source_turn.session != source or request.source_turn.turn_id != latest_prompt_id(path):
                    raise ValueError("Grok supports only a fork at the latest persisted prompt")
            cwd = source.native_locator["workdir"]
            response = process.request("_x.ai/session/fork", {
                "sourceSessionId": source.session_id, "sourceCwd": cwd,
                "newCwd": cwd, "newSessionId": target_id,
            }, timeout_s=60)
            if response.get("newSessionId") != target_id or response.get("parentSessionId") != source.session_id:
                raise GrokAcpError("Grok fork returned an unexpected session identity")
        target_path = path.parent / target_id
        if not target_path.is_dir():
            raise GrokAcpError("Grok fork did not persist a child session")
        native = dict(source.native_locator)
        native["session_relpath"] = str(target_path.relative_to(self.runtime_root))
        target = replace(source, session_id=target_id, created_at=utc_now_iso(), native_locator=native)
        return ProviderForkResult(
            source_session=source, target_session=target, status="forked", source_turn=request.source_turn,
            target_turn=None,
            artifact_locator=AgentArtifactLocator(provider_type="grok", home_id=target.home_id,
                session_id=target_id, adapter_version=GROK_ADAPTER_VERSION, native_primary_ref=native["session_relpath"]),
            limitations=("same Home and cwd; latest prompt only; workspace is shared",),
        )

    @contextmanager
    def maintenance(self, session_id: str):  # noqa: ANN201
        with self._lock:
            if not self.is_session_stable(session_id):
                raise RuntimeError("Grok session is active, under maintenance, or unclean")
            self._maintenance_sessions.add(session_id)
        try:
            yield
        finally:
            with self._lock:
                self._maintenance_sessions.discard(session_id)

    def mark_unstable(self, session_id: str) -> None:
        with self._lock:
            self._unstable_sessions.add(session_id)

    @contextmanager
    def artifact_boundary(self, session_id: str):  # noqa: ANN201
        with self._lock:
            if not self.is_session_stable(session_id):
                raise RuntimeError("Grok artifact operation requires a clean idle session")
            yield

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        with self._lock:
            handle = self._handles.get(request.run_id or "")
        if handle is None:
            return ProviderControlResult(
                action=request.action,
                accepted=False,
                terminal_confirmed=False,
                requested_at=request.requested_at,
                completed_at=utc_now_iso(),
                reason="unknown or expired Grok run_id",
            )
        return handle.control(request)

    def close_session(self, locator: ProviderSessionLocator) -> ProviderControlResult:
        now = utc_now_iso()
        active = [
            item
            for item in self._active_handles()
            if (session := item.session_locator()) is not None and session.session_id == locator.session_id
        ]
        for handle in active:
            handle.control(
                ProviderControlRequest(
                    action=ProviderControlAction.CANCEL,
                    requested_at=now,
                    run_id=handle.run_id,
                )
            )
        stable = self.is_session_stable(locator.session_id)
        return ProviderControlResult(
            action=ProviderControlAction.ARCHIVE_SESSION,
            accepted=bool(active),
            terminal_confirmed=stable,
            requested_at=now,
            completed_at=utc_now_iso(),
            session_locator=locator,
            reason=(
                "Grok session process-group cleanup is unconfirmed"
                if not stable
                else None if active else "Grok session had no active process; artifact was retained"
            ),
        )

    def is_session_active(self, session_id: str) -> bool:
        return any(
            (session := item.session_locator()) is not None
            and session.session_id == session_id
            and not item.poll_state().terminal
            for item in self._active_handles()
        )

    def is_session_stable(self, session_id: str) -> bool:
        with self._lock:
            return (not self.is_session_active(session_id)
                    and session_id not in self._unstable_sessions
                    and session_id not in self._maintenance_sessions)

    def close(self) -> None:
        for handle in self._active_handles():
            handle.close()

    def _active_handles(self) -> tuple[GrokProviderRunHandle, ...]:
        with self._lock:
            return tuple(self._handles.values())

    def _on_done(self, handle: GrokProviderRunHandle) -> None:
        session = handle.session_locator()
        with self._lock:
            self._handles.pop(handle.run_id, None)
            if session is not None and not handle.cleanup_confirmed:
                self._unstable_sessions.add(session.session_id)


def build_grok_command(context, *, model: ModelBackendIdentity | None = None) -> list[str]:  # noqa: ANN001
    runtime = context.runtime_payload
    if not isinstance(runtime, Mapping):
        raise ValueError("Grok execution context has no runtime configuration")
    command = [
        str(runtime["binary_path"]),
        "agent",
        "--no-leader",
        "--agent-profile",
        str(context.home_root / str(runtime["profile_relpath"])),
    ]
    selected_model = model.effective_model if model is not None else runtime.get("model")
    reasoning = model.reasoning_effort if model is not None else runtime.get("reasoning_effort")
    if selected_model:
        command.extend(["--model", str(selected_model)])
    if reasoning:
        command.extend(["--reasoning-effort", str(reasoning)])
    command.append("stdio")
    return command


def _validate_resume_identity(request: ProviderRunRequest, home_root: Path, workdir: Path) -> None:
    locator = request.session_locator
    if locator is None:
        return
    native = locator.native_locator
    if not isinstance(native, Mapping):
        raise ValueError("Grok session locator has no native identity")
    if native.get("workdir") != str(workdir):
        raise ValueError("Grok session belongs to a different workdir")
    if native.get("grok_home") != str(home_root / ".grok"):
        raise ValueError("Grok session belongs to a different managed Home")
    expected = _session_relpath(home_root, workdir, locator.session_id)
    if native.get("session_relpath") != expected:
        raise ValueError("Grok session artifact locator does not match Home/workdir/session identity")


def _session_relpath(home_root: Path, workdir: Path, session_id: str) -> str:
    session = home_root / ".grok" / "sessions" / quote(str(workdir), safe="") / session_id
    return str(session.relative_to(home_root.parents[2]))


def _verify_required_mcp(
    binary: str,
    required: tuple[str, ...],
    workdir: Path,
    env: Mapping[str, str],
    *,
    cancelled=None,  # noqa: ANN001
) -> None:
    process = subprocess.Popen(
        [binary, "mcp", "doctor", "--json"],
        cwd=str(workdir),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    pgid = process.pid
    deadline = monotonic() + 60
    try:
        while True:
            if callable(cancelled) and cancelled():
                _terminate_group(process, pgid)
                raise GrokAcpError("Grok run was cancelled during MCP verification")
            try:
                stdout, _ = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if monotonic() >= deadline:
                    _terminate_group(process, pgid)
                    raise RuntimeError("Grok MCP doctor timed out")
    finally:
        if _group_exists(pgid):
            _terminate_group(process, pgid)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Grok MCP doctor returned invalid JSON") from exc
    result = payload.get("result") if isinstance(payload, Mapping) else None
    source = result if isinstance(result, Mapping) else payload
    servers = source.get("servers") if isinstance(source, Mapping) else None
    by_name = {
        str(item.get("name")): item
        for item in servers or ()
        if isinstance(item, Mapping) and item.get("name") is not None
    }
    failed = [name for name in required if name not in by_name or by_name[name].get("healthy") is not True]
    if failed:
        raise RuntimeError(f"required Grok MCP servers are unhealthy or missing: {', '.join(failed)}")


def _terminate_group(process: subprocess.Popen[str], pgid: int) -> None:
    for sig, timeout in ((signal.SIGTERM, 2.0), (signal.SIGKILL, 3.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = monotonic() + timeout
        while _group_exists(pgid) and monotonic() < deadline:
            try:
                process.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                pass
            sleep(0.02)
        if not _group_exists(pgid):
            return
    raise GrokProcessCleanupError("Grok MCP doctor process group could not be cleaned up")


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _compose_prompt(request: ProviderRunRequest) -> str:
    parts: list[str] = []
    if request.system_instructions:
        parts.append(f"<system-instructions>\n{request.system_instructions}\n</system-instructions>")
    if request.developer_instructions:
        parts.append(f"<developer-instructions>\n{request.developer_instructions}\n</developer-instructions>")
    parts.append(request.prompt)
    return "\n\n".join(parts)


def _terminal_state(stop_reason: str, requested: ProviderControlAction | None) -> ProviderRunState:
    if stop_reason == "end_turn":
        return ProviderRunState.COMPLETED
    if stop_reason == "cancelled":
        return (
            ProviderRunState.INTERRUPTED
            if requested is ProviderControlAction.INTERRUPT
            else ProviderRunState.CANCELLED
        )
    return ProviderRunState.FAILED


def _normalize_usage(
    meta: Mapping[str, object],
    session: ProviderSessionLocator,
    turn: ProviderTurnLocator | None,
    stop_reason: str,
) -> tuple[AgentTurnUsage | None, tuple[()], ModelBackendIdentity | None]:
    raw = meta.get("usage")
    model_id = _string_or_none(meta.get("modelId"))
    base = session.backend_identity
    model = ModelBackendIdentity(
        api_provider="xai",
        api_mode="other",
        requested_model=base.requested_model if base is not None else model_id,
        resolved_model=model_id,
        reasoning_effort=base.reasoning_effort if base is not None else None,
    ) if model_id or base is None else base
    if not isinstance(raw, Mapping):
        return None, (), model
    tokens = TokenUsage(
        input_tokens=_nonnegative_int(raw.get("inputTokens")),
        output_tokens=_nonnegative_int(raw.get("outputTokens")),
        total_tokens=_nonnegative_int(raw.get("totalTokens")),
        cache_read_input_tokens=_nonnegative_int(raw.get("cachedReadTokens")),
        cache_creation_input_tokens=_nonnegative_int(raw.get("cacheCreationTokens")),
        reasoning_output_tokens=_nonnegative_int(raw.get("reasoningTokens")),
    )
    assert model is not None
    usage = AgentTurnUsage(
        request_count=_nonnegative_int(raw.get("modelCalls")),
        requests=(),
        token_usage=tokens,
        models_used=(model,),
        aggregate_complete=all(
            value is not None for value in (tokens.input_tokens, tokens.output_tokens, tokens.total_tokens)
        ),
    )
    return usage, (), model


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise GrokAcpError(f"Grok {label} must be an object")
    return dict(value)


def _mapping_or_empty(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _nonnegative_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) and value >= 0 else None
