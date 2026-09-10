from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from time import monotonic
from typing import Mapping

from ..provider_contracts import (
    AgentArtifactLocator,
    AgentEvent,
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
)
from ..store_utils import utc_now_iso
from .claude_code import ClaudeCodeProvider
from .claude_code_normalization import (
    find_session_file,
    latest_turn_id,
    normalize_stream_result,
)


class ClaudeCodeRunHandle:
    def __init__(
        self,
        provider: ClaudeCodeProvider,
        request: ProviderRunRequest,
        *,
        resume: bool,
    ) -> None:
        self.provider = provider
        self.request = request
        self.resume = resume
        self._run_id = f"r_{uuid.uuid4().hex}"
        self._started_at = utc_now_iso()
        self._started_monotonic = monotonic()
        session_id = (
            request.session_locator.session_id
            if request.session_locator is not None
            else str(uuid.uuid4())
        )
        self._session = (
            replace(
                request.session_locator,
                backend_identity=(
                    request.session_locator.backend_identity or request.model_overrides
                ),
            )
            if request.session_locator is not None
            else ProviderSessionLocator(
                provider_type="claude_code",
                session_id=session_id,
                home_id=request.home_id,
                created_at=self._started_at,
                backend_identity=request.model_overrides,
            )
        )
        self._turn: ProviderTurnLocator | None = None
        self._state = ProviderRunState.STARTING
        self._result: ProviderTurnResult | None = None
        self._error: BaseException | None = None
        self._events: list[AgentEvent] = []
        self._messages: list[object] = []
        self._terminal: object | None = None
        self._done = threading.Event()
        self._client_ready = threading.Event()
        self._lock = threading.RLock()
        self._sink_publish_lock = threading.Lock()
        self._sink_enabled = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: object | None = None
        self._interrupt_requested = False
        self._closed = False
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def run_id(self) -> str:
        return self._run_id

    def session_locator(self) -> ProviderSessionLocator:
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
            return ProviderEventBatch(
                events=tuple(self._events[start:]),
                next_cursor=str(len(self._events)),
                terminal=self._state.terminal,
            )

    def wait_terminal(self, timeout_s: float | None = None) -> ProviderTurnResult:
        # AgentService persists the early session locator immediately before it
        # calls wait_terminal(). Enabling the observational event sink here
        # prevents the sink's locator update from racing that first atomic
        # Agent-record write while still streaming all buffered/live events.
        self._enable_event_sink()
        if not self._done.wait(timeout_s):
            raise TimeoutError(self.run_id)
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("Claude run completed without a normalized result")
        return self._result

    def interrupt(self, timeout_s: float | None = None) -> ProviderControlResult:
        requested_at = utc_now_iso()
        if self._done.is_set():
            return ProviderControlResult(
                action=ProviderControlAction.INTERRUPT,
                accepted=False,
                terminal_confirmed=True,
                resulting_state=self.poll_state(),
                requested_at=requested_at,
                completed_at=utc_now_iso(),
                session_locator=self._session,
                turn_locator=self.turn_locator(),
                reason="Claude run is already terminal",
            )
        deadline = None if timeout_s is None else monotonic() + timeout_s
        remaining = None if deadline is None else max(0.0, deadline - monotonic())
        if not self._client_ready.wait(remaining):
            return ProviderControlResult(
                action=ProviderControlAction.INTERRUPT,
                accepted=False,
                terminal_confirmed=False,
                requested_at=requested_at,
                completed_at=utc_now_iso(),
                session_locator=self._session,
                reason="Claude client did not become ready before timeout",
            )
        loop = self._loop
        client = self._client
        if loop is None or client is None or self._done.is_set():
            return ProviderControlResult(
                action=ProviderControlAction.INTERRUPT,
                accepted=False,
                terminal_confirmed=self._done.is_set(),
                resulting_state=self.poll_state() if self._done.is_set() else None,
                requested_at=requested_at,
                completed_at=utc_now_iso(),
                session_locator=self._session,
                reason="Claude client is no longer active",
            )
        self._interrupt_requested = True
        with self._lock:
            self._state = ProviderRunState.RUNNING
            self._append_event("interrupt.requested", data=None)
        try:
            future = asyncio.run_coroutine_threadsafe(client.interrupt(), loop)
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            future.result(timeout=remaining)
        except BaseException as exc:
            self._interrupt_requested = False
            return ProviderControlResult(
                action=ProviderControlAction.INTERRUPT,
                accepted=False,
                terminal_confirmed=self._done.is_set(),
                resulting_state=self.poll_state() if self._done.is_set() else None,
                requested_at=requested_at,
                completed_at=utc_now_iso(),
                session_locator=self._session,
                reason=f"Claude interrupt request failed: {type(exc).__name__}",
            )
        remaining = None if deadline is None else max(0.0, deadline - monotonic())
        confirmed = self._done.wait(remaining)
        return ProviderControlResult(
            action=ProviderControlAction.INTERRUPT,
            accepted=True,
            terminal_confirmed=confirmed,
            resulting_state=self.poll_state() if confirmed else None,
            requested_at=requested_at,
            completed_at=utc_now_iso(),
            session_locator=self._session,
            turn_locator=self.turn_locator(),
            reason=None if confirmed else "Claude interrupt was sent but terminal Result was not observed",
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.action is ProviderControlAction.INTERRUPT:
            raw = request.options.get("timeout_s")
            return self.interrupt(float(raw) if isinstance(raw, (int, float)) else None)
        return ProviderControlResult(
            action=request.action,
            accepted=False,
            terminal_confirmed=self._done.is_set(),
            resulting_state=self.poll_state() if self._done.is_set() else None,
            requested_at=request.requested_at,
            completed_at=utc_now_iso(),
            session_locator=self._session,
            turn_locator=self.turn_locator(),
            reason="Claude Code v1 adapter does not implement this control action",
        )

    def close(self) -> None:
        self._closed = True
        if not self._done.is_set():
            self.interrupt(timeout_s=5)

    def _run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except BaseException as exc:
            self._error = exc
            with self._lock:
                self._state = ProviderRunState.FAILED
                self._append_event(
                    "terminal.failed",
                    terminal=True,
                    data={"error_type": type(exc).__name__},
                )
        finally:
            self._client_ready.set()
            self._done.set()

    async def _run_async(self) -> None:
        sdk = self.provider.sdk()
        context = self.request.execution_context
        if context is None or context.provider_type != "claude_code":
            raise ValueError("Claude runtime requires a Claude ProviderExecutionContext")
        options = _build_options(
            sdk,
            self.request,
            session_id=self._session.session_id,
            resume=self.resume,
        )
        client = sdk.ClaudeSDKClient(options=options)
        self._loop = asyncio.get_running_loop()
        self._client = client
        try:
            await client.connect()
            with self._lock:
                self._state = ProviderRunState.RUNNING
                self._append_event("session.started" if not self.resume else "session.resumed")
            self._client_ready.set()
            await client.query(self.request.prompt)
            async for message in client.receive_response():
                self._messages.append(message)
                terminal = type(message).__name__ == "ResultMessage"
                self._append_native_event(message, terminal=terminal)
                if terminal:
                    self._terminal = message
            if self._terminal is None:
                raise RuntimeError("Claude response stream ended without ResultMessage")
            completed_at = utc_now_iso()
            turn_id = latest_turn_id(context.home_root, self._session.session_id)
            turn_id = turn_id or str(getattr(self._terminal, "uuid", "") or f"turn-{self.run_id}")
            self._turn = ProviderTurnLocator(session=self._session, turn_id=turn_id)
            result = normalize_stream_result(
                run_id=self.run_id,
                session=self._session,
                turn_id=turn_id,
                messages=self._messages,
                terminal=self._terminal,
                started_at=self._started_at,
                completed_at=completed_at,
                duration_ms=(monotonic() - self._started_monotonic) * 1000,
                interrupted=self._interrupt_requested,
            )
            artifact = _artifact_locator(context.home_root, self._session)
            self._result = replace(result, artifact_locator=artifact)
            with self._lock:
                self._state = self._result.status
        finally:
            try:
                await client.disconnect()
            finally:
                self._client = None

    def _append_native_event(self, message: object, *, terminal: bool) -> None:
        name = type(message).__name__
        kind = {
            "AssistantMessage": "content.completed",
            "UserMessage": "tool.result",
            "SystemMessage": "system.message",
            "ResultMessage": "terminal.result",
            "StreamEvent": "content.delta",
        }.get(name, "provider.message")
        self._append_event(
            kind,
            terminal=terminal,
            data={
                "native_type": name,
                "subtype": getattr(message, "subtype", None),
                "is_error": getattr(message, "is_error", None),
            },
        )

    def _append_event(
        self,
        kind: str,
        *,
        terminal: bool = False,
        data: object | None = None,
    ) -> None:
        with self._lock:
            event = AgentEvent(
                provider_type="claude_code",
                session_id=self._session.session_id,
                turn_id=self._turn.turn_id if self._turn is not None else None,
                sequence=len(self._events),
                timestamp=utc_now_iso(),
                kind=kind,
                terminal=terminal,
                data=data,
            )
            self._events.append(event)
            publish = self._sink_enabled
        if publish:
            self._publish_event(event)

    def _enable_event_sink(self) -> None:
        if self.request.event_sink is None:
            return
        with self._sink_publish_lock:
            with self._lock:
                if self._sink_enabled:
                    return
                pending = tuple(self._events)
                self._sink_enabled = True
            for event in pending:
                self._publish_event_unlocked(event)

    def _publish_event(self, event: AgentEvent) -> None:
        with self._sink_publish_lock:
            self._publish_event_unlocked(event)

    def _publish_event_unlocked(self, event: AgentEvent) -> None:
        sink = self.request.event_sink
        if sink is None:
            return
        try:
            sink(event)
        except Exception:
            # Event delivery is observational. A sink failure must not convert
            # a provider-completed turn into a provider failure; callers can
            # still drain the handle-local event buffer.
            pass


class ClaudeCodeRuntimeAdapter:
    provider_type = "claude_code"

    def __init__(self, provider: ClaudeCodeProvider) -> None:
        self.provider = provider
        self._handles: dict[str, ClaudeCodeRunHandle] = {}
        self._lock = threading.RLock()

    def start(self, request: ProviderRunRequest) -> ClaudeCodeRunHandle:
        return self._start(request, resume=False)

    def resume(self, request: ProviderRunRequest) -> ClaudeCodeRunHandle:
        if request.session_locator is None:
            raise ValueError("Claude resume requires session_locator")
        return self._start(request, resume=True)

    def _start(self, request: ProviderRunRequest, *, resume: bool) -> ClaudeCodeRunHandle:
        if request.provider_type != self.provider_type:
            raise ValueError("Claude runtime received a different provider_type")
        handle = ClaudeCodeRunHandle(self.provider, request, resume=resume)
        with self._lock:
            self._handles[handle.run_id] = handle
        return handle

    def fork(self, request: ProviderForkRequest) -> ProviderForkResult:
        context = request.execution_context
        if context is None or context.provider_type != self.provider_type:
            raise ValueError("Claude fork requires a Claude ProviderExecutionContext")
        if request.target_home_id != request.source_session.home_id:
            raise ValueError("Claude v1 fork requires the source and target to use the same Home")
        source_turn = request.source_turn.turn_id if request.source_turn is not None else None
        if source_turn is not None:
            try:
                source_turn = str(uuid.UUID(source_turn))
            except ValueError:
                source_turn = None
        command = [
            sys.executable,
            "-c",
            (
                "import json,sys; from claude_agent_sdk import fork_session; "
                "r=fork_session(sys.argv[1], up_to_message_id=sys.argv[2] or None, "
                "title='ARK session fork'); print(json.dumps({'session_id': r.session_id}))"
            ),
            request.source_session.session_id,
            source_turn or "",
        ]
        completed = subprocess.run(
            command,
            env=dict(context.process_environment),
            cwd=context.workdir or None,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        try:
            payload = json.loads(completed.stdout)
            session_id = str(payload["session_id"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("Claude fork helper returned invalid JSON") from exc
        target = ProviderSessionLocator(
            provider_type=self.provider_type,
            session_id=session_id,
            home_id=request.target_home_id,
            created_at=utc_now_iso(),
            backend_identity=request.source_session.backend_identity,
        )
        path = find_session_file(context.home_root, session_id)
        if path is None:
            raise RuntimeError("Claude fork did not create a target transcript")
        artifact = AgentArtifactLocator(
            provider_type=self.provider_type,
            home_id=request.target_home_id,
            session_id=session_id,
            adapter_version="1",
            native_primary_ref=str(path.relative_to(context.home_root / ".claude")),
        )
        return ProviderForkResult(
            source_session=request.source_session,
            target_session=target,
            status="created",
            source_turn=request.source_turn,
            fork_mode="session_only",
            workspace_isolated=False,
            artifact_locator=artifact,
            limitations=(
                "Claude fork copies transcript history only",
                "workspace files and Claude undo/file-history checkpoints are not copied",
            ),
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        handle = self._handles.get(request.run_id or "")
        if handle is not None:
            return handle.control(request)
        return ProviderControlResult(
            action=request.action,
            accepted=False,
            terminal_confirmed=False,
            requested_at=request.requested_at,
            completed_at=utc_now_iso(),
            reason="unknown or expired Claude run_id",
        )

    def close_session(self, locator: ProviderSessionLocator) -> ProviderControlResult:
        now = utc_now_iso()
        return ProviderControlResult(
            action=ProviderControlAction.ARCHIVE_SESSION,
            accepted=False,
            terminal_confirmed=True,
            requested_at=now,
            completed_at=now,
            session_locator=locator,
            reason="Claude session archival is not implemented by the v1 adapter",
        )

    def close(self) -> None:
        with self._lock:
            handles = tuple(self._handles.values())
        for handle in handles:
            handle.close()
        self.provider.close()


def _build_options(
    sdk: object,
    request: ProviderRunRequest,
    *,
    session_id: str,
    resume: bool,
) -> object:
    context = request.execution_context
    assert context is not None
    config = context.runtime_payload
    if not isinstance(config, Mapping):
        raise RuntimeError("Claude execution context has no runtime config")
    home_prompt = config.get("system_prompt")
    prompt_parts = [
        str(value).strip()
        for value in (home_prompt, request.system_instructions, request.developer_instructions)
        if isinstance(value, str) and value.strip()
    ]
    settings_path = context.home_root / ".claude" / "settings.json"
    model = (
        request.model_overrides.requested_model
        if request.model_overrides is not None and request.model_overrides.requested_model
        else config.get("model")
    )
    max_turns = request.run_options.max_turns or config.get("max_turns")
    kwargs = {
        "cwd": request.workdir or context.workdir or str(context.home_root),
        "cli_path": config.get("cli_path"),
        "env": dict(context.process_environment),
        "settings": str(settings_path) if settings_path.is_file() else None,
        "setting_sources": config.get("setting_sources"),
        "system_prompt": "\n\n".join(prompt_parts) or None,
        "tools": config.get("tools"),
        "allowed_tools": list(config.get("allowed_tools") or []),
        "disallowed_tools": list(config.get("disallowed_tools") or []),
        "permission_mode": config.get("permission_mode"),
        "mcp_servers": dict(config.get("mcp_servers_resolved") or {}),
        "strict_mcp_config": bool(config.get("strict_mcp_config", True)),
        "skills": config.get("skills"),
        "model": model,
        "fallback_model": config.get("fallback_model"),
        "thinking": config.get("thinking"),
        "effort": config.get("effort"),
        "max_turns": int(max_turns) if max_turns is not None else None,
        "max_budget_usd": config.get("max_budget_usd"),
        "add_dirs": list(config.get("add_dirs") or []),
        "extra_args": dict(config.get("extra_args") or {}),
        "enable_file_checkpointing": False,
        "resume": session_id if resume else None,
        "session_id": None if resume else session_id,
    }
    return sdk.ClaudeAgentOptions(**kwargs)


def _artifact_locator(
    home_root: Path,
    session: ProviderSessionLocator,
) -> AgentArtifactLocator | None:
    path = find_session_file(home_root, session.session_id)
    if path is None:
        return None
    return AgentArtifactLocator(
        provider_type="claude_code",
        home_id=session.home_id,
        session_id=session.session_id,
        adapter_version="1",
        native_primary_ref=str(path.relative_to(home_root / ".claude")),
    )
