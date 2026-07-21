from __future__ import annotations

import os
import secrets
import shutil
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..provider_contracts import (
    AgentArtifactLocator,
    AgentError,
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
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .opencode_client import OpenCodeClient, OpenCodeClientError, event_properties
from .opencode_models import (
    ADAPTER_VERSION,
    OpenCodeNativeLocator,
    OpenCodeRunOptions,
    PROVIDER_TYPE,
    SUPPORTED_CLI_VERSION,
    parse_native_locator,
)
from .opencode_query import completed_turn_result


@dataclass
class OpenCodeServer:
    agent_id: str
    runtime_root: Path
    directory: str
    database_path: Path
    process: subprocess.Popen[str]
    client: OpenCodeClient
    password: str

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


class OpenCodeRuntimeRegistry:
    def __init__(self, runtime_root: Path, *, binary_path: str | Path = "opencode") -> None:
        self.runtime_root = Path(runtime_root)
        self.binary_path = str(binary_path)
        self._servers: dict[str, OpenCodeServer] = {}
        self._lock = threading.RLock()

    def ensure(self, request: ProviderRunRequest) -> OpenCodeServer:
        with self._lock:
            existing = self._servers.get(request.agent_id)
            if existing is not None and existing.process.poll() is None:
                if request.workdir and Path(existing.directory).resolve() != Path(request.workdir).resolve():
                    raise ValueError("OpenCode Agent runtime cannot change workdir while its server is active")
                return existing
            server = self._start_server(request)
            self._servers[request.agent_id] = server
            return server

    def client_for_locator(self, locator: ProviderSessionLocator) -> OpenCodeClient:
        native = parse_native_locator(locator.native_locator)
        with self._lock:
            server = self._servers.get(native.agent_id)
        if server is None or server.process.poll() is not None:
            raise RuntimeError(f"OpenCode server is not active for agent {native.agent_id}")
        if str(server.database_path) != native.database_path:
            raise RuntimeError("OpenCode locator database does not match active Agent runtime")
        return server.client

    def server_for_agent(self, agent_id: str) -> OpenCodeServer | None:
        with self._lock:
            return self._servers.get(agent_id)

    def close_agent(self, agent_id: str) -> None:
        with self._lock:
            server = self._servers.pop(agent_id, None)
        if server is not None:
            server.close()

    def close(self) -> None:
        with self._lock:
            servers = tuple(self._servers.values())
            self._servers.clear()
        for server in servers:
            server.close()

    def _start_server(self, request: ProviderRunRequest) -> OpenCodeServer:
        context = request.execution_context
        if context is None or context.provider_type != PROVIDER_TYPE:
            raise ValueError("OpenCode runtime requires an OpenCode ProviderExecutionContext")
        runtime = self.runtime_root / "providers" / PROVIDER_TYPE / "agents" / request.agent_id
        paths = {
            "home": runtime / "home",
            "config": runtime / "xdg-config",
            "data": runtime / "xdg-data",
            "cache": runtime / "xdg-cache",
            "state": runtime / "xdg-state",
            "tmp": runtime / "tmp",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        database = runtime / "opencode.db"
        directory = str(Path(request.workdir or context.workdir or os.getcwd()).resolve())
        password = secrets.token_urlsafe(32)
        port = _free_port()
        env = dict(context.process_environment)
        env.update(request.environment)
        env.pop("OPENCODE_CONFIG", None)
        env.pop("OPENCODE_CONFIG_CONTENT", None)
        env["OPENCODE_PURE"] = "1"
        if bool(context.runtime_payload.get("allow_project_config", False)):
            env.pop("OPENCODE_DISABLE_PROJECT_CONFIG", None)
        else:
            env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
        env.update(
            {
                "HOME": str(paths["home"]),
                "XDG_CONFIG_HOME": str(paths["config"]),
                "XDG_DATA_HOME": str(paths["data"]),
                "XDG_CACHE_HOME": str(paths["cache"]),
                "XDG_STATE_HOME": str(paths["state"]),
                "TMPDIR": str(paths["tmp"]),
                "OPENCODE_DB": str(database),
                "OPENCODE_SERVER_PASSWORD": password,
                "OPENCODE_CONFIG_DIR": str(context.home_root),
            }
        )
        binary = env.pop("ARK_OPENCODE_BINARY", self.binary_path)
        start_timeout_s = float(env.pop("ARK_OPENCODE_SERVER_START_TIMEOUT", "15"))
        process = subprocess.Popen(
            [binary, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
            cwd=directory,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        client = OpenCodeClient(
            f"http://127.0.0.1:{port}", password=password, directory=directory, timeout_s=5
        )
        deadline = time.monotonic() + start_timeout_s
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"OpenCode serve exited during startup with code {process.returncode}"
                )
            try:
                health = client.health()
                if health.get("healthy") is not True:
                    raise OpenCodeClientError("OpenCode health response did not report healthy=true")
                version = str(health.get("version") or "")
                if version != SUPPORTED_CLI_VERSION:
                    process.terminate()
                    process.wait(timeout=5)
                    raise RuntimeError(
                        f"unsupported OpenCode version: {version or 'unknown'}; "
                        f"expected {SUPPORTED_CLI_VERSION}"
                    )
                break
            except OpenCodeClientError as exc:
                last_error = exc
                time.sleep(0.05)
        else:
            process.terminate()
            raise TimeoutError(f"OpenCode serve health timeout: {last_error}")
        return OpenCodeServer(
            agent_id=request.agent_id,
            runtime_root=runtime,
            directory=directory,
            database_path=database,
            process=process,
            client=client,
            password=password,
        )


class OpenCodeProviderRunHandle:
    def __init__(self, registry: OpenCodeRuntimeRegistry, request: ProviderRunRequest, *, resume: bool) -> None:
        self.registry = registry
        self.request = request
        self.resume = resume
        self._run_id = f"r_{uuid.uuid4().hex}"
        self._started_at = utc_now_iso()
        self._state = ProviderRunState.STARTING
        self._session = request.session_locator
        self._turn: ProviderTurnLocator | None = None
        self._events: list[AgentEvent] = []
        self._result: ProviderTurnResult | None = None
        self._error: BaseException | None = None
        self._lock = threading.RLock()
        self._done = threading.Event()
        self._stop_sse = threading.Event()
        self._connected = threading.Event()
        self._turn_seen = threading.Event()
        self._armed = threading.Event()
        self._pending: tuple[str, str] | None = None
        self._provider_error: AgentError | None = None
        self._interrupt_requested = threading.Event()
        self._interaction_resolved = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    @property
    def run_id(self) -> str:
        return self._run_id

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
        start = int(after_cursor or 0)
        with self._lock:
            return ProviderEventBatch(
                events=tuple(self._events[start:]),
                next_cursor=str(len(self._events)),
                terminal=self._state.terminal,
            )

    def wait_terminal(self, timeout_s: float | None = None) -> ProviderTurnResult:
        if not self._done.wait(timeout_s):
            raise TimeoutError(self.run_id)
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result

    def interrupt(self, timeout_s: float | None = None) -> ProviderControlResult:
        requested = utc_now_iso()
        session = self.session_locator()
        if session is None:
            return _control_result(ProviderControlAction.INTERRUPT, requested, False, self, "session not created")
        if self.poll_state().terminal:
            return _control_result(ProviderControlAction.INTERRUPT, requested, False, self, "already terminal")
        server = self.registry.server_for_agent(self.request.agent_id)
        if server is None:
            return _control_result(ProviderControlAction.INTERRUPT, requested, False, self, "server not active")
        server.client.abort(session.session_id)
        self._interrupt_requested.set()
        confirmed = self._done.wait(timeout_s)
        return ProviderControlResult(
            action=ProviderControlAction.INTERRUPT,
            accepted=True,
            terminal_confirmed=confirmed,
            requested_at=requested,
            completed_at=utc_now_iso(),
            resulting_state=self.poll_state() if confirmed else None,
            session_locator=self.session_locator(),
            turn_locator=self.turn_locator(),
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.action is ProviderControlAction.INTERRUPT:
            timeout = request.options.get("timeout_s")
            return self.interrupt(float(timeout) if isinstance(timeout, (int, float)) else None)
        pending = self._pending
        server = self.registry.server_for_agent(self.request.agent_id)
        if pending is None or server is None:
            return _control_result(request.action, request.requested_at, False, self, "no pending interaction")
        kind, interaction_id = pending
        if kind == "permission" and request.action is ProviderControlAction.RESPOND_APPROVAL:
            reply = str(request.content or request.options.get("response") or "once")
            if reply not in {"once", "always", "reject"}:
                return _control_result(request.action, request.requested_at, False, self, "invalid permission response")
            payload: dict[str, object] = {"reply": reply}
            if request.options.get("message") is not None:
                payload["message"] = request.options["message"]
            server.client.reply_permission(interaction_id, payload)
        elif kind == "question" and request.action is ProviderControlAction.RESPOND_INPUT:
            answers = request.content
            if not isinstance(answers, (list, tuple)):
                return _control_result(request.action, request.requested_at, False, self, "question answers must be a sequence")
            server.client.reply_question(interaction_id, {"answers": answers})
        elif kind == "question" and request.action is ProviderControlAction.REJECT_INPUT:
            server.client.reject_question(interaction_id)
        else:
            return _control_result(request.action, request.requested_at, False, self, "control action does not match pending interaction")
        self._pending = None
        with self._lock:
            self._state = ProviderRunState.RUNNING
            self._result = None
            self._done.clear()
        self._interaction_resolved.set()
        return _control_result(request.action, request.requested_at, True, self, None)

    def close(self) -> None:
        self._stop_sse.set()

    def _run(self) -> None:
        try:
            server = self.registry.ensure(self.request)
            client = server.client
            if self.resume:
                if self._session is None:
                    raise ValueError("OpenCode resume requires session_locator")
                client.get_session(self._session.session_id)
            else:
                session = client.create_session()
                session_id = str(session.get("id") or "")
                if not session_id:
                    raise RuntimeError("OpenCode created a session without id")
                native = OpenCodeNativeLocator(
                    agent_id=self.request.agent_id,
                    directory=server.directory,
                    database_path=str(server.database_path),
                    runtime_relpath=str(server.runtime_root.relative_to(self.registry.runtime_root)),
                )
                self._session = ProviderSessionLocator(
                    provider_type=PROVIDER_TYPE,
                    session_id=session_id,
                    home_id=self.request.home_id,
                    created_at=utc_now_iso(),
                    backend_identity=self.request.model_overrides or self.request.execution_context.resolved_defaults,
                    native_locator=native.as_dict(),
                )
            assert self._session is not None
            turn_id = _message_id()
            self._turn = ProviderTurnLocator(session=self._session, turn_id=turn_id)
            sse_thread = threading.Thread(target=self._consume_sse, args=(client,), daemon=True)
            sse_thread.start()
            if not self._connected.wait(5):
                raise TimeoutError("OpenCode SSE did not report server.connected")
            with self._lock:
                self._state = ProviderRunState.RUNNING
            client.prompt_async(self._session.session_id, self._prompt_payload(turn_id))
            self._armed.set()
            deadline = time.monotonic() + (self.request.run_options.timeout_s or 3600)
            while time.monotonic() < deadline:
                messages = client.list_messages(self._session.session_id)
                if _turn_activity(messages, turn_id):
                    self._turn_seen.set()
                status = _status(client.session_status(), self._session.session_id)
                if self._pending is not None:
                    self._publish_needs_input(messages)
                    while self._pending is not None and time.monotonic() < deadline:
                        self._interaction_resolved.wait(0.25)
                    self._interaction_resolved.clear()
                    if self._pending is not None:
                        raise TimeoutError("OpenCode interaction was not answered before run timeout")
                    continue
                if self._armed.is_set() and status == "idle" and self._provider_error is not None:
                    self._finish(ProviderRunState.FAILED, messages=messages)
                    return
                if self._interrupt_requested.is_set() and status == "idle":
                    self._finish(ProviderRunState.INTERRUPTED, messages=messages)
                    return
                if (
                    self._armed.is_set()
                    and self._turn_seen.is_set()
                    and status == "idle"
                    and _turn_complete(messages, turn_id)
                ):
                    self._finish(ProviderRunState.COMPLETED, messages=messages)
                    return
                time.sleep(0.25)
            raise TimeoutError(f"OpenCode run timed out: {self.run_id}")
        except BaseException as exc:
            self._error = exc
            with self._lock:
                self._state = ProviderRunState.FAILED
            self._append_event("terminal.failed", terminal=True, data={"error_type": type(exc).__name__})
            self._done.set()
        finally:
            self._stop_sse.set()

    def _consume_sse(self, client: OpenCodeClient) -> None:
        try:
            for event in client.iter_events(self._stop_sse):
                event_type, properties = event_properties(event.data)
                if event_type == "server.connected":
                    self._connected.set()
                session_id = _event_session_id(properties)
                if self._session is not None and session_id not in {None, self._session.session_id}:
                    continue
                if _event_is_turn_activity(event_type, properties, self._turn):
                    self._turn_seen.set()
                if event_type == "permission.asked":
                    self._pending = ("permission", str(properties.get("id") or ""))
                elif event_type == "question.asked":
                    self._pending = ("question", str(properties.get("id") or ""))
                elif event_type == "session.error":
                    raw_error = properties.get("error") or properties
                    self._provider_error = AgentError(
                        error_type="opencode_session_error",
                        message=str(raw_error),
                        provider_payload=build_provider_payload(
                            provider_type=PROVIDER_TYPE,
                            payload_type="session_error",
                            data=raw_error,
                            adapter_version=ADAPTER_VERSION,
                        ),
                    )
                self._append_event(event_type or event.event, data=properties)
        except BaseException as exc:
            if not self._stop_sse.is_set() and not self._done.is_set():
                self._append_event("stream.disconnected", data={"error_type": type(exc).__name__})

    def _prompt_payload(self, turn_id: str) -> dict[str, object]:
        options = self.request.provider_options
        options = options if isinstance(options, OpenCodeRunOptions) else OpenCodeRunOptions()
        backend = self.request.model_overrides or self.request.execution_context.resolved_defaults
        provider_id = options.provider_id or (backend.api_provider if backend else None)
        model_id = options.model_id or (backend.effective_model if backend else None)
        if not provider_id or not model_id:
            raise ValueError("OpenCode run requires provider_id and model_id")
        payload: dict[str, object] = {
            "messageID": turn_id,
            "model": {"providerID": provider_id, "modelID": model_id},
            "parts": [{"type": "text", "text": self.request.prompt}],
        }
        if options.agent:
            payload["agent"] = options.agent
        if options.variant:
            payload["variant"] = options.variant
        if options.tools:
            payload["tools"] = dict(options.tools)
        if options.output_format is not None:
            payload["format"] = options.output_format
        instructions = self.request.system_instructions or self.request.developer_instructions
        if instructions:
            payload["system"] = instructions
        return payload

    def _finish(
        self,
        state: ProviderRunState,
        *,
        messages: list[object] | None = None,
    ) -> None:
        if self._done.is_set():
            return
        session = self.session_locator()
        turn = self.turn_locator()
        if session is None or turn is None:
            return
        server = self.registry.server_for_agent(self.request.agent_id)
        values = messages
        if values is None and server is not None:
            values = server.client.list_messages(session.session_id)
        artifact = AgentArtifactLocator(
            provider_type=PROVIDER_TYPE,
            home_id=session.home_id,
            session_id=session.session_id,
            adapter_version=ADAPTER_VERSION,
            native_primary_ref=parse_native_locator(session.native_locator).database_path,
        )
        error = self._provider_error
        if state is ProviderRunState.FAILED and error is None:
            error = AgentError(error_type="opencode_run_error", message="OpenCode run failed")
        self._result = completed_turn_result(
            session=session,
            messages=values or [],
            turn_id=turn.turn_id,
            run_id=self.run_id,
            started_at=self._started_at,
            status=state,
            error=error,
            artifact_locator=artifact,
        )
        with self._lock:
            self._state = state
        self._append_event("terminal." + state.value, terminal=state.terminal)
        self._done.set()

    def _publish_needs_input(self, messages: list[object]) -> None:
        session = self.session_locator()
        turn = self.turn_locator()
        if session is None or turn is None:
            return
        artifact = AgentArtifactLocator(
            provider_type=PROVIDER_TYPE,
            home_id=session.home_id,
            session_id=session.session_id,
            adapter_version=ADAPTER_VERSION,
            native_primary_ref=parse_native_locator(session.native_locator).database_path,
        )
        self._result = completed_turn_result(
            session=session,
            messages=messages,
            turn_id=turn.turn_id,
            run_id=self.run_id,
            started_at=self._started_at,
            status=ProviderRunState.NEEDS_INPUT,
            artifact_locator=artifact,
        )
        with self._lock:
            self._state = ProviderRunState.NEEDS_INPUT
        self._append_event("run.needs_input")
        self._done.set()

    def _append_event(self, kind: str, *, terminal: bool = False, data: object | None = None) -> None:
        with self._lock:
            event = AgentEvent(
                provider_type=PROVIDER_TYPE,
                sequence=len(self._events),
                timestamp=utc_now_iso(),
                kind=kind,
                session_id=self._session.session_id if self._session else None,
                turn_id=self._turn.turn_id if self._turn else None,
                terminal=terminal,
                data=data,
                provider_payload=build_provider_payload(
                    provider_type=PROVIDER_TYPE,
                    payload_type="sse_event",
                    data=data,
                    adapter_version=ADAPTER_VERSION,
                ) if data is not None else None,
            )
            self._events.append(event)
        if self.request.event_sink is not None:
            self.request.event_sink(event)


class OpenCodeRuntimeAdapter:
    provider_type = PROVIDER_TYPE

    def __init__(self, registry: OpenCodeRuntimeRegistry) -> None:
        self.registry = registry
        self._handles: dict[str, OpenCodeProviderRunHandle] = {}
        self._lock = threading.RLock()

    def start(self, request: ProviderRunRequest) -> OpenCodeProviderRunHandle:
        return self._start(request, resume=False)

    def resume(self, request: ProviderRunRequest) -> OpenCodeProviderRunHandle:
        return self._start(request, resume=True)

    def _start(self, request: ProviderRunRequest, *, resume: bool) -> OpenCodeProviderRunHandle:
        if request.provider_type != self.provider_type:
            raise ValueError("OpenCode runtime received another provider_type")
        handle = OpenCodeProviderRunHandle(self.registry, request, resume=resume)
        with self._lock:
            self._handles[handle.run_id] = handle
        return handle

    def fork(self, request: ProviderForkRequest) -> ProviderForkResult:
        source_native = parse_native_locator(request.source_session.native_locator)
        source_client = self.registry.client_for_locator(request.source_session)
        forked = source_client.fork(request.source_session.session_id, {})
        target_id = str(forked.get("id") or "")
        if not target_id:
            raise RuntimeError("OpenCode fork returned no session id")
        source_db = Path(source_native.database_path)
        target_runtime = self.registry.runtime_root / "providers" / PROVIDER_TYPE / "agents" / request.target_agent_id
        self.registry.close_agent(request.target_agent_id)
        target_runtime.mkdir(parents=True, exist_ok=True)
        target_db = target_runtime / "opencode.db"
        _backup_sqlite(source_db, target_db)
        source_runtime = self.registry.runtime_root / source_native.runtime_relpath
        for relative in (
            Path("xdg-data/opencode/tool-output"),
            Path("xdg-data/opencode/plans"),
        ):
            source_path = source_runtime / relative
            target_path = target_runtime / relative
            if source_path.is_dir():
                if target_path.exists():
                    shutil.rmtree(target_path)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_path, target_path)
        target_native = OpenCodeNativeLocator(
            agent_id=request.target_agent_id,
            directory=source_native.directory,
            database_path=str(target_db),
            runtime_relpath=str(target_runtime.relative_to(self.registry.runtime_root)),
        )
        target = ProviderSessionLocator(
            provider_type=PROVIDER_TYPE,
            session_id=target_id,
            home_id=request.target_home_id,
            created_at=utc_now_iso(),
            backend_identity=request.source_session.backend_identity,
            native_locator=target_native.as_dict(),
        )
        return ProviderForkResult(
            source_session=request.source_session,
            target_session=target,
            status="created",
            source_turn=request.source_turn,
            fork_mode="session_only",
            workspace_isolated=False,
            artifact_locator=AgentArtifactLocator(
                provider_type=PROVIDER_TYPE,
                home_id=request.target_home_id,
                session_id=target_id,
                adapter_version=ADAPTER_VERSION,
                native_primary_ref=str(target_db),
            ),
            limitations=(
                "workspace files are not isolated or rolled back",
                "historical absolute tool-output references may still point at the source Agent runtime",
            ),
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.run_id is None:
            now = utc_now_iso()
            return ProviderControlResult(
                action=request.action,
                accepted=False,
                terminal_confirmed=False,
                requested_at=request.requested_at,
                completed_at=now,
                reason="run_id is required",
            )
        with self._lock:
            handle = self._handles.get(request.run_id)
        if handle is None:
            now = utc_now_iso()
            return ProviderControlResult(
                action=request.action,
                accepted=False,
                terminal_confirmed=False,
                requested_at=request.requested_at,
                completed_at=now,
                reason="unknown or expired run_id",
            )
        return handle.control(request)

    def close_session(self, locator: ProviderSessionLocator) -> ProviderControlResult:
        native = parse_native_locator(locator.native_locator)
        self.registry.close_agent(native.agent_id)
        now = utc_now_iso()
        return ProviderControlResult(
            action=ProviderControlAction.ARCHIVE_SESSION,
            accepted=True,
            terminal_confirmed=True,
            requested_at=now,
            completed_at=now,
            resulting_state=ProviderRunState.COMPLETED,
            session_locator=locator,
        )

    def close(self) -> None:
        for handle in tuple(self._handles.values()):
            handle.close()
        self.registry.close()


def _control_result(
    action: ProviderControlAction,
    requested_at: str,
    accepted: bool,
    handle: OpenCodeProviderRunHandle,
    reason: str | None,
) -> ProviderControlResult:
    state = handle.poll_state()
    return ProviderControlResult(
        action=action,
        accepted=accepted,
        terminal_confirmed=state.terminal,
        requested_at=requested_at,
        completed_at=utc_now_iso(),
        resulting_state=state if state.terminal else None,
        session_locator=handle.session_locator(),
        turn_locator=handle.turn_locator(),
        reason=reason,
    )


def _status(statuses: Mapping[str, object], session_id: str) -> str:
    value = statuses.get(session_id)
    if not isinstance(value, Mapping):
        return "idle"
    return str(value.get("type") or value.get("status") or "unknown")


def _event_session_id(properties: Mapping[str, object]) -> str | None:
    value = properties.get("sessionID") or properties.get("sessionId")
    if value is not None:
        return str(value)
    info = properties.get("info") or properties.get("message") or properties.get("part")
    if isinstance(info, Mapping):
        value = info.get("sessionID") or info.get("sessionId")
        return str(value) if value is not None else None
    return None


def _turn_activity(messages: list[object], turn_id: str) -> bool:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        info = message.get("info") if isinstance(message.get("info"), Mapping) else message
        if info.get("role") == "assistant" and str(info.get("parentID") or "") == turn_id:
            return True
        parts = message.get("parts") if isinstance(message.get("parts"), list) else ()
        if any(
            isinstance(part, Mapping) and part.get("type") in {"retry", "compaction"}
            for part in parts
        ) and str(info.get("parentID") or info.get("id") or "") == turn_id:
            return True
    return False


def _turn_complete(messages: list[object], turn_id: str) -> bool:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        info = message.get("info") if isinstance(message.get("info"), Mapping) else message
        if info.get("role") != "assistant" or str(info.get("parentID") or "") != turn_id:
            continue
        timestamp = info.get("time")
        completed = isinstance(timestamp, Mapping) and timestamp.get("completed") is not None
        if completed or info.get("finish") is not None or info.get("error") is not None:
            return True
    return False


def _event_is_turn_activity(
    event_type: str | None,
    properties: Mapping[str, object],
    turn: ProviderTurnLocator | None,
) -> bool:
    if event_type in {"session.error", "permission.asked", "question.asked"}:
        return True
    if event_type == "session.status":
        status = properties.get("status")
        if isinstance(status, Mapping):
            return str(status.get("type") or status.get("status") or "") not in {"", "idle"}
        return str(status or "") not in {"", "idle"}
    if event_type == "message.updated":
        info = properties.get("info") or properties.get("message")
        return (
            isinstance(info, Mapping)
            and info.get("role") == "assistant"
            and turn is not None
            and str(info.get("parentID") or "") == turn.turn_id
        )
    if event_type in {"message.part.updated", "message.part.delta"}:
        part = properties.get("part")
        return isinstance(part, Mapping) and part.get("type") in {"retry", "compaction"}
    return False


def _message_id() -> str:
    global _last_id_timestamp, _id_counter
    timestamp = int(time.time() * 1000)
    with _id_lock:
        if timestamp != _last_id_timestamp:
            _last_id_timestamp = timestamp
            _id_counter = 0
        _id_counter += 1
        encoded = timestamp * 0x1000 + _id_counter
    # Node's 6-byte Buffer keeps only the low 48 bits of the bigint.
    time_hex = (encoded & ((1 << 48) - 1)).to_bytes(6, byteorder="big", signed=False).hex()
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    random_suffix = "".join(secrets.choice(alphabet) for _ in range(14))
    return "msg_" + time_hex + random_suffix


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _backup_sqlite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    with sqlite3.connect(source) as source_conn, sqlite3.connect(temp) as target_conn:
        source_conn.backup(target_conn)
        result = target_conn.execute("pragma integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError("OpenCode SQLite backup failed integrity_check")
    os.replace(temp, target)
    for suffix in ("-wal", "-shm"):
        stale = Path(str(target) + suffix)
        if stale.exists():
            stale.unlink()


_id_lock = threading.Lock()
_last_id_timestamp = 0
_id_counter = 0
