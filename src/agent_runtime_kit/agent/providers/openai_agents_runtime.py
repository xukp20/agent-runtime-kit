from __future__ import annotations

import asyncio
import json
import threading
import uuid
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path
from typing import Mapping

from ..provider_contracts import (
    AgentArtifactLocator,
    AgentEvent,
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
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .openai_agents import (
    OpenAIAgentsBuildContext,
    OpenAIAgentsControlOptions,
    OpenAIAgentsResourceRegistry,
    OpenAIAgentsRunOptions,
)
from .openai_agents_normalization import normalize_error, normalize_items, normalize_usage
from .openai_agents_storage import OpenAIAgentsSessionStore, safe_session_id


class OpenAIAgentsRunHandle:
    provider_type = "openai_agents"

    def __init__(self, *, runtime: "OpenAIAgentsRuntimeAdapter", request: ProviderRunRequest) -> None:
        self.runtime = runtime
        self.request = request
        self._run_id = f"oai-run-{uuid.uuid4().hex}"
        session_id = (
            request.session_locator.session_id
            if request.session_locator is not None
            else f"oai-session-{uuid.uuid4().hex}"
        )
        self._started_at = utc_now_iso()
        self._session = ProviderSessionLocator(
            provider_type=self.provider_type,
            session_id=safe_session_id(session_id),
            home_id=request.home_id,
            created_at=(request.session_locator.created_at if request.session_locator else self._started_at),
            backend_identity=_effective_model_identity(request),
            native_locator={"sqlite_relpath": f"sessions/{session_id}.sqlite3"},
        )
        self._turn = ProviderTurnLocator(
            session=self._session,
            turn_id=f"oai-turn-{uuid.uuid4().hex}",
        )
        home_root = _execution_context(request).home_root
        self.store = OpenAIAgentsSessionStore(
            OpenAIAgentsSessionStore.path_for(home_root, session_id),
            session_id=session_id,
            home_id=request.home_id,
        )
        turn_sequence = self.store.begin_turn(
            turn_id=self._turn.turn_id,
            run_id=self._run_id,
            started_at=self._started_at,
            backend=self._session.backend_identity,
        )
        self._turn = replace(self._turn, sequence=turn_sequence)
        self._state = ProviderRunState.STARTING
        self._events: list[AgentEvent] = []
        self._result: ProviderTurnResult | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream_result: object | None = None
        self._interrupt_requested = False
        self._closed = False
        self._condition = threading.Condition()
        self._thread = threading.Thread(target=self._thread_main, name=self._run_id, daemon=True)
        self._thread.start()

    @property
    def run_id(self) -> str:
        return self._run_id

    def session_locator(self) -> ProviderSessionLocator:
        return self._session

    def turn_locator(self) -> ProviderTurnLocator:
        return self._turn

    def poll_state(self) -> ProviderRunState:
        with self._condition:
            return self._state

    def drain_events(self, after_cursor: str | None = None) -> ProviderEventBatch:
        start = int(after_cursor) if after_cursor is not None else 0
        if start < 0:
            raise ValueError("event cursor must not be negative")
        with self._condition:
            events = tuple(self._events[start:])
            state = self._state
        return ProviderEventBatch(
            events=events,
            next_cursor=str(start + len(events)),
            terminal=state.terminal,
        )

    def wait_terminal(self, timeout_s: float | None = None) -> ProviderTurnResult:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._result is not None,
                timeout=timeout_s,
            )
            if not ready or self._result is None:
                raise TimeoutError(f"OpenAI Agents run did not reach a stable boundary: {self.run_id}")
            return self._result

    def interrupt(self, timeout_s: float | None = None) -> ProviderControlResult:
        requested_at = utc_now_iso()
        with self._condition:
            if self._state.terminal:
                return _control_result(
                    ProviderControlAction.INTERRUPT,
                    False,
                    requested_at,
                    self._state,
                    self._session,
                    self._turn,
                    reason="run is already terminal",
                )
            self._interrupt_requested = True
            loop = self._loop
            stream = self._stream_result
        if loop is not None and stream is not None:
            loop.call_soon_threadsafe(getattr(stream, "cancel"), "immediate")
        try:
            result = self.wait_terminal(timeout_s)
        except TimeoutError:
            return _control_result(
                ProviderControlAction.INTERRUPT,
                True,
                requested_at,
                None,
                self._session,
                self._turn,
                reason="interrupt requested but terminal barrier timed out",
            )
        return _control_result(
            ProviderControlAction.INTERRUPT,
            True,
            requested_at,
            result.status,
            self._session,
            self._turn,
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.action in {ProviderControlAction.INTERRUPT, ProviderControlAction.CANCEL}:
            result = self.interrupt(
                request.options.get("timeout_s")
                if isinstance(request.options.get("timeout_s"), (int, float))
                else None
            )
            return replace(result, action=request.action)
        if request.action not in {
            ProviderControlAction.RESPOND_APPROVAL,
            ProviderControlAction.RESPOND_INPUT,
            ProviderControlAction.REJECT_INPUT,
        }:
            return _control_result(
                request.action,
                False,
                request.requested_at,
                self.poll_state(),
                self._session,
                self._turn,
                reason="control action is unsupported by OpenAI Agents provider",
            )
        if self.poll_state() is not ProviderRunState.NEEDS_INPUT:
            return _control_result(
                request.action,
                False,
                request.requested_at,
                self.poll_state(),
                self._session,
                self._turn,
                reason="run is not waiting for approval/input",
            )
        try:
            resumed = self.runtime._resume_pending(self.request, self._session, request)
            result = resumed.wait_terminal(
                request.options.get("timeout_s") if isinstance(request.options.get("timeout_s"), (int, float)) else None
            )
        except BaseException as exc:
            return ProviderControlResult(
                action=request.action,
                accepted=False,
                terminal_confirmed=False,
                requested_at=request.requested_at,
                completed_at=utc_now_iso(),
                resulting_state=self.poll_state(),
                session_locator=self._session,
                turn_locator=self._turn,
                error=normalize_error(exc),
            )
        with self._condition:
            self._state = result.status
            self._result = result
            self._condition.notify_all()
        return _control_result(
            request.action,
            True,
            request.requested_at,
            result.status,
            result.session_locator,
            result.turn_locator,
        )

    def close(self) -> None:
        if not self.poll_state().terminal and self.poll_state() is not ProviderRunState.NEEDS_INPUT:
            raise RuntimeError("cannot close an active OpenAI Agents run handle")
        self._thread.join(timeout=1)
        self._closed = True

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:
            self._finish_failure(exc)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        with self._condition:
            self._state = ProviderRunState.RUNNING
        self._emit("turn.started", data={"run_id": self.run_id})
        config = _provider_config(self.request)
        client, model = _build_model(self.request, config)
        async with AsyncExitStack() as stack:
            mcp_servers = await _build_mcp_servers(config, self.request, stack)
            agent = _build_agent(self.request, config, model, mcp_servers)
            session = _build_sdk_session(self.store.path, self._session.session_id)
            try:
                run_input: object = self.request.prompt
                pending = self.request.metadata.get("openai_agents_pending_state")
                if isinstance(pending, Mapping):
                    run_input = await _restore_and_decide_state(agent, pending)
                run_options = self.request.provider_options
                if run_options is not None and not isinstance(run_options, OpenAIAgentsRunOptions):
                    raise TypeError("provider run options must be OpenAIAgentsRunOptions")
                kwargs: dict[str, object] = {"session": session}
                if self.request.run_options.max_turns is not None:
                    kwargs["max_turns"] = self.request.run_options.max_turns
                if isinstance(run_options, OpenAIAgentsRunOptions):
                    if run_options.context is not None:
                        kwargs["context"] = run_options.context
                    if run_options.hooks is not None:
                        kwargs["hooks"] = run_options.hooks
                    if run_options.run_config is not None:
                        kwargs["run_config"] = run_options.run_config
                from agents import Runner

                stream = Runner.run_streamed(agent, run_input, **kwargs)
                self._stream_result = stream
                if self._interrupt_requested:
                    stream.cancel("immediate")
                async for event in stream.stream_events():
                    self._emit_stream_event(event)
                run_exception = getattr(stream, "run_loop_exception", None)
                if run_exception is not None:
                    raise run_exception
                await self._finish_sdk_result(stream, config)
            finally:
                close = getattr(session, "close", None)
                if callable(close):
                    close()
                close_client = getattr(client, "close", None)
                if callable(close_client):
                    value = close_client()
                    if asyncio.iscoroutine(value):
                        await value

    async def _finish_sdk_result(self, result: object, config: Mapping[str, object]) -> None:
        completed_at = utc_now_iso()
        interruptions = list(getattr(result, "interruptions", []) or [])
        if self._interrupt_requested:
            status = ProviderRunState.INTERRUPTED
        elif interruptions:
            status = ProviderRunState.NEEDS_INPUT
        else:
            status = ProviderRunState.COMPLETED
        items = list(getattr(result, "new_items", []) or [])
        blocks, calls = normalize_items(items, turn_id=self._turn.turn_id)
        requests, turn_usage = normalize_usage(
            list(getattr(result, "raw_responses", []) or []),
            model_identity=self._session.backend_identity or _effective_model_identity(self.request),
            session_id=self._session.session_id,
            turn_id=self._turn.turn_id,
        )
        if interruptions:
            state = result.to_state()
            state_payload = state.to_json()
            state_id = f"oai-state-{uuid.uuid4().hex}"
            self.store.save_pending_state(
                state_id=state_id,
                turn_id=self._turn.turn_id,
                state_json=json.dumps(state_payload, sort_keys=True),
                interruptions=[_plain_interruption(item) for item in interruptions],
                factory_ref=str(config["agent_factory_ref"]),
                factory_version=str(config["agent_factory_version"]),
                resource_fingerprint=_optional_str(config.get("resource_fingerprint")),
            )
        final_output = getattr(result, "final_output", None)
        turn_result = ProviderTurnResult(
            provider_type=self.provider_type,
            run_id=self.run_id,
            session_locator=self._session,
            turn_locator=replace(
                self._turn,
                request_ids=tuple(item.request_id for item in requests if item.request_id),
            ),
            status=status,
            started_at=self._started_at,
            completed_at=completed_at,
            final_text=final_output if isinstance(final_output, str) else None,
            structured_output=None if isinstance(final_output, str) else final_output,
            content_blocks=blocks,
            tool_calls=calls,
            request_usages=requests,
            turn_usage=turn_usage,
            event_cursor=str(len(self._events)),
            artifact_locator=self.runtime.artifact_locator(self._session),
            provider_payload=build_provider_payload(
                provider_type=self.provider_type,
                payload_type="run_result",
                data={
                    "last_response_id": getattr(result, "last_response_id", None),
                    "interruption_count": len(interruptions),
                },
                adapter_version="1",
                sdk_or_cli_version="0.18.3",
            ),
        )
        self._emit(f"terminal.{status.value}", terminal=status.terminal, data={"status": status.value})
        self.store.finish_turn(
            turn_id=self._turn.turn_id,
            status=status,
            completed_at=completed_at,
            result=turn_result,
        )
        with self._condition:
            self._state = status
            self._result = turn_result
            self._condition.notify_all()

    def _finish_failure(self, exc: BaseException) -> None:
        completed_at = utc_now_iso()
        error = normalize_error(exc)
        result = ProviderTurnResult(
            provider_type=self.provider_type,
            run_id=self.run_id,
            session_locator=self._session,
            turn_locator=self._turn,
            status=ProviderRunState.FAILED,
            started_at=self._started_at,
            completed_at=completed_at,
            error=error,
            event_cursor=str(len(self._events)),
            artifact_locator=self.runtime.artifact_locator(self._session),
        )
        try:
            self._emit("terminal.failed", terminal=True, data={"error_type": error.error_type})
            self.store.finish_turn(
                turn_id=self._turn.turn_id,
                status=ProviderRunState.FAILED,
                completed_at=completed_at,
                result=result,
                error=error,
            )
        finally:
            with self._condition:
                self._state = ProviderRunState.FAILED
                self._result = result
                self._condition.notify_all()

    def _emit_stream_event(self, event: object) -> None:
        event_type = str(getattr(event, "type", "unknown"))
        name = getattr(event, "name", None)
        data = getattr(event, "data", None)
        item = getattr(event, "item", None)
        self._emit(
            str(name or getattr(data, "type", None) or event_type),
            data={
                "stream_event_type": event_type,
                "data": data,
                "item_type": getattr(item, "type", None),
            },
        )

    def _emit(self, kind: str, *, terminal: bool = False, data: object | None = None) -> None:
        sequence = self.store.next_event_sequence()
        event = AgentEvent(
            provider_type=self.provider_type,
            session_id=self._session.session_id,
            turn_id=self._turn.turn_id,
            sequence=sequence,
            timestamp=utc_now_iso(),
            kind=kind,
            terminal=terminal,
            data=data,
            provider_payload=build_provider_payload(
                provider_type=self.provider_type,
                payload_type="stream_event",
                data=data,
                adapter_version="1",
                sdk_or_cli_version="0.18.3",
            ),
        )
        self.store.append_event(event)
        with self._condition:
            self._events.append(event)
            self._condition.notify_all()
        if self.request.event_sink is not None:
            self.request.event_sink(event)


class OpenAIAgentsRuntimeAdapter:
    provider_type = "openai_agents"

    def __init__(self, *, runtime_root: Path, registry: OpenAIAgentsResourceRegistry) -> None:
        self.runtime_root = Path(runtime_root)
        self.registry = registry
        self._handles: dict[str, OpenAIAgentsRunHandle] = {}
        self._lock = threading.RLock()

    def start(self, request: ProviderRunRequest) -> OpenAIAgentsRunHandle:
        self._validate_request(request)
        handle = OpenAIAgentsRunHandle(runtime=self, request=request)
        with self._lock:
            self._handles[handle.run_id] = handle
        return handle

    def resume(self, request: ProviderRunRequest) -> OpenAIAgentsRunHandle:
        if request.session_locator is None:
            raise ValueError("OpenAI Agents resume requires session_locator")
        return self.start(request)

    def fork(self, request: ProviderForkRequest) -> ProviderForkResult:
        if request.source_turn is not None:
            raise NotImplementedError("OpenAI Agents first version only forks latest session history")
        ctx = request.execution_context
        if ctx is None:
            raise ValueError("OpenAI Agents fork requires execution_context")
        if request.target_home_id != request.source_session.home_id or ctx.home_id != request.target_home_id:
            raise ValueError("OpenAI Agents first version only forks within the same Home")
        source_path = OpenAIAgentsSessionStore.path_for(ctx.home_root, request.source_session.session_id)
        source = OpenAIAgentsSessionStore(source_path, session_id=request.source_session.session_id, home_id=request.source_session.home_id)
        if not source.is_quiescent():
            raise RuntimeError("cannot fork a running OpenAI Agents session")
        target_id = f"oai-session-{uuid.uuid4().hex}"
        target_path = OpenAIAgentsSessionStore.path_for(ctx.home_root, target_id)
        source.backup_to(target_path)
        with sqlite_connection(target_path) as conn:
            now = utc_now_iso()
            conn.execute("update ark_session set session_id=?,home_id=?,created_at=?,updated_at=?,fork_json=?", (
                target_id,
                request.target_home_id,
                now,
                now,
                json.dumps({"source_session_id": request.source_session.session_id, "fork_mode": "session_only", "workspace_isolated": False}),
            ))
            conn.execute("update agent_sessions set session_id=?", (target_id,))
        target = ProviderSessionLocator(
            provider_type=self.provider_type,
            session_id=target_id,
            home_id=request.target_home_id,
            created_at=utc_now_iso(),
            backend_identity=request.source_session.backend_identity,
            native_locator={"sqlite_relpath": f"sessions/{target_id}.sqlite3"},
        )
        return ProviderForkResult(
            source_session=request.source_session,
            target_session=target,
            status="forked",
            fork_mode="session_only",
            workspace_isolated=False,
            artifact_locator=self.artifact_locator(target),
            limitations=("fork copies Agent session history only; workspace files are not isolated",),
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.run_id:
            with self._lock:
                handle = self._handles.get(request.run_id)
            if handle is not None:
                return handle.control(request)
        if request.action in {
            ProviderControlAction.RESPOND_APPROVAL,
            ProviderControlAction.RESPOND_INPUT,
            ProviderControlAction.REJECT_INPUT,
        } and request.session_id:
            options = request.provider_options
            if not isinstance(options, OpenAIAgentsControlOptions):
                return ProviderControlResult(
                    action=request.action,
                    accepted=False,
                    terminal_confirmed=False,
                    requested_at=request.requested_at,
                    completed_at=utc_now_iso(),
                    reason="durable approval requires OpenAIAgentsControlOptions",
                )
            session = ProviderSessionLocator(
                provider_type=self.provider_type,
                session_id=request.session_id,
                home_id=options.home_id,
                created_at=utc_now_iso(),
                backend_identity=getattr(options.execution_context, "resolved_defaults", None),
                native_locator={"sqlite_relpath": f"sessions/{request.session_id}.sqlite3"},
            )
            original = ProviderRunRequest(
                agent_id=options.agent_id,
                scope_id=options.scope_id,
                agent_type=options.agent_type,
                provider_type=self.provider_type,
                home_id=options.home_id,
                prompt="",
                session_locator=session,
                execution_context=options.execution_context,
            )
            try:
                handle = self._resume_pending(original, session, request)
                result = handle.wait_terminal(
                    request.options.get("timeout_s")
                    if isinstance(request.options.get("timeout_s"), (int, float))
                    else None
                )
            except BaseException as exc:
                return ProviderControlResult(
                    action=request.action,
                    accepted=False,
                    terminal_confirmed=False,
                    requested_at=request.requested_at,
                    completed_at=utc_now_iso(),
                    session_locator=session,
                    error=normalize_error(exc),
                )
            return _control_result(
                request.action,
                True,
                request.requested_at,
                result.status,
                result.session_locator,
                result.turn_locator,
            )
        return ProviderControlResult(
            action=request.action,
            accepted=False,
            terminal_confirmed=False,
            requested_at=request.requested_at,
            completed_at=utc_now_iso(),
            reason="OpenAI Agents control requires a known live run handle",
        )

    def close_session(self, locator: ProviderSessionLocator) -> ProviderControlResult:
        return ProviderControlResult(
            action=ProviderControlAction.ARCHIVE_SESSION,
            accepted=True,
            terminal_confirmed=True,
            requested_at=utc_now_iso(),
            completed_at=utc_now_iso(),
            resulting_state=ProviderRunState.COMPLETED,
            session_locator=locator,
            reason="SQLite session is already durable; no live resource remains",
        )

    def close(self) -> None:
        with self._lock:
            handles = tuple(self._handles.values())
        for handle in handles:
            if not handle.poll_state().terminal and handle.poll_state() is not ProviderRunState.NEEDS_INPUT:
                handle.interrupt(5)
            handle.close()

    def artifact_locator(self, session: ProviderSessionLocator) -> AgentArtifactLocator:
        return AgentArtifactLocator(
            provider_type=self.provider_type,
            home_id=session.home_id,
            session_id=session.session_id,
            adapter_version="1",
            # The artifact adapter derives the SQLite path from the session locator.
            native_primary_ref=None,
        )

    def _resume_pending(
        self,
        original: ProviderRunRequest,
        session: ProviderSessionLocator,
        control: ProviderControlRequest,
    ) -> OpenAIAgentsRunHandle:
        ctx = _execution_context(original)
        store = OpenAIAgentsSessionStore(
            OpenAIAgentsSessionStore.path_for(ctx.home_root, session.session_id),
            session_id=session.session_id,
            home_id=session.home_id,
        )
        row = store.pending_state(_optional_str(control.options.get("state_id")))
        if row is None:
            raise RuntimeError("no pending OpenAI Agents RunState")
        config = _provider_config(original)
        expected = (
            str(config["agent_factory_ref"]),
            str(config["agent_factory_version"]),
            _optional_str(config.get("resource_fingerprint")),
        )
        actual = (
            str(row["factory_ref"]),
            str(row["factory_version"]),
            _optional_str(row["resource_fingerprint"]),
        )
        if actual != expected:
            raise RuntimeError("pending OpenAI Agents RunState resource identity mismatch")
        content = control.content if isinstance(control.content, Mapping) else {}
        decision = "reject" if control.action is ProviderControlAction.REJECT_INPUT else str(content.get("decision") or "approve")
        pending = {
            "state_id": row["state_id"],
            "state_json": json.loads(row["state_json"]),
            "approval_id": content.get("approval_id"),
            "decision": decision,
            "always": bool(content.get("always", False)),
            "rejection_message": content.get("rejection_message"),
        }
        if not store.consume_pending_state(str(row["state_id"])):
            raise RuntimeError("pending OpenAI Agents RunState was already consumed")
        request = replace(
            original,
            prompt="",
            session_locator=session,
            metadata={**original.metadata, "openai_agents_pending_state": pending},
        )
        return self.resume(request)

    @staticmethod
    def _validate_request(request: ProviderRunRequest) -> None:
        if request.provider_type != "openai_agents":
            raise ValueError(f"OpenAI Agents runtime received {request.provider_type}")
        _execution_context(request)


def _execution_context(request: ProviderRunRequest):  # noqa: ANN202
    ctx = request.execution_context
    if ctx is None or ctx.provider_type != "openai_agents":
        raise ValueError("OpenAI Agents runtime requires its ProviderExecutionContext")
    return ctx


def _provider_config(request: ProviderRunRequest) -> Mapping[str, object]:
    payload = _execution_context(request).runtime_payload
    if not isinstance(payload, Mapping):
        raise RuntimeError("OpenAI Agents execution context lacks provider config")
    return payload


def _effective_model_identity(request: ProviderRunRequest) -> ModelBackendIdentity:
    model = request.model_overrides or _execution_context(request).resolved_defaults
    if model is None:
        raise ValueError("OpenAI Agents run requires model backend identity")
    return model


def _build_model(request: ProviderRunRequest, config: Mapping[str, object]) -> tuple[object, object]:
    from openai import AsyncOpenAI
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from agents.models.openai_responses import OpenAIResponsesModel

    ctx = _execution_context(request)
    identity = _effective_model_identity(request)
    model_name = identity.effective_model
    if not model_name:
        raise ValueError("OpenAI Agents backend requires a model name")
    api_key_env = str(config["api_key_env"])
    api_key = ctx.process_environment.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"missing OpenAI Agents API key env: {api_key_env}")
    base_url = _optional_str(config.get("base_url"))
    base_url_env = _optional_str(config.get("base_url_env"))
    if base_url_env:
        base_url = ctx.process_environment.get(base_url_env)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    if identity.api_mode == "responses":
        return client, OpenAIResponsesModel(model_name, client)
    if identity.api_mode == "chat_completions":
        return client, OpenAIChatCompletionsModel(model_name, client)
    raise ValueError(f"unsupported OpenAI Agents API mode: {identity.api_mode}")


async def _build_mcp_servers(config: Mapping[str, object], request: ProviderRunRequest, stack: AsyncExitStack) -> tuple[object, ...]:
    raw_servers = config.get("mcp_servers", [])
    if not isinstance(raw_servers, list) or not any(
        isinstance(item, Mapping) and item.get("enabled", True) for item in raw_servers
    ):
        return ()
    from agents.mcp import MCPServerSse, MCPServerStdio, MCPServerStreamableHttp

    ctx = _execution_context(request)
    values: list[object] = []
    for raw in raw_servers:
        if not isinstance(raw, Mapping) or not raw.get("enabled", True):
            continue
        transport = str(raw.get("transport") or "http")
        name = _optional_str(raw.get("name"))
        if transport == "stdio":
            env = dict(raw.get("env") or {})
            for variable in raw.get("env_vars") or []:
                if ctx.process_environment.get(str(variable)):
                    env[str(variable)] = ctx.process_environment[str(variable)]
            server = MCPServerStdio(
                {"command": str(raw["command"]), "args": list(raw.get("args") or []), "env": env or None, "cwd": raw.get("cwd")},
                name=name,
            )
        else:
            headers = dict(raw.get("http_headers") or {})
            for header, variable in dict(raw.get("env_http_headers") or {}).items():
                value = ctx.process_environment.get(str(variable))
                if value:
                    headers[str(header)] = value
            token_env = raw.get("bearer_token_env_var")
            if token_env and ctx.process_environment.get(str(token_env)):
                headers["Authorization"] = f"Bearer {ctx.process_environment[str(token_env)]}"
            params = {"url": str(raw["url"]), "headers": headers}
            server = MCPServerSse(params, name=name) if transport == "sse" else MCPServerStreamableHttp(params, name=name)
        await stack.enter_async_context(server)
        values.append(server)
    return tuple(values)


def _build_agent(request: ProviderRunRequest, config: Mapping[str, object], model: object, mcp_servers: tuple[object, ...]) -> object:
    ctx = _execution_context(request)
    registry = next((item for item in ctx.resource_handles if isinstance(item, OpenAIAgentsResourceRegistry)), None)
    if registry is None:
        raise RuntimeError("OpenAI Agents execution context lacks resource registry")
    instructions = "\n\n".join(str(item) for item in config.get("instructions", []) if str(item).strip())
    if request.system_instructions:
        instructions = _join_instructions(instructions, request.system_instructions)
    if request.developer_instructions:
        instructions = request.developer_instructions if request.replace_developer_instructions else _join_instructions(instructions, request.developer_instructions)
    factory = registry.resolve_agent_factory(
        str(config["agent_factory_ref"]),
        version=str(config["agent_factory_version"]),
        fingerprint=_optional_str(config.get("resource_fingerprint")),
    )
    build_ctx = OpenAIAgentsBuildContext(
        home_id=request.home_id,
        home_root=ctx.home_root,
        workdir=request.workdir or ctx.workdir,
        model=model,
        model_identity=_effective_model_identity(request),
        instructions=instructions,
        skills_root=ctx.home_root / "skills",
        mcp_servers=mcp_servers,
        provider_config=config,
    )
    agent = factory(build_ctx)
    clone = getattr(agent, "clone", None)
    if not callable(clone):
        raise TypeError("OpenAI Agents factory must return an Agent with clone()")
    overrides: dict[str, object] = {"model": model}
    if mcp_servers:
        overrides["mcp_servers"] = list(mcp_servers)
    if instructions:
        overrides["instructions"] = instructions
    settings = dict(config.get("model_settings") or {})
    settings["store"] = bool(config.get("store", False))
    if settings:
        from agents import ModelSettings

        overrides["model_settings"] = ModelSettings(**settings)
    return clone(**overrides)


def _build_sdk_session(path: Path, session_id: str) -> object:
    from agents.memory import SQLiteSession

    return SQLiteSession(session_id, db_path=path)


async def _restore_and_decide_state(agent: object, pending: Mapping[str, object]) -> object:
    from agents import RunState

    state_json = pending.get("state_json")
    if not isinstance(state_json, dict):
        raise ValueError("pending OpenAI Agents state_json must be an object")
    state = await RunState.from_json(agent, state_json)
    interruptions = state.get_interruptions()
    approval_id = _optional_str(pending.get("approval_id"))
    selected = None
    for item in interruptions:
        raw = getattr(item, "raw_item", None)
        call_id = getattr(raw, "call_id", None) or getattr(raw, "id", None)
        if approval_id is None or str(call_id) == approval_id:
            selected = item
            break
    if selected is None:
        raise RuntimeError("requested approval id is not pending")
    if pending.get("decision") == "reject":
        state.reject(
            selected,
            always_reject=bool(pending.get("always", False)),
            rejection_message=_optional_str(pending.get("rejection_message")),
        )
    else:
        state.approve(selected, always_approve=bool(pending.get("always", False)))
    return state


def _plain_interruption(item: object) -> object:
    raw = getattr(item, "raw_item", None)
    dump = getattr(raw, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    if isinstance(raw, Mapping):
        return dict(raw)
    return {"tool_name": getattr(item, "tool_name", None), "raw_type": type(raw).__name__}


def _join_instructions(left: str, right: str) -> str:
    return "\n\n".join(item.strip() for item in (left, right) if item.strip())


def _control_result(action: ProviderControlAction, accepted: bool, requested_at: str, state: ProviderRunState | None, session: ProviderSessionLocator, turn: ProviderTurnLocator | None, *, reason: str | None = None) -> ProviderControlResult:
    return ProviderControlResult(
        action=action,
        accepted=accepted,
        terminal_confirmed=bool(state and state.terminal),
        requested_at=requested_at,
        completed_at=utc_now_iso(),
        resulting_state=state,
        session_locator=session,
        turn_locator=turn,
        reason=reason,
    )


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


class sqlite_connection:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection = None

    def __enter__(self):  # noqa: ANN204
        import sqlite3

        self.connection = sqlite3.connect(self.path)
        return self.connection

    def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
        assert self.connection is not None
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()
