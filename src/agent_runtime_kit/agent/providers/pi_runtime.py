from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from time import monotonic

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
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .pi_rpc import PiRpcError, PiRpcProcess
from .pi_session import (
    PI_ADAPTER_VERSION,
    PI_CLI_VERSION,
    PiSessionTranscript,
    find_pi_session,
)


class PiProviderRunHandle:
    def __init__(
        self,
        *,
        runtime_root: Path,
        request: ProviderRunRequest,
        resume: bool,
        on_done,
    ) -> None:  # noqa: ANN001
        self.runtime_root = Path(runtime_root)
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
        self._transport: PiRpcProcess | None = None
        self._events: list[AgentEvent] = []
        self._live_record_cursor = 0
        self._lock = threading.RLock()
        self._collect_lock = threading.RLock()
        self._done = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)

    def begin(self) -> None:
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
        self._collect_live_events()
        with self._lock:
            return self._state

    def drain_events(self, after_cursor: str | None = None) -> ProviderEventBatch:
        self._collect_live_events()
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
        requested_at = utc_now_iso()
        effective_timeout = timeout_s if timeout_s is not None else 10.0
        transport = self._transport
        if transport is None or self._done.is_set():
            return ProviderControlResult(
                action=ProviderControlAction.INTERRUPT,
                accepted=False,
                terminal_confirmed=self._done.is_set(),
                resulting_state=self.poll_state() if self._done.is_set() else None,
                requested_at=requested_at,
                completed_at=utc_now_iso(),
                session_locator=self.session_locator(),
                turn_locator=self.turn_locator(),
                reason="Pi run is not active",
            )
        try:
            transport.command("abort", timeout_s=effective_timeout)
            confirmed = self._done.wait(effective_timeout)
        except BaseException as exc:
            transport.terminate()
            confirmed = self._done.wait(5)
            if not confirmed:
                return ProviderControlResult(
                    action=ProviderControlAction.INTERRUPT,
                    accepted=True,
                    terminal_confirmed=False,
                    requested_at=requested_at,
                    completed_at=utc_now_iso(),
                    session_locator=self.session_locator(),
                    turn_locator=self.turn_locator(),
                    reason=f"Pi abort required process termination: {type(exc).__name__}",
                )
        return ProviderControlResult(
            action=ProviderControlAction.INTERRUPT,
            accepted=True,
            terminal_confirmed=confirmed,
            resulting_state=self.poll_state() if confirmed else None,
            requested_at=requested_at,
            completed_at=utc_now_iso(),
            session_locator=self.session_locator(),
            turn_locator=self.turn_locator(),
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        if request.action in {ProviderControlAction.INTERRUPT, ProviderControlAction.CANCEL}:
            timeout = request.options.get("timeout_s")
            return self.interrupt(float(timeout) if isinstance(timeout, (int, float)) else None)
        transport = self._transport
        command = {
            ProviderControlAction.STEER: "steer",
            ProviderControlAction.FOLLOW_UP: "follow_up",
        }.get(request.action)
        if transport is not None and command is not None and isinstance(request.content, str):
            try:
                transport.command(command, {"message": request.content}, timeout_s=10)
                return ProviderControlResult(
                    action=request.action,
                    accepted=True,
                    terminal_confirmed=False,
                    resulting_state=self.poll_state(),
                    requested_at=request.requested_at,
                    completed_at=utc_now_iso(),
                    session_locator=self.session_locator(),
                    turn_locator=self.turn_locator(),
                )
            except BaseException as exc:
                reason = str(exc)
        else:
            reason = "Pi control requires an active run and string content"
        return ProviderControlResult(
            action=request.action,
            accepted=False,
            terminal_confirmed=self._done.is_set(),
            resulting_state=self.poll_state() if self._done.is_set() else None,
            requested_at=request.requested_at,
            completed_at=utc_now_iso(),
            session_locator=self.session_locator(),
            turn_locator=self.turn_locator(),
            reason=reason,
        )

    def close(self) -> None:
        transport = self._transport
        if transport is not None:
            transport.close()

    def _run(self) -> None:
        transport: PiRpcProcess | None = None
        try:
            context = self.request.execution_context
            if context is None or context.provider_type != "pi":
                raise ValueError("Pi runtime requires a Pi ProviderExecutionContext")
            session_root = context.home_root / ".pi" / "sessions"
            session_root.mkdir(parents=True, exist_ok=True)
            requested_session_id = (
                self.request.session_locator.session_id
                if self.resume and self.request.session_locator is not None
                else str(uuid.uuid4())
            )
            session_path = None
            if self.resume:
                session_path = find_pi_session(session_root, requested_session_id)
                if session_path is None:
                    raise KeyError(f"unknown Pi session: {requested_session_id}")
            command = build_pi_command(
                context,
                session_path=session_path,
                session_id=None if self.resume else requested_session_id,
                model=self.request.model_overrides,
            )
            transport = PiRpcProcess(
                command,
                cwd=Path(self.request.workdir or context.workdir or context.home_root),
                env=context.process_environment,
            )
            self._transport = transport
            with self._lock:
                self._state = ProviderRunState.RUNNING
            state = _response_data(transport.command("get_state", timeout_s=15))
            session_id = str(state.get("sessionId") or requested_session_id)
            native_file = state.get("sessionFile")
            self._session = ProviderSessionLocator(
                provider_type="pi",
                session_id=session_id,
                home_id=self.request.home_id,
                created_at=self._started_at,
                backend_identity=self.request.model_overrides,
                native_locator={
                    "session_relpath": (
                        str(Path(str(native_file)).relative_to(session_root))
                        if native_file and Path(str(native_file)).is_relative_to(session_root)
                        else None
                    )
                },
            )
            self._append_event("session.started", data={"session_id": session_id})
            baseline = _response_data(transport.command("get_entries", timeout_s=15))
            baseline_leaf = baseline.get("leafId")
            prompt = _compose_prompt(self.request)
            transport.command("prompt", {"message": prompt}, timeout_s=15)
            self._append_event("turn.accepted")
            transport.wait_for(
                lambda item: item.get("type") == "agent_settled",
                timeout_s=self.request.run_options.timeout_s or 3600,
            )
            self._collect_live_events()
            final_state = _response_data(transport.command("get_state", timeout_s=15))
            if bool(final_state.get("isStreaming")) or bool(final_state.get("isCompacting")):
                raise PiRpcError("Pi emitted agent_settled but remained busy")
            final_entries = _response_data(transport.command("get_entries", timeout_s=15))
            session_file = final_state.get("sessionFile") or native_file
            path = Path(str(session_file)) if session_file else find_pi_session(session_root, session_id)
            if path is None or not path.is_file():
                raise RuntimeError("Pi completed without a persisted session JSONL")
            transcript = PiSessionTranscript.read(path)
            turns = list(transcript.turns())
            if not turns:
                raise RuntimeError("Pi completed without a persisted user turn")
            turn = _select_new_turn(turns, baseline_leaf)
            result = transcript.turn_result(turn, home_id=self.request.home_id, run_id=self.run_id)
            artifact_ref = str(path.relative_to(self.runtime_root))
            result = _with_artifact(result, artifact_ref)
            with self._lock:
                self._session = result.session_locator
                self._turn = result.turn_locator
                self._result = result
                self._state = result.status
            self._append_event("terminal." + result.status.value, terminal=True)
        except BaseException as exc:
            self._error = exc
            with self._lock:
                self._state = ProviderRunState.FAILED
            self._append_event(
                "terminal.failed",
                terminal=True,
                data={"error_type": type(exc).__name__, "message": str(exc)},
            )
        finally:
            if transport is not None:
                transport.close()
            self._done.set()
            self._on_done(self)

    def _collect_live_events(self) -> None:
        with self._collect_lock:
            transport = self._transport
            if transport is None:
                return
            records = transport.records
            with self._lock:
                new_records = records[self._live_record_cursor :]
                self._live_record_cursor = len(records)
            for record in new_records:
                if record.get("type") == "response":
                    continue
                self._append_event(
                    _live_event_kind(record),
                    terminal=record.get("type") == "agent_settled",
                    data=record,
                    provider_record=record,
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
                provider_type="pi",
                session_id=self._session.session_id if self._session is not None else None,
                turn_id=self._turn.turn_id if self._turn is not None else None,
                sequence=len(self._events),
                timestamp=utc_now_iso(),
                kind=kind,
                terminal=terminal,
                data=data,
                provider_payload=(
                    build_provider_payload(
                        provider_type="pi",
                        payload_type="rpc_event",
                        data=provider_record,
                        adapter_version=PI_ADAPTER_VERSION,
                        sdk_or_cli_version=PI_CLI_VERSION,
                    )
                    if provider_record is not None
                    else None
                ),
            )
            self._events.append(event)
        if self.request.event_sink is not None:
            self.request.event_sink(event)


class PiRuntimeAdapter:
    provider_type = "pi"

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)
        self._handles: dict[str, PiProviderRunHandle] = {}
        self._lock = threading.RLock()

    def start(self, request: ProviderRunRequest) -> PiProviderRunHandle:
        return self._start(request, resume=False)

    def resume(self, request: ProviderRunRequest) -> PiProviderRunHandle:
        if request.session_locator is None:
            raise ValueError("Pi resume requires session_locator")
        return self._start(request, resume=True)

    def _start(self, request: ProviderRunRequest, *, resume: bool) -> PiProviderRunHandle:
        if request.provider_type != "pi":
            raise ValueError("PiRuntimeAdapter received a different provider_type")
        handle = PiProviderRunHandle(
            runtime_root=self.runtime_root,
            request=request,
            resume=resume,
            on_done=self._on_done,
        )
        with self._lock:
            self._handles[handle.run_id] = handle
        handle.begin()
        return handle

    def fork(self, request: ProviderForkRequest) -> ProviderForkResult:
        context = request.execution_context
        if context is None or context.provider_type != "pi":
            raise ValueError("Pi fork requires a Pi ProviderExecutionContext")
        source_root = self.runtime_root / "homes" / "pi" / request.source_session.home_id / ".pi" / "sessions"
        source = find_pi_session(source_root, request.source_session.session_id)
        if source is None:
            raise KeyError(f"unknown Pi source session: {request.source_session.session_id}")
        target_id = str(uuid.uuid4())
        historical_command: tuple[str, dict[str, str]] | None = None
        if request.source_turn is None:
            args = build_pi_command(
                context,
                fork_path=source,
                session_id=target_id,
                model=request.source_session.backend_identity,
            )
        else:
            # Pi's RPC fork operates on the currently opened source session and
            # creates the historical branch directly in the configured session
            # directory. Starting with --fork here would first create a full
            # clone and then fork that clone, leaving an abandoned artifact.
            args = build_pi_command(
                context,
                session_path=source,
                model=request.source_session.backend_identity,
            )
            historical_command = _historical_fork_command(
                PiSessionTranscript.read(source),
                request.source_turn.turn_id,
            )
        rpc = PiRpcProcess(
            args,
            cwd=Path(context.workdir or context.home_root),
            env=context.process_environment,
        )
        try:
            state = _response_data(rpc.command("get_state", timeout_s=20))
            if historical_command is not None:
                command, payload = historical_command
                response = rpc.command(command, payload, timeout_s=30)
                if bool(_response_data(response).get("cancelled")):
                    raise RuntimeError("Pi historical session fork was cancelled")
                state = _response_data(rpc.command("get_state", timeout_s=20))
            session_id = str(state.get("sessionId") or target_id)
            session_file = Path(str(state["sessionFile"])) if state.get("sessionFile") else None
        finally:
            rpc.close()
        if session_file is None or not session_file.is_file():
            target_root = context.home_root / ".pi" / "sessions"
            session_file = find_pi_session(target_root, session_id)
        if session_file is None:
            raise RuntimeError("Pi fork did not produce a session artifact")
        transcript = PiSessionTranscript.read(session_file)
        target = transcript.locator(request.target_home_id)
        target_turn = None
        if transcript.turns():
            latest = transcript.turns()[-1]
            target_turn = ProviderTurnLocator(session=target, turn_id=latest.turn_id, sequence=latest.sequence)
        artifact_ref = str(session_file.relative_to(self.runtime_root))
        return ProviderForkResult(
            source_session=request.source_session,
            target_session=target,
            source_turn=request.source_turn,
            target_turn=target_turn,
            status="created",
            fork_mode="session_only",
            workspace_isolated=False,
            artifact_locator=AgentArtifactLocator(
                provider_type="pi",
                home_id=request.target_home_id,
                session_id=target.session_id,
                adapter_version=PI_ADAPTER_VERSION,
                native_primary_ref=artifact_ref,
            ),
            limitations=("workspace files are not isolated or rolled back",),
        )

    def control(self, request: ProviderControlRequest) -> ProviderControlResult:
        with self._lock:
            handle = self._handles.get(request.run_id or "")
        if handle is None:
            now = utc_now_iso()
            return ProviderControlResult(
                action=request.action,
                accepted=False,
                terminal_confirmed=False,
                requested_at=request.requested_at,
                completed_at=now,
                reason="unknown or expired Pi run_id",
            )
        return handle.control(request)

    def close_session(self, locator: ProviderSessionLocator) -> ProviderControlResult:
        now = utc_now_iso()
        active = [
            handle
            for handle in self._active_handles()
            if (session := handle.session_locator()) is not None and session.session_id == locator.session_id
        ]
        for handle in active:
            handle.interrupt(10)
        return ProviderControlResult(
            action=ProviderControlAction.ARCHIVE_SESSION,
            accepted=bool(active),
            terminal_confirmed=not self.is_session_active(locator.session_id),
            requested_at=now,
            completed_at=utc_now_iso(),
            session_locator=locator,
            reason=None if active else "Pi session had no active process; artifact was retained",
        )

    def is_session_active(self, session_id: str) -> bool:
        return any(
            (session := item.session_locator()) is not None
            and session.session_id == session_id
            and not item.poll_state().terminal
            for item in self._active_handles()
        )

    def close(self) -> None:
        for handle in self._active_handles():
            handle.close()

    def _active_handles(self) -> tuple[PiProviderRunHandle, ...]:
        with self._lock:
            return tuple(self._handles.values())

    def _on_done(self, handle: PiProviderRunHandle) -> None:
        with self._lock:
            self._handles.pop(handle.run_id, None)


def build_pi_command(
    context,
    *,
    session_path: Path | None = None,
    fork_path: Path | None = None,
    session_id: str | None = None,
    model=None,
) -> list[str]:  # noqa: ANN001
    runtime = context.runtime_payload
    if not isinstance(runtime, Mapping):
        raise ValueError("Pi execution context has no runtime configuration")
    cli = runtime.get("pi_cli_path")
    if not cli:
        raise RuntimeError("Pi runtime configuration has no CLI path")
    cli_path = Path(str(cli))
    node = runtime.get("node_executable")
    command = [str(node), str(cli_path)] if node else [str(cli_path)]
    command.extend(
        [
            "--mode",
            "rpc",
            "--approve" if runtime.get("approve_project_resources") is True else "--no-approve",
            "--session-dir",
            str(context.home_root / ".pi" / "sessions"),
        ]
    )
    if runtime.get("offline") is True:
        command.append("--offline")
    if session_path is not None:
        command.extend(["--session", str(session_path)])
    if fork_path is not None:
        command.extend(["--fork", str(fork_path)])
    if session_id is not None:
        command.extend(["--session-id", session_id])
    if model is not None:
        if getattr(model, "api_provider", None):
            command.extend(["--provider", str(model.api_provider)])
        if getattr(model, "effective_model", None):
            command.extend(["--model", str(model.effective_model)])
    tools = runtime.get("tools")
    if isinstance(tools, list) and tools:
        command.extend(["--tools", ",".join(str(item) for item in tools)])
    instructions = runtime.get("instructions")
    if instructions:
        command.extend(
            [
                "--append-system-prompt",
                (context.home_root / str(instructions)).read_text(encoding="utf-8"),
            ]
        )
    for relpath in runtime.get("extensions") or ():
        command.extend(["--extension", str(context.home_root / str(relpath))])
    mcp_manifest = runtime.get("mcp_manifest")
    if mcp_manifest:
        bridge = context.home_root / ".pi" / "extensions" / "ark_pi_mcp_bridge.mjs"
        command.extend(["--extension", str(bridge)])
    command.extend(str(item) for item in runtime.get("extra_cli_args") or ())
    return command


def _compose_prompt(request: ProviderRunRequest) -> str:
    prefixes = []
    if request.system_instructions:
        prefixes.append(f"<system-instructions>\n{request.system_instructions}\n</system-instructions>")
    if request.developer_instructions:
        prefixes.append(
            f"<developer-instructions>\n{request.developer_instructions}\n</developer-instructions>"
        )
    prefixes.append(request.prompt)
    return "\n\n".join(prefixes)


def _response_data(response: Mapping[str, object]) -> dict[str, object]:
    value = response.get("data")
    return dict(value) if isinstance(value, Mapping) else {}


def _select_new_turn(turns, baseline_leaf):  # noqa: ANN001, ANN202
    if baseline_leaf is None:
        return turns[-1]
    for turn in turns:
        if any(entry.get("parentId") == baseline_leaf for entry in turn.entries):
            return turn
    return turns[-1]


def _historical_fork_command(
    transcript: PiSessionTranscript,
    source_turn_id: str,
) -> tuple[str, dict[str, str]]:
    turns = transcript.turns()
    for index, turn in enumerate(turns):
        if turn.turn_id != source_turn_id:
            continue
        # Pi's RPC `fork` branches immediately before the selected user
        # message. Selecting the following turn therefore retains the complete
        # requested turn. For the active leaf, `clone` is the equivalent
        # operation and preserves the full branch through that leaf.
        if index + 1 < len(turns):
            return "fork", {"entryId": turns[index + 1].turn_id}
        return "clone", {}
    raise KeyError(f"Pi source turn is not on the active session branch: {source_turn_id}")


def _live_event_kind(record: Mapping[str, object]) -> str:
    value = str(record.get("type") or "unknown")
    return {
        "agent_settled": "terminal.settled",
        "tool_execution_start": "tool.started",
        "tool_execution_end": "tool.completed",
        "message_start": "message.started",
        "message_end": "message.completed",
        "compaction_start": "compact.started",
        "compaction_end": "compact.completed",
    }.get(value, value.replace("_", "."))


def _with_artifact(result: ProviderTurnResult, artifact_ref: str) -> ProviderTurnResult:
    from dataclasses import replace

    return replace(
        result,
        artifact_locator=AgentArtifactLocator(
            provider_type="pi",
            home_id=result.session_locator.home_id,
            session_id=result.session_locator.session_id,
            adapter_version=PI_ADAPTER_VERSION,
            native_primary_ref=artifact_ref,
        ),
    )
