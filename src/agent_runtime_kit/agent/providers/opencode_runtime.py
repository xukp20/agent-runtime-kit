from __future__ import annotations

import hashlib
import os
import re
import secrets
import shutil
import socket
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
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
    ProviderExecutionContext,
    ProviderRunRequest,
    ProviderRunState,
    ProviderSessionLocator,
    ProviderTurnLocator,
    ProviderTurnResult,
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .opencode_client import OpenCodeClient, OpenCodeClientError, _safe_body, event_properties
from .opencode_home import MUTABLE_CONFIG_RUNTIME_NAMES
from .opencode_models import (
    ADAPTER_VERSION,
    OpenCodeNativeLocator,
    OpenCodeRunOptions,
    OpenCodeTransientRetryPolicy,
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
    config_root: Path
    process: subprocess.Popen[str]
    client: OpenCodeClient
    password: str
    environment_fingerprint: str
    validated_variants: set[tuple[str, str, str]] = field(default_factory=set)
    ready_mcp_names: set[str] = field(default_factory=set)

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


@dataclass(frozen=True)
class OpenCodeMcpReadinessPolicy:
    max_attempts: int = 3
    initial_delay_s: float = 0.25
    max_delay_s: float = 0.5

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("OpenCode MCP readiness max_attempts must be positive")
        if self.initial_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("OpenCode MCP readiness delays must be non-negative")


@dataclass(frozen=True)
class OpenCodeMcpReadinessResult:
    required_mcp_names: tuple[str, ...]
    attempts: int
    elapsed_s: float
    reconnected: tuple[str, ...] = ()
    cached: bool = False


class OpenCodeMcpReadinessError(RuntimeError):
    def __init__(
        self,
        *,
        agent_id: str,
        required_mcp_names: tuple[str, ...],
        statuses: Mapping[str, str],
        attempts: int,
        elapsed_s: float,
        retryable: bool,
        reconnected: tuple[str, ...] = (),
        reason: str,
    ) -> None:
        self.agent_id = agent_id
        self.required_mcp_names = required_mcp_names
        self.statuses = dict(statuses)
        self.attempts = attempts
        self.elapsed_s = elapsed_s
        self.retryable = retryable
        self.reconnected = reconnected
        self.process_healthy = True
        self.model_turn_started = False
        super().__init__(
            "OpenCode required MCP readiness failed before model turn "
            f"for {agent_id} after {attempts} attempt(s): {_safe_body(reason)}"
        )

    def event_data(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "required_mcp_names": list(self.required_mcp_names),
            "statuses": dict(self.statuses),
            "attempts": self.attempts,
            "elapsed_s": self.elapsed_s,
            "retryable": self.retryable,
            "reconnected": list(self.reconnected),
            "process_healthy": self.process_healthy,
            "model_turn_started": self.model_turn_started,
        }


def _ensure_required_mcp_ready(
    server: OpenCodeServer,
    *,
    required_mcp_names: tuple[str, ...],
    policy: OpenCodeMcpReadinessPolicy | None = None,
) -> OpenCodeMcpReadinessResult:
    required = tuple(sorted(set(required_mcp_names)))
    pending = tuple(name for name in required if name not in server.ready_mcp_names)
    if not pending:
        return OpenCodeMcpReadinessResult(
            required_mcp_names=required,
            attempts=0,
            elapsed_s=0.0,
            cached=True,
        )
    resolved_policy = policy or OpenCodeMcpReadinessPolicy()
    started = time.monotonic()
    reconnected: set[str] = set()
    last_statuses: dict[str, str] = {}
    last_reason = "required MCP status was not observed"
    for attempt in range(1, resolved_policy.max_attempts + 1):
        try:
            status_map = server.client.mcp_status()
        except OpenCodeClientError as exc:
            retryable = _opencode_client_error_is_retryable(exc)
            last_reason = str(exc)
            if not retryable or attempt == resolved_policy.max_attempts:
                raise OpenCodeMcpReadinessError(
                    agent_id=server.agent_id,
                    required_mcp_names=required,
                    statuses=last_statuses,
                    attempts=attempt,
                    elapsed_s=time.monotonic() - started,
                    retryable=retryable,
                    reconnected=tuple(sorted(reconnected)),
                    reason=last_reason,
                ) from exc
            _sleep_mcp_backoff(resolved_policy, attempt)
            continue

        missing = [name for name in pending if name not in status_map]
        if missing:
            raise OpenCodeMcpReadinessError(
                agent_id=server.agent_id,
                required_mcp_names=required,
                statuses=last_statuses,
                attempts=attempt,
                elapsed_s=time.monotonic() - started,
                retryable=False,
                reconnected=tuple(sorted(reconnected)),
                reason=f"required MCP server is not configured: {', '.join(missing)}",
            )

        retry_names: list[str] = []
        deterministic: list[str] = []
        for name in pending:
            raw = status_map[name]
            if not isinstance(raw, Mapping):
                deterministic.append(f"{name}=invalid_status")
                last_statuses[name] = "invalid_status"
                continue
            status = str(raw.get("status") or "unknown")
            last_statuses[name] = status
            if status == "connected":
                continue
            if status in {"disabled", "needs_auth", "needs_client_registration"}:
                deterministic.append(f"{name}={status}")
                continue
            if status == "failed" and not _mcp_failure_is_retryable(str(raw.get("error") or "")):
                deterministic.append(f"{name}=failed")
                last_reason = str(raw.get("error") or "deterministic MCP connection failure")
                continue
            if status in {"failed", "connecting"}:
                retry_names.append(name)
                last_reason = str(raw.get("error") or status)
                continue
            deterministic.append(f"{name}={status}")

        if deterministic:
            raise OpenCodeMcpReadinessError(
                agent_id=server.agent_id,
                required_mcp_names=required,
                statuses=last_statuses,
                attempts=attempt,
                elapsed_s=time.monotonic() - started,
                retryable=False,
                reconnected=tuple(sorted(reconnected)),
                reason=last_reason if "failed" in deterministic else ", ".join(deterministic),
            )
        if not retry_names:
            server.ready_mcp_names.update(required)
            return OpenCodeMcpReadinessResult(
                required_mcp_names=required,
                attempts=attempt,
                elapsed_s=time.monotonic() - started,
                reconnected=tuple(sorted(reconnected)),
            )
        if attempt == resolved_policy.max_attempts:
            raise OpenCodeMcpReadinessError(
                agent_id=server.agent_id,
                required_mcp_names=required,
                statuses=last_statuses,
                attempts=attempt,
                elapsed_s=time.monotonic() - started,
                retryable=True,
                reconnected=tuple(sorted(reconnected)),
                reason=last_reason,
            )
        for name in retry_names:
            try:
                server.client.connect_mcp(name)
            except OpenCodeClientError as exc:
                if not _opencode_client_error_is_retryable(exc):
                    raise OpenCodeMcpReadinessError(
                        agent_id=server.agent_id,
                        required_mcp_names=required,
                        statuses=last_statuses,
                        attempts=attempt,
                        elapsed_s=time.monotonic() - started,
                        retryable=False,
                        reconnected=tuple(sorted(reconnected)),
                        reason=str(exc),
                    ) from exc
                last_reason = str(exc)
            reconnected.add(name)
        _sleep_mcp_backoff(resolved_policy, attempt)
    raise AssertionError("unreachable OpenCode MCP readiness state")


def _sleep_mcp_backoff(policy: OpenCodeMcpReadinessPolicy, attempt: int) -> None:
    delay = min(policy.max_delay_s, policy.initial_delay_s * (2 ** (attempt - 1)))
    if delay > 0:
        time.sleep(delay)


def _opencode_client_error_is_retryable(error: OpenCodeClientError) -> bool:
    return error.status is None or error.status in {408, 409, 429} or (
        error.status is not None and 500 <= error.status < 600
    )


def _mcp_failure_is_retryable(reason: str) -> bool:
    normalized = reason.casefold()
    deterministic_markers = (
        "401",
        "403",
        "authentication",
        "unauthorized",
        "forbidden",
        "invalid url",
        "malformed",
    )
    return not any(marker in normalized for marker in deterministic_markers)


class OpenCodeRuntimeRegistry:
    provider_type = PROVIDER_TYPE

    def __init__(
        self,
        runtime_root: Path,
        *,
        binary_path: str | Path = "opencode",
        transient_retry_policy: OpenCodeTransientRetryPolicy | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.binary_path = str(binary_path)
        self.transient_retry_policy = (
            transient_retry_policy or OpenCodeTransientRetryPolicy()
        )
        self._servers: dict[str, OpenCodeServer] = {}
        self._lock = threading.RLock()

    def ensure(self, request: ProviderRunRequest) -> OpenCodeServer:
        with self._lock:
            existing = self._servers.get(request.agent_id)
            if existing is not None and existing.process.poll() is None:
                if request.workdir and Path(existing.directory).resolve() != Path(request.workdir).resolve():
                    raise ValueError("OpenCode Agent runtime cannot change workdir while its server is active")
                if request.session_locator is not None:
                    native = parse_native_locator(request.session_locator.native_locator)
                    if str(existing.database_path) != native.database_path:
                        raise RuntimeError(
                            "OpenCode session locator database does not match active Agent runtime"
                        )
                desired_environment = _environment_fingerprint(request)
                if (
                    existing.environment_fingerprint == desired_environment
                    or not _has_ark_runtime_identity(request)
                ):
                    return existing
                existing.close()
                self._servers.pop(request.agent_id, None)
            server = self._start_server(request)
            if request.session_locator is not None:
                native = parse_native_locator(request.session_locator.native_locator)
                if str(server.database_path) != native.database_path:
                    server.close()
                    raise RuntimeError(
                        "OpenCode session locator database does not match Agent runtime"
                    )
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

    def ensure_client_for_locator(
        self,
        locator: ProviderSessionLocator,
        *,
        agent_id: str,
        execution_context: ProviderExecutionContext,
    ) -> OpenCodeClient:
        native = parse_native_locator(locator.native_locator)
        if agent_id != native.agent_id:
            raise ValueError("OpenCode context Agent does not match session locator")
        if (
            execution_context.provider_type != PROVIDER_TYPE
            or execution_context.home_id != locator.home_id
        ):
            raise ValueError("OpenCode context does not match session locator")
        context_workdir = execution_context.workdir
        if context_workdir and Path(str(context_workdir)).resolve() != Path(native.directory).resolve():
            raise ValueError("OpenCode context workdir does not match session locator")
        server = self.ensure(
            ProviderRunRequest(
                agent_id=agent_id,
                scope_id="provider-session-bootstrap",
                agent_type="provider-session-bootstrap",
                provider_type=PROVIDER_TYPE,
                home_id=locator.home_id,
                prompt="",
                session_locator=locator,
                workdir=native.directory,
                environment=execution_context.process_environment,
                model_overrides=execution_context.resolved_defaults,
                execution_context=execution_context,
            )
        )
        if str(server.database_path) != native.database_path:
            raise RuntimeError("OpenCode locator database does not match bootstrapped Agent runtime")
        return server.client

    def prepare_session_access(
        self,
        locator: ProviderSessionLocator,
        *,
        agent_id: str,
        execution_context: ProviderExecutionContext,
    ) -> None:
        self.ensure_client_for_locator(
            locator,
            agent_id=agent_id,
            execution_context=execution_context,
        )

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
        config_root = self._prepare_config_runtime(context)
        paths = {
            "home": runtime / "home",
            "data": runtime / "xdg-data",
            "cache": runtime / "xdg-cache",
            "state": runtime / "xdg-state",
            "tmp": runtime / "tmp",
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        shared_cache_root = self.runtime_root / "providers" / PROVIDER_TYPE / "shared-cache"
        npm_cache = shared_cache_root / "npm"
        bun_cache = shared_cache_root / "bun"
        npm_cache.mkdir(parents=True, exist_ok=True)
        bun_cache.mkdir(parents=True, exist_ok=True)
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
                "XDG_CONFIG_HOME": str(config_root.parent),
                "XDG_DATA_HOME": str(paths["data"]),
                "XDG_CACHE_HOME": str(paths["cache"]),
                "XDG_STATE_HOME": str(paths["state"]),
                "TMPDIR": str(paths["tmp"]),
                "OPENCODE_DB": str(database),
                "OPENCODE_SERVER_PASSWORD": password,
                "OPENCODE_CONFIG_DIR": str(config_root),
                "NPM_CONFIG_CACHE": str(npm_cache),
                "npm_config_cache": str(npm_cache),
                "BUN_INSTALL_CACHE_DIR": str(bun_cache),
            }
        )
        auth_source = context.home_root / ".opencode" / "auth.json"
        auth_target = paths["data"] / "opencode" / "auth.json"
        if auth_source.is_file():
            auth_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(auth_source, auth_target)
            auth_target.chmod(0o600)
        elif auth_target.exists():
            auth_target.unlink()
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
            config_root=config_root,
            process=process,
            client=client,
            password=password,
            environment_fingerprint=_environment_fingerprint(request),
        )

    def _prepare_config_runtime(self, context: ProviderExecutionContext) -> Path:
        manifest_hash = str(context.runtime_payload.get("materialization_manifest_hash") or "")
        if len(manifest_hash) != 64 or any(char not in "0123456789abcdef" for char in manifest_hash):
            raise RuntimeError("OpenCode runtime requires the current Home materialization hash")
        runtime = (
            self.runtime_root
            / "providers"
            / PROVIDER_TYPE
            / "home-runtimes"
            / manifest_hash
        )
        config_root = runtime / "xdg-config" / "opencode"
        marker = runtime / "source-home-manifest"
        if (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == manifest_hash
            and config_root.joinpath("opencode.json").is_file()
        ):
            return config_root

        config_root.mkdir(parents=True, exist_ok=True)
        for path in tuple(config_root.iterdir()):
            if path.name in MUTABLE_CONFIG_RUNTIME_NAMES:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        for source in context.home_root.iterdir():
            if source.name in MUTABLE_CONFIG_RUNTIME_NAMES or source.name in {".ark", ".opencode"}:
                continue
            target = config_root / source.name
            if source.is_dir() and not source.is_symlink():
                shutil.copytree(source, target, symlinks=True)
            else:
                shutil.copy2(source, target, follow_symlinks=False)
        marker.write_text(manifest_hash + "\n", encoding="utf-8")
        return config_root


def _environment_fingerprint(request: ProviderRunRequest) -> str:
    context = request.execution_context
    if context is None:
        raise ValueError("OpenCode runtime requires a ProviderExecutionContext")
    environment = dict(context.process_environment)
    environment.update(request.environment)
    digest = hashlib.sha256()
    for name, value in sorted(environment.items()):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _has_ark_runtime_identity(request: ProviderRunRequest) -> bool:
    environment = (
        dict(request.execution_context.process_environment)
        if request.execution_context
        else {}
    )
    environment.update(request.environment)
    return all(environment.get(name) for name in ("ARK_STEP_ID", "ARK_FLOW_ID", "ARK_AGENT_ID"))


def _required_mcp_names(context: ProviderExecutionContext | None) -> tuple[str, ...]:
    if context is None or not isinstance(context.runtime_payload, Mapping):
        return ()
    raw = context.runtime_payload.get("required_mcp_names", ())
    if not isinstance(raw, (list, tuple)) or any(
        not isinstance(name, str) or not name.strip()
        for name in raw
    ):
        raise RuntimeError("OpenCode execution context has invalid required MCP names")
    return tuple(sorted(set(raw)))


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
            required_mcp_names = _required_mcp_names(self.request.execution_context)
            try:
                readiness = _ensure_required_mcp_ready(
                    server,
                    required_mcp_names=required_mcp_names,
                )
            except OpenCodeMcpReadinessError as exc:
                self._append_event("mcp.readiness_failed", data=exc.event_data())
                raise
            if required_mcp_names:
                self._append_event(
                    "mcp.ready",
                    data={
                        "required_mcp_names": list(required_mcp_names),
                        "attempts": readiness.attempts,
                        "elapsed_s": readiness.elapsed_s,
                        "reconnected": list(readiness.reconnected),
                        "cached": readiness.cached,
                        "model_turn_started": False,
                    },
                )
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
            expected_variant = self._submit_prompt_attempt(server, turn_id)
            attempt = 1
            retry_policy = getattr(
                self.registry,
                "transient_retry_policy",
                OpenCodeTransientRetryPolicy(),
            )
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
                    classification = _classify_transient_opencode_error(self._provider_error)
                    has_tool_side_effect = _turn_has_tool_side_effect(messages, turn_id)
                    if classification is not None and has_tool_side_effect:
                        self._append_event(
                            "turn.retry_blocked",
                            data={
                                "classification": classification,
                                "failed_attempt": attempt,
                                "max_attempts": retry_policy.max_attempts,
                                "reason": "tool_side_effect_or_uncertain_tool_state",
                                "session_id": self._session.session_id,
                                "turn_id": turn_id,
                            },
                        )
                    elif classification is not None and attempt < retry_policy.max_attempts:
                        delay_s = min(
                            retry_policy.max_delay_s,
                            retry_policy.initial_delay_s * (2 ** (attempt - 1)),
                        )
                        self._append_event(
                            "turn.retry_scheduled",
                            data={
                                "classification": classification,
                                "failed_attempt": attempt,
                                "next_attempt": attempt + 1,
                                "max_attempts": retry_policy.max_attempts,
                                "delay_s": delay_s,
                                "session_id": self._session.session_id,
                                "turn_id": turn_id,
                                "has_tool_side_effect": False,
                            },
                        )
                        if delay_s > 0:
                            time.sleep(delay_s)
                        attempt += 1
                        self._provider_error = None
                        self._turn_seen.clear()
                        self._armed.clear()
                        turn_id = _message_id()
                        self._turn = ProviderTurnLocator(
                            session=self._session,
                            turn_id=turn_id,
                        )
                        expected_variant = self._submit_prompt_attempt(server, turn_id)
                        continue
                    elif classification is not None:
                        original = self._provider_error
                        self._provider_error = AgentError(
                            error_type="opencode_transient_provider_exhausted",
                            message=(
                                f"OpenCode transient provider failure exhausted "
                                f"{attempt} attempts: {_safe_body(original.message)}"
                            ),
                            provider_payload=build_provider_payload(
                                provider_type=PROVIDER_TYPE,
                                payload_type="transient_provider_exhausted",
                                data={
                                    "classification": classification,
                                    "attempts": attempt,
                                    "last_error_type": original.error_type,
                                },
                                adapter_version=ADAPTER_VERSION,
                            ),
                        )
                        self._append_event(
                            "turn.retry_exhausted",
                            data={
                                "classification": classification,
                                "attempts": attempt,
                                "max_attempts": retry_policy.max_attempts,
                                "session_id": self._session.session_id,
                                "turn_id": turn_id,
                            },
                        )
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
                    if expected_variant is not None:
                        _require_persisted_variant(messages, turn_id, expected_variant)
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

    def _submit_prompt_attempt(self, server: OpenCodeServer, turn_id: str) -> str | None:
        assert self._session is not None
        prompt_payload = self._prompt_payload(turn_id)
        expected_variant = _prompt_variant(prompt_payload)
        if expected_variant is not None:
            provider_id, model_id = _prompt_model(prompt_payload)
            variant_key = (provider_id, model_id, expected_variant)
            if variant_key not in server.validated_variants:
                _require_supported_variant(
                    server.client.list_providers(),
                    provider_id=provider_id,
                    model_id=model_id,
                    variant=expected_variant,
                )
                server.validated_variants.add(variant_key)
        server.client.prompt_async(self._session.session_id, prompt_payload)
        self._armed.set()
        return expected_variant

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
        configured_variant = backend.reasoning_effort if backend is not None else None
        if options.variant and configured_variant and options.variant != configured_variant:
            raise ValueError(
                "OpenCode run variant conflicts with configured model reasoning effort: "
                f"{options.variant!r} != {configured_variant!r}"
            )
        effective_variant = options.variant or configured_variant
        if effective_variant:
            payload["variant"] = effective_variant
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
        if request.execution_context is None:
            raise ValueError("OpenCode fork requires ProviderExecutionContext")
        source_client = self.registry.ensure_client_for_locator(
            request.source_session,
            agent_id=request.source_agent_id,
            execution_context=request.execution_context,
        )
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


def _turn_has_tool_side_effect(messages: list[object], turn_id: str) -> bool:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        info = message.get("info") if isinstance(message.get("info"), Mapping) else message
        if info.get("role") != "assistant" or str(info.get("parentID") or "") != turn_id:
            continue
        parts = message.get("parts") if isinstance(message.get("parts"), list) else ()
        for part in parts:
            if not isinstance(part, Mapping) or part.get("type") != "tool":
                continue
            state = part.get("state") if isinstance(part.get("state"), Mapping) else {}
            if str(state.get("status") or "unknown") != "pending":
                return True
    return False


def _classify_transient_opencode_error(error: AgentError) -> str | None:
    message = error.message.casefold()
    non_retryable_markers = (
        "unauthorized",
        "forbidden",
        "invalid api key",
        "invalid model",
        "model not found",
        "content filter",
        "content_filter",
        "bad request",
        "permission",
        "question",
        "mcp",
        "tool error",
    )
    if any(marker in message for marker in non_retryable_markers):
        return None
    if "model unavailable" in message or "model is unavailable" in message:
        return "model_unavailable"
    if "capacity" in message or "temporarily unavailable" in message:
        return "provider_capacity"
    if "rate limit" in message or re.search(r"\b429\b", message):
        return "rate_limited"
    status = re.search(r"\b(408|409|5\d\d)\b", message)
    if status is not None:
        return f"provider_http_{status.group(1)}"
    if "request timeout" in message or "timed out" in message:
        return "provider_timeout"
    return None


def _prompt_model(payload: Mapping[str, object]) -> tuple[str, str]:
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError("OpenCode prompt payload is missing model identity")
    provider_id = model.get("providerID")
    model_id = model.get("modelID")
    if not isinstance(provider_id, str) or not provider_id:
        raise RuntimeError("OpenCode prompt payload is missing providerID")
    if not isinstance(model_id, str) or not model_id:
        raise RuntimeError("OpenCode prompt payload is missing modelID")
    return provider_id, model_id


def _prompt_variant(payload: Mapping[str, object]) -> str | None:
    value = payload.get("variant")
    return value if isinstance(value, str) and value else None


def _require_supported_variant(
    catalog: Mapping[str, object],
    *,
    provider_id: str,
    model_id: str,
    variant: str,
) -> None:
    providers = catalog.get("all")
    if not isinstance(providers, list):
        raise RuntimeError("OpenCode provider catalog is missing the all list")
    provider = next(
        (
            item
            for item in providers
            if isinstance(item, Mapping) and item.get("id") == provider_id
        ),
        None,
    )
    if provider is None:
        raise ValueError(f"OpenCode provider catalog does not contain {provider_id!r}")
    models = provider.get("models")
    model = models.get(model_id) if isinstance(models, Mapping) else None
    if not isinstance(model, Mapping):
        raise ValueError(
            f"OpenCode provider {provider_id!r} does not contain model {model_id!r}"
        )
    variants = model.get("variants")
    if not isinstance(variants, Mapping) or variant not in variants:
        raise ValueError(
            f"OpenCode model {provider_id}/{model_id} does not support variant {variant!r}"
        )


def _require_persisted_variant(
    messages: list[object],
    turn_id: str,
    expected_variant: str,
) -> None:
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        info = message.get("info") if isinstance(message.get("info"), Mapping) else message
        if info.get("role") != "user" or str(info.get("id") or "") != turn_id:
            continue
        model = info.get("model")
        persisted_variant = model.get("variant") if isinstance(model, Mapping) else None
        if persisted_variant != expected_variant:
            raise RuntimeError(
                "OpenCode persisted user message variant does not match the requested variant: "
                f"{persisted_variant!r} != {expected_variant!r}"
            )
        return
    raise RuntimeError("OpenCode completed turn is missing its persisted user message")


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
