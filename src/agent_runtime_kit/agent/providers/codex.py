from __future__ import annotations

import importlib
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Callable, Mapping

from ..provider_contracts import ProviderContextCompactionResult, ProviderContextUsage
from ..models import (
    AgentContextCompactionEvidenceError,
    AgentContextCompactionRequestUnknown,
    AgentContextCompactionTimeout,
)
from ..store_utils import read_json, utc_now_iso, write_json_atomic
from .codex_context import (
    CodexCompactBaseline,
    capture_codex_compact_baseline,
    inspect_codex_compact_evidence,
    inspect_codex_rollout_context,
)


class CodexSdkUnavailable(RuntimeError):
    pass


@dataclass
class CodexTurnResult:
    thread_id: str
    rollout_relpath: str | None
    turn_result: object
    thread: object | None = None


@dataclass
class CodexForkResult:
    thread_id: str
    rollout_relpath: str | None
    thread: object | None = None


@dataclass
class CodexHomeInitializationRecord:
    home_id: str
    home_root: str
    initialized_at: str
    marker_path: str


@dataclass
class _CodexAgentRun:
    agent_id: str
    home_id: str
    thread_id: str | None = None
    turn_id: str | None = None
    turn_handle: object | None = None


class CodexProvider:
    """Synchronous wrapper around the OpenAI Codex Python SDK."""

    provider_type = "codex"

    REQUIRED_STATE_DATABASES = (
        "state_5.sqlite",
        "logs_2.sqlite",
        "goals_1.sqlite",
        "memories_1.sqlite",
    )

    def __init__(
        self,
        *,
        runtime_root: Path | None = None,
        codex_bin: str | None = None,
        sdk_python_root: Path | None = None,
        model: str | None = None,
        thread_config: dict[str, object] | None = None,
        approval_mode: str = "deny_all",
    ) -> None:
        self.runtime_root = Path(runtime_root) if runtime_root is not None else None
        self.codex_bin = codex_bin
        self.sdk_python_root = Path(sdk_python_root) if sdk_python_root is not None else None
        self.model = model
        self.thread_config = dict(thread_config or {})
        self.approval_mode = approval_mode
        self._agent_runs: dict[str, _CodexAgentRun] = {}
        self._home_init_locks: dict[tuple[str, str], threading.Lock] = {}
        self._lock = threading.RLock()

    def build_provider_bundle(self, *, runtime_root: Path):
        from .codex_bundle import build_codex_provider_bundle

        return build_codex_provider_bundle(self, runtime_root=runtime_root)

    def start_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        workdir: str | None,
        prompt: str,
        developer_instructions: str | None,
        agent_id: str,
        overwrite_developer_instructions: bool = False,
        on_thread_started: Callable[[str], None] | None = None,
        on_turn_started: Callable[[str, str], None] | None = None,
    ) -> CodexTurnResult:
        self.ensure_home_initialized(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        self._begin_agent_run(home_id=home_id, agent_id=agent_id)
        try:
            sdk = self._sdk()
            with self._new_codex(sdk, env=env, workdir=workdir) as codex:
                thread = codex.thread_start(
                    cwd=workdir,
                    developer_instructions=developer_instructions,
                    model=self.model,
                    config=self.thread_config or None,
                    approval_mode=self._sdk_approval_mode(sdk),
                )
                self._update_agent_run_locator(agent_id, thread_id=thread.id)
                if on_thread_started is not None:
                    on_thread_started(thread.id)
                turn_result = self._run_turn_with_control_handle(
                    thread,
                    prompt=prompt,
                    workdir=workdir,
                    agent_id=agent_id,
                    on_turn_started=on_turn_started,
                )
                rollout_relpath = _find_rollout_relpath(Path(home_root) / ".codex", thread.id)
                return CodexTurnResult(
                    thread_id=thread.id,
                    rollout_relpath=rollout_relpath,
                    turn_result=turn_result,
                    thread=thread,
                )
        finally:
            self._finish_agent_run(agent_id)

    def resume_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str,
        workdir: str | None,
        prompt: str,
        developer_instructions: str | None,
        agent_id: str,
        overwrite_developer_instructions: bool = False,
        on_turn_started: Callable[[str, str], None] | None = None,
    ) -> CodexTurnResult:
        self.ensure_home_initialized(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        self._begin_agent_run(home_id=home_id, agent_id=agent_id, thread_id=thread_id)
        try:
            sdk = self._sdk()
            with self._new_codex(sdk, env=env, workdir=workdir) as codex:
                if overwrite_developer_instructions:
                    thread, turn_result = self._resume_and_run_with_developer_instructions_override(
                        sdk=sdk,
                        codex=codex,
                        thread_id=thread_id,
                        workdir=workdir,
                        prompt=prompt,
                        developer_instructions=developer_instructions,
                        agent_id=agent_id,
                        on_turn_started=on_turn_started,
                    )
                    rollout_relpath = _find_rollout_relpath(Path(home_root) / ".codex", thread.id)
                    return CodexTurnResult(
                        thread_id=thread.id,
                        rollout_relpath=rollout_relpath,
                        turn_result=turn_result,
                        thread=thread,
                    )
                thread = codex.thread_resume(
                    thread_id,
                    cwd=workdir,
                    developer_instructions=developer_instructions,
                    model=self.model,
                    config=self.thread_config or None,
                )
                turn_result = self._run_turn_with_control_handle(
                    thread,
                    prompt=prompt,
                    workdir=workdir,
                    agent_id=agent_id,
                    on_turn_started=on_turn_started,
                )
                rollout_relpath = _find_rollout_relpath(Path(home_root) / ".codex", thread.id)
                return CodexTurnResult(
                    thread_id=thread.id,
                    rollout_relpath=rollout_relpath,
                    turn_result=turn_result,
                    thread=thread,
                )
        finally:
            self._finish_agent_run(agent_id)

    def _resume_and_run_with_developer_instructions_override(
        self,
        *,
        sdk,
        codex: object,
        thread_id: str,
        workdir: str | None,
        prompt: str,
        developer_instructions: str | None,
        agent_id: str,
        on_turn_started: Callable[[str, str], None] | None,
    ) -> tuple[object, object]:
        client = getattr(codex, "_client", None)
        if client is None:
            raise RuntimeError(
                "Codex developer instruction overwrite requires a Codex SDK object with _client"
            )
        resumed = client.thread_resume(
            thread_id,
            {
                "cwd": workdir,
                "model": self.model,
                "config": self.thread_config or None,
            },
        )
        response_thread = getattr(resumed, "thread", None)
        resumed_thread_id = str(getattr(response_thread, "id", thread_id))
        model = self.model or getattr(resumed, "model", None)
        if not model:
            raise RuntimeError(
                "Codex developer instruction overwrite requires a resolved model from "
                "CodexProvider.model or thread/resume"
            )
        params = {
            "cwd": workdir,
            "model": model,
            "collaborationMode": {
                "mode": "default",
                "settings": {
                    "model": model,
                    "developer_instructions": developer_instructions,
                },
            },
        }
        started = client.turn_start(resumed_thread_id, prompt, params=params)
        turn_handle_type = getattr(sdk, "TurnHandle", None)
        thread_type = getattr(sdk, "Thread", None)
        if turn_handle_type is None or thread_type is None:
            raise RuntimeError(
                "Codex developer instruction overwrite requires a Codex SDK with "
                "Thread and TurnHandle"
            )
        thread = thread_type(client, resumed_thread_id)
        turn = getattr(started, "turn", None)
        turn_id = str(getattr(turn, "id"))
        turn_handle = turn_handle_type(client, resumed_thread_id, turn_id)
        self._update_agent_run_locator(
            agent_id,
            thread_id=resumed_thread_id,
            turn_id=turn_id,
            turn_handle=turn_handle,
        )
        if on_turn_started is not None:
            on_turn_started(resumed_thread_id, turn_id)
        turn_result = turn_handle.run()
        return thread, turn_result

    def _run_turn_with_control_handle(
        self,
        thread: object,
        *,
        prompt: str,
        workdir: str | None,
        agent_id: str,
        on_turn_started: Callable[[str, str], None] | None,
    ) -> object:
        start_turn = getattr(thread, "turn", None)
        if not callable(start_turn):
            raise TypeError("supported Codex SDK must expose Thread.turn()")
        turn_handle = start_turn(prompt, cwd=workdir, model=self.model)
        self._update_agent_run_locator(
            agent_id,
            thread_id=str(getattr(thread, "id")),
            turn_id=str(getattr(turn_handle, "id")),
            turn_handle=turn_handle,
        )
        if on_turn_started is not None:
            on_turn_started(str(getattr(thread, "id")), str(getattr(turn_handle, "id")))
        return turn_handle.run()

    def fork_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str,
        agent_id: str,
    ) -> CodexForkResult:
        self.ensure_home_initialized(home_id=home_id, home_root=home_root, env=env, workdir=None)
        self._begin_agent_run(home_id=home_id, agent_id=agent_id, thread_id=thread_id)
        try:
            sdk = self._sdk()
            with self._new_codex(sdk, env=env, workdir=None) as codex:
                thread = codex.thread_fork(
                    thread_id,
                    model=self.model,
                    config=self.thread_config or None,
                )
                rollout_relpath = _find_rollout_relpath(Path(home_root) / ".codex", thread.id)
                return CodexForkResult(thread_id=thread.id, rollout_relpath=rollout_relpath, thread=thread)
        finally:
            self._finish_agent_run(agent_id)

    def inspect_thread_context(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str,
        workdir: str | None,
        agent_id: str,
    ) -> ProviderContextUsage:
        del home_id, env, workdir, agent_id
        rollout_path = self._rollout_path_for_thread(home_root=home_root, thread_id=thread_id)
        if rollout_path is None:
            return ProviderContextUsage(
                session_id=thread_id,
                observed_at=utc_now_iso(),
                source="artifact",
                available=False,
                measurement="unavailable",
                reason="rollout_missing",
            )
        return inspect_codex_rollout_context(rollout_path, session_id=thread_id)

    def compact_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str,
        workdir: str | None,
        agent_id: str,
        timeout_s: float,
        on_compaction_started: Callable[[dict[str, object], str | None], None] | None = None,
    ) -> ProviderContextCompactionResult:
        self.ensure_home_initialized(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        rollout_path = self._rollout_path_for_thread(home_root=home_root, thread_id=thread_id)
        if rollout_path is None:
            raise AgentContextCompactionEvidenceError(f"Codex rollout is missing for agent: {agent_id}")
        baseline = capture_codex_compact_baseline(rollout_path, session_id=thread_id)
        self._begin_agent_run(home_id=home_id, agent_id=agent_id, thread_id=thread_id)
        try:
            sdk = self._sdk()
            with self._new_codex(sdk, env=env, workdir=workdir) as codex:
                thread = codex.thread_resume(
                    thread_id,
                    cwd=workdir,
                    model=self.model,
                    config=self.thread_config or None,
                )
                status = _codex_thread_status_type(thread.read(include_turns=False))
                if status != "idle":
                    raise AgentContextCompactionEvidenceError(
                        f"Codex thread is not idle before compaction: {thread_id} ({status or 'unknown'})"
                    )
                started_at = utc_now_iso()
                try:
                    response = thread.compact()
                except BaseException as exc:
                    raise AgentContextCompactionRequestUnknown(
                        f"Codex compaction request terminal state is unknown: {agent_id}"
                    ) from exc
                operation_id = _optional_operation_id(response)
                if on_compaction_started is not None:
                    on_compaction_started(baseline.to_dict(), operation_id)
                deadline = monotonic() + timeout_s
                while True:
                    try:
                        evidence = inspect_codex_compact_evidence(
                            rollout_path,
                            session_id=thread_id,
                            baseline=baseline,
                        )
                    except ValueError as exc:
                        raise AgentContextCompactionEvidenceError(str(exc)) from exc
                    if evidence.complete:
                        status = _codex_thread_status_type(thread.read(include_turns=False))
                        if status == "systemError":
                            raise AgentContextCompactionEvidenceError(
                                f"Codex thread entered systemError after compaction: {thread_id}"
                            )
                        if status == "idle":
                            return ProviderContextCompactionResult(
                                session_id=thread_id,
                                status="compacted",
                                reason="provider_confirmed",
                                usage_after=evidence.usage,
                                started_at=started_at,
                                completed_at=utc_now_iso(),
                                provider_operation_id=operation_id,
                            )
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise AgentContextCompactionTimeout(
                            f"Codex compaction did not reach a confirmed idle terminal state: {agent_id}"
                        )
                    sleep(min(0.1, remaining))
        finally:
            self._finish_agent_run(agent_id)

    def reconcile_thread_compaction(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str,
        workdir: str | None,
        agent_id: str,
        baseline: dict[str, object],
        provider_operation_id: str | None,
    ) -> ProviderContextCompactionResult | None:
        del provider_operation_id
        rollout_path = self._rollout_path_for_thread(home_root=home_root, thread_id=thread_id)
        if rollout_path is None:
            return None
        try:
            parsed_baseline = CodexCompactBaseline.from_dict(baseline)
            evidence = inspect_codex_compact_evidence(
                rollout_path,
                session_id=thread_id,
                baseline=parsed_baseline,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentContextCompactionEvidenceError(str(exc)) from exc
        if not evidence.complete:
            return None
        self.ensure_home_initialized(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        self._begin_agent_run(home_id=home_id, agent_id=agent_id, thread_id=thread_id)
        try:
            sdk = self._sdk()
            with self._new_codex(sdk, env=env, workdir=workdir) as codex:
                thread = codex.thread_resume(
                    thread_id,
                    cwd=workdir,
                    model=self.model,
                    config=self.thread_config or None,
                )
                if _codex_thread_status_type(thread.read(include_turns=False)) != "idle":
                    return None
        finally:
            self._finish_agent_run(agent_id)
        now = utc_now_iso()
        return ProviderContextCompactionResult(
            session_id=thread_id,
            status="compacted",
            reason="provider_confirmed",
            usage_after=evidence.usage,
            started_at=now,
            completed_at=now,
        )

    def interrupt_agent(self, agent_id: str) -> bool:
        with self._lock:
            run = self._agent_runs.get(agent_id)
            handle = run.turn_handle if run is not None else None
        if handle is None:
            return False
        return self.interrupt(handle)

    def interrupt(self, handle: object) -> bool:
        if hasattr(handle, "interrupt"):
            result = handle.interrupt()
            return bool(result)
        return False

    def list_active_agents(self, home_id: str | None = None) -> list[str]:
        with self._lock:
            if home_id is None:
                return sorted(self._agent_runs)
            return sorted(
                agent_id
                for agent_id, run in self._agent_runs.items()
                if run.home_id == home_id
            )

    def close(self) -> None:
        return None

    def ensure_home_initialized(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        workdir: str | None,
    ) -> CodexHomeInitializationRecord:
        resolved_home_root = Path(home_root)
        existing = self._read_initialization_marker(home_id, resolved_home_root)
        if existing is not None:
            return existing
        init_lock = self._home_init_lock(home_id, resolved_home_root)
        with init_lock:
            existing = self._read_initialization_marker(home_id, resolved_home_root)
            if existing is not None:
                return existing
            sdk = self._sdk()
            codex_root = resolved_home_root / ".codex"
            codex_root.mkdir(parents=True, exist_ok=True)
            with self._new_codex(sdk, env=env, workdir=workdir) as codex:
                account = getattr(codex, "account", None)
                if callable(account):
                    account()
            missing = self._missing_state_databases(resolved_home_root)
            if missing:
                joined = ", ".join(missing)
                raise RuntimeError(f"Codex home initialization did not create required state databases: {joined}")
            initialized_at = utc_now_iso()
            marker_path = self._home_marker_path(resolved_home_root)
            payload = {
                "schema_version": 1,
                "object_type": "codex_home_initialization",
                "home_id": home_id,
                "home_root": str(resolved_home_root),
                "initialized_at": initialized_at,
                "required_state_databases": list(self.REQUIRED_STATE_DATABASES),
            }
            write_json_atomic(marker_path, payload)
            return CodexHomeInitializationRecord(
                home_id=home_id,
                home_root=str(resolved_home_root),
                initialized_at=initialized_at,
                marker_path=str(marker_path),
            )

    def _begin_agent_run(
        self,
        *,
        home_id: str,
        agent_id: str,
        thread_id: str | None = None,
    ) -> None:
        with self._lock:
            if agent_id in self._agent_runs:
                raise RuntimeError(f"agent is already active in CodexProvider: {agent_id}")
            self._agent_runs[agent_id] = _CodexAgentRun(
                agent_id=agent_id,
                home_id=home_id,
                thread_id=thread_id,
            )

    def _finish_agent_run(
        self,
        agent_id: str,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        with self._lock:
            run = self._agent_runs.pop(agent_id, None)
            if run is None:
                return
            if thread_id is not None:
                run.thread_id = thread_id
            if turn_id is not None:
                run.turn_id = turn_id

    def _update_agent_run_locator(
        self,
        agent_id: str,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
        turn_handle: object | None = None,
    ) -> None:
        with self._lock:
            run = self._agent_runs.get(agent_id)
            if run is None:
                return
            if thread_id is not None:
                run.thread_id = thread_id
            if turn_id is not None:
                run.turn_id = turn_id
            if turn_handle is not None:
                run.turn_handle = turn_handle

    def _home_init_lock(self, home_id: str, home_root: Path) -> threading.Lock:
        key = (home_id, str(home_root.resolve()))
        with self._lock:
            lock = self._home_init_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._home_init_locks[key] = lock
            return lock

    def _read_initialization_marker(
        self,
        home_id: str,
        home_root: Path,
    ) -> CodexHomeInitializationRecord | None:
        marker_path = self._home_marker_path(home_root)
        if not marker_path.exists():
            return None
        missing = self._missing_state_databases(home_root)
        if missing:
            return None
        payload = read_json(marker_path)
        if str(payload.get("home_id", "")) != home_id:
            return None
        return CodexHomeInitializationRecord(
            home_id=str(payload["home_id"]),
            home_root=str(payload.get("home_root", home_root)),
            initialized_at=str(payload.get("initialized_at", "")),
            marker_path=str(marker_path),
        )

    def _missing_state_databases(self, home_root: Path) -> list[str]:
        codex_root = Path(home_root) / ".codex"
        return [name for name in self.REQUIRED_STATE_DATABASES if not (codex_root / name).exists()]

    def _home_marker_path(self, home_root: Path) -> Path:
        return Path(home_root) / ".ark" / "codex_home_initialized.json"

    @contextmanager
    def _new_codex(self, sdk, *, env: Mapping[str, str], workdir: str | None) -> Iterator[object]:
        codex = sdk.Codex(config=self._sdk_config(sdk, env=env, workdir=workdir))
        enter = getattr(codex, "__enter__", None)
        if callable(enter):
            with codex as entered:
                yield entered
            return
        try:
            yield codex
        finally:
            self._close_codex_object(codex)

    def _close_codex_object(self, codex: object) -> None:
        close = getattr(codex, "close", None)
        if callable(close):
            close()

    def _sdk(self):
        if self.sdk_python_root is not None:
            src = self.sdk_python_root / "src"
            if str(src) not in sys.path:
                sys.path.insert(0, str(src))
        try:
            return importlib.import_module("openai_codex")
        except ImportError as exc:
            raise CodexSdkUnavailable(
                "OpenAI Codex Python SDK is not available. Install openai-codex "
                "or pass sdk_python_root pointing at the Codex repo sdk/python directory."
            ) from exc

    def _sdk_config(self, sdk, *, env: Mapping[str, str], workdir: str | None):
        codex_bin = self.codex_bin or shutil.which("codex")
        return sdk.CodexConfig(
            codex_bin=codex_bin,
            cwd=workdir,
            env=dict(env),
        )

    def _sdk_approval_mode(self, sdk):
        approval_mode_type = getattr(sdk, "ApprovalMode", None)
        if approval_mode_type is None:
            return self.approval_mode
        try:
            return approval_mode_type(self.approval_mode)
        except (TypeError, ValueError):
            return getattr(approval_mode_type, self.approval_mode)

    def _rollout_path_for_thread(self, *, home_root: Path, thread_id: str) -> Path | None:
        rollout_relpath = _find_rollout_relpath(
            Path(home_root) / ".codex",
            thread_id,
            allow_fallback=False,
        )
        if rollout_relpath is None:
            return None
        return Path(home_root) / ".codex" / rollout_relpath

    def find_rollout_relpath(self, *, home_root: Path, thread_id: str) -> str | None:
        return _find_rollout_relpath(Path(home_root) / ".codex", thread_id)


def _find_rollout_relpath(
    codex_home: Path,
    thread_id: str,
    *,
    allow_fallback: bool = True,
) -> str | None:
    sessions_root = codex_home / "sessions"
    if not sessions_root.exists():
        return None
    candidates = sorted(sessions_root.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        if thread_id in path.name:
            return str(path.relative_to(codex_home))
    needle = f'"{thread_id}"'
    for path in candidates:
        if _file_contains(path, needle):
            return str(path.relative_to(codex_home))
    if allow_fallback and candidates:
        return str(candidates[0].relative_to(codex_home))
    return None


def _codex_thread_status_type(response: object) -> str | None:
    thread = getattr(response, "thread", response)
    status = getattr(thread, "status", None)
    root = getattr(status, "root", status)
    value = getattr(root, "type", None)
    return value if isinstance(value, str) else None


def _optional_operation_id(response: object) -> str | None:
    for name in ("operation_id", "operationId", "id"):
        value = getattr(response, name, None)
        if isinstance(value, str) and value:
            return value
    return None


def _file_contains(path: Path, needle: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if needle in line:
                    return True
    except OSError:
        return False
    return False
