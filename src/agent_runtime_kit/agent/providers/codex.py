from __future__ import annotations

import importlib
import json
import shutil
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import Callable, Mapping

from ..store_utils import read_json, utc_now_iso, write_json_atomic


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
class CodexStoredTurnResult:
    id: str
    status: object | None
    error: object | None = None
    started_at: int | None = None
    completed_at: int | None = None
    duration_ms: int | None = None
    final_response: str | None = None
    items: list[object] | None = None
    usage: object | None = None


@dataclass
class CodexThreadSnapshot:
    id: str
    turns: list[CodexStoredTurnResult]
    raw_response: object | None = None
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


class CodexProvider:
    """Synchronous wrapper around the OpenAI Codex Python SDK."""

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
                turn_result = thread.run(prompt, cwd=workdir, model=self.model)
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
                turn_result = thread.run(prompt, cwd=workdir, model=self.model)
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
        turn_result = turn_handle_type(client, resumed_thread_id, turn_id).run()
        return thread, turn_result

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

    def read_thread(
        self,
        agent,
        *,
        home_root: Path,
        env: Mapping[str, str],
        include_turns: bool = True,
    ) -> object:
        if not getattr(agent, "thread_id", None):
            raise RuntimeError("agent has no thread_id")
        self.ensure_home_initialized(
            home_id=str(agent.home_id),
            home_root=home_root,
            env=env,
            workdir=None,
        )
        sdk = self._sdk()
        with self._new_codex(sdk, env=env, workdir=None) as codex:
            thread = codex.thread_resume(str(agent.thread_id), model=self.model)
            raw_response = thread.read(include_turns=include_turns)
        raw_thread = getattr(raw_response, "thread", raw_response)
        raw_turns = list(getattr(raw_thread, "turns", []) or [])
        turns = _coerce_turns(raw_turns)
        if include_turns and not turns:
            rollout = self._agent_rollout_path(agent, home_root=home_root)
            turns = _turns_from_rollout(rollout) if rollout is not None else []
        return CodexThreadSnapshot(
            id=str(getattr(raw_thread, "id", agent.thread_id)),
            turns=turns,
            raw_response=raw_response,
            thread=raw_thread,
        )

    def read_latest_turn_result(
        self,
        agent,
        *,
        home_root: Path,
        env: Mapping[str, str],
    ) -> CodexStoredTurnResult:
        snapshot = self.read_thread(agent, home_root=home_root, env=env, include_turns=True)
        turns = list(getattr(snapshot, "turns", []) or [])
        if not turns:
            raise RuntimeError("thread has no turns")
        return turns[-1]

    def interrupt_agent(self, agent_id: str) -> bool:
        return agent_id in self._agent_runs and False

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
    ) -> None:
        with self._lock:
            run = self._agent_runs.get(agent_id)
            if run is None:
                return
            if thread_id is not None:
                run.thread_id = thread_id
            if turn_id is not None:
                run.turn_id = turn_id

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

    def _agent_home_root(self, agent) -> Path:
        if self.runtime_root is None:
            raise RuntimeError("CodexProvider.runtime_root is required for thread reads")
        return self.runtime_root / "homes" / str(agent.cli_type) / str(agent.home_id)

    def _agent_rollout_path(self, agent, *, home_root: Path | None = None) -> Path | None:
        rollout_relpath = getattr(agent, "rollout_relpath", None)
        resolved_home_root = Path(home_root) if home_root is not None else self._agent_home_root(agent)
        if not rollout_relpath and getattr(agent, "thread_id", None):
            rollout_relpath = _find_rollout_relpath(
                resolved_home_root / ".codex",
                str(agent.thread_id),
            )
        if not rollout_relpath:
            return None
        return resolved_home_root / ".codex" / str(rollout_relpath)

    def find_rollout_relpath(self, *, home_root: Path, thread_id: str) -> str | None:
        return _find_rollout_relpath(Path(home_root) / ".codex", thread_id)


def _find_rollout_relpath(codex_home: Path, thread_id: str) -> str | None:
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
    return str(candidates[0].relative_to(codex_home)) if candidates else None


def _file_contains(path: Path, needle: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if needle in line:
                    return True
    except OSError:
        return False
    return False


def _coerce_turns(turns: list[object]) -> list[CodexStoredTurnResult]:
    return [
        CodexStoredTurnResult(
            id=str(getattr(turn, "id", "")),
            status=getattr(turn, "status", None),
            error=getattr(turn, "error", None),
            started_at=getattr(turn, "started_at", None),
            completed_at=getattr(turn, "completed_at", None),
            duration_ms=getattr(turn, "duration_ms", None),
            final_response=_final_response_from_turn_items(list(getattr(turn, "items", []) or [])),
            items=list(getattr(turn, "items", []) or []),
            usage=getattr(turn, "usage", None),
        )
        for turn in turns
        if getattr(turn, "id", None)
    ]


def _turns_from_rollout(path: Path) -> list[CodexStoredTurnResult]:
    if not path.exists():
        return []
    turns: dict[str, CodexStoredTurnResult] = {}
    order: list[str] = []
    current_turn_id: str | None = None
    for line in path.open("r", encoding="utf-8", errors="ignore"):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        payload = event.get("payload") if isinstance(event, dict) else None
        payload = payload if isinstance(payload, dict) else {}
        event_type = event.get("type")
        payload_type = payload.get("type")
        turn_id = payload.get("turn_id")
        if event_type == "turn_context" and turn_id:
            current_turn_id = str(turn_id)
            _ensure_rollout_turn(turns, order, current_turn_id, status="inProgress")
            continue
        if payload_type == "task_started" and turn_id:
            current_turn_id = str(turn_id)
            _ensure_rollout_turn(turns, order, current_turn_id, status="inProgress")
            continue
        if payload_type == "agent_message" and current_turn_id:
            message = payload.get("message")
            if isinstance(message, str):
                _ensure_rollout_turn(turns, order, current_turn_id, status="inProgress").final_response = message
            continue
        if payload_type == "task_complete" and turn_id:
            current_turn_id = str(turn_id)
            turn = _ensure_rollout_turn(turns, order, current_turn_id, status="completed")
            turn.status = "completed"
            turn.completed_at = payload.get("completed_at")
            turn.duration_ms = payload.get("duration_ms")
            message = payload.get("last_agent_message")
            if isinstance(message, str):
                turn.final_response = message
            continue
        if payload_type == "turn_aborted" and turn_id:
            current_turn_id = str(turn_id)
            _ensure_rollout_turn(turns, order, current_turn_id, status="interrupted").status = "interrupted"
    return [turns[turn_id] for turn_id in order]


def _ensure_rollout_turn(
    turns: dict[str, CodexStoredTurnResult],
    order: list[str],
    turn_id: str,
    *,
    status: str,
) -> CodexStoredTurnResult:
    if turn_id not in turns:
        turns[turn_id] = CodexStoredTurnResult(id=turn_id, status=status, items=[])
        order.append(turn_id)
    return turns[turn_id]


def _final_response_from_turn_items(items: list[object]) -> str | None:
    for item in reversed(items):
        root = getattr(item, "root", item)
        item_type = getattr(root, "type", None)
        phase = getattr(root, "phase", None)
        if item_type == "agentMessage" and (phase is None or str(phase).endswith("final_answer")):
            text = getattr(root, "text", None)
            if isinstance(text, str):
                return text
    return None
