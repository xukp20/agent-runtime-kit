from __future__ import annotations

import importlib
import json
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..store_utils import utc_now_iso


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
class _CodexHomeState:
    home_id: str
    home_root: Path
    codex: object
    active_agent_ids: set[str] = field(default_factory=set)
    last_used_at: str = ""
    closed: bool = False


@dataclass
class _CodexAgentRun:
    agent_id: str
    home_id: str
    thread_id: str | None = None
    turn_id: str | None = None


class CodexProvider:
    """Synchronous wrapper around the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        runtime_root: Path | None = None,
        codex_bin: str | None = None,
        sdk_python_root: Path | None = None,
        model: str | None = None,
        thread_config: dict[str, object] | None = None,
        max_idle_homes: int = 8,
    ) -> None:
        self.runtime_root = Path(runtime_root) if runtime_root is not None else None
        self.codex_bin = codex_bin
        self.sdk_python_root = Path(sdk_python_root) if sdk_python_root is not None else None
        self.model = model
        self.thread_config = dict(thread_config or {})
        self.max_idle_homes = max_idle_homes
        self._homes: dict[str, _CodexHomeState] = {}
        self._agent_runs: dict[str, _CodexAgentRun] = {}
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
    ) -> CodexTurnResult:
        home = self._get_or_start_home(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        self._begin_agent_run(home, agent_id)
        try:
            thread = home.codex.thread_start(
                cwd=workdir,
                developer_instructions=developer_instructions,
                model=self.model,
                config=self.thread_config or None,
            )
            turn_result = thread.run(prompt, cwd=workdir, model=self.model)
            rollout_relpath = _find_rollout_relpath(home_root / ".codex", thread.id)
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
    ) -> CodexTurnResult:
        home = self._get_or_start_home(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        self._begin_agent_run(home, agent_id, thread_id=thread_id)
        try:
            thread = home.codex.thread_resume(
                thread_id,
                cwd=workdir,
                developer_instructions=developer_instructions,
                model=self.model,
                config=self.thread_config or None,
            )
            turn_result = thread.run(prompt, cwd=workdir, model=self.model)
            rollout_relpath = _find_rollout_relpath(home_root / ".codex", thread.id)
            return CodexTurnResult(
                thread_id=thread.id,
                rollout_relpath=rollout_relpath,
                turn_result=turn_result,
                thread=thread,
            )
        finally:
            self._finish_agent_run(agent_id)

    def fork_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str,
        agent_id: str,
    ) -> CodexForkResult:
        home = self._get_or_start_home(home_id=home_id, home_root=home_root, env=env, workdir=None)
        self._begin_agent_run(home, agent_id, thread_id=thread_id)
        try:
            thread = home.codex.thread_fork(
                thread_id,
                model=self.model,
                config=self.thread_config or None,
            )
            rollout_relpath = _find_rollout_relpath(home_root / ".codex", thread.id)
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
        home = self._get_or_start_home(
            home_id=str(agent.home_id),
            home_root=home_root,
            env=env,
            workdir=None,
        )
        thread = home.codex.thread_resume(str(agent.thread_id), model=self.model)
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

    def close_home(self, home_id: str, *, force: bool = False) -> bool:
        with self._lock:
            return self._close_home_locked(home_id, force=force)

    def close_idle_homes(self) -> int:
        with self._lock:
            return self._evict_idle_homes_locked(force=True)

    def close_all(self, *, force: bool = False) -> None:
        with self._lock:
            for home_id in list(self._homes):
                self._close_home_locked(home_id, force=force)

    def list_active_agents(self, home_id: str | None = None) -> list[str]:
        with self._lock:
            if home_id is not None:
                state = self._homes.get(home_id)
                if state is None:
                    return []
                return sorted(state.active_agent_ids)
            return sorted(self._agent_runs)

    def _get_or_start_home(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        workdir: str | None,
    ) -> _CodexHomeState:
        resolved_home_root = Path(home_root)
        with self._lock:
            state = self._homes.get(home_id)
            if state is not None and not state.closed:
                if state.home_root != resolved_home_root:
                    raise ValueError(
                        f"home_id {home_id!r} was already started at {state.home_root}, "
                        f"cannot reuse it at {resolved_home_root}"
                    )
                state.last_used_at = utc_now_iso()
                return state
            sdk = self._sdk()
            codex = sdk.Codex(config=self._sdk_config(sdk, env=env, workdir=workdir))
            state = _CodexHomeState(
                home_id=home_id,
                home_root=resolved_home_root,
                codex=codex,
                last_used_at=utc_now_iso(),
            )
            self._homes[home_id] = state
            self._evict_idle_homes_locked()
            return state

    def _begin_agent_run(
        self,
        home: _CodexHomeState,
        agent_id: str,
        thread_id: str | None = None,
    ) -> None:
        with self._lock:
            if agent_id in self._agent_runs:
                raise RuntimeError(f"agent is already active in CodexProvider: {agent_id}")
            home.active_agent_ids.add(agent_id)
            self._agent_runs[agent_id] = _CodexAgentRun(
                agent_id=agent_id,
                home_id=home.home_id,
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
            home = self._homes.get(run.home_id)
            if home is not None:
                home.active_agent_ids.discard(agent_id)
                home.last_used_at = utc_now_iso()
            self._evict_idle_homes_locked()

    def _evict_idle_homes_locked(self, *, force: bool = False) -> int:
        if not force and len(self._homes) <= self.max_idle_homes:
            return 0
        idle = sorted(
            [state for state in self._homes.values() if not state.active_agent_ids],
            key=lambda state: state.last_used_at,
        )
        closed = 0
        for state in idle:
            if not force and len(self._homes) <= self.max_idle_homes:
                break
            if self._close_home_locked(state.home_id, force=False):
                closed += 1
        return closed

    def _close_home_locked(self, home_id: str, *, force: bool) -> bool:
        state = self._homes.get(home_id)
        if state is None:
            return True
        if state.active_agent_ids and not force:
            return False
        self._close_codex_object(state.codex)
        state.closed = True
        self._homes.pop(home_id, None)
        if force:
            for agent_id, run in list(self._agent_runs.items()):
                if run.home_id == home_id:
                    self._agent_runs.pop(agent_id, None)
        return True

    def _close_codex_object(self, codex: object) -> None:
        close = getattr(codex, "close", None)
        if callable(close):
            close()
            return
        exit_method = getattr(codex, "__exit__", None)
        if callable(exit_method):
            exit_method(None, None, None)

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

    def _agent_home_root(self, agent) -> Path:
        if self.runtime_root is None:
            raise RuntimeError("CodexProvider.runtime_root is required for thread reads")
        return self.runtime_root / "homes" / str(agent.cli_type) / str(agent.home_id)

    def _agent_rollout_path(self, agent, *, home_root: Path | None = None) -> Path | None:
        rollout_relpath = getattr(agent, "rollout_relpath", None)
        if not rollout_relpath:
            return None
        resolved_home_root = Path(home_root) if home_root is not None else self._agent_home_root(agent)
        return resolved_home_root / ".codex" / str(rollout_relpath)


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
