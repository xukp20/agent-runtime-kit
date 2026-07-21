from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent_runtime_kit.agent.context import (
    ProviderContextCompactionResult,
    ProviderContextUsage,
)
from agent_runtime_kit.agent.store_utils import utc_now_iso


@dataclass
class FakeTurnResult:
    id: str
    thread_id: str
    prompt: str
    developer_instructions: str | None
    status: str = "completed"
    final_response: str = ""


@dataclass
class FakeThread:
    thread_id: str
    turns: list[FakeTurnResult] = field(default_factory=list)


@dataclass
class FakeRunResult:
    thread_id: str
    rollout_relpath: str
    turn_result: FakeTurnResult
    thread: FakeThread


@dataclass
class FakeForkResult:
    thread_id: str
    rollout_relpath: str


class FakeProvider:
    def __init__(self, runtime_root: Path, *, run_delay_s: float = 0.0) -> None:
        self.runtime_root = Path(runtime_root)
        self.next_thread = 1
        self.next_turn = 1
        self.calls: list[dict[str, object]] = []
        self.ensure_home_initialized_calls: list[dict[str, object]] = []
        self.close_calls = 0
        self.interrupt_calls: list[str] = []
        self.active_by_home: dict[str, set[str]] = {}
        self.max_active_by_home: dict[str, int] = {}
        self.run_delay_s = run_delay_s
        self.operation_order: list[str] = []
        self.compact_calls: list[dict[str, object]] = []
        self.context_usage = ProviderContextUsage(
            session_id=None,
            total_tokens=90,
            context_window=100,
            observed_at=utc_now_iso(),
            source="provider_api",
            available=True,
        )
        self.compact_usage_after = ProviderContextUsage(
            session_id=None,
            total_tokens=20,
            context_window=100,
            observed_at=utc_now_iso(),
            source="provider_api",
            available=True,
        )
        self.compact_error_before_start: BaseException | None = None
        self.compact_error_after_start: BaseException | None = None
        self.compact_started_event: threading.Event | None = None
        self.compact_release_event: threading.Event | None = None
        self.reconcile_confirmed = True
        self._lock = threading.RLock()

    def start_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: dict[str, str],
        workdir: str | None,
        prompt: str,
        developer_instructions: str | None,
        agent_id: str,
        overwrite_developer_instructions: bool = False,
    ) -> FakeRunResult:
        with self._lock:
            thread_id = f"thread-{self.next_thread}"
            self.next_thread += 1
        return self._run_with_active_record(
            home_id=home_id,
            agent_id=agent_id,
            fn=lambda: self._append_turn(
                home_id=home_id,
                agent_id=agent_id,
                home_root=home_root,
                env=env,
                workdir=workdir,
                thread_id=thread_id,
                prompt=prompt,
                developer_instructions=developer_instructions,
                overwrite_developer_instructions=overwrite_developer_instructions,
            ),
        )

    def resume_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: dict[str, str],
        thread_id: str,
        workdir: str | None,
        prompt: str,
        developer_instructions: str | None,
        agent_id: str,
        overwrite_developer_instructions: bool = False,
    ) -> FakeRunResult:
        return self._run_with_active_record(
            home_id=home_id,
            agent_id=agent_id,
            fn=lambda: self._append_turn(
                home_id=home_id,
                agent_id=agent_id,
                home_root=home_root,
                env=env,
                workdir=workdir,
                thread_id=thread_id,
                prompt=prompt,
                developer_instructions=developer_instructions,
                overwrite_developer_instructions=overwrite_developer_instructions,
            ),
        )

    def fork_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: dict[str, str],
        thread_id: str,
        agent_id: str,
    ) -> FakeForkResult:
        with self._lock:
            new_thread_id = f"thread-{self.next_thread}"
            self.next_thread += 1
        source = home_root / ".codex" / _rollout_relpath(thread_id)
        target = home_root / ".codex" / _rollout_relpath(new_thread_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        _append_jsonl(
            target,
            {
                "type": "fork",
                "source_thread_id": thread_id,
                "thread_id": new_thread_id,
                "env_home": env.get("HOME"),
                "home_id": home_id,
                "agent_id": agent_id,
            },
        )
        return FakeForkResult(thread_id=new_thread_id, rollout_relpath=_rollout_relpath(new_thread_id))

    def read_thread(
        self,
        agent: object,
        *,
        home_root: Path | None = None,
        env: dict[str, str] | None = None,
        include_turns: bool = True,
    ) -> FakeThread:
        thread_id = str(getattr(agent, "thread_id"))
        turns = self._read_turns(agent) if include_turns else []
        return FakeThread(thread_id=thread_id, turns=turns)

    def read_latest_turn_result(
        self,
        agent: object,
        *,
        home_root: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> FakeTurnResult:
        turns = self._read_turns(agent)
        if not turns:
            raise RuntimeError("no fake turns")
        return turns[-1]

    def interrupt_agent(self, agent_id: str) -> bool:
        self.interrupt_calls.append(agent_id)
        return False

    def inspect_thread_context(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: dict[str, str],
        thread_id: str,
        workdir: str | None,
        agent_id: str,
    ) -> ProviderContextUsage:
        del home_id, home_root, env, workdir, agent_id
        usage = self.context_usage
        return ProviderContextUsage(
            session_id=thread_id,
            total_tokens=usage.total_tokens,
            context_window=usage.context_window,
            observed_at=usage.observed_at,
            source=usage.source,
            available=usage.available,
            reason=usage.reason,
        )

    def compact_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: dict[str, str],
        thread_id: str,
        workdir: str | None,
        agent_id: str,
        timeout_s: float,
        on_compaction_started,
    ) -> ProviderContextCompactionResult:
        if self.compact_error_before_start is not None:
            raise self.compact_error_before_start
        call = {
            "home_id": home_id,
            "home_root": str(home_root),
            "env": dict(env),
            "thread_id": thread_id,
            "workdir": workdir,
            "agent_id": agent_id,
            "timeout_s": timeout_s,
        }
        self.compact_calls.append(call)
        self.operation_order.append("compact")
        on_compaction_started({"offset": 12}, "fake-operation")
        if self.compact_started_event is not None:
            self.compact_started_event.set()
        if self.compact_release_event is not None:
            assert self.compact_release_event.wait(timeout=5)
        if self.compact_error_after_start is not None:
            raise self.compact_error_after_start
        usage = self.compact_usage_after
        return ProviderContextCompactionResult(
            session_id=thread_id,
            usage_after=ProviderContextUsage(
                session_id=thread_id,
                total_tokens=usage.total_tokens,
                context_window=usage.context_window,
                observed_at=usage.observed_at,
                source=usage.source,
                available=usage.available,
                reason=usage.reason,
            ),
            started_at=utc_now_iso(),
            completed_at=utc_now_iso(),
            provider_operation_id="fake-operation",
        )

    def reconcile_thread_compaction(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: dict[str, str],
        thread_id: str,
        workdir: str | None,
        agent_id: str,
        baseline: dict[str, object],
        provider_operation_id: str | None,
    ) -> ProviderContextCompactionResult | None:
        del home_id, home_root, env, workdir, agent_id, baseline, provider_operation_id
        if not self.reconcile_confirmed:
            return None
        return ProviderContextCompactionResult(
            session_id=thread_id,
            usage_after=self.compact_usage_after,
            started_at=utc_now_iso(),
            completed_at=utc_now_iso(),
            provider_operation_id="fake-operation",
        )

    def ensure_home_initialized(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: dict[str, str],
        workdir: str | None,
    ) -> dict[str, object]:
        call = {
            "home_id": home_id,
            "home_root": str(home_root),
            "env": dict(env),
            "workdir": workdir,
        }
        self.ensure_home_initialized_calls.append(call)
        return call

    def close(self) -> None:
        self.close_calls += 1

    def list_active_agents(self, home_id: str | None = None) -> list[str]:
        with self._lock:
            if home_id is not None:
                return sorted(self.active_by_home.get(home_id, set()))
            active = set()
            for agent_ids in self.active_by_home.values():
                active.update(agent_ids)
            return sorted(active)

    def _append_turn(
        self,
        *,
        home_id: str,
        agent_id: str,
        home_root: Path,
        env: dict[str, str],
        workdir: str | None,
        thread_id: str,
        prompt: str,
        developer_instructions: str | None,
        overwrite_developer_instructions: bool,
    ) -> FakeRunResult:
        self.operation_order.append("turn")
        if self.run_delay_s:
            time.sleep(self.run_delay_s)
        with self._lock:
            turn_id = f"turn-{self.next_turn}"
            self.next_turn += 1
        rollout_relpath = _rollout_relpath(thread_id)
        path = home_root / ".codex" / rollout_relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "type": "turn_result",
            "thread_id": thread_id,
            "turn_id": turn_id,
            "prompt": prompt,
            "developer_instructions": developer_instructions,
            "overwrite_developer_instructions": overwrite_developer_instructions,
            "workdir": workdir,
            "env_home": env.get("HOME"),
            "env_codex_home": env.get("CODEX_HOME"),
            "env": dict(env),
            "home_id": home_id,
            "agent_id": agent_id,
        }
        _append_jsonl(path, event)
        result = FakeTurnResult(
            id=turn_id,
            thread_id=thread_id,
            prompt=prompt,
            developer_instructions=developer_instructions,
            final_response=f"response for {prompt}",
        )
        self.calls.append(event)
        return FakeRunResult(
            thread_id=thread_id,
            rollout_relpath=rollout_relpath,
            turn_result=result,
            thread=FakeThread(thread_id=thread_id, turns=self._read_turns_from_path(path)),
        )

    def _run_with_active_record(self, *, home_id: str, agent_id: str, fn):
        with self._lock:
            active = self.active_by_home.setdefault(home_id, set())
            active.add(agent_id)
            self.max_active_by_home[home_id] = max(self.max_active_by_home.get(home_id, 0), len(active))
        try:
            return fn()
        finally:
            with self._lock:
                active = self.active_by_home.setdefault(home_id, set())
                active.discard(agent_id)

    def _read_turns(self, agent: object) -> list[FakeTurnResult]:
        rollout_relpath = getattr(agent, "rollout_relpath")
        if not rollout_relpath:
            return []
        path = (
            self.runtime_root
            / "homes"
            / str(getattr(agent, "cli_type"))
            / str(getattr(agent, "home_id"))
            / ".codex"
            / str(rollout_relpath)
        )
        return self._read_turns_from_path(path)

    def _read_turns_from_path(self, path: Path) -> list[FakeTurnResult]:
        if not path.exists():
            return []
        turns: list[FakeTurnResult] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("type") == "turn_result":
                turns.append(
                    FakeTurnResult(
                        id=str(payload["turn_id"]),
                        thread_id=str(payload["thread_id"]),
                        prompt=str(payload["prompt"]),
                        developer_instructions=payload.get("developer_instructions"),
                        final_response=f"response for {payload['prompt']}",
                    )
                )
        return turns


def _rollout_relpath(thread_id: str) -> str:
    return f"sessions/fake/rollout-{thread_id}.jsonl"


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
