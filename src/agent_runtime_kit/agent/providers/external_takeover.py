from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Mapping
import uuid

from ..store_utils import read_json, utc_now_iso, write_json_atomic


class ExternalTakeoverCancelled(RuntimeError):
    """Raised when an external takeover turn is cancelled."""


@dataclass(frozen=True)
class ExternalTakeoverTurnResult:
    id: str
    status: str
    final_response: str | None = None
    handoff_id: str | None = None
    completed_at: str | None = None
    metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class ExternalTakeoverProviderResult:
    thread_id: str
    rollout_relpath: str | None
    turn_result: ExternalTakeoverTurnResult


@dataclass(frozen=True)
class ExternalTakeoverForkResult:
    thread_id: str
    rollout_relpath: str | None = None


@dataclass(frozen=True)
class ExternalTakeoverThreadSnapshot:
    id: str
    turns: list[ExternalTakeoverTurnResult]
    raw_handoffs: list[dict[str, object]]


@dataclass(frozen=True)
class ExternalTakeoverHomeInitializationRecord:
    home_id: str
    home_root: str
    initialized_at: str
    marker_path: str


class ExternalTakeoverProvider:
    """Provider that hands agent turns to an external controller via files."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        poll_interval_s: float = 0.1,
        default_timeout_s: float | None = None,
        handoff_dirname: str = "external_turns",
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.poll_interval_s = poll_interval_s
        self.default_timeout_s = default_timeout_s
        self.handoff_dirname = handoff_dirname

    @property
    def handoff_root(self) -> Path:
        return self.runtime_root / self.handoff_dirname

    def ensure_home_initialized(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        workdir: str | None,
    ) -> ExternalTakeoverHomeInitializationRecord:
        del env, workdir
        marker_path = Path(home_root) / ".ark" / "external_takeover_home_initialized.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        if marker_path.exists():
            payload = read_json(marker_path)
            return ExternalTakeoverHomeInitializationRecord(
                home_id=str(payload.get("home_id", home_id)),
                home_root=str(payload.get("home_root", home_root)),
                initialized_at=str(payload.get("initialized_at", "")),
                marker_path=str(marker_path),
            )
        initialized_at = utc_now_iso()
        payload = {
            "schema_version": 1,
            "object_type": "external_takeover_home_initialization",
            "home_id": home_id,
            "home_root": str(home_root),
            "initialized_at": initialized_at,
        }
        write_json_atomic(marker_path, payload)
        return ExternalTakeoverHomeInitializationRecord(
            home_id=home_id,
            home_root=str(home_root),
            initialized_at=initialized_at,
            marker_path=str(marker_path),
        )

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
    ) -> ExternalTakeoverProviderResult:
        self.ensure_home_initialized(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        return self._run_external_turn(
            home_id=home_id,
            home_root=home_root,
            env=env,
            thread_id=None,
            workdir=workdir,
            prompt=prompt,
            developer_instructions=developer_instructions,
            overwrite_developer_instructions=overwrite_developer_instructions,
            agent_id=agent_id,
        )

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
    ) -> ExternalTakeoverProviderResult:
        self.ensure_home_initialized(home_id=home_id, home_root=home_root, env=env, workdir=workdir)
        return self._run_external_turn(
            home_id=home_id,
            home_root=home_root,
            env=env,
            thread_id=thread_id,
            workdir=workdir,
            prompt=prompt,
            developer_instructions=developer_instructions,
            overwrite_developer_instructions=overwrite_developer_instructions,
            agent_id=agent_id,
        )

    def fork_thread(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str,
        agent_id: str,
    ) -> ExternalTakeoverForkResult:
        del home_id, home_root, env, agent_id
        return ExternalTakeoverForkResult(thread_id=f"{thread_id}-fork-{uuid.uuid4().hex[:8]}")

    def read_thread(
        self,
        agent,
        *,
        home_root: Path,
        env: Mapping[str, str],
        include_turns: bool = True,
    ) -> ExternalTakeoverThreadSnapshot:
        del home_root, env
        thread_id = str(getattr(agent, "thread_id", "") or "")
        if not thread_id:
            raise RuntimeError("agent has no thread_id")
        handoffs = self._handoffs_for_thread(thread_id)
        turns = [self._turn_result_from_handoff(handoff) for handoff in handoffs] if include_turns else []
        return ExternalTakeoverThreadSnapshot(id=thread_id, turns=turns, raw_handoffs=handoffs)

    def read_latest_turn_result(
        self,
        agent,
        *,
        home_root: Path,
        env: Mapping[str, str],
    ) -> ExternalTakeoverTurnResult:
        snapshot = self.read_thread(agent, home_root=home_root, env=env, include_turns=True)
        if not snapshot.turns:
            raise RuntimeError("thread has no turns")
        return snapshot.turns[-1]

    def interrupt_agent(self, agent_id: str) -> bool:
        del agent_id
        return False

    def close(self) -> None:
        return None

    def _run_external_turn(
        self,
        *,
        home_id: str,
        home_root: Path,
        env: Mapping[str, str],
        thread_id: str | None,
        workdir: str | None,
        prompt: str,
        developer_instructions: str | None,
        overwrite_developer_instructions: bool,
        agent_id: str,
    ) -> ExternalTakeoverProviderResult:
        handoff_id = f"h_{uuid.uuid4().hex}"
        handoff_dir = self._handoff_dir(handoff_id)
        handoff = {
            "schema_version": 1,
            "handoff_id": handoff_id,
            "status": "pending",
            "created_at": utc_now_iso(),
            "home_id": home_id,
            "home_root": str(home_root),
            "agent_id": agent_id,
            "thread_id": thread_id,
            "workdir": workdir,
            "prompt": prompt,
            "developer_instructions": developer_instructions,
            "overwrite_developer_instructions": overwrite_developer_instructions,
            "env": dict(env),
            "metadata": {},
        }
        write_json_atomic(handoff_dir / "handoff.json", handoff)
        completion = self._wait_for_completion(handoff_id)
        return self._provider_result_from_completion(handoff, completion)

    def _wait_for_completion(self, handoff_id: str) -> dict[str, object]:
        completion_path = self._handoff_dir(handoff_id) / "completion.json"
        deadline = None if self.default_timeout_s is None else monotonic() + self.default_timeout_s
        while True:
            if completion_path.exists():
                completion = read_json(completion_path)
                if str(completion.get("handoff_id", handoff_id)) != handoff_id:
                    raise RuntimeError(f"completion handoff_id does not match {handoff_id}")
                return completion
            if deadline is not None and monotonic() >= deadline:
                raise TimeoutError(handoff_id)
            sleep(self.poll_interval_s)

    def _provider_result_from_completion(
        self,
        handoff: dict[str, object],
        completion: dict[str, object],
    ) -> ExternalTakeoverProviderResult:
        status = str(completion.get("status", "")).strip()
        handoff_id = str(handoff["handoff_id"])
        if status == "failed":
            raise RuntimeError(str(completion.get("error") or completion.get("final_response") or f"external handoff failed: {handoff_id}"))
        if status == "cancelled":
            raise ExternalTakeoverCancelled(f"external handoff cancelled: {handoff_id}")
        if status != "completed":
            raise RuntimeError(f"unsupported external completion status {status!r} for {handoff_id}")
        thread_id = str(
            completion.get("thread_id")
            or handoff.get("thread_id")
            or f"external-thread-{handoff_id}"
        )
        completed_at = str(completion.get("completed_at") or utc_now_iso())
        turn_id = str(completion.get("turn_id") or f"external-turn-{handoff_id}")
        metadata = completion.get("metadata")
        turn_result = ExternalTakeoverTurnResult(
            id=turn_id,
            status=status,
            final_response=_optional_str(completion.get("final_response")),
            handoff_id=handoff_id,
            completed_at=completed_at,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )
        return ExternalTakeoverProviderResult(
            thread_id=thread_id,
            rollout_relpath=_optional_str(completion.get("rollout_relpath")),
            turn_result=turn_result,
        )

    def _turn_result_from_handoff(self, handoff: dict[str, object]) -> ExternalTakeoverTurnResult:
        completion = handoff.get("completion")
        if not isinstance(completion, dict):
            raise RuntimeError(f"handoff has no completion: {handoff.get('handoff_id')}")
        return self._provider_result_from_completion(handoff, completion).turn_result

    def _handoffs_for_thread(self, thread_id: str) -> list[dict[str, object]]:
        handoffs: list[dict[str, object]] = []
        for handoff_path in sorted(self.handoff_root.glob("*/handoff.json")):
            completion_path = handoff_path.with_name("completion.json")
            if not completion_path.exists():
                continue
            handoff = read_json(handoff_path)
            completion = read_json(completion_path)
            completed_thread_id = str(
                completion.get("thread_id")
                or handoff.get("thread_id")
                or f"external-thread-{handoff.get('handoff_id')}"
            )
            if completed_thread_id != thread_id and str(handoff.get("thread_id") or "") != thread_id:
                continue
            handoff["completion"] = completion
            handoffs.append(handoff)
        return sorted(handoffs, key=lambda item: str(item.get("created_at") or ""))

    def _handoff_dir(self, handoff_id: str) -> Path:
        return self.handoff_root / handoff_id


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
