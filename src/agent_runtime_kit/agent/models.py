from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class AgentRuntimeKitError(Exception):
    """Base exception for agent-runtime-kit."""


class AgentClosedError(AgentRuntimeKitError):
    pass


class AgentAlreadyRunningError(AgentRuntimeKitError):
    pass


class AgentPausedError(AgentRuntimeKitError):
    pass


class AgentIncompleteError(AgentRuntimeKitError):
    def __init__(self, agent_id: str, record: "AgentCompletionRecord") -> None:
        super().__init__(f"agent did not complete: {agent_id}")
        self.agent_id = agent_id
        self.record = record


class AgentCompletionCheckError(AgentRuntimeKitError):
    pass


class AgentHasNoCompletedTurn(AgentRuntimeKitError):
    pass


class AgentContextMaintenanceError(AgentRuntimeKitError):
    pass


class AgentContextMaintenanceUnsupported(AgentContextMaintenanceError):
    pass


class AgentContextCompactionTimeout(AgentContextMaintenanceError):
    pass


class AgentContextCompactionRequestUnknown(AgentContextMaintenanceError):
    pass


class AgentContextCompactionEvidenceError(AgentContextMaintenanceError):
    pass


class AgentContextMaintenanceBlocked(AgentContextMaintenanceError):
    pass


class AgentContextUsageUnavailable(AgentContextMaintenanceError):
    pass


class MissingProviderEnvError(AgentRuntimeKitError):
    def __init__(self, name: str) -> None:
        super().__init__(f"missing provider environment variable: {name}")
        self.name = name


@dataclass(frozen=True)
class CompletionDecision:
    complete: bool
    reason: str | None = None
    continue_prompt: str | None = None
    close_agent: bool = False


@dataclass
class AgentCompletionRecord:
    turn_id: str
    decision: CompletionDecision
    status: str
    auto_continue_count: int
    checked_at: str
    error_message: str | None = None


@dataclass
class Agent:
    agent_id: str
    scope_id: str
    agent_type: str
    cli_type: str
    home_id: str
    thread_id: str | None = None
    rollout_relpath: str | None = None
    status: str = "idle"
    last_completion: AgentCompletionRecord | None = None
    fork_source_agent_id: str | None = None
    fork_source_thread_id: str | None = None
    created_at: str = ""
    updated_at: str = ""


@dataclass
class WaitAgentsResult:
    completed: dict[str, object]
    errors: dict[str, BaseException]
    pending: tuple[str, ...]
    timeout: bool

    @property
    def clean(self) -> bool:
        return not self.errors and not self.pending and not self.timeout


@dataclass
class ScopeSnapshotResult:
    snapshot_id: str | None
    scope_id: str
    status: str
    running_agent_ids: tuple[str, ...] = ()
    running_step_ids: tuple[str, ...] = ()
    errors: dict[str, BaseException] = field(default_factory=dict)
    snapshot_relpath: str | None = None


@dataclass
class RuntimeSnapshotResult:
    snapshot_id: str | None
    status: str
    scope_snapshot_ids: dict[str, str] = field(default_factory=dict)
    blocked_scope_ids: tuple[str, ...] = ()
    running_step_ids: tuple[str, ...] = ()
    errors: dict[str, BaseException] = field(default_factory=dict)
    snapshot_relpath: str | None = None
    pruned_scope_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScopeSnapshotInfo:
    snapshot_id: str
    scope_id: str
    scope_key: str
    status: str
    snapshot_relpath: str
    created_at: str


@dataclass(frozen=True)
class RuntimeSnapshotInfo:
    snapshot_id: str
    status: str
    snapshot_relpath: str
    created_at: str
    scope_count: int


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {
            key.value if isinstance(key, Enum) else str(key): to_jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def completion_record_from_dict(payload: dict[str, Any] | None) -> AgentCompletionRecord | None:
    if payload is None:
        return None
    decision_payload = dict(payload["decision"])
    decision = CompletionDecision(
        complete=bool(decision_payload["complete"]),
        reason=decision_payload.get("reason"),
        continue_prompt=decision_payload.get("continue_prompt"),
        close_agent=bool(decision_payload.get("close_agent", False)),
    )
    return AgentCompletionRecord(
        turn_id=str(payload["turn_id"]),
        decision=decision,
        status=str(payload["status"]),
        auto_continue_count=int(payload["auto_continue_count"]),
        checked_at=str(payload["checked_at"]),
        error_message=payload.get("error_message"),
    )


def agent_from_dict(payload: dict[str, Any]) -> Agent:
    return Agent(
        agent_id=str(payload["agent_id"]),
        scope_id=str(payload["scope_id"]),
        agent_type=str(payload["agent_type"]),
        cli_type=str(payload["cli_type"]),
        home_id=str(payload["home_id"]),
        thread_id=payload.get("thread_id"),
        rollout_relpath=payload.get("rollout_relpath"),
        status=str(payload.get("status", "idle")),
        last_completion=completion_record_from_dict(payload.get("last_completion")),
        fork_source_agent_id=payload.get("fork_source_agent_id"),
        fork_source_thread_id=payload.get("fork_source_thread_id"),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
    )
