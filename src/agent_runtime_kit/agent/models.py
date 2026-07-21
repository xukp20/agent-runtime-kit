from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from .provider_contracts import (
    AgentArtifactLocator,
    ModelBackendIdentity,
    ProviderSessionLocator,
    ProviderTurnLocator,
)


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
class AgentForkInfo:
    source_agent_id: str
    source_session_id: str
    created_at: str
    source_turn_id: str | None = None
    fork_mode: str = "session_only"
    workspace_isolated: bool = False
    source_workspace_revision: str | None = None

    def __post_init__(self) -> None:
        if self.fork_mode != "session_only" or self.workspace_isolated:
            raise ValueError("Agent fork records must use session_only without workspace isolation")


@dataclass
class Agent:
    agent_id: str
    scope_id: str
    agent_type: str
    cli_type: str
    home_id: str
    schema_version: int = 2
    provider_type: str | None = None
    session_locator: ProviderSessionLocator | None = None
    latest_turn_locator: ProviderTurnLocator | None = None
    artifact_locator: AgentArtifactLocator | None = None
    fork_info: AgentForkInfo | None = None
    thread_id: str | None = None
    rollout_relpath: str | None = None
    status: str = "idle"
    last_completion: AgentCompletionRecord | None = None
    fork_source_agent_id: str | None = None
    fork_source_thread_id: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.normalize_compat_fields()

    def normalize_compat_fields(self) -> None:
        provider_type = self.provider_type or self.cli_type
        if provider_type != self.cli_type:
            raise ValueError("provider_type and legacy cli_type conflict")
        self.provider_type = provider_type
        if self.schema_version not in {1, 2}:
            raise ValueError(f"unsupported Agent schema_version: {self.schema_version}")
        if self.session_locator is None and self.thread_id:
            # COMPAT(legacy-codex-agent-record): records written before schema
            # v2 only have cli_type/thread_id/rollout_relpath. Remove after all
            # supported snapshots and LC callers persist v2 locators.
            self.session_locator = ProviderSessionLocator(
                provider_type=provider_type,
                session_id=str(self.thread_id),
                home_id=self.home_id,
                created_at=self.created_at,
                native_locator={"rollout_relpath": self.rollout_relpath},
            )
        if self.session_locator is not None:
            if self.session_locator.provider_type != provider_type:
                raise ValueError("Agent and session locator provider_type conflict")
            if self.session_locator.home_id != self.home_id:
                raise ValueError("Agent and session locator home_id conflict")
            if self.thread_id is not None and self.thread_id != self.session_locator.session_id:
                raise ValueError("session_locator and legacy thread_id conflict")
            if provider_type == "codex":
                self.thread_id = self.session_locator.session_id
                native = self.session_locator.native_locator
                native_rollout = native.get("rollout_relpath") if isinstance(native, dict) else None
                if (
                    native_rollout is not None
                    and self.rollout_relpath is not None
                    and str(native_rollout) != self.rollout_relpath
                ):
                    raise ValueError("session_locator and legacy rollout_relpath conflict")
        if self.artifact_locator is None and provider_type == "codex" and self.thread_id and self.rollout_relpath:
            self.artifact_locator = AgentArtifactLocator(
                provider_type="codex",
                home_id=self.home_id,
                session_id=self.thread_id,
                adapter_version="legacy-record-v1",
                native_primary_ref=self.rollout_relpath,
            )
        if self.artifact_locator is not None:
            if self.artifact_locator.provider_type != provider_type:
                raise ValueError("Agent and artifact locator provider_type conflict")
            if self.artifact_locator.home_id != self.home_id:
                raise ValueError("Agent and artifact locator home_id conflict")
            if self.session_locator is not None and (
                self.artifact_locator.session_id != self.session_locator.session_id
            ):
                raise ValueError("session and artifact locator session_id conflict")
            if (
                provider_type == "codex"
                and self.rollout_relpath is not None
                and self.artifact_locator.native_primary_ref is not None
                and self.rollout_relpath != self.artifact_locator.native_primary_ref
            ):
                raise ValueError("artifact_locator and legacy rollout_relpath conflict")
            if provider_type == "codex" and self.rollout_relpath is None:
                self.rollout_relpath = self.artifact_locator.native_primary_ref
        if self.latest_turn_locator is not None and self.session_locator is not None:
            if self.latest_turn_locator.session != self.session_locator:
                raise ValueError("latest turn and Agent session locator conflict")
        if self.fork_info is None and self.fork_source_agent_id and self.fork_source_thread_id:
            self.fork_info = AgentForkInfo(
                source_agent_id=self.fork_source_agent_id,
                source_session_id=self.fork_source_thread_id,
                created_at=self.created_at,
            )
        if self.fork_info is not None:
            if self.fork_source_agent_id not in {None, self.fork_info.source_agent_id}:
                raise ValueError("fork_info and legacy fork_source_agent_id conflict")
            if self.fork_source_thread_id not in {None, self.fork_info.source_session_id}:
                raise ValueError("fork_info and legacy fork_source_thread_id conflict")
            self.fork_source_agent_id = self.fork_info.source_agent_id
            self.fork_source_thread_id = self.fork_info.source_session_id
        self.schema_version = 2


@dataclass(frozen=True)
class AgentStatusWaitResult:
    agent: Agent
    changed: bool
    timed_out: bool
    observed_at: str


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
    provider_type = str(payload.get("provider_type") or payload.get("cli_type") or "")
    cli_type = str(payload.get("cli_type") or provider_type)
    return Agent(
        agent_id=str(payload["agent_id"]),
        scope_id=str(payload["scope_id"]),
        agent_type=str(payload["agent_type"]),
        cli_type=cli_type,
        home_id=str(payload["home_id"]),
        schema_version=int(payload.get("schema_version", 1)),
        provider_type=provider_type,
        session_locator=_session_locator_from_dict(payload.get("session_locator")),
        latest_turn_locator=_turn_locator_from_dict(payload.get("latest_turn_locator")),
        artifact_locator=_artifact_locator_from_dict(payload.get("artifact_locator")),
        fork_info=_fork_info_from_dict(payload.get("fork_info")),
        thread_id=payload.get("thread_id"),
        rollout_relpath=payload.get("rollout_relpath"),
        status=str(payload.get("status", "idle")),
        last_completion=completion_record_from_dict(payload.get("last_completion")),
        fork_source_agent_id=payload.get("fork_source_agent_id"),
        fork_source_thread_id=payload.get("fork_source_thread_id"),
        created_at=str(payload.get("created_at", "")),
        updated_at=str(payload.get("updated_at", "")),
    )


def _session_locator_from_dict(payload: object) -> ProviderSessionLocator | None:
    if not isinstance(payload, dict):
        return None
    return ProviderSessionLocator(
        provider_type=str(payload["provider_type"]),
        session_id=str(payload["session_id"]),
        home_id=str(payload["home_id"]),
        created_at=str(payload.get("created_at", "")),
        backend_identity=_backend_identity_from_dict(payload.get("backend_identity")),
        native_locator=payload.get("native_locator"),
    )


def _backend_identity_from_dict(payload: object) -> ModelBackendIdentity | None:
    if not isinstance(payload, dict):
        return None
    return ModelBackendIdentity(
        api_provider=str(payload["api_provider"]),
        api_mode=str(payload["api_mode"]),
        endpoint_id=str(payload["endpoint_id"]) if payload.get("endpoint_id") is not None else None,
        requested_model=(
            str(payload["requested_model"]) if payload.get("requested_model") is not None else None
        ),
        resolved_model=(
            str(payload["resolved_model"]) if payload.get("resolved_model") is not None else None
        ),
        model_version=(
            str(payload["model_version"]) if payload.get("model_version") is not None else None
        ),
        service_tier=(
            str(payload["service_tier"]) if payload.get("service_tier") is not None else None
        ),
        reasoning_effort=(
            str(payload["reasoning_effort"])
            if payload.get("reasoning_effort") is not None
            else None
        ),
        tokenizer_id=(
            str(payload["tokenizer_id"]) if payload.get("tokenizer_id") is not None else None
        ),
        model_config_hash=(
            str(payload["model_config_hash"])
            if payload.get("model_config_hash") is not None
            else None
        ),
    )


def _turn_locator_from_dict(payload: object) -> ProviderTurnLocator | None:
    if not isinstance(payload, dict):
        return None
    session = _session_locator_from_dict(payload.get("session"))
    if session is None:
        raise ValueError("turn locator requires a session locator")
    return ProviderTurnLocator(
        session=session,
        turn_id=str(payload["turn_id"]),
        request_ids=tuple(str(item) for item in payload.get("request_ids") or ()),
        sequence=int(payload["sequence"]) if payload.get("sequence") is not None else None,
    )


def _artifact_locator_from_dict(payload: object) -> AgentArtifactLocator | None:
    if not isinstance(payload, dict):
        return None
    return AgentArtifactLocator(
        provider_type=str(payload["provider_type"]),
        home_id=str(payload["home_id"]),
        session_id=str(payload["session_id"]),
        adapter_version=str(payload["adapter_version"]),
        manifest_relpath=(
            str(payload["manifest_relpath"]) if payload.get("manifest_relpath") is not None else None
        ),
        native_primary_ref=(
            str(payload["native_primary_ref"])
            if payload.get("native_primary_ref") is not None
            else None
        ),
    )


def _fork_info_from_dict(payload: object) -> AgentForkInfo | None:
    if not isinstance(payload, dict):
        return None
    return AgentForkInfo(
        source_agent_id=str(payload["source_agent_id"]),
        source_session_id=str(payload["source_session_id"]),
        source_turn_id=(
            str(payload["source_turn_id"]) if payload.get("source_turn_id") is not None else None
        ),
        fork_mode=str(payload.get("fork_mode", "session_only")),
        workspace_isolated=bool(payload.get("workspace_isolated", False)),
        source_workspace_revision=(
            str(payload["source_workspace_revision"])
            if payload.get("source_workspace_revision") is not None
            else None
        ),
        created_at=str(payload.get("created_at", "")),
    )
