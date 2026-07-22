from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from .provider_contracts import (
    AgentArtifactLocator,
    ModelBackendIdentity,
    ProviderPayload,
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
    provider_type: str
    home_id: str
    schema_version: int = 3
    session_locator: ProviderSessionLocator | None = None
    latest_turn_locator: ProviderTurnLocator | None = None
    artifact_locator: AgentArtifactLocator | None = None
    fork_info: AgentForkInfo | None = None
    status: str = "idle"
    last_completion: AgentCompletionRecord | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError(f"unsupported Agent schema_version: {self.schema_version}")
        provider_type = self.provider_type
        if not provider_type.strip():
            raise ValueError("Agent provider_type must not be empty")
        if self.session_locator is not None:
            if self.session_locator.provider_type != provider_type:
                raise ValueError("Agent and session locator provider_type conflict")
            if self.session_locator.home_id != self.home_id:
                raise ValueError("Agent and session locator home_id conflict")
        if self.artifact_locator is not None:
            if self.session_locator is None:
                raise ValueError("artifact locator requires a session locator")
            if self.artifact_locator.provider_type != provider_type:
                raise ValueError("Agent and artifact locator provider_type conflict")
            if self.artifact_locator.home_id != self.home_id:
                raise ValueError("Agent and artifact locator home_id conflict")
            if self.session_locator is not None and (
                self.artifact_locator.session_id != self.session_locator.session_id
            ):
                raise ValueError("session and artifact locator session_id conflict")
        if self.latest_turn_locator is not None:
            if self.session_locator is None:
                raise ValueError("latest turn locator requires a session locator")
            if self.latest_turn_locator.session != self.session_locator:
                raise ValueError("latest turn and Agent session locator conflict")


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
    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 3:
        raise ValueError(f"unsupported Agent schema_version: {schema_version}; expected 3")
    return Agent(
        agent_id=str(payload["agent_id"]),
        scope_id=str(payload["scope_id"]),
        agent_type=str(payload["agent_type"]),
        provider_type=str(payload["provider_type"]),
        home_id=str(payload["home_id"]),
        schema_version=schema_version,
        session_locator=_session_locator_from_dict(payload.get("session_locator")),
        latest_turn_locator=_turn_locator_from_dict(payload.get("latest_turn_locator")),
        artifact_locator=_artifact_locator_from_dict(payload.get("artifact_locator")),
        fork_info=_fork_info_from_dict(payload.get("fork_info")),
        status=str(payload.get("status", "idle")),
        last_completion=completion_record_from_dict(payload.get("last_completion")),
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
        provider_payload=_provider_payload_from_dict(payload.get("provider_payload")),
    )


def _provider_payload_from_dict(payload: object) -> ProviderPayload | None:
    if not isinstance(payload, dict):
        return None
    return ProviderPayload(
        provider_type=str(payload["provider_type"]),
        payload_type=str(payload["payload_type"]),
        payload_schema_version=int(payload.get("payload_schema_version", 1)),
        adapter_version=(
            str(payload["adapter_version"]) if payload.get("adapter_version") is not None else None
        ),
        sdk_or_cli_version=(
            str(payload["sdk_or_cli_version"])
            if payload.get("sdk_or_cli_version") is not None
            else None
        ),
        sanitized_data=payload.get("sanitized_data"),
        truncated=bool(payload.get("truncated", False)),
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
