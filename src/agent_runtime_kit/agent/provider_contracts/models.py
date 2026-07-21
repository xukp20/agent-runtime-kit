from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping

from .capabilities import CapabilitySupport
from .homes import ProviderExecutionContext
from .identities import ModelBackendIdentity, ProviderPayload
from .locators import AgentArtifactLocator, ProviderSessionLocator, ProviderTurnLocator
from .usage import AgentTurnUsage, ModelRequestUsage


class ProviderRunState(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    NEEDS_INPUT = "needs_input"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            ProviderRunState.COMPLETED,
            ProviderRunState.INTERRUPTED,
            ProviderRunState.CANCELLED,
            ProviderRunState.FAILED,
        }


class ProviderControlAction(str, Enum):
    INTERRUPT = "interrupt"
    CANCEL = "cancel"
    STEER = "steer"
    FOLLOW_UP = "follow_up"
    STOP_TASK = "stop_task"
    UPDATE_MODEL = "update_model"
    UPDATE_PERMISSION_MODE = "update_permission_mode"
    ARCHIVE_SESSION = "archive_session"
    RESPOND_APPROVAL = "respond_approval"
    RESPOND_INPUT = "respond_input"
    REJECT_INPUT = "reject_input"


@dataclass(frozen=True)
class ProviderRunOptions:
    max_turns: int | None = None
    timeout_s: float | None = None
    stream: bool | None = None

    def __post_init__(self) -> None:
        if self.max_turns is not None and self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


@dataclass(frozen=True)
class ProviderRunRequest:
    agent_id: str
    scope_id: str
    agent_type: str
    provider_type: str
    home_id: str
    prompt: str
    session_locator: ProviderSessionLocator | None = None
    developer_instructions: str | None = None
    system_instructions: str | None = None
    replace_developer_instructions: bool = False
    workdir: str | None = None
    environment: Mapping[str, str] = field(default_factory=dict, repr=False)
    model_overrides: ModelBackendIdentity | None = None
    run_options: ProviderRunOptions = field(default_factory=ProviderRunOptions)
    provider_options: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    event_sink: Callable[["AgentEvent"], None] | None = field(default=None, repr=False, compare=False)
    execution_context: ProviderExecutionContext | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("agent_id", "scope_id", "agent_type", "provider_type", "home_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
        if self.session_locator is not None:
            if self.session_locator.provider_type != self.provider_type:
                raise ValueError("request and session locator provider_type differ")
            if self.session_locator.home_id != self.home_id:
                raise ValueError("request and session locator home_id differ")


@dataclass(frozen=True)
class AgentError:
    error_type: str
    message: str
    code: str | None = None
    retryable: bool | None = None
    provider_payload: ProviderPayload | None = None


@dataclass(frozen=True)
class AgentContentBlock:
    kind: str
    data: object
    block_id: str | None = None
    sequence: int | None = None
    parent_id: str | None = None
    call_id: str | None = None
    model_request_id: str | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("content block kind must not be empty")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("content block sequence must not be negative")


@dataclass(frozen=True)
class AgentToolCall:
    call_id: str
    tool_name: str
    tool_kind: str
    status: str
    parent_call_id: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    display_name: str | None = None
    server_name: str | None = None
    arguments: object | None = None
    result: object | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_ms: float | None = None
    error: AgentError | None = None
    approval: object | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("tool call id and name must not be empty")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must not be negative")


@dataclass(frozen=True)
class AgentEvent:
    provider_type: str
    sequence: int
    timestamp: str
    kind: str
    schema_version: int = 1
    session_id: str | None = None
    turn_id: str | None = None
    request_id: str | None = None
    phase: str | None = None
    block_id: str | None = None
    call_id: str | None = None
    parent_id: str | None = None
    terminal: bool = False
    data: object | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence must not be negative")
        if not self.provider_type.strip() or not self.kind.strip():
            raise ValueError("event provider_type and kind must not be empty")


@dataclass(frozen=True)
class ProviderEventBatch:
    events: tuple[AgentEvent, ...]
    next_cursor: str | None = None
    terminal: bool = False


@dataclass(frozen=True)
class ContextUsageCategory:
    kind: str
    name: str
    tokens: int | None = None
    deferred: bool | None = None
    measurement: str = "unavailable"
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if self.tokens is not None and self.tokens < 0:
            raise ValueError("category tokens must not be negative")


@dataclass(frozen=True)
class ProviderContextUsage:
    session_id: str | None
    observed_at: str
    source: str
    available: bool
    used_tokens: int | None = None
    context_window_tokens: int | None = None
    effective_context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    reserved_output_tokens: int | None = None
    remaining_tokens: int | None = None
    categories: tuple[ContextUsageCategory, ...] = ()
    auto_compact_enabled: bool | None = None
    auto_compact_threshold_tokens: int | None = None
    compact_capability: CapabilitySupport | None = None
    measurement: str = "unavailable"
    as_of_turn_id: str | None = None
    stale: bool = False
    reason: str | None = None
    model_identity: ModelBackendIdentity | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        for name in (
            "used_tokens",
            "max_output_tokens",
            "reserved_output_tokens",
            "remaining_tokens",
            "auto_compact_threshold_tokens",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")
        for name in ("context_window_tokens", "effective_context_window_tokens"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.available and self.used_tokens is None:
            raise ValueError("available context usage requires used_tokens")

    @property
    def usage_ratio(self) -> float | None:
        window = self.effective_context_window_tokens or self.context_window_tokens
        if not self.available or self.used_tokens is None or window is None:
            return None
        return self.used_tokens / window

    def for_agent(self, *, agent_id: str, provider_type: str) -> "AgentContextUsage":
        return AgentContextUsage(
            session_id=self.session_id,
            observed_at=self.observed_at,
            source=self.source,
            available=self.available,
            used_tokens=self.used_tokens,
            context_window_tokens=self.context_window_tokens,
            effective_context_window_tokens=self.effective_context_window_tokens,
            max_output_tokens=self.max_output_tokens,
            reserved_output_tokens=self.reserved_output_tokens,
            remaining_tokens=self.remaining_tokens,
            categories=self.categories,
            auto_compact_enabled=self.auto_compact_enabled,
            auto_compact_threshold_tokens=self.auto_compact_threshold_tokens,
            compact_capability=self.compact_capability,
            measurement=self.measurement,
            as_of_turn_id=self.as_of_turn_id,
            stale=self.stale,
            reason=self.reason,
            model_identity=self.model_identity,
            provider_payload=self.provider_payload,
            agent_id=agent_id,
            provider_type=provider_type,
        )


@dataclass(frozen=True)
class AgentContextUsage(ProviderContextUsage):
    agent_id: str = ""
    provider_type: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.agent_id.strip() or not self.provider_type.strip():
            raise ValueError("agent_id and provider_type must not be empty")


@dataclass(frozen=True)
class ProviderTurnResult:
    provider_type: str
    run_id: str
    session_locator: ProviderSessionLocator
    status: ProviderRunState
    started_at: str
    completed_at: str
    schema_version: int = 1
    turn_locator: ProviderTurnLocator | None = None
    duration_ms: float | None = None
    final_text: str | None = None
    structured_output: object | None = None
    content_blocks: tuple[AgentContentBlock, ...] = ()
    tool_calls: tuple[AgentToolCall, ...] = ()
    request_usages: tuple[ModelRequestUsage, ...] = ()
    turn_usage: AgentTurnUsage | None = None
    context_after: ProviderContextUsage | None = None
    error: AgentError | None = None
    event_cursor: str | None = None
    artifact_locator: AgentArtifactLocator | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if self.provider_type != self.session_locator.provider_type:
            raise ValueError("result and session locator provider_type differ")
        if not self.status.terminal and self.status != ProviderRunState.NEEDS_INPUT:
            raise ValueError("turn result must be terminal or needs_input")
        if self.status == ProviderRunState.FAILED and self.error is None:
            raise ValueError("failed turn result requires error")
        if self.duration_ms is not None and self.duration_ms < 0:
            raise ValueError("duration_ms must not be negative")


@dataclass(frozen=True)
class AgentTurnResult:
    agent_id: str
    scope_id: str
    agent_type: str
    home_id: str
    provider_result: ProviderTurnResult
    completion: object | None = None

    @property
    def provider_type(self) -> str:
        return self.provider_result.provider_type

    @property
    def run_id(self) -> str:
        return self.provider_result.run_id

    @property
    def session_locator(self) -> ProviderSessionLocator:
        return self.provider_result.session_locator

    @property
    def turn_locator(self) -> ProviderTurnLocator | None:
        return self.provider_result.turn_locator

    @property
    def status(self) -> ProviderRunState:
        return self.provider_result.status

    @property
    def final_text(self) -> str | None:
        return self.provider_result.final_text


@dataclass(frozen=True)
class ProviderControlRequest:
    action: ProviderControlAction
    requested_at: str
    session_id: str | None = None
    turn_id: str | None = None
    run_id: str | None = None
    content: object | None = None
    options: Mapping[str, object] = field(default_factory=dict)
    provider_options: object | None = None


@dataclass(frozen=True)
class ProviderControlResult:
    action: ProviderControlAction
    accepted: bool
    terminal_confirmed: bool
    requested_at: str
    completed_at: str
    resulting_state: ProviderRunState | None = None
    session_locator: ProviderSessionLocator | None = None
    turn_locator: ProviderTurnLocator | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()
    error: AgentError | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if self.terminal_confirmed and self.resulting_state is not None and not self.resulting_state.terminal:
            raise ValueError("terminal_confirmed requires a terminal resulting_state")


@dataclass(frozen=True)
class AgentControlResult:
    agent_id: str
    scope_id: str
    provider_type: str
    provider_result: ProviderControlResult


@dataclass(frozen=True)
class ProviderContextCompactionResult:
    session_id: str | None
    status: str
    reason: str
    started_at: str
    completed_at: str
    usage_after: ProviderContextUsage | None = None
    provider_operation_id: str | None = None
    provider_payload: ProviderPayload | None = None


@dataclass(frozen=True)
class ProviderForkRequest:
    source_agent_id: str
    source_session: ProviderSessionLocator
    target_agent_id: str
    target_scope_id: str
    target_home_id: str
    source_turn: ProviderTurnLocator | None = None
    fork_mode: str = "session_only"
    source_workspace_revision: str | None = None
    provider_options: object | None = None
    execution_context: ProviderExecutionContext | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.fork_mode != "session_only":
            raise ValueError("ARK currently defines provider fork as session_only")


@dataclass(frozen=True)
class ProviderForkResult:
    source_session: ProviderSessionLocator
    target_session: ProviderSessionLocator
    status: str
    source_turn: ProviderTurnLocator | None = None
    target_turn: ProviderTurnLocator | None = None
    fork_mode: str = "session_only"
    workspace_isolated: bool = False
    artifact_locator: AgentArtifactLocator | None = None
    limitations: tuple[str, ...] = ()
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if self.fork_mode != "session_only" or self.workspace_isolated:
            raise ValueError("provider fork must not claim workspace isolation")


@dataclass(frozen=True)
class Page:
    items: tuple[object, ...]
    next_cursor: str | None = None
