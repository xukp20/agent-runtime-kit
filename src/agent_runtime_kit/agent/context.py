from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentContextCompactionStatus(str, Enum):
    COMPACTED = "compacted"
    SKIPPED = "skipped"
    UNSUPPORTED = "unsupported"


class AgentContextMaintenanceJournalStatus(str, Enum):
    PREPARED = "prepared"
    STARTED = "started"
    CONFIRMED = "confirmed"
    UNKNOWN_TERMINAL = "unknown_terminal"


@dataclass(frozen=True)
class AgentContextUsage:
    agent_id: str
    provider_type: str
    session_id: str | None
    total_tokens: int | None
    context_window: int | None
    observed_at: str
    source: str
    available: bool
    reason: str | None = None
    usage_ratio: float | None = field(init=False)

    def __post_init__(self) -> None:
        _validate_usage_values(
            total_tokens=self.total_tokens,
            context_window=self.context_window,
            available=self.available,
        )
        ratio = None
        if self.available:
            assert self.total_tokens is not None
            assert self.context_window is not None
            ratio = self.total_tokens / self.context_window
        object.__setattr__(self, "usage_ratio", ratio)


@dataclass(frozen=True)
class ProviderContextUsage:
    session_id: str | None
    total_tokens: int | None
    context_window: int | None
    observed_at: str
    source: str
    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        _validate_usage_values(
            total_tokens=self.total_tokens,
            context_window=self.context_window,
            available=self.available,
        )

    def for_agent(self, *, agent_id: str, provider_type: str) -> AgentContextUsage:
        return AgentContextUsage(
            agent_id=agent_id,
            provider_type=provider_type,
            session_id=self.session_id,
            total_tokens=self.total_tokens,
            context_window=self.context_window,
            observed_at=self.observed_at,
            source=self.source,
            available=self.available,
            reason=self.reason,
        )


@dataclass(frozen=True)
class AgentContextMaintenancePolicy:
    enabled: bool = True
    threshold: float = 0.80
    timeout_s: float = 120.0

    def __post_init__(self) -> None:
        if not 0 < self.threshold <= 1:
            raise ValueError("context maintenance threshold must be in (0, 1]")
        if self.timeout_s <= 0:
            raise ValueError("context maintenance timeout_s must be positive")


@dataclass(frozen=True)
class AgentContextCompactionResult:
    agent_id: str
    provider_type: str
    session_id: str | None
    status: AgentContextCompactionStatus
    reason: str
    usage_before: AgentContextUsage
    usage_after: AgentContextUsage | None
    started_at: str
    completed_at: str
    provider_operation_id: str | None = None


@dataclass(frozen=True)
class ProviderContextCompactionResult:
    session_id: str | None
    usage_after: ProviderContextUsage | None
    started_at: str
    completed_at: str
    provider_operation_id: str | None = None


@dataclass(frozen=True)
class AgentContextMaintenanceJournal:
    agent_id: str
    provider_type: str
    session_id: str | None
    status: AgentContextMaintenanceJournalStatus
    trigger: str
    prepared_at: str
    started_at: str | None = None
    completed_at: str | None = None
    provider_operation_id: str | None = None
    baseline: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "provider_type": self.provider_type,
            "session_id": self.session_id,
            "status": self.status.value,
            "trigger": self.trigger,
            "prepared_at": self.prepared_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "provider_operation_id": self.provider_operation_id,
            "baseline": self.baseline,
            "error_type": self.error_type,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentContextMaintenanceJournal":
        schema_version = int(payload.get("schema_version", 1))
        if schema_version != 1:
            raise ValueError(f"unsupported context maintenance journal schema: {schema_version}")
        return cls(
            schema_version=schema_version,
            agent_id=str(payload["agent_id"]),
            provider_type=str(payload["provider_type"]),
            session_id=_optional_str(payload.get("session_id")),
            status=AgentContextMaintenanceJournalStatus(str(payload["status"])),
            trigger=str(payload["trigger"]),
            prepared_at=str(payload["prepared_at"]),
            started_at=_optional_str(payload.get("started_at")),
            completed_at=_optional_str(payload.get("completed_at")),
            provider_operation_id=_optional_str(payload.get("provider_operation_id")),
            baseline=dict(payload.get("baseline") or {}),
            error_type=_optional_str(payload.get("error_type")),
        )

    @property
    def unresolved(self) -> bool:
        return self.status in {
            AgentContextMaintenanceJournalStatus.PREPARED,
            AgentContextMaintenanceJournalStatus.STARTED,
            AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL,
        }


def _validate_usage_values(
    *,
    total_tokens: int | None,
    context_window: int | None,
    available: bool,
) -> None:
    if total_tokens is not None and total_tokens < 0:
        raise ValueError("total_tokens must not be negative")
    if context_window is not None and context_window <= 0:
        raise ValueError("context_window must be positive")
    if available and (total_tokens is None or context_window is None):
        raise ValueError("available context usage requires total_tokens and context_window")


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
