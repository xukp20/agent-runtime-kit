from __future__ import annotations

from dataclasses import dataclass

from .locators import ProviderSessionLocator, ProviderTurnLocator
from .models import AgentEvent, AgentToolCall, ProviderTurnResult
from .usage import AgentSessionUsage, AgentTurnUsage


@dataclass(frozen=True)
class AgentSessionView:
    locator: ProviderSessionLocator
    status: str | None = None
    turns: tuple["AgentTurnView", ...] = ()
    usage: AgentSessionUsage | None = None


@dataclass(frozen=True)
class AgentTurnView:
    locator: ProviderTurnLocator
    result: ProviderTurnResult | None = None
    events: tuple[AgentEvent, ...] = ()
    tool_calls: tuple[AgentToolCall, ...] = ()
    usage: AgentTurnUsage | None = None


@dataclass(frozen=True)
class ProviderSessionListQuery:
    home_id: str | None = None
    cursor: str | None = None
    limit: int = 100


@dataclass(frozen=True)
class ProviderSessionQuery:
    locator: ProviderSessionLocator
    include_turns: bool = True


@dataclass(frozen=True)
class ProviderTurnQuery:
    session: ProviderSessionLocator
    turn: ProviderTurnLocator | None = None
    cursor: str | None = None
    limit: int = 100
    latest: bool = False


@dataclass(frozen=True)
class ProviderEventQuery(ProviderTurnQuery):
    kind: str | None = None


@dataclass(frozen=True)
class ProviderToolQuery(ProviderTurnQuery):
    call_id: str | None = None


@dataclass(frozen=True)
class ProviderUsageQuery(ProviderTurnQuery):
    include_session_aggregate: bool = False


@dataclass(frozen=True)
class ProviderContextQuery:
    session: ProviderSessionLocator


@dataclass(frozen=True)
class ProviderContextCompactionRequest:
    session: ProviderSessionLocator
    trigger: str
    timeout_s: float | None = None
    provider_options: object | None = None


@dataclass(frozen=True)
class ProviderContextReconcileRequest:
    session: ProviderSessionLocator
    operation_id: str | None = None
    baseline: object | None = None
