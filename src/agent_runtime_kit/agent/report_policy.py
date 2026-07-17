from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TraceReportPersistence(StrEnum):
    DISABLED = "disabled"
    LATEST_ONLY = "latest_only"
    LATEST_AND_TURNS = "latest_and_turns"


@dataclass(frozen=True)
class AgentTraceReportPolicy:
    persistence: TraceReportPersistence = TraceReportPersistence.LATEST_ONLY
    include_in_snapshots: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "persistence", TraceReportPersistence(self.persistence))
