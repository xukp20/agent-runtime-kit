from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TraceReportPersistence(StrEnum):
    DISABLED = "disabled"
    LATEST_ONLY = "latest_only"
    LATEST_AND_TURNS = "latest_and_turns"


@dataclass(frozen=True)
class AgentTraceReportPolicy:
    # Raw provider artifacts are the canonical trace evidence.  Materialized
    # reports are an explicit operator opt-in so report serialization does not
    # sit on the Agent completion critical path by default.
    persistence: TraceReportPersistence = TraceReportPersistence.DISABLED
    include_in_snapshots: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "persistence", TraceReportPersistence(self.persistence))
