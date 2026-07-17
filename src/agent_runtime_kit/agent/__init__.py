"""Agent runtime primitives."""

from .instructions import InstructionService, TextFragment
from .models import (
    Agent,
    AgentCompletionRecord,
    AgentRuntimeKitError,
    CompletionDecision,
    WaitAgentsResult,
)
from .report_policy import AgentTraceReportPolicy, TraceReportPersistence
from .service import AgentCompletionContext
from .skills import SkillService, SkillSpec, write_skill_spec
from .snapshots import AgentSnapshotService
from .templates import TemplateVariableError, render_template
from .trace import (
    AgentArtifactView,
    AgentResponseTextView,
    AgentRolloutInfo,
    AgentToolCallView,
    AgentTraceEventView,
    AgentTraceReader,
    AgentTraceReport,
    AgentTurnSummary,
)

__all__ = [
    "Agent",
    "AgentCompletionContext",
    "AgentCompletionRecord",
    "AgentArtifactView",
    "AgentResponseTextView",
    "AgentRuntimeKitError",
    "AgentTraceReportPolicy",
    "AgentSnapshotService",
    "AgentRolloutInfo",
    "AgentToolCallView",
    "AgentTraceEventView",
    "AgentTraceReader",
    "AgentTraceReport",
    "AgentTurnSummary",
    "CompletionDecision",
    "InstructionService",
    "SkillService",
    "SkillSpec",
    "TemplateVariableError",
    "TextFragment",
    "TraceReportPersistence",
    "WaitAgentsResult",
    "render_template",
    "write_skill_spec",
]
