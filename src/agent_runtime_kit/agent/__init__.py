"""Agent runtime primitives."""

from .context import (
    AgentContextCompactionResult,
    AgentContextCompactionStatus,
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
    AgentContextMaintenancePolicy,
)
from .instructions import InstructionService, TextFragment
from .models import (
    Agent,
    AgentCompletionRecord,
    AgentForkInfo,
    AgentContextCompactionEvidenceError,
    AgentContextCompactionRequestUnknown,
    AgentContextCompactionTimeout,
    AgentContextMaintenanceBlocked,
    AgentContextMaintenanceError,
    AgentContextMaintenanceUnsupported,
    AgentContextUsageUnavailable,
    AgentRuntimeKitError,
    AgentStatusWaitResult,
    CompletionDecision,
    WaitAgentsResult,
)
from .report_policy import AgentTraceReportPolicy, TraceReportPersistence
from .service import AgentCompletionContext
from .skills import SkillService, SkillSpec, write_skill_spec
from .snapshots import AgentSnapshotService
from .templates import TemplateVariableError, render_template
from .provider_contracts import AgentContextUsage, ProviderContextCompactionResult, ProviderContextUsage
from .trace import AgentTraceReport, AgentTraceReportPaths

__all__ = [
    "Agent",
    "AgentCompletionContext",
    "AgentCompletionRecord",
    "AgentForkInfo",
    "AgentContextCompactionEvidenceError",
    "AgentContextCompactionResult",
    "AgentContextCompactionRequestUnknown",
    "AgentContextCompactionStatus",
    "AgentContextCompactionTimeout",
    "AgentContextMaintenanceBlocked",
    "AgentContextMaintenanceError",
    "AgentContextMaintenanceJournal",
    "AgentContextMaintenanceJournalStatus",
    "AgentContextMaintenancePolicy",
    "AgentContextMaintenanceUnsupported",
    "AgentContextUsage",
    "AgentContextUsageUnavailable",
    "AgentRuntimeKitError",
    "AgentStatusWaitResult",
    "AgentTraceReportPolicy",
    "AgentSnapshotService",
    "AgentTraceReport",
    "AgentTraceReportPaths",
    "CompletionDecision",
    "InstructionService",
    "ProviderContextCompactionResult",
    "ProviderContextUsage",
    "SkillService",
    "SkillSpec",
    "TemplateVariableError",
    "TextFragment",
    "TraceReportPersistence",
    "WaitAgentsResult",
    "render_template",
    "write_skill_spec",
]
