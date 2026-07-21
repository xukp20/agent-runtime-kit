"""Legacy Codex trace API compatibility exports.

Provider-neutral callers should use ``AgentService.query_*``.  The rollout
parser is implemented by the Codex Query adapter; these names remain here so
existing ARK and Lean Constellation imports keep their original behavior.
"""

# COMPAT(legacy-codex-trace-module): remove when callers no longer import the
# Codex rollout projection from agent.trace and all report APIs use QueryAdapter.
# Covered by tests/unit/agent/test_trace.py and LC observability tests.
from .providers.codex_trace import (
    AgentArtifactView,
    AgentResponseTextView,
    AgentRolloutInfo,
    AgentToolCallView,
    AgentTraceEventView,
    AgentTraceReader,
    AgentTraceReport,
    AgentTraceReportPaths,
    AgentTurnSummary,
)

__all__ = [
    "AgentArtifactView",
    "AgentResponseTextView",
    "AgentRolloutInfo",
    "AgentToolCallView",
    "AgentTraceEventView",
    "AgentTraceReader",
    "AgentTraceReport",
    "AgentTraceReportPaths",
    "AgentTurnSummary",
]
