"""Agent runtime primitives."""

from .instructions import InstructionService, TextFragment
from .models import (
    Agent,
    AgentCompletionRecord,
    AgentRuntimeKitError,
    CompletionDecision,
    WaitAgentsResult,
)
from .service import AgentCompletionContext
from .snapshots import AgentSnapshotService
from .templates import TemplateVariableError, render_template

__all__ = [
    "Agent",
    "AgentCompletionContext",
    "AgentCompletionRecord",
    "AgentRuntimeKitError",
    "AgentSnapshotService",
    "CompletionDecision",
    "InstructionService",
    "TemplateVariableError",
    "TextFragment",
    "WaitAgentsResult",
    "render_template",
]
