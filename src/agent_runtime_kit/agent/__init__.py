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
from .skills import SkillService, SkillSpec, write_skill_spec
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
    "SkillService",
    "SkillSpec",
    "TemplateVariableError",
    "TextFragment",
    "WaitAgentsResult",
    "render_template",
    "write_skill_spec",
]
