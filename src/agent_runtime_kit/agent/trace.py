from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .models import to_jsonable
from .provider_contracts import (
    AgentEvent,
    AgentSessionUsage,
    AgentToolCall,
    AgentTurnUsage,
    AgentTurnView,
    ProviderSessionLocator,
)


@dataclass(frozen=True)
class AgentTraceReport:
    """Provider-neutral, query-adapter-backed Agent observation report."""

    agent_id: str
    provider_type: str
    session: ProviderSessionLocator
    turns: tuple[AgentTurnView, ...] = ()
    events: tuple[AgentEvent, ...] = ()
    tool_calls: tuple[AgentToolCall, ...] = ()
    usage: AgentTurnUsage | AgentSessionUsage | None = None
    warnings: tuple[str, ...] = ()

    @property
    def latest_turn(self) -> AgentTurnView | None:
        return self.turns[-1] if self.turns else None


@dataclass(frozen=True)
class AgentTraceReportPaths:
    agent_id: str
    reports_root: str
    latest_json_path: str
    latest_markdown_path: str
    turn_json_path: str | None = None
    turn_markdown_path: str | None = None
    written_paths: tuple[str, ...] = field(default_factory=tuple)


def export_trace_report(report: AgentTraceReport, output_path: Path, format: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "json":
        path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    if format != "markdown":
        raise ValueError(f"unsupported trace report format: {format}")
    latest = report.latest_turn
    lines = [
        f"# Agent report: {report.agent_id}",
        "",
        f"- Provider: `{report.provider_type}`",
        f"- Session: `{report.session.session_id}`",
        f"- Turns: {len(report.turns)}",
        f"- Events: {len(report.events)}",
        f"- Tool calls: {len(report.tool_calls)}",
    ]
    if latest is not None:
        lines.extend(
            [
                f"- Latest turn: `{latest.locator.turn_id}`",
                f"- Latest status: `{latest.result.status.value if latest.result is not None else 'unknown'}`",
            ]
        )
        if latest.result is not None and latest.result.final_text:
            lines.extend(["", "## Latest response", "", latest.result.final_text])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = ["AgentTraceReport", "AgentTraceReportPaths", "export_trace_report"]
