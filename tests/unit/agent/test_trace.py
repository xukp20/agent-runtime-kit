from __future__ import annotations

import json

from agent_runtime_kit.agent.provider_contracts import (
    AgentSessionView,
    AgentTurnView,
    ProviderSessionLocator,
    ProviderTurnLocator,
)
from agent_runtime_kit.agent.trace import AgentTraceReport, export_trace_report


def test_provider_neutral_trace_report_exports_json_and_markdown(tmp_path) -> None:  # noqa: ANN001
    session = ProviderSessionLocator(
        provider_type="fake",
        session_id="session-1",
        home_id="home-1",
        created_at="2026-07-22T00:00:00Z",
    )
    turn = AgentTurnView(locator=ProviderTurnLocator(session=session, turn_id="turn-1"))
    report = AgentTraceReport(
        agent_id="agent-1",
        provider_type="fake",
        session=session,
        turns=(turn,),
    )
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"

    export_trace_report(report, json_path, "json")
    export_trace_report(report, markdown_path, "markdown")

    assert json.loads(json_path.read_text())["session"]["session_id"] == "session-1"
    assert "Provider: `fake`" in markdown_path.read_text()
    assert report.latest_turn == turn
