from __future__ import annotations

import json

from agent_runtime_kit.agent.provider_contracts import (
    AgentTurnView,
    ProviderSessionLocator,
    ProviderTurnLocator,
)
from agent_runtime_kit.agent.providers.codex_trace import AgentTraceReader
from agent_runtime_kit.agent.report_policy import AgentTraceReportPolicy, TraceReportPersistence
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


def test_trace_report_policy_defaults_to_disabled() -> None:
    assert AgentTraceReportPolicy().persistence is TraceReportPersistence.DISABLED


def test_codex_trace_synthesizes_end_only_mcp_call() -> None:
    reader = AgentTraceReader(
        events=[
            {
                "timestamp": "2026-01-01T00:00:01.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "mcp_tool_call_end",
                    "turn_id": "turn-1",
                    "call_id": "call-1",
                    "invocation": {
                        "server": "lc_app",
                        "tool": "diagnostics",
                        "arguments": {"path": "Main.lean"},
                    },
                    "duration": {"secs": 1, "nanos": 500_000_000},
                    "result": {"Ok": {"content": []}},
                },
            }
        ]
    )

    calls = reader.list_tool_calls()

    assert len(calls) == 1
    call = calls[0]
    assert call.tool_kind == "mcp"
    assert call.server_name == "lc_app"
    assert call.tool_name == "diagnostics"
    assert call.duration_ms == 1500
    assert call.started_at == "2025-12-31T23:59:59.500000Z"
    assert call.completed_at == "2026-01-01T00:00:01.000Z"
