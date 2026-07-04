from __future__ import annotations

import json
from pathlib import Path

from agent_runtime_kit.agent.models import to_jsonable
from agent_runtime_kit.agent.store import AgentStoreService


def test_trace_parses_single_turn_response_and_tool_call(tmp_path: Path) -> None:
    store, agent_id = _store_with_rollout(
        tmp_path,
        [
            {
                "timestamp": "2026-07-02T01:00:00.000Z",
                "type": "turn_context",
                "payload": {"turn_id": "turn-1"},
            },
            {
                "timestamp": "2026-07-02T01:00:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "run_lean_f_abcdef",
                    "call_id": "call-1",
                    "arguments": "{\"file_path\":\"A.lean\"}",
                },
            },
            {
                "timestamp": "2026-07-02T01:00:01.050Z",
                "type": "event_msg",
                "payload": {
                    "type": "mcp_tool_call_end",
                    "call_id": "call-1",
                    "invocation": {
                        "server": "lean-constellation-tools-application",
                        "tool": "run_lean_file_diagnostics",
                        "arguments": {"file_path": "A.lean"},
                    },
                    "duration": {"secs": 0, "nanos": 10_000_000},
                    "result": {"Ok": {"structuredContent": {"ok": True, "summary": "passed"}}},
                },
            },
            {
                "timestamp": "2026-07-02T01:00:03.250Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "{\"ok\":true,\"summary\":\"passed\"}",
                },
            },
            {
                "timestamp": "2026-07-02T01:00:04.000Z",
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "done"},
            },
            {
                "timestamp": "2026-07-02T01:00:05.000Z",
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "turn-1",
                    "last_agent_message": "final done",
                    "duration_ms": 5000,
                },
            },
        ],
    )

    info = store.get_rollout_info(agent_id)
    turns = store.list_trace_turns(agent_id)
    responses = store.list_response_texts(agent_id, latest=True)
    calls = store.list_tool_calls(agent_id, latest=True)

    assert info.exists is True
    assert info.event_count == 6
    assert len(turns) == 1
    assert turns[0].turn_id == "turn-1"
    assert turns[0].status == "completed"
    assert turns[0].final_response == "final done"
    assert [response.text for response in responses] == ["done", "final done"]
    assert len(calls) == 1
    assert calls[0].call_id == "call-1"
    assert calls[0].tool_name == "run_lean_file_diagnostics"
    assert calls[0].display_name == "run_lean_f_abcdef"
    assert calls[0].arguments == {"file_path": "A.lean"}
    assert calls[0].output == {"Ok": {"structuredContent": {"ok": True, "summary": "passed"}}}
    assert calls[0].ok is True
    assert calls[0].duration_ms == 10


def test_trace_parses_multi_turn_latest_and_index_accessors(tmp_path: Path) -> None:
    store, agent_id = _store_with_rollout(
        tmp_path,
        [
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "first"}},
            {
                "type": "event_msg",
                "payload": {"type": "task_complete", "turn_id": "turn-1", "last_agent_message": "first final"},
            },
            {"type": "turn_context", "payload": {"turn_id": "turn-2"}},
            {"type": "event_msg", "payload": {"type": "agent_message", "message": "second"}},
            {
                "type": "event_msg",
                "payload": {"type": "task_complete", "turn_id": "turn-2", "last_agent_message": "second final"},
            },
        ],
    )

    assert store.get_trace_turn(agent_id, index=0).turn_id == "turn-1"
    assert store.get_trace_turn(agent_id, latest=True).turn_id == "turn-2"
    assert store.get_trace_event(agent_id, last=True).payload_type == "task_complete"
    assert [event.index for event in store.tail_trace_events(agent_id, limit=2)] == [4, 5]
    assert [event.index for event in store.tail_trace_events(agent_id, limit=4, payload_type="agent_message")] == [1, 4]
    assert store.get_latest_response_text(agent_id) == "second final"


def test_trace_preserves_unknown_event_and_reports_missing_rollout(tmp_path: Path) -> None:
    store, agent_id = _store_with_rollout(
        tmp_path,
        [
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            {"type": "mystery", "payload": {"type": "unknown_payload", "turn_id": "turn-1", "value": 3}},
        ],
    )

    event = store.get_trace_event(agent_id, index=1)
    assert event.event_type == "mystery"
    assert event.payload_type == "unknown_payload"
    assert event.raw_event["payload"]["value"] == 3

    missing_agent = store.create_agent_record(
        scope_id="scope",
        agent_type="worker",
        cli_type="codex",
        home_id="worker",
        thread_id="thread-missing",
        rollout_relpath="sessions/missing.jsonl",
    )
    report = store.build_trace_report(missing_agent.agent_id)
    assert report.rollout.exists is False
    assert report.turns == []
    assert "rollout file is missing" in report.warnings


def test_trace_pairs_out_of_order_tool_output_with_call(tmp_path: Path) -> None:
    store, agent_id = _store_with_rollout(
        tmp_path,
        [
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            {
                "type": "response_item",
                "payload": {"type": "function_call_output", "call_id": "late-call", "output": "ok"},
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "submit_repo_ready",
                    "call_id": "late-call",
                    "arguments": {"summary": "ready"},
                },
            },
        ],
    )

    call = store.get_tool_call(agent_id, call_id="late-call")

    assert call is not None
    assert call.tool_name == "submit_repo_ready"
    assert call.output == "ok"
    assert store.build_trace_report(agent_id).warnings == []


def test_trace_builds_report_and_exports_json_and_markdown(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "prompt_marker_seen": "marker",
                "skill_keys_seen": ["skill-a"],
                "application_tools_called": ["tool-a"],
                "tool_results": {"tool-a": {"ok": True}},
            }
        ),
        encoding="utf-8",
    )
    store, agent_id = _store_with_rollout(
        tmp_path,
        [
            {"type": "turn_context", "payload": {"turn_id": "turn-1"}},
            {"type": "response_item", "payload": {"type": "function_call", "name": "tool-a", "call_id": "call-a"}},
            {
                "type": "event_msg",
                "payload": {"type": "task_complete", "turn_id": "turn-1", "last_agent_message": "ok"},
            },
        ],
    )
    json_path = tmp_path / "trace.json"
    markdown_path = tmp_path / "trace.md"

    report = store.export_trace_report(agent_id, output_path=json_path, artifact_path=artifact_path)
    store.export_trace_report(agent_id, output_path=markdown_path, format="markdown", artifact_path=artifact_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == to_jsonable(report)
    assert payload["artifact"]["summary"]["skill_keys_seen"]["items"] == ["skill-a"]
    assert "Agent Trace Report" in markdown_path.read_text(encoding="utf-8")


def test_trace_exports_and_reads_default_reports(tmp_path: Path) -> None:
    store, agent_id = _store_with_rollout(
        tmp_path,
        [
            {"type": "turn_context", "payload": {"turn_id": "turn/with/slash"}},
            {
                "type": "event_msg",
                "payload": {"type": "task_complete", "turn_id": "turn/with/slash", "last_agent_message": "ok"},
            },
        ],
    )

    paths = store.export_default_trace_reports(agent_id)

    assert Path(paths.latest_json_path).exists()
    assert Path(paths.latest_markdown_path).exists()
    assert paths.turn_json_path is not None
    assert paths.turn_markdown_path is not None
    assert Path(paths.turn_json_path).name == "turn_with_slash.json"
    assert store.read_default_trace_report(agent_id)["latest_turn"]["final_response"] == "ok"
    assert "Agent Trace Report" in store.read_default_trace_report(agent_id, format="markdown")


def _store_with_rollout(
    tmp_path: Path,
    events: list[dict[str, object]],
) -> tuple[AgentStoreService, str]:
    runtime_root = tmp_path / ".agent_runtime"
    rollout = runtime_root / "homes" / "codex" / "worker" / ".codex" / "sessions" / "trace.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    store = AgentStoreService(runtime_root)
    agent = store.create_agent_record(
        scope_id="scope",
        agent_type="worker",
        cli_type="codex",
        home_id="worker",
        thread_id="thread-1",
        rollout_relpath="sessions/trace.jsonl",
    )
    return store, agent.agent_id
