from __future__ import annotations

import json
from pathlib import Path

from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry


class _QueryAgentType(AgentType):
    agent_type = "query-agent"
    start_prompt_template = "query"


def test_codex_standard_query_matches_legacy_trace_projection(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(_QueryAgentType())
    service = AgentService(
        runtime_root,
        agent_types=registry,
        providers={"codex": CodexProvider(runtime_root=runtime_root)},
    )
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="query-agent"))
    agent = service.create_agent("scope", "query-agent")
    rollout_relpath = "sessions/trace.jsonl"
    service.store.update_thread_locator(
        agent.agent_id,
        thread_id="thread-1",
        rollout_relpath=rollout_relpath,
    )
    rollout = runtime_root / "homes" / "codex" / "query-agent" / ".codex" / rollout_relpath
    rollout.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "type": "turn_context",
            "payload": {
                "turn_id": "turn-1",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "lean_check",
                "call_id": "call-1",
                "arguments": {"file": "Main.lean"},
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": {"ok": True},
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": "turn-1",
                "last_agent_message": "done",
            },
        },
    ]
    rollout.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    turns = service.query_turns(agent.agent_id)
    latest = service.query_turn(agent.agent_id, latest=True)
    first_events = service.query_events(agent.agent_id, limit=2)
    second_events = service.query_events(agent.agent_id, cursor=first_events.next_cursor, limit=2)
    tools = service.query_tool_calls(agent.agent_id)
    usage = service.query_usage(agent.agent_id, latest=True)

    assert len(turns.items) == 1
    assert latest.locator.turn_id == "turn-1"
    assert latest.result.final_text == "done"
    assert [event.sequence for event in first_events.items] == [0, 1]
    assert [event.sequence for event in second_events.items] == [2, 3]
    assert len(tools.items) == 1
    assert tools.items[0].call_id == "call-1"
    assert tools.items[0].tool_name == "lean_check"
    assert tools.items[0].result == {"ok": True}
    assert usage.token_usage.total_tokens == 15

    # COMPAT: existing rollout/trace entry points keep their previous shapes.
    assert service.list_trace_turns(agent.agent_id)[0].turn_id == "turn-1"
    assert service.list_tool_calls(agent.agent_id)[0].tool_name == "lean_check"
