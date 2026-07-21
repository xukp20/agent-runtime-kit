from __future__ import annotations

import json
from pathlib import Path

from agent_runtime_kit.agent.provider_contracts import ProviderRunState
from agent_runtime_kit.agent.providers.pi_session import PiSessionTranscript, find_pi_session


def _write_session(path: Path) -> None:
    records = [
        {
            "type": "session",
            "version": 3,
            "id": "session-1",
            "timestamp": "2026-07-21T00:00:00Z",
            "cwd": "/workspace",
        },
        {
            "type": "model_change",
            "id": "model-entry",
            "parentId": None,
            "timestamp": "2026-07-21T00:00:00Z",
            "provider": "deepseek",
            "modelId": "deepseek-chat",
        },
        {
            "type": "message",
            "id": "user-1",
            "parentId": "model-entry",
            "timestamp": "2026-07-21T00:00:01Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "write"}]},
        },
        {
            "type": "message",
            "id": "assistant-1",
            "parentId": "user-1",
            "timestamp": "2026-07-21T00:00:02Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "reason"},
                    {
                        "type": "toolCall",
                        "id": "call-1",
                        "name": "mcp__demo__echo",
                        "arguments": {"text": "hello"},
                    },
                ],
                "api": "openai-completions",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "responseId": "response-1",
                "usage": {
                    "input": 100,
                    "output": 20,
                    "cacheRead": 30,
                    "cacheWrite": 4,
                    "cacheWrite1h": 2,
                    "reasoning": 5,
                    "totalTokens": 154,
                    "cost": {
                        "input": 0.1,
                        "output": 0.2,
                        "cacheRead": 0.01,
                        "cacheWrite": 0.02,
                        "total": 0.33,
                    },
                },
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "id": "tool-1",
            "parentId": "assistant-1",
            "timestamp": "2026-07-21T00:00:03Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call-1",
                "toolName": "mcp__demo__echo",
                "content": [{"type": "text", "text": "hello"}],
                "isError": False,
            },
        },
        {
            "type": "message",
            "id": "assistant-2",
            "parentId": "tool-1",
            "timestamp": "2026-07-21T00:00:04Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "api": "openai-completions",
                "provider": "deepseek",
                "model": "deepseek-chat",
                "responseId": "response-2",
                "usage": {
                    "input": 10,
                    "output": 2,
                    "cacheRead": 100,
                    "cacheWrite": 0,
                    "reasoning": 0,
                    "totalTokens": 112,
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                },
                "stopReason": "stop",
            },
        },
    ]
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_pi_session_projects_turn_tools_usage_and_model(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_session(path)

    transcript = PiSessionTranscript.read(path)
    turns = transcript.turns()
    assert transcript.session_id == "session-1"
    assert len(turns) == 1

    result = transcript.turn_result(turns[0], home_id="home-1")
    assert result.status is ProviderRunState.COMPLETED
    assert result.final_text == "done"
    assert result.session_locator.backend_identity is not None
    assert result.session_locator.backend_identity.api_provider == "deepseek"
    assert result.session_locator.backend_identity.api_mode == "chat_completions"
    assert result.turn_usage is not None
    assert result.turn_usage.request_count == 2
    assert result.turn_usage.token_usage.input_tokens == 110
    assert result.turn_usage.token_usage.reasoning_output_tokens == 5
    assert result.request_usages[0].response_id == "response-1"
    assert result.request_usages[0].token_usage.cache_creation_1h_input_tokens == 2
    assert result.request_usages[0].reported_cost is not None
    assert result.request_usages[0].reported_cost.total_cost == "0.33"
    assert result.tool_calls[0].tool_kind == "mcp"
    assert result.tool_calls[0].result == [{"type": "text", "text": "hello"}]


def test_pi_session_active_branch_excludes_abandoned_entries(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    _write_session(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records.insert(
        -1,
        {
            "type": "message",
            "id": "abandoned",
            "parentId": "user-1",
            "timestamp": "2026-07-21T00:00:03Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "wrong"}]},
        },
    )
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")

    transcript = PiSessionTranscript.read(path)
    assert "abandoned" not in {str(entry["id"]) for entry in transcript.active_entries()}


def test_find_pi_session_matches_header_id(tmp_path: Path) -> None:
    path = tmp_path / "opaque-name.jsonl"
    _write_session(path)
    assert find_pi_session(tmp_path, "session-1") == path
    assert find_pi_session(tmp_path, "missing") is None

