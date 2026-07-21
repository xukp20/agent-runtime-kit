from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent_runtime_kit.agent.provider_contracts import (
    ModelBackendIdentity,
    ProviderRunState,
    ProviderSessionLocator,
)
from agent_runtime_kit.agent.providers.claude_code_normalization import (
    group_transcript_turns,
    normalize_stream_result,
)


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking: str) -> None:
        self.thinking = thinking
        self.signature = "sig"


class ToolUseBlock:
    def __init__(self) -> None:
        self.id = "call-1"
        self.name = "mcp__lean__diagnostics"
        self.input = {"file": "Main.lean"}


class ToolResultBlock:
    def __init__(self) -> None:
        self.tool_use_id = "call-1"
        self.content = "ok"
        self.is_error = False


class AssistantMessage:
    def __init__(self, content: list[object], message_id: str) -> None:
        self.content = content
        self.message_id = message_id
        self.model = "deepseek-v4"
        self.stop_reason = "end_turn"
        self.error = None
        self.usage = {
            "input_tokens": 10,
            "output_tokens": 4,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 2,
                "ephemeral_1h_input_tokens": 0,
            },
            "service_tier": "standard",
            "server_tool_use": {"web_search_requests": 0},
        }


class UserMessage:
    def __init__(self, content: list[object]) -> None:
        self.content = content


class ResultMessage:
    subtype = "success"
    is_error = False
    result = "done"
    structured_output = None
    total_cost_usd = 0.01
    num_turns = 1
    stop_reason = "end_turn"
    api_error_status = None
    usage = None
    model_usage = None
    deferred_tool_use = None
    errors = None


def test_claude_normalization_deduplicates_request_usage_and_maps_tools() -> None:
    session = ProviderSessionLocator(
        provider_type="claude_code",
        session_id="11111111-1111-4111-8111-111111111111",
        home_id="worker",
        created_at="2026-07-21T00:00:00Z",
        backend_identity=ModelBackendIdentity(
            api_provider="deepseek",
            api_mode="anthropic_messages",
            requested_model="deepseek-chat",
        ),
    )
    messages = [
        AssistantMessage([ThinkingBlock("think"), ToolUseBlock()], "response-1"),
        AssistantMessage([TextBlock("done")], "response-1"),
        UserMessage([ToolResultBlock()]),
    ]

    result = normalize_stream_result(
        run_id="run-1",
        session=session,
        turn_id="turn-1",
        messages=messages,
        terminal=ResultMessage(),
        started_at="2026-07-21T00:00:00Z",
        completed_at="2026-07-21T00:00:01Z",
        duration_ms=1000,
        interrupted=False,
    )

    assert result.status is ProviderRunState.COMPLETED
    assert result.final_text == "done"
    assert len(result.request_usages) == 1
    usage = result.request_usages[0]
    assert usage.model_identity.api_provider == "deepseek"
    assert usage.model_identity.resolved_model == "deepseek-v4"
    assert usage.token_usage.total_tokens is None
    assert usage.token_usage.cache_read_input_tokens == 3
    assert usage.token_usage.reasoning_output_tokens is None
    assert result.tool_calls[0].tool_kind == "mcp"
    assert result.tool_calls[0].server_name == "lean"
    assert result.tool_calls[0].tool_name == "diagnostics"
    assert result.turn_usage.reported_costs[0].total_cost == "0.01"


def test_claude_transcript_turn_grouping_ignores_tool_results_and_compact_summary() -> None:
    records = [
        {"type": "user", "uuid": "turn-1", "message": {"content": "first"}},
        {
            "type": "user",
            "uuid": "tool-result",
            "message": {"content": [{"type": "tool_result", "tool_use_id": "c"}]},
        },
        {
            "type": "user",
            "uuid": "summary",
            "isCompactSummary": True,
            "message": {"content": "summary"},
        },
        {
            "type": "user",
            "uuid": "compact-command",
            "message": {"content": "<command-name>/compact</command-name>"},
        },
        {
            "type": "user",
            "uuid": "compact-output",
            "message": {"content": "<local-command-stdout>Compacted</local-command-stdout>"},
        },
        {"type": "user", "uuid": "turn-2", "message": {"content": "second"}},
    ]

    turns = group_transcript_turns(records)

    assert [turn.turn_id for turn in turns] == ["turn-1", "turn-2"]
    assert len(turns[0].records) == 5
