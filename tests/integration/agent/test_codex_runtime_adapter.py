from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from agent_runtime_kit.agent.provider_contracts import (
    ProviderExecutionContext,
    ProviderRunRequest,
    ProviderRunState,
    TokenUsage,
)
from agent_runtime_kit.agent.providers.codex import CodexTurnResult
from agent_runtime_kit.agent.providers.codex_runtime import CodexRuntimeAdapter


@dataclass
class _UsageBreakdown:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_input_tokens: int
    reasoning_output_tokens: int


@dataclass
class _ThreadUsage:
    last: _UsageBreakdown
    total: _UsageBreakdown
    model_context_window: int


class _CodexProviderStub:
    model = "gpt-example"

    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.started = threading.Event()
        self.release = threading.Event()
        self.interrupt_calls: list[str] = []

    def start_thread(self, *, on_thread_started, on_turn_started, **kwargs):  # noqa: ANN001, ANN201
        on_thread_started("thread-1")
        on_turn_started("thread-1", "turn-1")
        self.started.set()
        if self.block:
            assert self.release.wait(timeout=5)
        usage = _ThreadUsage(
            last=_UsageBreakdown(80, 20, 100, 30, 10),
            total=_UsageBreakdown(120, 30, 150, 40, 15),
            model_context_window=200,
        )
        return CodexTurnResult(
            thread_id="thread-1",
            rollout_relpath="sessions/rollout.jsonl",
            turn_result=SimpleNamespace(
                id="turn-1",
                status="completed",
                error=None,
                started_at=None,
                completed_at=None,
                duration_ms=12,
                final_response="done",
                items=[
                    SimpleNamespace(
                        id="message-1",
                        type="agentMessage",
                        text="done",
                        phase="final_answer",
                    ),
                    SimpleNamespace(
                        id="tool-1",
                        type="mcpToolCall",
                        server="lean",
                        tool="diagnostics",
                        arguments={"file": "Main.lean"},
                        result={"ok": True},
                        status="completed",
                        duration_ms=8,
                        error=None,
                    ),
                ],
                usage=usage,
            ),
        )

    def resume_thread(self, **kwargs):  # noqa: ANN003, ANN201
        return self.start_thread(
            on_thread_started=lambda _thread_id: None,
            **kwargs,
        )

    def interrupt_agent(self, agent_id: str) -> bool:
        self.interrupt_calls.append(agent_id)
        self.release.set()
        return self.block

    def close(self) -> None:
        return None


def _request(tmp_path: Path) -> ProviderRunRequest:
    return ProviderRunRequest(
        agent_id="a1",
        scope_id="scope",
        agent_type="Worker",
        provider_type="codex",
        home_id="worker",
        prompt="work",
        execution_context=ProviderExecutionContext(
            provider_type="codex",
            home_id="worker",
            home_root=tmp_path / "home",
            process_environment={"CODEX_HOME": str(tmp_path / "home" / ".codex")},
        ),
    )


def test_codex_runtime_adapter_normalizes_result_usage_and_context(tmp_path: Path) -> None:
    provider = _CodexProviderStub()
    handle = CodexRuntimeAdapter(provider).start(_request(tmp_path))  # type: ignore[arg-type]

    result = handle.wait_terminal(timeout_s=5)

    assert result.status is ProviderRunState.COMPLETED
    assert result.session_locator.session_id == "thread-1"
    assert result.turn_locator is not None and result.turn_locator.turn_id == "turn-1"
    assert result.final_text == "done"
    assert [block.kind for block in result.content_blocks] == ["text", "tool_call"]
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_kind == "mcp"
    assert result.tool_calls[0].server_name == "lean"
    assert result.tool_calls[0].tool_name == "diagnostics"
    assert result.turn_usage is not None
    assert result.turn_usage.request_count is None
    assert result.turn_usage.token_usage == TokenUsage(
        input_tokens=120,
        output_tokens=30,
        total_tokens=150,
        cached_input_tokens=40,
        reasoning_output_tokens=15,
        semantics={
            "cached_input_tokens": "subset_of_input_tokens",
            "reasoning_output_tokens": "subset_of_output_tokens",
        },
    )
    assert result.context_after is not None
    assert result.context_after.used_tokens == 100
    assert result.context_after.context_window == 200
    assert handle.legacy_turn_result.final_response == "done"


def test_codex_runtime_handle_interrupt_confirms_terminal(tmp_path: Path) -> None:
    provider = _CodexProviderStub(block=True)
    handle = CodexRuntimeAdapter(provider).start(_request(tmp_path))  # type: ignore[arg-type]
    assert provider.started.wait(timeout=5)

    control = handle.interrupt(timeout_s=5)

    assert control.accepted is True
    assert control.terminal_confirmed is True
    assert handle.wait_terminal(timeout_s=1).status is ProviderRunState.COMPLETED
    assert provider.interrupt_calls == ["a1"]
