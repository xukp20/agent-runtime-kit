from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime_kit.agent.providers.claude_code_runtime import _build_options


class _RecordingOptions:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.mark.parametrize(
    "strict_mcp_config",
    [True, False],
)
def test_claude_options_use_native_strict_mcp_config(
    tmp_path: Path,
    strict_mcp_config: bool,
) -> None:
    sdk = SimpleNamespace(ClaudeAgentOptions=_RecordingOptions)
    context = SimpleNamespace(
        runtime_payload={
            "strict_mcp_config": strict_mcp_config,
            "extra_args": {"debug-to-stderr": None},
        },
        home_root=tmp_path,
        workdir=None,
        process_environment={},
    )
    request = SimpleNamespace(
        execution_context=context,
        system_instructions=None,
        developer_instructions=None,
        model_overrides=None,
        run_options=SimpleNamespace(max_turns=None),
        workdir=None,
    )

    options = _build_options(sdk, request, session_id="session-1", resume=False)

    assert options.kwargs["strict_mcp_config"] is strict_mcp_config
    assert options.kwargs["extra_args"] == {"debug-to-stderr": None}


@pytest.mark.parametrize("tools", [None, [], ["Read"]])
def test_claude_options_forward_tools_tristate(tmp_path: Path, tools: list[str] | None) -> None:
    sdk = SimpleNamespace(ClaudeAgentOptions=_RecordingOptions)
    request = _request(tmp_path, runtime_payload={"tools": tools})

    options = _build_options(sdk, request, session_id="session-1", resume=False)

    assert options.kwargs["tools"] == tools


def _request(tmp_path: Path, *, runtime_payload: dict[str, object] | None = None) -> object:
    context = SimpleNamespace(
        provider_type="claude_code",
        runtime_payload=runtime_payload or {},
        home_root=tmp_path,
        workdir=None,
        process_environment={},
    )
    return SimpleNamespace(
        provider_type="claude_code",
        home_id="worker",
        execution_context=context,
        session_locator=None,
        model_overrides=None,
        prompt="test",
        system_instructions=None,
        developer_instructions=None,
        run_options=SimpleNamespace(max_turns=None),
        workdir=None,
        event_sink=None,
    )
