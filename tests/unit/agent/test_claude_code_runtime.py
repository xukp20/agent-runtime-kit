from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime_kit.agent.providers.claude_code_runtime import _build_options


class _RecordingOptions:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("strict_mcp_config", "expected_strict_flag"),
    [(True, True), (False, False)],
)
def test_claude_options_route_strict_mcp_config_through_extra_args(
    tmp_path: Path,
    strict_mcp_config: bool,
    expected_strict_flag: bool,
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

    assert "strict_mcp_config" not in options.kwargs
    assert options.kwargs["extra_args"]["debug-to-stderr"] is None
    assert ("strict-mcp-config" in options.kwargs["extra_args"]) is expected_strict_flag
    if expected_strict_flag:
        assert options.kwargs["extra_args"]["strict-mcp-config"] is None
