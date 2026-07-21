from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from agent_runtime_kit.agent.provider_contracts import ModelBackendIdentity, ProviderHomeSpec
from agent_runtime_kit.agent.providers.claude_code import ClaudeCodeProvider
from agent_runtime_kit.agent.providers.claude_code_home import ClaudeCodeHomeOptions
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService


pytestmark = [
    pytest.mark.real,
    pytest.mark.real_claude,
    pytest.mark.skipif(
        os.getenv("ARK_RUN_REAL_CLAUDE") != "1",
        reason="set ARK_RUN_REAL_CLAUDE=1 to run real Claude tests",
    ),
]


class RealClaudeWorker(AgentType):
    agent_type = "real-claude-worker"
    developer_instructions_template = "Follow the requested output format exactly."
    start_prompt_template = "Reply with exactly ARK_CLAUDE_REAL_ONE."
    continue_prompt_template = "Reply with exactly ARK_CLAUDE_REAL_NEXT."


class RealClaudeInterruptWorker(AgentType):
    agent_type = "real-claude-interrupt-worker"
    developer_instructions_template = "You must use the Bash tool exactly as requested."
    start_prompt_template = "Use Bash to run sleep 30, then reply DONE."
    continue_prompt_template = "Continue."


def test_real_claude_deepseek_runtime_context_fork_and_snapshot(tmp_path: Path) -> None:
    service, env = _service(tmp_path, RealClaudeWorker(), tools=())
    agent = service.create_agent(
        "real-claude-scope",
        RealClaudeWorker.agent_type,
        cli_type="claude_code",
        home_id="worker",
    )

    service.start_agent(agent.agent_id, env=env, workdir=str(tmp_path))
    first = service.wait_agent_result(agent.agent_id, timeout_s=180).provider_result
    service.start_agent(
        agent.agent_id,
        prompt="Reply with exactly ARK_CLAUDE_REAL_TWO.",
        env=env,
        workdir=str(tmp_path),
    )
    second = service.wait_agent_result(agent.agent_id, timeout_s=180).provider_result
    usage = service.inspect_agent_context_result(
        agent.agent_id,
        env=env,
        workdir=str(tmp_path),
    )
    compacted = service.compact_agent(
        agent.agent_id,
        timeout_s=180,
        env=env,
        workdir=str(tmp_path),
    )

    snapshot_service = AgentSnapshotService(
        service.runtime_root,
        store=service.store,
        agent_service=service,
    )
    snapshot = snapshot_service.create_scope_snapshot("real-claude-scope")
    assert snapshot.snapshot_id is not None
    transcript = _transcript(service, agent.agent_id)
    expected = transcript.read_bytes()
    transcript.write_bytes(b"corrupted\n")
    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)
    latest = service.query_turn(agent.agent_id, latest=True)
    service.resume_runs("real-claude-scope")
    forked = service.fork_agent(agent.agent_id, target_scope_id="real-claude-fork")

    assert first.status.value == "completed"
    assert second.status.value == "completed"
    assert first.session_locator.session_id == second.session_locator.session_id
    assert usage.available and usage.used_tokens is not None and usage.used_tokens > 0
    assert compacted.status.value == "compacted"
    assert snapshot.status == "created"
    assert restored.status == "created"
    assert transcript.read_bytes() == expected
    assert latest is not None and latest.result is not None
    assert forked.fork_info is not None
    assert forked.fork_info.fork_mode == "session_only"
    assert not forked.fork_info.workspace_isolated
    service.close()


def test_real_claude_interrupt_waits_for_terminal_result(tmp_path: Path) -> None:
    service, env = _service(
        tmp_path,
        RealClaudeInterruptWorker(),
        tools=("Bash",),
        allowed_tools=("Bash",),
    )
    agent = service.create_agent(
        "real-claude-interrupt-scope",
        RealClaudeInterruptWorker.agent_type,
        cli_type="claude_code",
        home_id="worker",
    )

    service.start_agent(agent.agent_id, env=env, workdir=str(tmp_path))
    time.sleep(2)
    accepted = service.interrupt_agent(agent.agent_id, timeout_s=30)
    result = service.wait_agent_result(agent.agent_id, timeout_s=30).provider_result

    assert accepted
    assert result.status.value == "interrupted"
    assert result.artifact_locator is not None
    service.close()


def _service(
    tmp_path: Path,
    agent_type: AgentType,
    *,
    tools: tuple[str, ...],
    allowed_tools: tuple[str, ...] = (),
) -> tuple[AgentService, dict[str, str]]:
    settings_path = Path(os.getenv("ARK_CLAUDE_SETTINGS_PATH", "/root/.claude/settings.json"))
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    env = {str(key): str(value) for key, value in dict(settings.get("env") or {}).items()}
    cli_path = os.getenv(
        "ARK_CLAUDE_CLI_PATH",
        "/root/code/worktrees/agent-runtime-kit-provider-research/data/provider_research/"
        "claude-runtime/claude-node22",
    )
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(agent_type)
    service = AgentService(
        runtime_root,
        agent_types=registry,
        providers={"claude_code": ClaudeCodeProvider(runtime_root=runtime_root)},
    )
    service.create_home(
        ProviderHomeSpec(
            provider_type="claude_code",
            home_id="worker",
            model_config=ModelBackendIdentity(
                api_provider="deepseek",
                api_mode="anthropic_messages",
                requested_model=env.get("ANTHROPIC_MODEL"),
            ),
            provider_options=ClaudeCodeHomeOptions(
                cli_path=cli_path,
                tools=tools,
                allowed_tools=allowed_tools,
            ),
        ),
        env=env,
        workdir=str(tmp_path),
    )
    return service, env


def _transcript(service: AgentService, agent_id: str) -> Path:
    agent = service.get_agent(agent_id)
    assert agent.artifact_locator is not None
    assert agent.artifact_locator.native_primary_ref is not None
    return (
        service.runtime_root
        / "homes"
        / "claude_code"
        / agent.home_id
        / ".claude"
        / agent.artifact_locator.native_primary_ref
    )
