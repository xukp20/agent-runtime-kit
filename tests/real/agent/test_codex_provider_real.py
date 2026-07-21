from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

from agent_runtime_kit.agent.context import (
    AgentContextCompactionStatus,
    AgentContextMaintenanceJournalStatus,
)
from agent_runtime_kit.agent.homes import ProviderHomeSpec
from agent_runtime_kit.agent.models import AgentPausedError, CompletionDecision
from agent_runtime_kit.agent.provider_contracts import BaseConfigSource, ProviderRegistry
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.providers.codex_home import CodexHomeOptions
from agent_runtime_kit.agent.service import AgentCompletionContext, AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService


pytestmark = pytest.mark.real_codex


class RealCodexSmokeAgentType(AgentType):
    agent_type = "node_worker"
    developer_instructions_template = (
        "You are running a real agent-runtime-kit smoke test. "
        "Keep the answer short and avoid tool calls unless absolutely required."
    )
    start_prompt_template = "Reply with exactly this token and no extra text: {{token}}"
    continue_prompt_template = "Reply again with exactly this token and no extra text: {{token}}"

    def check_completion(self, ctx: AgentCompletionContext) -> CompletionDecision:
        if not ctx.turn_result.run_id:
            return CompletionDecision(complete=False, reason="turn result has no run id")
        return CompletionDecision(complete=True)


def test_real_codex_minimal_run_resume_store_and_pause(tmp_path: Path) -> None:
    runtime_root = tmp_path / "project" / ".agent_runtime"
    service = _service(runtime_root)
    try:
        _create_codex_home(service, "node_worker")
        agent = service.create_agent("repo-a:node-root", "node_worker")

        service.pause_runs(agent.scope_id)
        with pytest.raises(AgentPausedError):
            service.start_agent(agent.agent_id, variables={"token": "ARK_PAUSED"})
        service.resume_runs(agent.scope_id)

        first = service.wait_agent(
            service.start_agent(agent.agent_id, variables={"token": "ARK_OK_FIRST"}).agent_id,
            timeout_s=600,
        )
        restored = service.get_agent(agent.agent_id)
        assert restored.status == "idle"
        assert restored.session_locator is not None
        assert restored.artifact_locator is not None
        assert first.run_id
        assert service.query_events(agent.agent_id).items
        assert service.query_turns(agent.agent_id).items

        before_events = len(service.query_events(agent.agent_id).items)
        second = service.wait_agent(
            service.start_agent(agent.agent_id, variables={"token": "ARK_OK_SECOND"}).agent_id,
            timeout_s=600,
        )
        resumed = service.get_agent(agent.agent_id)
        assert resumed.session_locator == restored.session_locator
        assert resumed.artifact_locator == restored.artifact_locator
        assert second.run_id
        assert len(service.query_events(agent.agent_id).items) > before_events
        assert service.query_turn(agent.agent_id, latest=True) is not None
    finally:
        service.close()


def test_real_codex_developer_instruction_override_on_resume(tmp_path: Path) -> None:
    runtime_root = tmp_path / "project" / ".agent_runtime"
    service = _service(runtime_root)
    try:
        _create_codex_home(service, "node_worker")
        agent = service.create_agent("repo-a:node-root", "node_worker")
        prompt = "Reply with the current ARK dynamic instruction sentinel and no extra text."

        service.wait_agent(
            service.start_agent(
                agent.agent_id,
                variables={},
                prompt=prompt,
                developer_instructions_template_override=(
                    "This is an ARK dynamic instruction override test. "
                    "For this turn, the current ARK dynamic instruction sentinel is "
                    "ARK_DYNAMIC_INSTRUCTION_FIRST. Reply with exactly that sentinel when asked."
                ),
            ).agent_id,
            timeout_s=600,
        )
        first_agent = service.get_agent(agent.agent_id)
        assert first_agent.session_locator is not None
        first_text = _latest_text(service, agent.agent_id)
        assert "ARK_DYNAMIC_INSTRUCTION_FIRST" in first_text

        service.wait_agent(
            service.start_agent(
                agent.agent_id,
                variables={},
                prompt=prompt,
                developer_instructions_template_override=(
                    "This is an ARK dynamic instruction override test. "
                    "For this turn, the current ARK dynamic instruction sentinel is "
                    "ARK_DYNAMIC_INSTRUCTION_SECOND. This current instruction supersedes any earlier "
                    "sentinel in the same thread. Reply with exactly that sentinel when asked."
                ),
            ).agent_id,
            timeout_s=600,
        )
        second_agent = service.get_agent(agent.agent_id)
        second_text = _latest_text(service, agent.agent_id)
        assert second_agent.session_locator == first_agent.session_locator
        assert "ARK_DYNAMIC_INSTRUCTION_SECOND" in second_text
    finally:
        service.close()


def test_real_codex_multi_scope_snapshot_flow(tmp_path: Path) -> None:
    runtime_root = tmp_path / "project" / ".agent_runtime"
    service = _service(runtime_root)
    try:
        _create_codex_home(service, "node_worker")
        root_agent = service.create_agent("repo-a:node-root", "node_worker")
        child_agent = service.create_agent("repo-a:node-child", "node_worker")

        service.start_agent(root_agent.agent_id, variables={"token": "ARK_ROOT"})
        service.start_agent(child_agent.agent_id, variables={"token": "ARK_CHILD"})
        waited = service.wait_agents([root_agent.agent_id, child_agent.agent_id], timeout_s=600)
        assert waited.clean
        assert service.get_agent(root_agent.agent_id).session_locator is not None
        assert service.get_agent(child_agent.agent_id).session_locator is not None
        assert service.get_agent(root_agent.agent_id).artifact_locator is not None
        assert service.get_agent(child_agent.agent_id).artifact_locator is not None
        assert not service.list_running_agents()
        assert not service.has_running_agents()
        assert service.is_stable()
        assert set(service.store.list_scope_ids()) == {"repo-a:node-root", "repo-a:node-child"}
        assert service.wait_scope_agents("repo-a:node-root").clean
        assert service.wait_all_active_agents().clean

        snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
        root_scope_snapshot = snapshot_service.create_scope_snapshot("repo-a:node-root")
        assert root_scope_snapshot.status == "created"
        assert root_scope_snapshot.snapshot_id is not None
        assert snapshot_service.get_latest_scope_snapshot("repo-a:node-root") is not None

        causal = snapshot_service.create_runtime_snapshot_causal()
        assert causal.status == "created"
        assert "repo-a:node-root" in causal.scope_snapshot_ids

        synchronized = snapshot_service.create_runtime_snapshot_synchronized(timeout_s=600)
        assert synchronized.status == "created"
        assert set(synchronized.scope_snapshot_ids) == {"repo-a:node-root", "repo-a:node-child"}

        service.close_agent(child_agent.agent_id)
        assert service.get_agent(child_agent.agent_id).status == "closed"
        restored = snapshot_service.restore_runtime_snapshot(synchronized.snapshot_id)
        assert restored.status == "created"
        assert service.get_agent(child_agent.agent_id).status == "idle"
        assert snapshot_service.list_scope_snapshots()
        assert snapshot_service.list_runtime_snapshots()
    finally:
        service.close()


def test_real_codex_context_compact_resume_and_snapshot_restore(tmp_path: Path) -> None:
    runtime_root = tmp_path / "project" / ".agent_runtime"
    service = _service(runtime_root)
    try:
        _create_codex_home(service, "node_worker")
        agent = service.create_agent("repo-a:node-compact", "node_worker")
        service.wait_agent(
            service.start_agent(agent.agent_id, variables={"token": "ARK_COMPACT_BEFORE"}).agent_id,
            timeout_s=600,
        )

        before = service.inspect_agent_context(agent.agent_id)
        assert before.available
        assert before.used_tokens is not None
        compacted = service.compact_agent(agent.agent_id, timeout_s=600)
        assert compacted.status is AgentContextCompactionStatus.COMPACTED
        assert compacted.usage_after is not None
        journal = service.store.read_context_maintenance(agent.agent_id)
        assert journal is not None
        assert journal.status is AgentContextMaintenanceJournalStatus.CONFIRMED

        snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
        snapshot = snapshot_service.create_scope_snapshot(agent.scope_id)
        assert snapshot.status == "created"
        assert snapshot.snapshot_id is not None

        service.wait_agent(
            service.start_agent(agent.agent_id, variables={"token": "ARK_COMPACT_AFTER"}).agent_id,
            timeout_s=600,
        )
        restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id, leave_paused=False)
        assert restored.status == "created"
        restored_journal = service.store.read_context_maintenance(agent.agent_id)
        assert restored_journal is not None
        assert restored_journal.status is AgentContextMaintenanceJournalStatus.CONFIRMED
        service.wait_agent(
            service.start_agent(agent.agent_id, variables={"token": "ARK_COMPACT_RESTORED"}).agent_id,
            timeout_s=600,
        )
    finally:
        service.close()


def _service(runtime_root: Path) -> AgentService:
    _ensure_real_codex_enabled()
    registry = AgentTypeRegistry()
    registry.register(RealCodexSmokeAgentType())
    provider = CodexProvider(
        runtime_root=runtime_root,
        codex_bin=os.environ.get("ARK_CODEX_BIN") or shutil.which("codex"),
        sdk_python_root=_sdk_python_root(),
        model=os.environ.get("ARK_REAL_CODEX_MODEL"),
    )
    return AgentService(
        runtime_root,
        agent_types=registry,
        provider_registry=ProviderRegistry((provider.build_provider_bundle(runtime_root=runtime_root),)),
    )


def _create_codex_home(service: AgentService, home_id: str) -> None:
    config_dir = _config_dir()
    skills_dir = config_dir / "skills"
    service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="codex",
            home_id=home_id,
            base_config=BaseConfigSource(path=str(config_dir / "config.toml")),
            provider_options=CodexHomeOptions(
                auth_json_path=config_dir / "auth.json",
                skill_paths={
                    path.name: path
                    for path in sorted(skills_dir.iterdir())
                    if path.is_dir() and (path / "SKILL.md").exists()
                }
                if skills_dir.exists()
                else {},
            ),
        )
    )


def _latest_text(service: AgentService, agent_id: str) -> str:
    latest = service.query_turn(agent_id, latest=True)
    text = latest.result.final_text if latest is not None and latest.result is not None else None
    assert isinstance(text, str), f"latest turn has no final response: {latest!r}"
    return text


def _ensure_real_codex_enabled() -> None:
    if os.environ.get("ARK_RUN_REAL_CODEX") != "1":
        pytest.skip("set ARK_RUN_REAL_CODEX=1 to run real Codex SDK tests")
    if shutil.which("codex") is None and not os.environ.get("ARK_CODEX_BIN"):
        pytest.skip("codex binary is not available")
    sdk_root = _sdk_python_root()
    if importlib.util.find_spec("openai_codex") is None and sdk_root is None:
        pytest.skip("openai_codex is not installed and no local SDK root is available")


def _sdk_python_root() -> Path | None:
    value = os.environ.get("ARK_CODEX_SDK_PYTHON_ROOT")
    root = Path(value) if value else Path("/root/code/tools/codex/sdk/python")
    if not root.exists():
        return None
    src = root / "src"
    if not (src / "openai_codex").exists():
        pytest.skip(f"invalid Codex SDK Python root: {root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def _config_dir() -> Path:
    path = Path(os.environ.get("ARK_CODEX_CONFIG_DIR", "data/configs/codex"))
    if not path.exists():
        pytest.skip(f"Codex config dir does not exist: {path}")
    if not (path / "config.toml").exists() or not (path / "auth.json").exists():
        pytest.skip(f"Codex config dir must contain config.toml and auth.json: {path}")
    return path
