from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.service import AgentCompletionContext, AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.models import CompletionDecision


class SmokeAgentType(AgentType):
    agent_type = "node_worker"
    developer_instructions_template = "You are running an agent-runtime-kit smoke test. Keep responses short."
    start_prompt_template = "Reply with exactly this token and no extra text: {{token}}"
    continue_prompt_template = "Reply with exactly this token and no extra text: {{token}}"

    def check_completion(self, ctx: AgentCompletionContext) -> CompletionDecision:
        return CompletionDecision(complete=bool(getattr(ctx.turn_result, "id", None)))


def main() -> int:
    config_dir = Path(os.environ.get("ARK_CODEX_CONFIG_DIR", REPO_ROOT / "data" / "configs" / "codex"))
    sdk_root = os.environ.get("ARK_CODEX_SDK_PYTHON_ROOT")
    if sdk_root:
        sdk_src = Path(sdk_root) / "src"
        if str(sdk_src) not in sys.path:
            sys.path.insert(0, str(sdk_src))
    if not (config_dir / "config.toml").exists() or not (config_dir / "auth.json").exists():
        raise RuntimeError(f"missing config.toml/auth.json under {config_dir}")

    with tempfile.TemporaryDirectory(prefix="ark-real-codex-smoke-") as temp_root:
        runtime_root = Path(temp_root) / "project" / ".agent_runtime"
        registry = AgentTypeRegistry()
        registry.register(SmokeAgentType())
        provider = CodexProvider(
            runtime_root=runtime_root,
            codex_bin=os.environ.get("ARK_CODEX_BIN") or shutil.which("codex"),
            sdk_python_root=Path(sdk_root) if sdk_root else None,
            model=os.environ.get("ARK_REAL_CODEX_MODEL"),
        )
        service = AgentService(runtime_root, agent_types=registry, providers={"codex": provider})
        service.home_service.create_home(
            HomeCreateSpec(
                cli_type="codex",
                home_id="node_worker",
                base_config_path=config_dir / "config.toml",
                auth_json_path=config_dir / "auth.json",
            )
        )
        agent = service.create_agent("repo-a:node-root", "node_worker")
        sibling = service.create_agent("repo-a:node-child", "node_worker")
        service.start_agent(agent.agent_id, variables={"token": "ARK_SMOKE_A"})
        service.start_agent(sibling.agent_id, variables={"token": "ARK_SMOKE_B"})
        waited = service.wait_agents([agent.agent_id, sibling.agent_id], timeout_s=600)
        if not waited.clean:
            raise RuntimeError(f"concurrent smoke wait failed: {waited}")
        restored = service.get_agent(agent.agent_id)
        restored_sibling = service.get_agent(sibling.agent_id)
        snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
        scope_snapshot = snapshot_service.create_scope_snapshot(restored.scope_id)
        runtime_snapshot = snapshot_service.create_runtime_snapshot_synchronized(timeout_s=600)

        print("runtime_root:", runtime_root)
        print("wait_agents_clean:", waited.clean)
        print("agent_id:", restored.agent_id)
        print("sibling_agent_id:", restored_sibling.agent_id)
        print("thread_id:", restored.thread_id)
        print("sibling_thread_id:", restored_sibling.thread_id)
        print("turn_id:", getattr(waited.completed[agent.agent_id], "id", None))
        print("sibling_turn_id:", getattr(waited.completed[sibling.agent_id], "id", None))
        print("rollout_relpath:", restored.rollout_relpath)
        print("sibling_rollout_relpath:", restored_sibling.rollout_relpath)
        print("turns:", len(service.list_turns(agent.agent_id)))
        print("events:", len(service.read_rollout_events(agent.agent_id)))
        print("scope_snapshot_id:", scope_snapshot.snapshot_id)
        print("runtime_snapshot_id:", runtime_snapshot.snapshot_id)
        service.close(force_provider_homes=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
