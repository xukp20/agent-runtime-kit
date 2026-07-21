from __future__ import annotations

import json
from pathlib import Path

from agent_runtime_kit.agent.provider_contracts import (
    AgentArtifactLocator,
    ProviderHomeSpec,
    ProviderSessionLocator,
)
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService


class _Worker(AgentType):
    agent_type = "worker"
    start_prompt_template = "work"


def _runtime(tmp_path: Path):  # noqa: ANN202
    root = tmp_path / ".agent_runtime"
    types = AgentTypeRegistry()
    types.register(_Worker())
    service = AgentService(root, agent_types=types)
    service.home_service.create_home(ProviderHomeSpec(provider_type="codex", home_id="worker"))
    snapshots = AgentSnapshotService(root, store=service.store, agent_service=service)
    return root, service, snapshots


def test_scope_snapshot_restores_exact_codex_artifact_manifest(tmp_path: Path) -> None:
    root, service, snapshots = _runtime(tmp_path)
    agent = service.create_agent("scope-1", "worker")
    relpath = "sessions/2026/07/22/rollout-session-1.jsonl"
    rollout = root / "homes" / "codex" / "worker" / ".codex" / relpath
    rollout.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps({"type": "session_meta", "payload": {"id": "session-1"}}) + "\n"
    rollout.write_text(original, encoding="utf-8")
    session = ProviderSessionLocator(
        provider_type="codex",
        session_id="session-1",
        home_id="worker",
        created_at="2026-07-22T00:00:00Z",
        native_locator={"rollout_relpath": relpath},
    )
    service.store.update_session_locators(
        agent.agent_id,
        session_locator=session,
        artifact_locator=AgentArtifactLocator(
            provider_type="codex",
            home_id="worker",
            session_id="session-1",
            adapter_version="1",
            native_primary_ref=relpath,
        ),
    )

    created = snapshots.create_scope_snapshot("scope-1")
    assert created.status == "created"
    rollout.write_text("mutated\n", encoding="utf-8")
    restored = snapshots.restore_scope_snapshot(created.snapshot_id, leave_paused=False)

    assert restored.status == "created"
    assert rollout.read_text(encoding="utf-8") == original


def test_scope_restore_rejects_pre_manifest_schema(tmp_path: Path) -> None:
    root, service, snapshots = _runtime(tmp_path)
    service.create_agent("scope-1", "worker")
    created = snapshots.create_scope_snapshot("scope-1")
    manifest_path = root / created.snapshot_relpath / "snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest.pop("provider_artifacts")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    restored = snapshots.restore_scope_snapshot(created.snapshot_id)

    assert restored.status == "failed"
    assert "unsupported scope snapshot schema" in str(restored.errors["snapshot_archive"])
