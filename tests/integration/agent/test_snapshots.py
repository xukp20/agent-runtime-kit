from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    AgentArtifactLocator,
    ProviderArtifactEntry,
    ProviderArtifactManifest,
    ProviderHomeSpec,
    ProviderRegistry,
    ProviderSessionLocator,
)
from agent_runtime_kit.agent.providers.opencode_bundle import build_opencode_provider_bundle
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


def test_scope_snapshot_rejects_provider_artifact_path_collision() -> None:
    manifest = ProviderArtifactManifest(
        provider_type="fake",
        home_id="home",
        session_id="session",
        adapter_version="1",
        stable=True,
        entries=(
            ProviderArtifactEntry(
                artifact_id="database",
                kind="provider_database",
                authority="provider_native",
                capture_strategy="copy",
                snapshot_relpath="provider.db",
                required_for_resume=True,
            ),
        ),
    )
    owners: dict[str, str] = {}
    AgentSnapshotService._reserve_provider_artifact_paths(
        agent_id="agent-1",
        manifest=manifest,
        owners=owners,
    )

    with pytest.raises(RuntimeError, match="provider artifact snapshot path collision"):
        AgentSnapshotService._reserve_provider_artifact_paths(
            agent_id="agent-2",
            manifest=manifest,
            owners=owners,
        )


def test_scope_snapshot_restores_two_opencode_agent_databases(tmp_path: Path) -> None:
    root = tmp_path / ".agent_runtime"
    types = AgentTypeRegistry()
    types.register(_Worker())
    service = AgentService(
        root,
        agent_types=types,
        provider_registry=ProviderRegistry((build_opencode_provider_bundle(runtime_root=root),)),
    )
    service.home_service.create_home(ProviderHomeSpec(provider_type="opencode", home_id="worker"))
    databases: list[Path] = []
    for value in ("first", "second"):
        agent = service.create_agent(
            "scope-opencode",
            "worker",
            provider_type="opencode",
            home_id="worker",
        )
        runtime_relpath = f"providers/opencode/agents/{agent.agent_id}"
        database = root / runtime_relpath / "opencode.db"
        database.parent.mkdir(parents=True)
        with sqlite3.connect(database) as conn:
            conn.execute("create table sessions(id text primary key, value text)")
            conn.execute("insert into sessions values ('session', ?)", (value,))
        service.store.update_session_locators(
            agent.agent_id,
            session_locator=ProviderSessionLocator(
                provider_type="opencode",
                session_id=f"session-{agent.agent_id}",
                home_id="worker",
                created_at="2026-07-21T00:00:00Z",
                native_locator={
                    "agent_id": agent.agent_id,
                    "directory": str(tmp_path),
                    "database_path": str(database),
                    "runtime_relpath": runtime_relpath,
                },
            ),
        )
        databases.append(database)

    snapshots = AgentSnapshotService(root, store=service.store, agent_service=service)
    created = snapshots.create_scope_snapshot("scope-opencode")

    assert created.status == "created"
    manifest = json.loads((root / created.snapshot_relpath / "snapshot.json").read_text())
    relpaths = [
        record["manifest"]["entries"][0]["snapshot_relpath"]
        for record in manifest["provider_artifacts"]
    ]
    assert len(relpaths) == len(set(relpaths)) == 2
    for database in databases:
        with sqlite3.connect(database) as conn:
            conn.execute("update sessions set value='changed'")

    restored = snapshots.restore_scope_snapshot(created.snapshot_id, leave_paused=False)

    assert restored.status == "created"
    for database, expected in zip(databases, ("first", "second"), strict=True):
        with sqlite3.connect(database) as conn:
            assert conn.execute("select value from sessions").fetchone()[0] == expected
    service.close()
