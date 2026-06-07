from pathlib import Path

from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.store_utils import encode_scope_id

from .fakes import FakeProvider


class SnapshotAgentType(AgentType):
    agent_type = "worker"
    developer_instructions_template = "Developer."
    start_prompt_template = "Start {{item}}."
    continue_prompt_template = "Continue {{item}}."


def test_scope_snapshot_copies_scope_metadata_and_only_scope_rollouts(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent_a = service.create_agent("scope-a", "worker")
    agent_b = service.create_agent("scope-b", "worker")
    service.start_agent(agent_a.agent_id, variables={"item": "a"})
    service.wait_agent(agent_a.agent_id)
    service.start_agent(agent_b.agent_id, variables={"item": "b"})
    service.wait_agent(agent_b.agent_id)
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)

    result = snapshot_service.create_scope_snapshot("scope-a")

    assert result.snapshot_id is not None
    snapshot_dir = runtime_root / result.snapshot_relpath
    assert (snapshot_dir / "files" / "scopes" / encode_scope_id("scope-a") / "scope.json").exists()
    assert (
        snapshot_dir
        / "files"
        / "homes"
        / "codex"
        / "worker"
        / ".codex"
        / service.get_agent(agent_a.agent_id).rollout_relpath
    ).exists()
    assert not (
        snapshot_dir
        / "files"
        / "homes"
        / "codex"
        / "worker"
        / ".codex"
        / service.get_agent(agent_b.agent_id).rollout_relpath
    ).exists()


def test_scope_restore_replaces_scope_and_rollout_from_snapshot(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.start_agent(agent.agent_id, variables={"item": "first"})
    service.wait_agent(agent.agent_id)
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.snapshot_id is not None
    rollout_path = service.store.locate_rollout(agent.agent_id)
    assert rollout_path is not None
    rollout_path.write_text('{"type": "turn_result", "thread_id": "bad", "turn_id": "bad", "prompt": "bad"}\n', encoding="utf-8")

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    provider = service.providers["codex"]
    assert provider.close_home_calls == [{"home_id": "worker", "force": False}]
    events = service.read_rollout_events(agent.agent_id)
    assert events[-1]["prompt"] == "Start first."
    assert service.get_agent(agent.agent_id).scope_id == "scope-a"


def test_runtime_synchronized_snapshot_and_restore(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent_a = service.create_agent("scope-a", "worker")
    agent_b = service.create_agent("scope-b", "worker")
    for agent, item in [(agent_a, "a"), (agent_b, "b")]:
        service.start_agent(agent.agent_id, variables={"item": item})
        service.wait_agent(agent.agent_id)
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)

    runtime_snapshot = snapshot_service.create_runtime_snapshot_synchronized()

    assert runtime_snapshot.status == "created"
    assert set(runtime_snapshot.scope_snapshot_ids) == {"scope-a", "scope-b"}
    service.store.close_agent(agent_b.agent_id)
    assert service.get_agent(agent_b.agent_id).status == "closed"
    restored = snapshot_service.restore_runtime_snapshot(runtime_snapshot.snapshot_id)

    assert restored.status == "created"
    provider = service.providers["codex"]
    assert len(provider.close_home_calls) >= 1
    assert {"home_id": "worker", "force": False} in provider.close_home_calls
    assert service.get_agent(agent_b.agent_id).status == "idle"
    assert len(snapshot_service.list_runtime_snapshots()) == 1


def test_scope_snapshot_blocks_when_scope_has_running_agent(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.store.patch_agent(agent.agent_id, status="running")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)

    result = snapshot_service.create_scope_snapshot("scope-a")

    assert result.status == "blocked"
    assert result.running_agent_ids == (agent.agent_id,)


def _make_service(runtime_root: Path) -> AgentService:
    registry = AgentTypeRegistry()
    registry.register(SnapshotAgentType())
    provider = FakeProvider(runtime_root)
    return AgentService(runtime_root, agent_types=registry, providers={"codex": provider})
