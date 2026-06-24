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
    assert service.get_agent(agent_b.agent_id).status == "idle"
    assert len(snapshot_service.list_runtime_snapshots()) == 1


def test_runtime_snapshot_for_scopes_validates_parameters(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    service.create_agent("scope-a", "worker")
    service.create_agent("scope-b", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)

    empty = snapshot_service.create_runtime_snapshot_for_scopes(refresh_scope_ids=[])
    unknown = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-c"],
        scope_ids=["scope-a", "scope-b"],
    )
    missing_latest = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a", "scope-b"],
    )
    partial = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a", "scope-b"],
        reuse_latest_for_other_scopes=False,
    )

    assert empty.status == "failed"
    assert "refresh_scope_ids" in empty.errors
    assert unknown.status == "failed"
    assert "refresh_scope_ids" in unknown.errors
    assert missing_latest.status == "failed"
    assert "latest_scope_snapshot" in missing_latest.errors
    assert partial.status == "created"
    assert partial.snapshot_id is not None
    assert set(partial.scope_snapshot_ids) == {"scope-a"}


def test_runtime_snapshot_for_scopes_refreshes_only_selected_scope(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    service.create_agent("scope-a", "worker")
    service.create_agent("scope-b", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    initial_a = snapshot_service.create_scope_snapshot("scope-a")
    initial_b = snapshot_service.create_scope_snapshot("scope-b")
    assert initial_a.snapshot_id is not None
    assert initial_b.snapshot_id is not None

    runtime_snapshot = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a", "scope-b"],
    )

    assert runtime_snapshot.status == "created"
    assert runtime_snapshot.snapshot_id is not None
    assert set(runtime_snapshot.scope_snapshot_ids) == {"scope-a", "scope-b"}
    assert runtime_snapshot.scope_snapshot_ids["scope-a"] != initial_a.snapshot_id
    assert runtime_snapshot.scope_snapshot_ids["scope-b"] == initial_b.snapshot_id
    assert len(snapshot_service.list_scope_snapshots("scope-b")) == 1


def test_runtime_snapshot_for_scopes_does_not_block_on_unrefreshed_running_scope(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    service.create_agent("scope-a", "worker")
    running_agent = service.create_agent("scope-b", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    initial_a = snapshot_service.create_scope_snapshot("scope-a")
    initial_b = snapshot_service.create_scope_snapshot("scope-b")
    assert initial_a.snapshot_id is not None
    assert initial_b.snapshot_id is not None
    service.store.patch_agent(running_agent.agent_id, status="running")

    runtime_snapshot = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a", "scope-b"],
        wait=False,
    )

    assert runtime_snapshot.status == "created"
    assert runtime_snapshot.scope_snapshot_ids["scope-a"] != initial_a.snapshot_id
    assert runtime_snapshot.scope_snapshot_ids["scope-b"] == initial_b.snapshot_id


def test_runtime_snapshot_for_scopes_blocks_on_refreshed_running_scope(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    running_agent = service.create_agent("scope-a", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    service.store.patch_agent(running_agent.agent_id, status="running")

    runtime_snapshot = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a"],
        wait=False,
    )

    assert runtime_snapshot.status == "blocked"
    assert runtime_snapshot.blocked_scope_ids == ("scope-a",)


def test_runtime_snapshot_for_scopes_under_global_pause_does_not_leak_direct_scope_pause(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    service.create_agent("scope-a", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    assert snapshot_service.ark.pause_controller is not None
    snapshot_service.ark.pause_controller.pause(None)

    runtime_snapshot = snapshot_service.create_runtime_snapshot_for_scopes(
        refresh_scope_ids=["scope-a"],
        scope_ids=["scope-a"],
    )

    assert runtime_snapshot.status == "created"
    assert snapshot_service.ark.pause_controller.is_paused("scope-a") is True
    assert snapshot_service.ark.pause_controller.is_scope_directly_paused("scope-a") is False
    snapshot_service.ark.pause_controller.resume(None)
    assert snapshot_service.ark.pause_controller.is_paused("scope-a") is False


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
