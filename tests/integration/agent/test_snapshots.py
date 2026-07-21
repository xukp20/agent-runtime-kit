import json
import threading
from pathlib import Path

import pytest

from agent_runtime_kit.agent.context import (
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
)
from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.models import (
    AgentAlreadyRunningError,
    AgentContextMaintenanceBlocked,
)
from agent_runtime_kit.agent.provider_contracts import ModelBackendIdentity, ProviderSessionLocator
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.report_policy import AgentTraceReportPolicy
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.store_utils import encode_scope_id

from .fakes import FakeProvider


class SnapshotAgentType(AgentType):
    agent_type = "worker"
    developer_instructions_template = "Developer."
    start_prompt_template = "Start {{item}}."
    continue_prompt_template = "Continue {{item}}."


def test_snapshot_provider_session_preserves_exact_native_locator(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = AgentService(runtime_root, providers={"contract_fake": object()})
    agent = service.store.create_agent_record(
        scope_id="scope-a",
        agent_type="worker",
        cli_type="contract_fake",
        home_id="worker",
    )
    exact = ProviderSessionLocator(
        provider_type="contract_fake",
        session_id="session-exact",
        home_id="worker",
        created_at="2026-07-21T10:00:00Z",
        native_locator={"database": "provider/session.db", "rollout_relpath": "exact.jsonl"},
        backend_identity=ModelBackendIdentity(
            api_provider="backend-a",
            api_mode="responses",
            requested_model="model-a",
        ),
    )
    persisted = service.store.update_thread_locator(
        agent.agent_id,
        thread_id=exact.session_id,
        rollout_relpath="legacy-placeholder.jsonl",
        session_locator=exact,
    )

    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)

    assert snapshot_service._provider_session(persisted) == exact


def test_codex_snapshot_uses_provider_artifact_manifest_for_restore(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(SnapshotAgentType())
    service = AgentService(
        runtime_root,
        agent_types=registry,
        providers={"codex": CodexProvider(runtime_root=runtime_root)},
    )
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    rollout_relpath = "sessions/2026/07/21/rollout-session-1.jsonl"
    service.store.update_thread_locator(
        agent.agent_id,
        thread_id="session-1",
        rollout_relpath=rollout_relpath,
    )
    codex_root = runtime_root / "homes" / "codex" / "worker" / ".codex"
    rollout = codex_root / rollout_relpath
    rollout.parent.mkdir(parents=True, exist_ok=True)
    rollout.write_text('{"type":"event_msg","payload":{"type":"task_complete"}}\n', encoding="utf-8")
    state_db = codex_root / "state_5.sqlite"
    state_db.write_bytes(b"rebuildable")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)

    snapshot = snapshot_service.create_scope_snapshot("scope-a")

    assert snapshot.snapshot_id is not None
    snapshot_root = runtime_root / str(snapshot.snapshot_relpath)
    manifest = json.loads((snapshot_root / "snapshot.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["provider_artifacts"][0]["manifest"]["provider_type"] == "codex"
    assert manifest["provider_artifacts"][0]["manifest"]["entries"][0]["kind"] == "session_transcript"
    assert not (snapshot_root / "files" / state_db.relative_to(runtime_root)).exists()

    rollout.write_text("corrupted\n", encoding="utf-8")
    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    assert "task_complete" in rollout.read_text(encoding="utf-8")
    assert not state_db.exists()


def test_codex_snapshot_keeps_legacy_placeholder_session_compatibility(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(SnapshotAgentType())
    service = AgentService(
        runtime_root,
        agent_types=registry,
        providers={"codex": CodexProvider(runtime_root=runtime_root)},
    )
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.store.create_agent_record(
        scope_id="scope-a",
        agent_type="worker",
        cli_type="codex",
        home_id="worker",
        thread_id="placeholder-thread",
    )
    assert agent.artifact_locator is None

    snapshot = AgentSnapshotService(
        runtime_root,
        store=service.store,
        agent_service=service,
    ).create_scope_snapshot("scope-a")

    assert snapshot.status == "created"
    assert snapshot.snapshot_id is not None


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
        / "reports"
        / "agents"
        / agent_a.agent_id
        / "latest.json"
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
    assert not (
        snapshot_dir
        / "files"
        / "reports"
        / "agents"
        / agent_b.agent_id
        / "latest.json"
    ).exists()


def test_scope_restore_replaces_scope_and_rollout_from_snapshot(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.start_agent(agent.agent_id, variables={"item": "first"})
    service.wait_agent(agent.agent_id)
    snapshot_service = AgentSnapshotService(
        runtime_root,
        store=service.store,
        agent_service=service,
        trace_report_policy=AgentTraceReportPolicy(include_in_snapshots=True),
    )
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.snapshot_id is not None
    rollout_path = service.store.locate_rollout(agent.agent_id)
    assert rollout_path is not None
    report_path = service.store.report_dir(agent.agent_id) / "latest.json"
    assert report_path.exists()
    rollout_path.write_text('{"type": "turn_result", "thread_id": "bad", "turn_id": "bad", "prompt": "bad"}\n', encoding="utf-8")
    report_path.write_text('{"bad": true}\n', encoding="utf-8")

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    events = service.read_rollout_events(agent.agent_id)
    assert events[-1]["prompt"] == "Start first."
    assert json.loads(report_path.read_text(encoding="utf-8"))["rollout"]["agent_id"] == agent.agent_id
    assert service.get_agent(agent.agent_id).scope_id == "scope-a"


def test_scope_snapshot_restores_unresolved_context_maintenance_fail_closed(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "first"}).agent_id)
    restored_agent = service.get_agent(agent.agent_id)
    service.store.write_context_maintenance(
        agent.agent_id,
        AgentContextMaintenanceJournal(
            agent_id=agent.agent_id,
            provider_type="codex",
            session_id=restored_agent.thread_id,
            status=AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL,
            trigger="manual",
            prepared_at="2026-07-21T00:00:00Z",
            started_at="2026-07-21T00:00:01Z",
            baseline={"offset": 12},
            error_type="TimeoutError",
        ),
    )
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.snapshot_id is not None
    service.store.write_context_maintenance(
        agent.agent_id,
        AgentContextMaintenanceJournal(
            agent_id=agent.agent_id,
            provider_type="codex",
            session_id=restored_agent.thread_id,
            status=AgentContextMaintenanceJournalStatus.CONFIRMED,
            trigger="manual",
            prepared_at="2026-07-21T00:00:00Z",
            completed_at="2026-07-21T00:00:02Z",
        ),
    )

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    journal = service.store.read_context_maintenance(agent.agent_id)
    assert journal is not None
    assert journal.status is AgentContextMaintenanceJournalStatus.UNKNOWN_TERMINAL
    with pytest.raises(AgentContextMaintenanceBlocked):
        service.start_agent(agent.agent_id, variables={"item": "blocked"})


def test_scope_restore_without_reports_removes_stale_live_reports(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.start_agent(agent.agent_id, variables={"item": "first"})
    service.wait_agent(agent.agent_id)
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.snapshot_id is not None
    report_dir = service.store.report_dir(agent.agent_id)
    assert report_dir.exists()
    assert not (runtime_root / str(snapshot.snapshot_relpath) / "files" / "reports").exists()

    (report_dir / "latest.json").write_text('{"stale": true}\n', encoding="utf-8")
    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    assert not report_dir.exists()
    rebuilt = service.export_default_trace_reports(agent.agent_id)
    assert Path(rebuilt.latest_json_path).exists()
    assert service.read_default_trace_report(agent.agent_id)["rollout"]["agent_id"] == agent.agent_id


def test_scope_snapshot_without_reports_reduces_report_heavy_payload_by_at_least_75_percent(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.start_agent(agent.agent_id, variables={"item": "first"})
    service.wait_agent(agent.agent_id)
    report_dir = service.store.report_dir(agent.agent_id)
    turns_dir = report_dir / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)
    (turns_dir / "large.json").write_bytes(b"x" * (4 * 1024 * 1024))

    with_reports = AgentSnapshotService(
        runtime_root,
        store=service.store,
        agent_service=service,
        trace_report_policy=AgentTraceReportPolicy(include_in_snapshots=True),
    ).create_scope_snapshot("scope-a")
    without_reports = AgentSnapshotService(
        runtime_root,
        store=service.store,
        agent_service=service,
    ).create_scope_snapshot("scope-a")

    assert with_reports.snapshot_id is not None
    assert without_reports.snapshot_id is not None
    with_reports_dir = runtime_root / str(with_reports.snapshot_relpath)
    without_reports_dir = runtime_root / str(without_reports.snapshot_relpath)
    with_reports_bytes = sum(path.stat().st_size for path in with_reports_dir.rglob("*") if path.is_file())
    without_reports_bytes = sum(path.stat().st_size for path in without_reports_dir.rglob("*") if path.is_file())
    assert without_reports_bytes <= with_reports_bytes * 0.25
    assert not (without_reports_dir / "files" / "reports").exists()


def test_scope_restore_accepts_legacy_v1_manifest_without_file_checksums(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.snapshot_id is not None
    manifest_path = runtime_root / str(snapshot.snapshot_relpath) / "snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest.pop("files")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    service.store.patch_agent(agent.agent_id, status="closed")

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    assert service.get_agent(agent.agent_id).status == "idle"


def test_scope_restore_rejects_corrupted_archive_before_mutating_live_scope(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.snapshot_id is not None
    service.store.patch_agent(agent.agent_id, status="closed")
    snapshot_agent = (
        runtime_root
        / str(snapshot.snapshot_relpath)
        / "files"
        / "scopes"
        / encode_scope_id("scope-a")
        / "agents"
        / agent.agent_id
        / "agent.json"
    )
    snapshot_agent.write_text("{}\n", encoding="utf-8")

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "failed"
    assert "snapshot_archive" in restored.errors
    assert service.get_agent(agent.agent_id).status == "closed"


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


def test_runtime_restore_can_prune_scopes_and_artifacts_created_after_snapshot(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    service.create_agent("scope-a", "worker")
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)
    runtime_snapshot = snapshot_service.create_runtime_snapshot_synchronized()
    assert runtime_snapshot.snapshot_id is not None

    late_agent = service.create_agent("scope-late", "worker")
    service.start_agent(late_agent.agent_id, variables={"item": "late"})
    service.wait_agent(late_agent.agent_id)
    late_rollout = service.store.locate_rollout(late_agent.agent_id)
    late_report = service.store.report_dir(late_agent.agent_id)
    assert late_rollout is not None and late_rollout.exists()
    assert late_report.exists()

    restored = snapshot_service.restore_runtime_snapshot(
        runtime_snapshot.snapshot_id,
        prune_extra_scopes=True,
    )

    assert restored.status == "created"
    assert restored.pruned_scope_ids == ("scope-late",)
    assert service.store.list_scope_ids() == ["scope-a"]
    assert not late_rollout.exists()
    assert not late_report.exists()


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


def test_manual_compaction_reservation_blocks_snapshot_and_new_turn(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service = _make_service(runtime_root)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "first"}).agent_id)
    provider = service.providers["codex"]
    assert isinstance(provider, FakeProvider)
    provider.compact_started_event = threading.Event()
    provider.compact_release_event = threading.Event()
    errors: list[BaseException] = []

    def compact() -> None:
        try:
            service.compact_agent(agent.agent_id)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=compact)
    worker.start()
    assert provider.compact_started_event.wait(timeout=5)
    snapshot_service = AgentSnapshotService(runtime_root, store=service.store, agent_service=service)

    snapshot = snapshot_service.create_scope_snapshot("scope-a", wait=False)

    assert snapshot.status == "blocked"
    assert snapshot.running_agent_ids == (agent.agent_id,)
    with pytest.raises(AgentAlreadyRunningError):
        service.start_agent(agent.agent_id, variables={"item": "concurrent"})
    provider.compact_release_event.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []


def _make_service(runtime_root: Path) -> AgentService:
    registry = AgentTypeRegistry()
    registry.register(SnapshotAgentType())
    provider = FakeProvider(runtime_root)
    return AgentService(runtime_root, agent_types=registry, providers={"codex": provider})
