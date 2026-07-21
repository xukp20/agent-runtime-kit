import json
from pathlib import Path

import pytest

from agent_runtime_kit.agent.context import (
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
)
from agent_runtime_kit.agent.models import CompletionDecision, AgentCompletionRecord
from agent_runtime_kit.agent.provider_contracts import ProviderTurnLocator
from agent_runtime_kit.agent.store import AgentStoreService
from agent_runtime_kit.agent.store_utils import encode_scope_id


def test_store_creates_agent_json_and_indexes(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")

    agent = store.create_agent_record(scope_id="repo:node/a", agent_type="node_worker", cli_type="codex")

    assert agent.home_id == "node_worker"
    assert agent.status == "idle"
    assert store.get_agent(agent.agent_id).scope_id == "repo:node/a"
    assert [item.agent_id for item in store.list_agents(scope_id="repo:node/a")] == [agent.agent_id]
    assert [item.agent_id for item in store.list_agents()] == [agent.agent_id]


def test_store_round_trips_context_maintenance_journal(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = store.create_agent_record(scope_id="repo:node/a", agent_type="node_worker")
    journal = AgentContextMaintenanceJournal(
        agent_id=agent.agent_id,
        provider_type="codex",
        session_id=None,
        status=AgentContextMaintenanceJournalStatus.PREPARED,
        trigger="manual",
        prepared_at="2026-07-21T00:00:00Z",
    )

    path = store.write_context_maintenance(agent.agent_id, journal)

    assert path == store.resolve_agent_path(agent.agent_id).parent / "context_maintenance.json"
    assert store.read_context_maintenance(agent.agent_id) == journal

    store.clear_context_maintenance(agent.agent_id)
    assert store.read_context_maintenance(agent.agent_id) is None


def test_store_persists_completion_record(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = store.create_agent_record(scope_id="scope", agent_type="planner")
    record = AgentCompletionRecord(
        turn_id="turn-1",
        decision=CompletionDecision(complete=False, reason="need more"),
        status="incomplete",
        auto_continue_count=0,
        checked_at="2026-06-06T00:00:00Z",
    )

    store.update_completion(agent.agent_id, record)
    restored = store.get_agent(agent.agent_id)

    assert restored.last_completion is not None
    assert restored.last_completion.decision.reason == "need more"
    assert restored.last_completion.status == "incomplete"


def test_store_rebuilds_indexes_from_scope_json_truth(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    store = AgentStoreService(runtime_root)
    agent = store.create_agent_record(scope_id="repo:node/a", agent_type="node_worker")
    scope_key = encode_scope_id("repo:node/a")

    (runtime_root / "index" / "global.sqlite").unlink()
    (runtime_root / "scopes" / scope_key / "index.sqlite").unlink()
    rebuilt = AgentStoreService(runtime_root)

    assert rebuilt.get_agent(agent.agent_id).agent_type == "node_worker"
    assert [item.agent_id for item in rebuilt.list_agents(scope_id="repo:node/a")] == [agent.agent_id]


def test_store_reads_rollout_jsonl_events(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    rollout = runtime_root / "homes" / "codex" / "worker" / ".codex" / "sessions" / "fake.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text('{"type": "turn", "id": "turn-1"}\n', encoding="utf-8")
    store = AgentStoreService(runtime_root)
    agent = store.create_agent_record(
        scope_id="scope",
        agent_type="worker",
        cli_type="codex",
        home_id="worker",
        thread_id="thread-1",
        rollout_relpath="sessions/fake.jsonl",
    )

    assert store.locate_rollout(agent.agent_id) == rollout
    assert store.read_rollout_events(agent.agent_id) == [{"type": "turn", "id": "turn-1"}]


def test_store_dual_writes_agent_record_v2_and_legacy_codex_locators(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = store.create_agent_record(
        scope_id="scope",
        agent_type="worker",
        cli_type="codex",
        home_id="worker",
        thread_id="thread-1",
        rollout_relpath="sessions/thread-1.jsonl",
    )

    payload = json.loads(store.resolve_agent_path(agent.agent_id).read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["provider_type"] == "codex"
    assert payload["session_locator"]["session_id"] == "thread-1"
    assert payload["artifact_locator"]["native_primary_ref"] == "sessions/thread-1.jsonl"
    assert payload["cli_type"] == "codex"
    assert payload["thread_id"] == "thread-1"
    assert payload["rollout_relpath"] == "sessions/thread-1.jsonl"


def test_store_reads_legacy_agent_record_without_rewriting_it(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = store.create_agent_record(scope_id="scope", agent_type="worker")
    path = store.resolve_agent_path(agent.agent_id)
    legacy = {
        "schema_version": 1,
        "object_type": "agent",
        "agent_id": agent.agent_id,
        "scope_id": "scope",
        "agent_type": "worker",
        "cli_type": "codex",
        "home_id": "worker",
        "thread_id": "legacy-thread",
        "rollout_relpath": "sessions/legacy.jsonl",
        "status": "idle",
        "created_at": "2026-07-20T00:00:00Z",
        "updated_at": "2026-07-20T00:00:00Z",
    }
    original = json.dumps(legacy, sort_keys=True) + "\n"
    path.write_text(original, encoding="utf-8")

    restored = store.get_agent(agent.agent_id)

    assert restored.schema_version == 2
    assert restored.provider_type == "codex"
    assert restored.session_locator is not None
    assert restored.session_locator.session_id == "legacy-thread"
    assert restored.artifact_locator is not None
    assert restored.artifact_locator.native_primary_ref == "sessions/legacy.jsonl"
    assert path.read_text(encoding="utf-8") == original


def test_store_rejects_conflicting_v2_and_legacy_provider_aliases(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = store.create_agent_record(scope_id="scope", agent_type="worker")
    path = store.resolve_agent_path(agent.agent_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provider_type"] = "claude-code"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provider_type and legacy cli_type conflict"):
        store.get_agent(agent.agent_id)


def test_store_reuses_exact_session_locator_for_resume_start_callback(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = store.create_agent_record(scope_id="scope", agent_type="worker")
    first = store.update_thread_locator(
        agent.agent_id,
        thread_id="thread-1",
        rollout_relpath="sessions/thread-1.jsonl",
    )
    turn = ProviderTurnLocator(session=first.session_locator, turn_id="turn-1")
    store.patch_agent(agent.agent_id, latest_turn_locator=turn)

    resumed = store.update_thread_locator(
        agent.agent_id,
        thread_id="thread-1",
        rollout_relpath="sessions/thread-1.jsonl",
    )

    assert resumed.session_locator == first.session_locator
    assert resumed.latest_turn_locator == turn
