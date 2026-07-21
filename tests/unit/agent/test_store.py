from pathlib import Path

from agent_runtime_kit.agent.context import (
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
)
from agent_runtime_kit.agent.models import CompletionDecision, AgentCompletionRecord
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
