import json
from pathlib import Path

import pytest

from agent_runtime_kit.agent.context import (
    AgentContextMaintenanceJournal,
    AgentContextMaintenanceJournalStatus,
)
from agent_runtime_kit.agent.models import AgentCompletionRecord, CompletionDecision
from agent_runtime_kit.agent.provider_contracts import (
    AgentArtifactLocator,
    ProviderSessionLocator,
    ProviderTurnLocator,
)
from agent_runtime_kit.agent.store import AgentStoreService
from agent_runtime_kit.agent.store_utils import encode_scope_id


def _create(store: AgentStoreService, *, scope_id: str = "scope"):  # noqa: ANN202
    return store.create_agent_record(
        scope_id=scope_id,
        agent_type="worker",
        provider_type="codex",
    )


def test_store_creates_schema_v3_agent_and_indexes(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = _create(store, scope_id="repo:node/a")

    payload = json.loads(store.resolve_agent_path(agent.agent_id).read_text(encoding="utf-8"))

    assert payload["schema_version"] == 3
    assert payload["provider_type"] == "codex"
    assert set(payload) == {
        "schema_version",
        "object_type",
        "agent_id",
        "scope_id",
        "agent_type",
        "provider_type",
        "home_id",
        "session_locator",
        "latest_turn_locator",
        "artifact_locator",
        "fork_info",
        "status",
        "last_completion",
        "created_at",
        "updated_at",
    }
    assert [item.agent_id for item in store.list_agents(scope_id="repo:node/a")] == [agent.agent_id]


def test_store_round_trips_exact_locators_and_completion(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = _create(store)
    session = ProviderSessionLocator(
        provider_type="codex",
        session_id="session-1",
        home_id="worker",
        created_at="2026-07-22T00:00:00Z",
        native_locator={"rollout_relpath": "sessions/one.jsonl", "opaque": {"partition": 7}},
    )
    turn = ProviderTurnLocator(session=session, turn_id="turn-1")
    artifact = AgentArtifactLocator(
        provider_type="codex",
        home_id="worker",
        session_id="session-1",
        adapter_version="1",
        native_primary_ref="sessions/one.jsonl",
    )
    store.update_session_locators(
        agent.agent_id,
        session_locator=session,
        latest_turn_locator=turn,
        artifact_locator=artifact,
    )
    record = AgentCompletionRecord(
        turn_id="turn-1",
        decision=CompletionDecision(complete=True),
        status="complete",
        auto_continue_count=0,
        checked_at="2026-07-22T00:00:01Z",
    )
    store.update_completion(agent.agent_id, record)

    restored = store.get_agent(agent.agent_id)

    assert restored.session_locator == session
    assert restored.latest_turn_locator == turn
    assert restored.artifact_locator == artifact
    assert restored.last_completion == record


def test_store_rejects_pre_v3_records(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = _create(store)
    path = store.resolve_agent_path(agent.agent_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="expected 3"):
        store.get_agent(agent.agent_id)


def test_store_round_trips_context_maintenance_journal(tmp_path: Path) -> None:
    store = AgentStoreService(tmp_path / ".agent_runtime")
    agent = _create(store)
    journal = AgentContextMaintenanceJournal(
        agent_id=agent.agent_id,
        provider_type="codex",
        session_id=None,
        status=AgentContextMaintenanceJournalStatus.PREPARED,
        trigger="manual",
        prepared_at="2026-07-22T00:00:00Z",
    )

    store.write_context_maintenance(agent.agent_id, journal)
    assert store.read_context_maintenance(agent.agent_id) == journal
    store.clear_context_maintenance(agent.agent_id)
    assert store.read_context_maintenance(agent.agent_id) is None


def test_store_rebuilds_indexes_from_schema_v3_json_truth(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    store = AgentStoreService(runtime_root)
    agent = _create(store, scope_id="repo:node/a")
    scope_key = encode_scope_id("repo:node/a")
    (runtime_root / "index" / "global.sqlite").unlink()
    (runtime_root / "scopes" / scope_key / "index.sqlite").unlink()

    rebuilt = AgentStoreService(runtime_root)

    assert rebuilt.get_agent(agent.agent_id).provider_type == "codex"
    assert [item.agent_id for item in rebuilt.list_agents(scope_id="repo:node/a")] == [agent.agent_id]
