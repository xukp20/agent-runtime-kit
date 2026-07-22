from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .models import (
    Agent,
    AgentCompletionRecord,
    AgentForkInfo,
    agent_from_dict,
    to_jsonable,
)
from .provider_contracts import AgentArtifactLocator, ProviderSessionLocator, ProviderTurnLocator
from .context import AgentContextMaintenanceJournal
from .store_utils import encode_scope_id, read_json, utc_now_iso, write_json_atomic


class AgentStoreService:
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)
        self.scopes_root = self.runtime_root / "scopes"
        self.index_root = self.runtime_root / "index"
        self.global_index_path = self.index_root / "global.sqlite"
        self.scopes_root.mkdir(parents=True, exist_ok=True)
        self.index_root.mkdir(parents=True, exist_ok=True)
        self._ensure_global_schema()

    def create_agent_record(
        self,
        *,
        scope_id: str,
        agent_type: str,
        provider_type: str,
        home_id: str | None = None,
        agent_id: str | None = None,
        session_locator: ProviderSessionLocator | None = None,
        latest_turn_locator: ProviderTurnLocator | None = None,
        artifact_locator: AgentArtifactLocator | None = None,
        fork_info: AgentForkInfo | None = None,
    ) -> Agent:
        now = utc_now_iso()
        agent = Agent(
            agent_id=agent_id or f"a_{uuid.uuid4().hex}",
            scope_id=scope_id,
            agent_type=agent_type,
            provider_type=provider_type,
            home_id=home_id or agent_type,
            session_locator=session_locator,
            latest_turn_locator=latest_turn_locator,
            artifact_locator=artifact_locator,
            fork_info=fork_info,
            status="idle",
            created_at=now,
            updated_at=now,
        )
        self._ensure_scope(scope_id)
        self._write_agent(agent)
        self._upsert_scope_index(agent)
        self._upsert_global_index(agent)
        return agent

    def get_agent(self, agent_id: str) -> Agent:
        return agent_from_dict(read_json(self.resolve_agent_path(agent_id)))

    def patch_agent(self, agent_id: str, **fields: object) -> Agent:
        if "agent_id" in fields or "created_at" in fields:
            raise ValueError("agent_id and created_at cannot be patched")
        agent = self.get_agent(agent_id)
        for key, value in fields.items():
            if not hasattr(agent, key):
                raise ValueError(f"unknown Agent field: {key}")
            setattr(agent, key, value)
        agent.updated_at = utc_now_iso()
        self._write_agent(agent)
        self._upsert_scope_index(agent)
        self._upsert_global_index(agent)
        return agent

    def list_agents(self, scope_id: str | None = None, status: str | None = None) -> list[Agent]:
        if scope_id is not None:
            scope_key = encode_scope_id(scope_id)
            index_path = self._scope_index_path(scope_key)
            index_exists = index_path.exists()
            self._ensure_scope_schema(scope_key)
            if not index_exists:
                self.rebuild_scope_index(scope_id)
            query = "select agent_relpath from agents"
            params: list[str] = []
            if status is not None:
                query += " where status=?"
                params.append(status)
            query += " order by created_at, agent_id"
            with sqlite3.connect(index_path) as conn:
                rows = conn.execute(query, params).fetchall()
            return [agent_from_dict(read_json(self.runtime_root / row[0])) for row in rows]

        if not self.global_index_path.exists():
            self.rebuild_global_index()
        query = "select agent_relpath from agents"
        params = []
        if status is not None:
            query += " where status=?"
            params.append(status)
        query += " order by created_at, agent_id"
        with sqlite3.connect(self.global_index_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [agent_from_dict(read_json(self.runtime_root / row[0])) for row in rows]

    def list_scope_ids(self) -> list[str]:
        scope_ids = []
        for scope_dir in sorted(self.scopes_root.iterdir()) if self.scopes_root.exists() else []:
            scope_json = scope_dir / "scope.json"
            if scope_json.exists():
                scope_ids.append(str(read_json(scope_json)["scope_id"]))
        return scope_ids

    def close_agent(self, agent_id: str) -> Agent:
        return self.patch_agent(agent_id, status="closed")

    def update_session_locators(
        self,
        agent_id: str,
        *,
        session_locator: ProviderSessionLocator,
        latest_turn_locator: ProviderTurnLocator | None = None,
        artifact_locator: AgentArtifactLocator | None = None,
    ) -> Agent:
        agent = self.get_agent(agent_id)
        return self.patch_agent(
            agent_id,
            session_locator=session_locator,
            latest_turn_locator=latest_turn_locator or agent.latest_turn_locator,
            artifact_locator=artifact_locator or agent.artifact_locator,
        )

    def update_completion(self, agent_id: str, record: AgentCompletionRecord | None) -> Agent:
        return self.patch_agent(agent_id, last_completion=record)

    def rebuild_scope_index(self, scope_id: str) -> None:
        scope_key = encode_scope_id(scope_id)
        self._ensure_scope_schema(scope_key)
        with sqlite3.connect(self._scope_index_path(scope_key)) as conn:
            conn.execute("delete from agents")
        agents_dir = self.scopes_root / scope_key / "agents"
        if not agents_dir.exists():
            return
        for agent_json in sorted(agents_dir.glob("*/agent.json")):
            agent = agent_from_dict(read_json(agent_json))
            self._upsert_scope_index(agent)

    def rebuild_global_index(self) -> None:
        self._ensure_global_schema()
        with sqlite3.connect(self.global_index_path) as conn:
            conn.execute("delete from agents")
        seen: set[str] = set()
        for scope_id in self.list_scope_ids():
            scope_key = encode_scope_id(scope_id)
            agents_dir = self.scopes_root / scope_key / "agents"
            if not agents_dir.exists():
                continue
            for agent_json in sorted(agents_dir.glob("*/agent.json")):
                agent = agent_from_dict(read_json(agent_json))
                if agent.agent_id in seen:
                    raise ValueError(f"duplicate agent_id while rebuilding global index: {agent.agent_id}")
                seen.add(agent.agent_id)
                self._upsert_global_index(agent)

    def resolve_agent_path(self, agent_id: str) -> Path:
        path = self._resolve_agent_path_from_global(agent_id)
        if path is not None and path.exists():
            return path
        self.rebuild_global_index()
        path = self._resolve_agent_path_from_global(agent_id)
        if path is None or not path.exists():
            raise KeyError(f"unknown agent: {agent_id}")
        return path

    def resolve_context_maintenance_path(self, agent_id: str) -> Path:
        return self.resolve_agent_path(agent_id).parent / "context_maintenance.json"

    def read_context_maintenance(self, agent_id: str) -> AgentContextMaintenanceJournal | None:
        path = self.resolve_context_maintenance_path(agent_id)
        if not path.exists():
            return None
        journal = AgentContextMaintenanceJournal.from_dict(read_json(path))
        if journal.agent_id != agent_id:
            raise ValueError(
                f"context maintenance journal agent mismatch: expected {agent_id}, got {journal.agent_id}"
            )
        return journal

    def write_context_maintenance(
        self,
        agent_id: str,
        journal: AgentContextMaintenanceJournal,
    ) -> Path:
        if journal.agent_id != agent_id:
            raise ValueError(
                f"context maintenance journal agent mismatch: expected {agent_id}, got {journal.agent_id}"
            )
        path = self.resolve_context_maintenance_path(agent_id)
        write_json_atomic(path, journal.to_dict())
        return path

    def clear_context_maintenance(self, agent_id: str) -> None:
        path = self.resolve_context_maintenance_path(agent_id)
        if path.exists():
            path.unlink()

    def _ensure_scope(self, scope_id: str) -> None:
        scope_key = encode_scope_id(scope_id)
        scope_dir = self.scopes_root / scope_key
        scope_dir.mkdir(parents=True, exist_ok=True)
        scope_json = scope_dir / "scope.json"
        now = utc_now_iso()
        if not scope_json.exists():
            write_json_atomic(
                scope_json,
                {
                    "schema_version": 1,
                    "object_type": "scope",
                    "scope_id": scope_id,
                    "scope_key": scope_key,
                    "created_at": now,
                    "updated_at": now,
                },
            )
        self._ensure_scope_schema(scope_key)

    def _write_agent(self, agent: Agent) -> None:
        agent.__post_init__()
        path = self._agent_json_path(agent)
        write_json_atomic(
            path,
            {
                "schema_version": 3,
                "object_type": "agent",
                **to_jsonable(agent),
            },
        )

    def _agent_json_path(self, agent: Agent) -> Path:
        scope_key = encode_scope_id(agent.scope_id)
        return self.scopes_root / scope_key / "agents" / agent.agent_id / "agent.json"

    def _agent_relpath(self, agent: Agent) -> str:
        return str(self._agent_json_path(agent).relative_to(self.runtime_root))

    def _scope_index_path(self, scope_key: str) -> Path:
        return self.scopes_root / scope_key / "index.sqlite"

    def _ensure_scope_schema(self, scope_key: str) -> None:
        path = self._scope_index_path(scope_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            _create_agents_schema(conn)

    def _ensure_global_schema(self) -> None:
        self.index_root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.global_index_path) as conn:
            _create_agents_schema(conn)

    def _upsert_scope_index(self, agent: Agent) -> None:
        scope_key = encode_scope_id(agent.scope_id)
        self._ensure_scope_schema(scope_key)
        with sqlite3.connect(self._scope_index_path(scope_key)) as conn:
            _upsert_agent_row(conn, agent, scope_key, self._agent_relpath(agent))

    def _upsert_global_index(self, agent: Agent) -> None:
        self._ensure_global_schema()
        with sqlite3.connect(self.global_index_path) as conn:
            _upsert_agent_row(conn, agent, encode_scope_id(agent.scope_id), self._agent_relpath(agent))

    def _resolve_agent_path_from_global(self, agent_id: str) -> Path | None:
        if not self.global_index_path.exists():
            return None
        with sqlite3.connect(self.global_index_path) as conn:
            row = conn.execute("select agent_relpath from agents where agent_id=?", (agent_id,)).fetchone()
        if row is None:
            return None
        return self.runtime_root / str(row[0])


def _create_agents_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists agents(
          agent_id text primary key,
          scope_id text not null,
          scope_key text,
          agent_type text not null,
          provider_type text not null,
          home_id text not null,
          status text not null,
          session_id text,
          last_completion_status text,
          fork_source_agent_id text,
          agent_relpath text not null,
          created_at text not null,
          updated_at text not null
        )
        """
    )
    conn.execute("create index if not exists idx_agents_type_status on agents(agent_type, status)")
    conn.execute("create index if not exists idx_agents_provider_home on agents(provider_type, home_id)")
    conn.execute("create index if not exists idx_agents_session on agents(session_id)")
    conn.execute("create index if not exists idx_agents_status_updated on agents(status, updated_at)")
    conn.execute("create index if not exists idx_agents_completion on agents(last_completion_status)")
    conn.execute("create index if not exists idx_agents_fork_source on agents(fork_source_agent_id)")
    conn.execute("create index if not exists idx_agents_scope_status on agents(scope_id, status)")


def _upsert_agent_row(conn: sqlite3.Connection, agent: Agent, scope_key: str, agent_relpath: str) -> None:
    completion_status = agent.last_completion.status if agent.last_completion is not None else None
    conn.execute(
        """
        insert into agents(
          agent_id, scope_id, scope_key, agent_type, provider_type, home_id, status,
          session_id, last_completion_status, fork_source_agent_id,
          agent_relpath, created_at, updated_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(agent_id) do update set
          scope_id=excluded.scope_id,
          scope_key=excluded.scope_key,
          agent_type=excluded.agent_type,
          provider_type=excluded.provider_type,
          home_id=excluded.home_id,
          status=excluded.status,
          session_id=excluded.session_id,
          last_completion_status=excluded.last_completion_status,
          fork_source_agent_id=excluded.fork_source_agent_id,
          agent_relpath=excluded.agent_relpath,
          updated_at=excluded.updated_at
        """,
        (
            agent.agent_id,
            agent.scope_id,
            scope_key,
            agent.agent_type,
            agent.provider_type,
            agent.home_id,
            agent.status,
            agent.session_locator.session_id if agent.session_locator is not None else None,
            completion_status,
            agent.fork_info.source_agent_id if agent.fork_info is not None else None,
            agent_relpath,
            agent.created_at,
            agent.updated_at,
        ),
    )
