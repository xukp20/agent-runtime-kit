from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .models import (
    Agent,
    AgentCompletionRecord,
    AgentForkInfo,
    AgentHasNoCompletedTurn,
    agent_from_dict,
    to_jsonable,
)
from .provider_contracts import AgentArtifactLocator, ProviderSessionLocator, ProviderTurnLocator
from .context import AgentContextMaintenanceJournal
from .store_utils import encode_scope_id, read_json, utc_now_iso, write_json_atomic
from .trace import (
    AgentRolloutInfo,
    AgentTraceEventView,
    AgentTraceReader,
    AgentTraceReport,
    AgentTraceReportPaths,
    AgentTurnSummary,
    AgentResponseTextView,
    AgentToolCallView,
)


class AgentStoreService:
    def __init__(self, runtime_root: Path, providers: dict[str, object] | None = None) -> None:
        self.runtime_root = Path(runtime_root)
        self.scopes_root = self.runtime_root / "scopes"
        self.index_root = self.runtime_root / "index"
        self.global_index_path = self.index_root / "global.sqlite"
        self.providers = providers or {}
        self.scopes_root.mkdir(parents=True, exist_ok=True)
        self.index_root.mkdir(parents=True, exist_ok=True)
        self._ensure_global_schema()

    def create_agent_record(
        self,
        *,
        scope_id: str,
        agent_type: str,
        cli_type: str = "codex",
        home_id: str | None = None,
        thread_id: str | None = None,
        rollout_relpath: str | None = None,
        fork_source_agent_id: str | None = None,
        fork_source_thread_id: str | None = None,
        agent_id: str | None = None,
        provider_type: str | None = None,
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
            cli_type=cli_type,
            home_id=home_id or agent_type,
            provider_type=provider_type or cli_type,
            session_locator=session_locator,
            latest_turn_locator=latest_turn_locator,
            artifact_locator=artifact_locator,
            fork_info=fork_info,
            thread_id=thread_id,
            rollout_relpath=rollout_relpath,
            status="idle",
            fork_source_agent_id=fork_source_agent_id,
            fork_source_thread_id=fork_source_thread_id,
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

    def update_thread_locator(
        self,
        agent_id: str,
        *,
        thread_id: str,
        rollout_relpath: str | None,
        session_locator: ProviderSessionLocator | None = None,
        latest_turn_locator: ProviderTurnLocator | None = None,
        artifact_locator: AgentArtifactLocator | None = None,
    ) -> Agent:
        agent = self.get_agent(agent_id)
        if session_locator is not None:
            resolved_session = session_locator
        elif agent.session_locator is not None and agent.session_locator.session_id == thread_id:
            # A live-start callback may repeat the current session before a new
            # turn exists. Keep the exact locator referenced by latest_turn.
            resolved_session = agent.session_locator
        else:
            resolved_session = ProviderSessionLocator(
                provider_type=agent.provider_type or agent.cli_type,
                session_id=thread_id,
                home_id=agent.home_id,
                created_at=utc_now_iso(),
                native_locator={"rollout_relpath": rollout_relpath},
            )
        resolved_artifact = artifact_locator
        if resolved_artifact is None and (agent.provider_type or agent.cli_type) == "codex" and rollout_relpath:
            resolved_artifact = AgentArtifactLocator(
                provider_type="codex",
                home_id=agent.home_id,
                session_id=thread_id,
                adapter_version="codex-artifact-v1",
                native_primary_ref=rollout_relpath,
            )
        return self.patch_agent(
            agent_id,
            thread_id=thread_id,
            rollout_relpath=rollout_relpath,
            session_locator=resolved_session,
            latest_turn_locator=latest_turn_locator or agent.latest_turn_locator,
            artifact_locator=resolved_artifact or agent.artifact_locator,
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

    def locate_rollout(self, agent_id: str) -> Path | None:
        agent = self.get_agent(agent_id)
        if not agent.thread_id or not agent.rollout_relpath:
            return None
        if agent.cli_type == "codex":
            return self.runtime_root / "homes" / "codex" / agent.home_id / ".codex" / agent.rollout_relpath
        return None

    def read_rollout_events(self, agent_id: str) -> list[dict]:
        path = self.locate_rollout(agent_id)
        if path is None or not path.exists():
            return []
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    def trace_reader(self, agent_id: str) -> AgentTraceReader:
        return AgentTraceReader(
            agent=self.get_agent(agent_id),
            rollout_path=self.locate_rollout(agent_id),
            events=self.read_rollout_events(agent_id),
        )

    def get_rollout_info(self, agent_id: str) -> AgentRolloutInfo:
        return self.trace_reader(agent_id).get_rollout_info()

    def list_trace_turns(self, agent_id: str) -> list[AgentTurnSummary]:
        return self.trace_reader(agent_id).list_turns()

    def get_trace_turn(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        index: int | None = None,
        latest: bool = False,
    ) -> AgentTurnSummary | None:
        return self.trace_reader(agent_id).get_turn(turn_id=turn_id, index=index, latest=latest)

    def get_trace_event(
        self,
        agent_id: str,
        *,
        index: int | None = None,
        last: bool = False,
    ) -> AgentTraceEventView | None:
        return self.trace_reader(agent_id).get_event(index=index, last=last)

    def tail_trace_events(
        self,
        agent_id: str,
        *,
        limit: int = 20,
        event_type: str | None = None,
        payload_type: str | None = None,
    ) -> list[AgentTraceEventView]:
        return self.trace_reader(agent_id).tail_events(
            limit=limit,
            event_type=event_type,
            payload_type=payload_type,
        )

    def list_response_texts(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ) -> list[AgentResponseTextView]:
        return self.trace_reader(agent_id).list_response_texts(turn_id=turn_id, latest=latest)

    def get_latest_response_text(self, agent_id: str) -> str | None:
        return self.trace_reader(agent_id).get_latest_response_text()

    def list_tool_calls(
        self,
        agent_id: str,
        *,
        turn_id: str | None = None,
        latest: bool = False,
    ) -> list[AgentToolCallView]:
        return self.trace_reader(agent_id).list_tool_calls(turn_id=turn_id, latest=latest)

    def get_tool_call(
        self,
        agent_id: str,
        *,
        call_id: str | None = None,
        index: int | None = None,
        last: bool = False,
    ) -> AgentToolCallView | None:
        return self.trace_reader(agent_id).get_tool_call(call_id=call_id, index=index, last=last)

    def build_trace_report(
        self,
        agent_id: str,
        *,
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
    ) -> AgentTraceReport:
        return self.trace_reader(agent_id).build_trace_report(
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )

    def export_trace_report(
        self,
        agent_id: str,
        *,
        output_path: str | Path,
        format: str = "json",
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
    ) -> AgentTraceReport:
        return self.trace_reader(agent_id).export_trace_report(
            output_path=output_path,
            format=format,
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )

    def get_default_trace_report_paths(self, agent_id: str) -> AgentTraceReportPaths:
        reports_root = self.report_dir(agent_id)
        return AgentTraceReportPaths(
            agent_id=agent_id,
            reports_root=str(reports_root),
            latest_json_path=str(reports_root / "latest.json"),
            latest_markdown_path=str(reports_root / "latest.md"),
        )

    def report_dir(self, agent_id: str) -> Path:
        return self.runtime_root / "reports" / "agents" / _safe_report_key(agent_id)

    def export_default_trace_reports(
        self,
        agent_id: str,
        *,
        artifact_path: str | Path | None = None,
        slow_call_limit: int = 20,
        include_turn_history: bool = True,
    ) -> AgentTraceReportPaths:
        reports_root = self.report_dir(agent_id)
        latest_json = reports_root / "latest.json"
        latest_markdown = reports_root / "latest.md"
        reports_root.mkdir(parents=True, exist_ok=True)
        report = self.export_trace_report(
            agent_id,
            output_path=latest_json,
            format="json",
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )
        self.export_trace_report(
            agent_id,
            output_path=latest_markdown,
            format="markdown",
            artifact_path=artifact_path,
            slow_call_limit=slow_call_limit,
        )
        written = [str(latest_json), str(latest_markdown)]
        turn_json = None
        turn_markdown = None
        if include_turn_history and report.latest_turn is not None:
            turn_key = _safe_report_key(report.latest_turn.turn_id)
            turn_json = reports_root / "turns" / f"{turn_key}.json"
            turn_markdown = reports_root / "turns" / f"{turn_key}.md"
            turn_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(latest_json, turn_json)
            shutil.copyfile(latest_markdown, turn_markdown)
            written.extend([str(turn_json), str(turn_markdown)])
        return AgentTraceReportPaths(
            agent_id=agent_id,
            reports_root=str(reports_root),
            latest_json_path=str(latest_json),
            latest_markdown_path=str(latest_markdown),
            turn_json_path=str(turn_json) if turn_json is not None else None,
            turn_markdown_path=str(turn_markdown) if turn_markdown is not None else None,
            written_paths=written,
        )

    def read_default_trace_report(self, agent_id: str, *, format: str = "json") -> object | None:
        paths = self.get_default_trace_report_paths(agent_id)
        if format == "json":
            path = Path(paths.latest_json_path)
            return read_json(path) if path.exists() else None
        if format == "markdown":
            path = Path(paths.latest_markdown_path)
            return path.read_text(encoding="utf-8") if path.exists() else None
        raise ValueError(f"unsupported trace report format: {format}")

    def read_thread(self, agent_id: str, include_turns: bool = True) -> object:
        agent = self.get_agent(agent_id)
        if not agent.thread_id:
            raise AgentHasNoCompletedTurn(agent_id)
        provider = self.providers.get(agent.cli_type)
        if provider is None:
            raise RuntimeError(f"no provider registered for {agent.cli_type}")
        return provider.read_thread(agent, include_turns=include_turns)

    def list_turns(self, agent_id: str) -> list[object]:
        thread = self.read_thread(agent_id, include_turns=True)
        return list(getattr(thread, "turns", []) or [])

    def read_latest_turn_result(self, agent_id: str) -> object:
        provider = self.providers.get(self.get_agent(agent_id).cli_type)
        if provider is not None and hasattr(provider, "read_latest_turn_result"):
            return provider.read_latest_turn_result(self.get_agent(agent_id))
        turns = self.list_turns(agent_id)
        completed = [turn for turn in turns if getattr(turn, "status", "completed") == "completed"]
        if not completed:
            raise AgentHasNoCompletedTurn(agent_id)
        return completed[-1]

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
        agent.normalize_compat_fields()
        path = self._agent_json_path(agent)
        write_json_atomic(
            path,
            {
                "schema_version": 2,
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
          cli_type text not null,
          home_id text not null,
          status text not null,
          thread_id text,
          rollout_relpath text,
          last_completion_status text,
          fork_source_agent_id text,
          fork_source_thread_id text,
          agent_relpath text not null,
          created_at text not null,
          updated_at text not null
        )
        """
    )
    conn.execute("create index if not exists idx_agents_type_status on agents(agent_type, status)")
    conn.execute("create index if not exists idx_agents_cli_home on agents(cli_type, home_id)")
    conn.execute("create index if not exists idx_agents_thread on agents(thread_id)")
    conn.execute("create index if not exists idx_agents_status_updated on agents(status, updated_at)")
    conn.execute("create index if not exists idx_agents_completion on agents(last_completion_status)")
    conn.execute("create index if not exists idx_agents_fork_source on agents(fork_source_agent_id)")
    conn.execute("create index if not exists idx_agents_scope_status on agents(scope_id, status)")


def _upsert_agent_row(conn: sqlite3.Connection, agent: Agent, scope_key: str, agent_relpath: str) -> None:
    completion_status = agent.last_completion.status if agent.last_completion is not None else None
    conn.execute(
        """
        insert into agents(
          agent_id, scope_id, scope_key, agent_type, cli_type, home_id, status,
          thread_id, rollout_relpath, last_completion_status,
          fork_source_agent_id, fork_source_thread_id,
          agent_relpath, created_at, updated_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        on conflict(agent_id) do update set
          scope_id=excluded.scope_id,
          scope_key=excluded.scope_key,
          agent_type=excluded.agent_type,
          cli_type=excluded.cli_type,
          home_id=excluded.home_id,
          status=excluded.status,
          thread_id=excluded.thread_id,
          rollout_relpath=excluded.rollout_relpath,
          last_completion_status=excluded.last_completion_status,
          fork_source_agent_id=excluded.fork_source_agent_id,
          fork_source_thread_id=excluded.fork_source_thread_id,
          agent_relpath=excluded.agent_relpath,
          updated_at=excluded.updated_at
        """,
        (
            agent.agent_id,
            agent.scope_id,
            scope_key,
            agent.agent_type,
            agent.cli_type,
            agent.home_id,
            agent.status,
            agent.thread_id,
            agent.rollout_relpath,
            completion_status,
            agent.fork_source_agent_id,
            agent.fork_source_thread_id,
            agent_relpath,
            agent.created_at,
            agent.updated_at,
        ),
    )


def _safe_report_key(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
