from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
import re
import sqlite3
from pathlib import Path
from ..models import to_jsonable
from ..provider_contracts import AgentEvent, ModelBackendIdentity, ProviderRunState
from ..store_utils import utc_now_iso


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


class OpenAIAgentsSessionStore:
    schema_version = 1

    def __init__(self, path: Path, *, session_id: str, home_id: str) -> None:
        self.path = Path(path)
        self.session_id = safe_session_id(session_id)
        self.home_id = home_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @classmethod
    def path_for(cls, home_root: Path, session_id: str) -> Path:
        return Path(home_root) / "sessions" / f"{safe_session_id(session_id)}.sqlite3"

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("pragma journal_mode=wal")
        conn.execute("pragma foreign_keys=on")
        return conn

    def _ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists ark_session(
                  session_id text primary key,
                  home_id text not null,
                  created_at text not null,
                  updated_at text not null,
                  status text not null,
                  backend_json text,
                  fork_json text
                );
                create table if not exists ark_turn(
                  turn_id text primary key,
                  run_id text not null unique,
                  sequence integer not null,
                  started_at text not null,
                  completed_at text,
                  status text not null,
                  result_json text,
                  error_json text
                );
                create table if not exists ark_event(
                  sequence integer primary key,
                  turn_id text,
                  kind text not null,
                  timestamp text not null,
                  event_json text not null
                );
                create table if not exists ark_pending_state(
                  state_id text primary key,
                  turn_id text not null,
                  state_json text not null,
                  interruptions_json text not null,
                  factory_ref text not null,
                  factory_version text not null,
                  resource_fingerprint text,
                  status text not null,
                  created_at text not null,
                  consumed_at text
                );
                create table if not exists ark_maintenance(
                  operation_id text primary key,
                  status text not null,
                  baseline_json text,
                  result_json text,
                  started_at text not null,
                  completed_at text
                );
                create table if not exists ark_meta(key text primary key, value text not null);
                """
            )
            now = utc_now_iso()
            conn.execute(
                """insert into ark_session(session_id,home_id,created_at,updated_at,status)
                   values(?,?,?,?,?) on conflict(session_id) do nothing""",
                (self.session_id, self.home_id, now, now, "idle"),
            )
            conn.execute(
                "insert into ark_meta(key,value) values('schema_version',?) on conflict(key) do update set value=excluded.value",
                (str(self.schema_version),),
            )

    def begin_turn(self, *, turn_id: str, run_id: str, started_at: str, backend: ModelBackendIdentity | None) -> int:
        with self.connect() as conn:
            row = conn.execute("select coalesce(max(sequence),-1)+1 from ark_turn").fetchone()
            sequence = int(row[0])
            conn.execute(
                "insert into ark_turn(turn_id,run_id,sequence,started_at,status) values(?,?,?,?,?)",
                (turn_id, run_id, sequence, started_at, ProviderRunState.RUNNING.value),
            )
            conn.execute(
                "update ark_session set updated_at=?,status=?,backend_json=? where session_id=?",
                (started_at, "running", _json(backend), self.session_id),
            )
            return sequence

    def append_event(self, event: AgentEvent) -> None:
        with self.connect() as conn:
            conn.execute(
                "insert into ark_event(sequence,turn_id,kind,timestamp,event_json) values(?,?,?,?,?)",
                (event.sequence, event.turn_id, event.kind, event.timestamp, _json(event)),
            )

    def next_event_sequence(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("select coalesce(max(sequence),-1)+1 from ark_event").fetchone()[0])

    def finish_turn(self, *, turn_id: str, status: ProviderRunState, completed_at: str, result: object, error: object | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "update ark_turn set completed_at=?,status=?,result_json=?,error_json=? where turn_id=?",
                (completed_at, status.value, _json(result), _json(error), turn_id),
            )
            conn.execute(
                "update ark_session set updated_at=?,status=? where session_id=?",
                (completed_at, status.value, self.session_id),
            )

    def save_pending_state(
        self,
        *,
        state_id: str,
        turn_id: str,
        state_json: str,
        interruptions: object,
        factory_ref: str,
        factory_version: str,
        resource_fingerprint: str | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """insert into ark_pending_state(
                     state_id,turn_id,state_json,interruptions_json,factory_ref,factory_version,
                     resource_fingerprint,status,created_at) values(?,?,?,?,?,?,?,?,?)""",
                (
                    state_id,
                    turn_id,
                    state_json,
                    _json(interruptions),
                    factory_ref,
                    factory_version,
                    resource_fingerprint,
                    "pending",
                    utc_now_iso(),
                ),
            )

    def pending_state(self, state_id: str | None = None) -> sqlite3.Row | None:
        with self.connect() as conn:
            if state_id is not None:
                return conn.execute(
                    "select * from ark_pending_state where state_id=? and status='pending'",
                    (state_id,),
                ).fetchone()
            return conn.execute(
                "select * from ark_pending_state where status='pending' order by created_at desc limit 1"
            ).fetchone()

    def consume_pending_state(self, state_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "update ark_pending_state set status='consumed',consumed_at=? where state_id=? and status='pending'",
                (utc_now_iso(), state_id),
            )
            return cursor.rowcount == 1

    def turn_rows(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute("select * from ark_turn order by sequence"))

    def event_rows(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute("select * from ark_event order by sequence"))

    def session_row(self) -> sqlite3.Row:
        with self.connect() as conn:
            row = conn.execute("select * from ark_session where session_id=?", (self.session_id,)).fetchone()
        if row is None:
            raise RuntimeError(f"missing OpenAI Agents session: {self.session_id}")
        return row

    def is_quiescent(self) -> bool:
        return str(self.session_row()["status"]) not in {"starting", "running", "compacting"}

    def set_session_status(self, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "update ark_session set status=?,updated_at=? where session_id=?",
                (status, utc_now_iso(), self.session_id),
            )

    def backup_to(self, target: Path) -> None:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        source_conn = self.connect()
        target_conn = sqlite3.connect(target)
        try:
            source_conn.backup(target_conn)
        finally:
            target_conn.close()
            source_conn.close()

    def integrity_check(self) -> None:
        with self.connect() as conn:
            value = conn.execute("pragma integrity_check").fetchone()[0]
        if value != "ok":
            raise RuntimeError(f"OpenAI Agents SQLite integrity check failed: {value}")


def safe_session_id(value: str) -> str:
    candidate = str(value).strip()
    if not _SAFE_ID.fullmatch(candidate):
        raise ValueError("session_id must use 1-192 safe filename characters")
    return candidate


def load_json(value: str | None) -> object | None:
    return json.loads(value) if value else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: object | None) -> str | None:
    if value is None:
        return None
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: object) -> object:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", exclude_none=True)
        except TypeError:
            return dump()
    if is_dataclass(value):
        return asdict(value)
    return repr(value)
