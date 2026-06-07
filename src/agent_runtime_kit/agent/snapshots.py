from __future__ import annotations

import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from .models import RuntimeSnapshotInfo, RuntimeSnapshotResult, ScopeSnapshotInfo, ScopeSnapshotResult
from .store import AgentStoreService
from .store_utils import encode_scope_id, read_json, utc_now_iso, write_json_atomic


class AgentSnapshotService:
    def __init__(
        self,
        runtime_root: Path,
        *,
        store: AgentStoreService,
        agent_service: object | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.store = store
        self.agent_service = agent_service
        self.snapshots_root = self.runtime_root / "snapshots"
        self.scope_snapshots_root = self.snapshots_root / "scopes"
        self.runtime_snapshots_root = self.snapshots_root / "runtime"
        self.scope_index_path = self.scope_snapshots_root / "index.sqlite"
        self.runtime_index_path = self.runtime_snapshots_root / "index.sqlite"
        self._lock = threading.RLock()
        self.scope_snapshots_root.mkdir(parents=True, exist_ok=True)
        self.runtime_snapshots_root.mkdir(parents=True, exist_ok=True)
        self._ensure_scope_index()
        self._ensure_runtime_index()

    def create_scope_snapshot(
        self,
        scope_id: str,
        *,
        wait: bool = False,
        timeout_s: float | None = None,
    ) -> ScopeSnapshotResult:
        with self._lock:
            self._pause(scope_id)
            try:
                running = self._running_agents(scope_id)
                if running and not wait:
                    return ScopeSnapshotResult(
                        snapshot_id=None,
                        scope_id=scope_id,
                        status="blocked",
                        running_agent_ids=tuple(agent.agent_id for agent in running),
                    )
                if running:
                    result = self._wait_scope(scope_id, timeout_s=timeout_s)
                    if not getattr(result, "clean", False):
                        return ScopeSnapshotResult(
                            snapshot_id=None,
                            scope_id=scope_id,
                            status="blocked",
                            running_agent_ids=tuple(result.pending),
                            errors=dict(result.errors),
                        )
                return self._create_scope_snapshot_unlocked(scope_id)
            finally:
                self._resume(scope_id)

    def create_runtime_snapshot_causal(self) -> RuntimeSnapshotResult:
        with self._lock:
            latest: dict[str, str] = {}
            for scope_id in self.store.list_scope_ids():
                info = self.get_latest_scope_snapshot(scope_id)
                if info is not None:
                    latest[scope_id] = info.snapshot_id
            return self._create_runtime_snapshot_unlocked(latest, status="created")

    def create_runtime_snapshot_synchronized(
        self,
        *,
        timeout_s: float | None = None,
    ) -> RuntimeSnapshotResult:
        with self._lock:
            self._pause(None)
            try:
                running = self._running_agents(None)
                if running:
                    result = self._wait_all(timeout_s=timeout_s)
                    if not getattr(result, "clean", False):
                        blocked_scope_ids = tuple(sorted({agent.scope_id for agent in running}))
                        return RuntimeSnapshotResult(
                            snapshot_id=None,
                            status="blocked",
                            blocked_scope_ids=blocked_scope_ids,
                            errors=dict(result.errors),
                        )
                scope_snapshot_ids: dict[str, str] = {}
                errors: dict[str, BaseException] = {}
                for scope_id in self.store.list_scope_ids():
                    try:
                        scope_result = self._create_scope_snapshot_unlocked(scope_id)
                        if scope_result.snapshot_id is not None:
                            scope_snapshot_ids[scope_id] = scope_result.snapshot_id
                    except BaseException as exc:
                        errors[scope_id] = exc
                if errors:
                    return RuntimeSnapshotResult(
                        snapshot_id=None,
                        status="failed",
                        scope_snapshot_ids=scope_snapshot_ids,
                        errors=errors,
                    )
                return self._create_runtime_snapshot_unlocked(scope_snapshot_ids, status="created")
            finally:
                self._resume(None)

    def restore_scope_snapshot(self, snapshot_id: str) -> ScopeSnapshotResult:
        with self._lock:
            manifest = self._read_scope_manifest(snapshot_id)
            scope_id = str(manifest["scope_id"])
            self._pause(scope_id)
            try:
                running = self._running_agents(scope_id)
                if running:
                    return ScopeSnapshotResult(
                        snapshot_id=snapshot_id,
                        scope_id=scope_id,
                        status="blocked",
                        running_agent_ids=tuple(agent.agent_id for agent in running),
                    )
                files_root = self._scope_snapshot_dir(snapshot_id) / "files"
                scope_key = str(manifest["scope_key"])
                restored_scope_dir = files_root / "scopes" / scope_key
                current_scope_dir = self.runtime_root / "scopes" / scope_key
                if not self._close_touched_provider_homes(files_root):
                    return ScopeSnapshotResult(
                        snapshot_id=snapshot_id,
                        scope_id=scope_id,
                        status="blocked",
                    )
                if current_scope_dir.exists():
                    shutil.rmtree(current_scope_dir)
                if restored_scope_dir.exists():
                    shutil.copytree(restored_scope_dir, current_scope_dir)
                self._restore_home_files(files_root)
                self._discard_codex_state_databases(files_root)
                self.store.rebuild_scope_index(scope_id)
                self.store.rebuild_global_index()
                return ScopeSnapshotResult(
                    snapshot_id=snapshot_id,
                    scope_id=scope_id,
                    status="created",
                    snapshot_relpath=str(self._scope_snapshot_dir(snapshot_id).relative_to(self.runtime_root)),
                )
            finally:
                self._resume(scope_id)

    def restore_runtime_snapshot(self, snapshot_id: str) -> RuntimeSnapshotResult:
        with self._lock:
            manifest = self._read_runtime_manifest(snapshot_id)
            self._pause(None)
            try:
                running = self._running_agents(None)
                if running:
                    return RuntimeSnapshotResult(
                        snapshot_id=snapshot_id,
                        status="blocked",
                        blocked_scope_ids=tuple(sorted({agent.scope_id for agent in running})),
                    )
                scope_snapshot_ids = dict(manifest["scope_snapshot_ids"])
                errors: dict[str, BaseException] = {}
                for scope_id, scope_snapshot_id in scope_snapshot_ids.items():
                    try:
                        scope_manifest = self._read_scope_manifest(str(scope_snapshot_id))
                        files_root = self._scope_snapshot_dir(str(scope_snapshot_id)) / "files"
                        if not self._close_touched_provider_homes(files_root):
                            return RuntimeSnapshotResult(
                                snapshot_id=snapshot_id,
                                status="blocked",
                                scope_snapshot_ids=scope_snapshot_ids,
                                blocked_scope_ids=(scope_id,),
                            )
                        scope_key = str(scope_manifest["scope_key"])
                        current_scope_dir = self.runtime_root / "scopes" / scope_key
                        restored_scope_dir = files_root / "scopes" / scope_key
                        if current_scope_dir.exists():
                            shutil.rmtree(current_scope_dir)
                        if restored_scope_dir.exists():
                            shutil.copytree(restored_scope_dir, current_scope_dir)
                        self._restore_home_files(files_root)
                    except BaseException as exc:
                        errors[scope_id] = exc
                self._discard_all_codex_state_databases()
                self.store.rebuild_global_index()
                for scope_id in scope_snapshot_ids:
                    try:
                        self.store.rebuild_scope_index(scope_id)
                    except BaseException:
                        pass
                if errors:
                    return RuntimeSnapshotResult(
                        snapshot_id=snapshot_id,
                        status="failed",
                        scope_snapshot_ids=scope_snapshot_ids,
                        errors=errors,
                    )
                return RuntimeSnapshotResult(
                    snapshot_id=snapshot_id,
                    status="created",
                    scope_snapshot_ids=scope_snapshot_ids,
                    snapshot_relpath=str(self._runtime_snapshot_dir(snapshot_id).relative_to(self.runtime_root)),
                )
            finally:
                self._resume(None)

    def list_scope_snapshots(self, scope_id: str | None = None) -> list[ScopeSnapshotInfo]:
        clauses: list[str] = []
        params: list[str] = []
        if scope_id is not None:
            clauses.append("scope_id=?")
            params.append(scope_id)
        where = f" where {' and '.join(clauses)}" if clauses else ""
        with sqlite3.connect(self.scope_index_path) as conn:
            rows = conn.execute(
                """
                select snapshot_id, scope_id, scope_key, status, snapshot_relpath, created_at
                from scope_snapshots
                """
                + where
                + " order by created_at, snapshot_id",
                params,
            ).fetchall()
        return [ScopeSnapshotInfo(*map(str, row)) for row in rows]

    def get_latest_scope_snapshot(self, scope_id: str) -> ScopeSnapshotInfo | None:
        with sqlite3.connect(self.scope_index_path) as conn:
            row = conn.execute(
                """
                select snapshot_id, scope_id, scope_key, status, snapshot_relpath, created_at
                from scope_snapshots
                where scope_id=?
                order by created_at desc, snapshot_id desc
                limit 1
                """,
                (scope_id,),
            ).fetchone()
        if row is None:
            return None
        return ScopeSnapshotInfo(*map(str, row))

    def list_runtime_snapshots(self) -> list[RuntimeSnapshotInfo]:
        with sqlite3.connect(self.runtime_index_path) as conn:
            rows = conn.execute(
                """
                select snapshot_id, status, snapshot_relpath, created_at, scope_count
                from runtime_snapshots
                order by created_at, snapshot_id
                """
            ).fetchall()
        return [
            RuntimeSnapshotInfo(
                snapshot_id=str(row[0]),
                status=str(row[1]),
                snapshot_relpath=str(row[2]),
                created_at=str(row[3]),
                scope_count=int(row[4]),
            )
            for row in rows
        ]

    def _create_scope_snapshot_unlocked(self, scope_id: str) -> ScopeSnapshotResult:
        snapshot_id = f"ss_{uuid.uuid4().hex}"
        snapshot_dir = self._scope_snapshot_dir(snapshot_id)
        files_root = snapshot_dir / "files"
        scope_key = encode_scope_id(scope_id)
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        current_scope_dir = self.runtime_root / "scopes" / scope_key
        if current_scope_dir.exists():
            shutil.copytree(current_scope_dir, files_root / "scopes" / scope_key)
        for agent in self.store.list_agents(scope_id=scope_id):
            rollout = self.store.locate_rollout(agent.agent_id)
            if rollout is None or not rollout.exists():
                continue
            relpath = rollout.relative_to(self.runtime_root)
            target = files_root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rollout, target)
        created_at = utc_now_iso()
        write_json_atomic(
            snapshot_dir / "snapshot.json",
            {
                "schema_version": 1,
                "object_type": "scope_snapshot",
                "snapshot_id": snapshot_id,
                "scope_id": scope_id,
                "scope_key": scope_key,
                "created_at": created_at,
            },
        )
        snapshot_relpath = str(snapshot_dir.relative_to(self.runtime_root))
        with sqlite3.connect(self.scope_index_path) as conn:
            conn.execute(
                """
                insert into scope_snapshots(
                  snapshot_id, scope_id, scope_key, status, snapshot_relpath, created_at
                )
                values (?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, scope_id, scope_key, "created", snapshot_relpath, created_at),
            )
        return ScopeSnapshotResult(
            snapshot_id=snapshot_id,
            scope_id=scope_id,
            status="created",
            snapshot_relpath=snapshot_relpath,
        )

    def _create_runtime_snapshot_unlocked(
        self,
        scope_snapshot_ids: dict[str, str],
        *,
        status: str,
    ) -> RuntimeSnapshotResult:
        snapshot_id = f"rs_{uuid.uuid4().hex}"
        snapshot_dir = self._runtime_snapshot_dir(snapshot_id)
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        created_at = utc_now_iso()
        write_json_atomic(
            snapshot_dir / "snapshot.json",
            {
                "schema_version": 1,
                "object_type": "runtime_snapshot",
                "snapshot_id": snapshot_id,
                "created_at": created_at,
                "scope_snapshot_ids": scope_snapshot_ids,
            },
        )
        snapshot_relpath = str(snapshot_dir.relative_to(self.runtime_root))
        with sqlite3.connect(self.runtime_index_path) as conn:
            conn.execute(
                """
                insert into runtime_snapshots(
                  snapshot_id, status, snapshot_relpath, created_at, scope_count
                )
                values (?, ?, ?, ?, ?)
                """,
                (snapshot_id, status, snapshot_relpath, created_at, len(scope_snapshot_ids)),
            )
        return RuntimeSnapshotResult(
            snapshot_id=snapshot_id,
            status=status,
            scope_snapshot_ids=scope_snapshot_ids,
            snapshot_relpath=snapshot_relpath,
        )

    def _restore_home_files(self, files_root: Path) -> None:
        homes_root = files_root / "homes"
        if not homes_root.exists():
            return
        shutil.copytree(homes_root, self.runtime_root / "homes", dirs_exist_ok=True)

    def _close_touched_provider_homes(self, files_root: Path) -> bool:
        if self.agent_service is None or not hasattr(self.agent_service, "close_provider_home"):
            return True
        codex_homes_root = files_root / "homes" / "codex"
        if not codex_homes_root.exists():
            return True
        for home_dir in codex_homes_root.iterdir():
            if not home_dir.is_dir():
                continue
            if not self.agent_service.close_provider_home("codex", home_dir.name, force=False):
                return False
        return True

    def _discard_codex_state_databases(self, files_root: Path) -> None:
        codex_homes_root = files_root / "homes" / "codex"
        if not codex_homes_root.exists():
            return
        for home_dir in codex_homes_root.iterdir():
            target_codex_root = self.runtime_root / "homes" / "codex" / home_dir.name / ".codex"
            self._discard_codex_state_databases_in(target_codex_root)

    def _discard_all_codex_state_databases(self) -> None:
        codex_homes_root = self.runtime_root / "homes" / "codex"
        if not codex_homes_root.exists():
            return
        for home_dir in codex_homes_root.iterdir():
            self._discard_codex_state_databases_in(home_dir / ".codex")

    def _discard_codex_state_databases_in(self, codex_root: Path) -> None:
        if not codex_root.exists():
            return
        for path in codex_root.glob("state_5.sqlite*"):
            if path.is_file():
                path.unlink()

    def _read_scope_manifest(self, snapshot_id: str) -> dict[str, Any]:
        return read_json(self._scope_snapshot_dir(snapshot_id) / "snapshot.json")

    def _read_runtime_manifest(self, snapshot_id: str) -> dict[str, Any]:
        return read_json(self._runtime_snapshot_dir(snapshot_id) / "snapshot.json")

    def _scope_snapshot_dir(self, snapshot_id: str) -> Path:
        return self.scope_snapshots_root / snapshot_id

    def _runtime_snapshot_dir(self, snapshot_id: str) -> Path:
        return self.runtime_snapshots_root / snapshot_id

    def _ensure_scope_index(self) -> None:
        with sqlite3.connect(self.scope_index_path) as conn:
            conn.execute(
                """
                create table if not exists scope_snapshots(
                  snapshot_id text primary key,
                  scope_id text not null,
                  scope_key text not null,
                  status text not null,
                  snapshot_relpath text not null,
                  created_at text not null
                )
                """
            )
            conn.execute("create index if not exists idx_scope_snapshots_scope on scope_snapshots(scope_id, created_at)")

    def _ensure_runtime_index(self) -> None:
        with sqlite3.connect(self.runtime_index_path) as conn:
            conn.execute(
                """
                create table if not exists runtime_snapshots(
                  snapshot_id text primary key,
                  status text not null,
                  snapshot_relpath text not null,
                  created_at text not null,
                  scope_count integer not null
                )
                """
            )

    def _pause(self, scope_id: str | None) -> None:
        if self.agent_service is not None and hasattr(self.agent_service, "pause_runs"):
            self.agent_service.pause_runs(scope_id)

    def _resume(self, scope_id: str | None) -> None:
        if self.agent_service is not None and hasattr(self.agent_service, "resume_runs"):
            self.agent_service.resume_runs(scope_id)

    def _running_agents(self, scope_id: str | None) -> list[Any]:
        if self.agent_service is not None and hasattr(self.agent_service, "list_running_agents"):
            return list(self.agent_service.list_running_agents(scope_id))
        return self.store.list_agents(scope_id=scope_id, status="running")

    def _wait_scope(self, scope_id: str, timeout_s: float | None) -> Any:
        if self.agent_service is None or not hasattr(self.agent_service, "wait_scope_agents"):
            raise RuntimeError("agent_service with wait_scope_agents is required")
        return self.agent_service.wait_scope_agents(scope_id, timeout_s=timeout_s)

    def _wait_all(self, timeout_s: float | None) -> Any:
        if self.agent_service is None or not hasattr(self.agent_service, "wait_all_active_agents"):
            raise RuntimeError("agent_service with wait_all_active_agents is required")
        return self.agent_service.wait_all_active_agents(timeout_s=timeout_s)
