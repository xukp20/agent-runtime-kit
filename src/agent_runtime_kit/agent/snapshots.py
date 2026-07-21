from __future__ import annotations

import hashlib
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from time import monotonic
from typing import Any

from agent_runtime_kit.runtime import ARKServices, AppServices, RuntimePauseController

from .models import (
    RuntimeSnapshotInfo,
    RuntimeSnapshotResult,
    ScopeSnapshotInfo,
    ScopeSnapshotResult,
    to_jsonable,
)
from .provider_contracts import (
    AgentArtifactLocator,
    ArtifactCaptureRequest,
    ArtifactDescribeRequest,
    ArtifactRestoreRequest,
    ArtifactStabilityRequest,
    ProviderArtifactEntry,
    ProviderArtifactManifest,
    ProviderSessionLocator,
)
from .report_policy import AgentTraceReportPolicy
from .store import AgentStoreService
from .store_utils import encode_scope_id, read_json, utc_now_iso, write_json_atomic


class AgentSnapshotService:
    def __init__(
        self,
        runtime_root: Path,
        *,
        store: AgentStoreService,
        agent_service: object | None = None,
        ark_services: ARKServices | None = None,
        app_services: AppServices | None = None,
        trace_report_policy: AgentTraceReportPolicy | None = None,
    ) -> None:
        self.runtime_root = Path(runtime_root)
        self.store = store
        inferred_ark = getattr(agent_service, "ark_services", None)
        self.ark = ark_services or inferred_ark or ARKServices()
        self.app = app_services or getattr(agent_service, "app_services", None) or AppServices()
        self.agent_service = agent_service or self.ark.agent_service
        inferred_report_policy = getattr(self.agent_service, "trace_report_policy", None)
        self.trace_report_policy = trace_report_policy or inferred_report_policy or AgentTraceReportPolicy()
        if self.agent_service is not None and self.ark.agent_service is None:
            self.ark.agent_service = self.agent_service
        if self.ark.pause_controller is None:
            self.ark.pause_controller = RuntimePauseController()
        self.ark.snapshot_service = self
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
            was_paused = self._is_scope_directly_paused(scope_id)
            deadline = self._deadline(timeout_s)
            self._pause(scope_id)
            try:
                running = self._running_agents(scope_id)
                running_steps = self._running_steps(scope_id)
                if running and not wait:
                    return ScopeSnapshotResult(
                        snapshot_id=None,
                        scope_id=scope_id,
                        status="blocked",
                        running_agent_ids=tuple(agent.agent_id for agent in running),
                        running_step_ids=tuple(running_steps),
                    )
                if running_steps and not wait:
                    return ScopeSnapshotResult(
                        snapshot_id=None,
                        scope_id=scope_id,
                        status="blocked",
                        running_agent_ids=tuple(agent.agent_id for agent in running),
                        running_step_ids=tuple(running_steps),
                    )
                if running:
                    result = self._wait_scope(scope_id, timeout_s=self._remaining(deadline))
                    if not getattr(result, "clean", False):
                        return ScopeSnapshotResult(
                            snapshot_id=None,
                            scope_id=scope_id,
                            status="blocked",
                            running_agent_ids=tuple(result.pending),
                            running_step_ids=tuple(self._running_steps(scope_id)),
                            errors=dict(result.errors),
                        )
                if running_steps:
                    pending_steps = self._wait_steps(running_steps, timeout_s=self._remaining(deadline))
                    if pending_steps:
                        return ScopeSnapshotResult(
                            snapshot_id=None,
                            scope_id=scope_id,
                            status="blocked",
                            running_step_ids=tuple(pending_steps),
                        )
                restorable_error = self._flow_restorable_error(scope_id)
                if restorable_error is not None:
                    return ScopeSnapshotResult(
                        snapshot_id=None,
                        scope_id=scope_id,
                        status="blocked",
                        errors={"flow_restorable": restorable_error},
                    )
                return self._create_scope_snapshot_unlocked(scope_id)
            finally:
                if not was_paused:
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
            was_paused = self._is_paused(None)
            deadline = self._deadline(timeout_s)
            self._pause(None)
            try:
                running = self._running_agents(None)
                running_steps = self._running_steps(None)
                if running or running_steps:
                    result = self._wait_all(timeout_s=self._remaining(deadline)) if running else None
                    pending_steps = self._wait_steps(running_steps, timeout_s=self._remaining(deadline)) if running_steps else []
                    if (result is not None and not getattr(result, "clean", False)) or pending_steps:
                        pending_agent_ids = list(getattr(result, "pending", ())) if result is not None else []
                        error_agent_ids = list(getattr(result, "errors", {}).keys()) if result is not None else []
                        blocked_scope_ids = tuple(
                            sorted(self._blocked_scope_ids_from_agent_ids(pending_agent_ids + error_agent_ids, pending_steps))
                        )
                        return RuntimeSnapshotResult(
                            snapshot_id=None,
                            status="blocked",
                            blocked_scope_ids=blocked_scope_ids,
                            running_step_ids=tuple(pending_steps),
                            errors=dict(getattr(result, "errors", {})),
                        )
                restorable_error = self._flow_restorable_error(None)
                if restorable_error is not None:
                    return RuntimeSnapshotResult(
                        snapshot_id=None,
                        status="blocked",
                        errors={"flow_restorable": restorable_error},
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
                if not was_paused:
                    self._resume(None)

    def create_runtime_snapshot_for_scopes(
        self,
        *,
        refresh_scope_ids: list[str],
        reuse_latest_for_other_scopes: bool = True,
        scope_ids: list[str] | None = None,
        wait: bool = False,
        timeout_s: float | None = None,
    ) -> RuntimeSnapshotResult:
        with self._lock:
            refresh_scope_ids = self._dedupe_scope_ids(refresh_scope_ids)
            if not refresh_scope_ids:
                return RuntimeSnapshotResult(
                    snapshot_id=None,
                    status="failed",
                    errors={"refresh_scope_ids": ValueError("refresh_scope_ids must not be empty")},
                )

            selected_scope_ids = self._dedupe_scope_ids(scope_ids if scope_ids is not None else self.store.list_scope_ids())
            selected_scope_set = set(selected_scope_ids)
            refresh_scope_set = set(refresh_scope_ids)
            unknown_refresh_scope_ids = sorted(refresh_scope_set - selected_scope_set)
            if unknown_refresh_scope_ids:
                return RuntimeSnapshotResult(
                    snapshot_id=None,
                    status="failed",
                    errors={
                        "refresh_scope_ids": ValueError(
                            "refresh_scope_ids must be contained in scope_ids: "
                            + ", ".join(unknown_refresh_scope_ids)
                        )
                    },
                )

            latest_scope_snapshot_ids: dict[str, str] = {}
            if reuse_latest_for_other_scopes:
                missing_latest_scope_ids: list[str] = []
                for selected_scope_id in selected_scope_ids:
                    if selected_scope_id in refresh_scope_set:
                        continue
                    latest = self.get_latest_scope_snapshot(selected_scope_id)
                    if latest is None:
                        missing_latest_scope_ids.append(selected_scope_id)
                    else:
                        latest_scope_snapshot_ids[selected_scope_id] = latest.snapshot_id
                if missing_latest_scope_ids:
                    return RuntimeSnapshotResult(
                        snapshot_id=None,
                        status="failed",
                        errors={
                            "latest_scope_snapshot": ValueError(
                                "missing latest scope snapshot for: " + ", ".join(missing_latest_scope_ids)
                            )
                        },
                    )

            was_directly_paused = {
                refresh_scope_id: self._is_scope_directly_paused(refresh_scope_id)
                for refresh_scope_id in refresh_scope_ids
            }
            deadline = self._deadline(timeout_s)
            for refresh_scope_id in refresh_scope_ids:
                self._pause(refresh_scope_id)
            try:
                running = self._running_agents_for_scopes(refresh_scope_ids)
                running_steps = self._running_steps_for_scopes(refresh_scope_ids)
                if (running or running_steps) and not wait:
                    return RuntimeSnapshotResult(
                        snapshot_id=None,
                        status="blocked",
                        blocked_scope_ids=tuple(sorted(self._blocked_scope_ids(running, running_steps))),
                        running_step_ids=tuple(running_steps),
                    )

                if running:
                    result = self._wait_selected_agents(
                        [str(agent.agent_id) for agent in running],
                        timeout_s=self._remaining(deadline),
                    )
                    if not getattr(result, "clean", False):
                        pending_agent_ids = list(getattr(result, "pending", ()))
                        error_agent_ids = list(getattr(result, "errors", {}).keys())
                        running_steps_after_agents = self._running_steps_for_scopes(refresh_scope_ids)
                        return RuntimeSnapshotResult(
                            snapshot_id=None,
                            status="blocked",
                            blocked_scope_ids=tuple(
                                sorted(
                                    self._blocked_scope_ids_from_agent_ids(
                                        pending_agent_ids + error_agent_ids,
                                        running_steps_after_agents,
                                    )
                                )
                            ),
                            running_step_ids=tuple(running_steps_after_agents),
                            errors=dict(getattr(result, "errors", {})),
                        )

                if running_steps:
                    pending_steps = self._wait_steps(running_steps, timeout_s=self._remaining(deadline))
                    pending_steps = [step_id for step_id in pending_steps if step_id in set(self._running_steps_for_scopes(refresh_scope_ids))]
                    if pending_steps:
                        return RuntimeSnapshotResult(
                            snapshot_id=None,
                            status="blocked",
                            blocked_scope_ids=tuple(sorted(self._blocked_scope_ids([], pending_steps))),
                            running_step_ids=tuple(pending_steps),
                        )

                restorable_errors: dict[str, BaseException] = {}
                for refresh_scope_id in refresh_scope_ids:
                    restorable_error = self._flow_restorable_error(refresh_scope_id)
                    if restorable_error is not None:
                        restorable_errors[refresh_scope_id] = restorable_error
                if restorable_errors:
                    return RuntimeSnapshotResult(
                        snapshot_id=None,
                        status="blocked",
                        blocked_scope_ids=tuple(sorted(restorable_errors)),
                        errors=restorable_errors,
                    )

                refreshed_scope_snapshot_ids: dict[str, str] = {}
                errors: dict[str, BaseException] = {}
                for refresh_scope_id in refresh_scope_ids:
                    try:
                        scope_result = self._create_scope_snapshot_unlocked(refresh_scope_id)
                        if scope_result.status != "created" or scope_result.snapshot_id is None:
                            errors[refresh_scope_id] = RuntimeError(
                                f"failed to create scope snapshot for {refresh_scope_id}: {scope_result.status}"
                            )
                        else:
                            refreshed_scope_snapshot_ids[refresh_scope_id] = scope_result.snapshot_id
                    except BaseException as exc:
                        errors[refresh_scope_id] = exc
                if errors:
                    return RuntimeSnapshotResult(
                        snapshot_id=None,
                        status="failed",
                        scope_snapshot_ids=refreshed_scope_snapshot_ids,
                        errors=errors,
                    )

                if reuse_latest_for_other_scopes:
                    scope_snapshot_ids = {
                        selected_scope_id: (
                            refreshed_scope_snapshot_ids[selected_scope_id]
                            if selected_scope_id in refreshed_scope_snapshot_ids
                            else latest_scope_snapshot_ids[selected_scope_id]
                        )
                        for selected_scope_id in selected_scope_ids
                    }
                else:
                    scope_snapshot_ids = {
                        selected_scope_id: refreshed_scope_snapshot_ids[selected_scope_id]
                        for selected_scope_id in selected_scope_ids
                        if selected_scope_id in refreshed_scope_snapshot_ids
                    }
                return self._create_runtime_snapshot_unlocked(scope_snapshot_ids, status="created")
            finally:
                for refresh_scope_id in refresh_scope_ids:
                    if not was_directly_paused[refresh_scope_id]:
                        self._resume(refresh_scope_id)

    def restore_scope_snapshot(self, snapshot_id: str, *, leave_paused: bool = True) -> ScopeSnapshotResult:
        with self._lock:
            manifest = self._read_scope_manifest(snapshot_id)
            scope_id = str(manifest["scope_id"])
            try:
                self._preflight_scope_snapshot(snapshot_id, manifest=manifest)
            except BaseException as exc:
                return ScopeSnapshotResult(
                    snapshot_id=snapshot_id,
                    scope_id=scope_id,
                    status="failed",
                    errors={"snapshot_archive": exc},
                )
            was_paused = self._is_scope_directly_paused(scope_id)
            self._pause(scope_id)
            try:
                running = self._running_agents(scope_id)
                running_steps = self._running_steps(scope_id)
                if running or running_steps:
                    return ScopeSnapshotResult(
                        snapshot_id=snapshot_id,
                        scope_id=scope_id,
                        status="blocked",
                        running_agent_ids=tuple(agent.agent_id for agent in running),
                        running_step_ids=tuple(running_steps),
                    )
                files_root = self._scope_snapshot_dir(snapshot_id) / "files"
                scope_key = str(manifest["scope_key"])
                restored_scope_dir = files_root / "scopes" / scope_key
                current_scope_dir = self.runtime_root / "scopes" / scope_key
                old_agents = list(self.store.list_agents(scope_id=scope_id))
                if current_scope_dir.exists():
                    shutil.rmtree(current_scope_dir)
                if restored_scope_dir.exists():
                    shutil.copytree(restored_scope_dir, current_scope_dir)
                self._remove_report_files_for_agents(old_agents)
                self._prepare_provider_artifacts(old_agents)
                self._restore_report_files(files_root)
                self._restore_provider_artifacts_or_legacy(manifest, files_root)
                self.store.rebuild_scope_index(scope_id)
                self.store.rebuild_global_index()
                self._rebuild_flow_indexes(scope_id)
                restorable_error = self._flow_restorable_error(scope_id)
                if restorable_error is not None:
                    return ScopeSnapshotResult(
                        snapshot_id=snapshot_id,
                        scope_id=scope_id,
                        status="failed",
                        errors={"flow_restorable": restorable_error},
                    )
                self._rebuild_scheduler_queues()
                return ScopeSnapshotResult(
                    snapshot_id=snapshot_id,
                    scope_id=scope_id,
                    status="created",
                    snapshot_relpath=str(self._scope_snapshot_dir(snapshot_id).relative_to(self.runtime_root)),
                )
            finally:
                if not leave_paused and not was_paused:
                    self._resume(scope_id)

    def restore_runtime_snapshot(
        self,
        snapshot_id: str,
        *,
        leave_paused: bool = True,
        prune_extra_scopes: bool = False,
    ) -> RuntimeSnapshotResult:
        with self._lock:
            manifest = self._read_runtime_manifest(snapshot_id)
            scope_snapshot_ids = dict(manifest["scope_snapshot_ids"])
            preflight_errors: dict[str, BaseException] = {}
            for scope_id, scope_snapshot_id in scope_snapshot_ids.items():
                try:
                    scope_manifest = self._read_scope_manifest(str(scope_snapshot_id))
                    if str(scope_manifest.get("scope_id")) != str(scope_id):
                        raise RuntimeError(
                            f"runtime snapshot scope {scope_id} points to a snapshot for "
                            f"{scope_manifest.get('scope_id')}"
                        )
                    self._preflight_scope_snapshot(str(scope_snapshot_id), manifest=scope_manifest)
                except BaseException as exc:
                    preflight_errors[str(scope_id)] = exc
            if preflight_errors:
                return RuntimeSnapshotResult(
                    snapshot_id=snapshot_id,
                    status="failed",
                    scope_snapshot_ids=scope_snapshot_ids,
                    errors=preflight_errors,
                )
            was_paused = self._is_paused(None)
            self._pause(None)
            try:
                running = self._running_agents(None)
                running_steps = self._running_steps(None)
                if running or running_steps:
                    return RuntimeSnapshotResult(
                        snapshot_id=snapshot_id,
                        status="blocked",
                        blocked_scope_ids=tuple(sorted(self._blocked_scope_ids(running, running_steps))),
                        running_step_ids=tuple(running_steps),
                    )
                errors: dict[str, BaseException] = {}
                pruned_scope_ids: list[str] = []
                if prune_extra_scopes:
                    extra_scope_ids = sorted(set(self.store.list_scope_ids()) - set(scope_snapshot_ids))
                    for scope_id in extra_scope_ids:
                        try:
                            old_agents = list(self.store.list_agents(scope_id=scope_id))
                            self._remove_report_files_for_agents(old_agents)
                            self._prepare_provider_artifacts(old_agents)
                            scope_dir = self.runtime_root / "scopes" / encode_scope_id(scope_id)
                            if scope_dir.exists():
                                shutil.rmtree(scope_dir)
                            pruned_scope_ids.append(scope_id)
                        except BaseException as exc:
                            errors[f"prune:{scope_id}"] = exc
                for scope_id, scope_snapshot_id in scope_snapshot_ids.items():
                    try:
                        scope_manifest = self._read_scope_manifest(str(scope_snapshot_id))
                        files_root = self._scope_snapshot_dir(str(scope_snapshot_id)) / "files"
                        scope_key = str(scope_manifest["scope_key"])
                        current_scope_dir = self.runtime_root / "scopes" / scope_key
                        restored_scope_dir = files_root / "scopes" / scope_key
                        old_agents = list(self.store.list_agents(scope_id=scope_id))
                        if current_scope_dir.exists():
                            shutil.rmtree(current_scope_dir)
                        if restored_scope_dir.exists():
                            shutil.copytree(restored_scope_dir, current_scope_dir)
                        self._remove_report_files_for_agents(old_agents)
                        self._prepare_provider_artifacts(old_agents)
                        self._restore_report_files(files_root)
                        self._restore_provider_artifacts_or_legacy(scope_manifest, files_root)
                    except BaseException as exc:
                        errors[scope_id] = exc
                self.store.rebuild_global_index()
                for scope_id in scope_snapshot_ids:
                    try:
                        self.store.rebuild_scope_index(scope_id)
                    except BaseException:
                        pass
                self._rebuild_flow_indexes(None)
                restorable_error = self._flow_restorable_error(None)
                if restorable_error is not None:
                    errors["flow_restorable"] = restorable_error
                self._rebuild_scheduler_queues()
                if errors:
                    return RuntimeSnapshotResult(
                        snapshot_id=snapshot_id,
                        status="failed",
                        scope_snapshot_ids=scope_snapshot_ids,
                        errors=errors,
                        pruned_scope_ids=tuple(pruned_scope_ids),
                    )
                return RuntimeSnapshotResult(
                    snapshot_id=snapshot_id,
                    status="created",
                    scope_snapshot_ids=scope_snapshot_ids,
                    snapshot_relpath=str(self._runtime_snapshot_dir(snapshot_id).relative_to(self.runtime_root)),
                    pruned_scope_ids=tuple(pruned_scope_ids),
                )
            finally:
                if not leave_paused and not was_paused:
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
        provider_artifacts: list[dict[str, object]] = []
        for agent in self.store.list_agents(scope_id=scope_id):
            artifact_manifest = self._capture_provider_artifacts(agent, files_root)
            if artifact_manifest is not None:
                provider_artifacts.append(
                    {
                        "agent_id": str(agent.agent_id),
                        "manifest": to_jsonable(artifact_manifest),
                    }
                )
            self._copy_report_files(agent.agent_id, files_root)
        created_at = utc_now_iso()
        files = self._snapshot_file_entries(files_root)
        write_json_atomic(
            snapshot_dir / "snapshot.json",
            {
                "schema_version": 3,
                "object_type": "scope_snapshot",
                "snapshot_id": snapshot_id,
                "scope_id": scope_id,
                "scope_key": scope_key,
                "created_at": created_at,
                "files": files,
                "provider_artifacts": provider_artifacts,
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

    def _copy_report_files(self, agent_id: str, files_root: Path) -> None:
        if not self.trace_report_policy.include_in_snapshots:
            return
        source = self.store.report_dir(agent_id)
        if not source.exists():
            return
        target = files_root / source.relative_to(self.runtime_root)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)

    def _restore_report_files(self, files_root: Path) -> None:
        reports_root = files_root / "reports"
        if not reports_root.exists():
            return
        shutil.copytree(reports_root, self.runtime_root / "reports", dirs_exist_ok=True)

    def _remove_report_files_for_agents(self, agents: list[Any]) -> None:
        for agent in agents:
            report_dir = self.store.report_dir(str(agent.agent_id))
            if report_dir.exists():
                shutil.rmtree(report_dir)

    def _capture_provider_artifacts(
        self,
        agent: Any,
        files_root: Path,
    ) -> ProviderArtifactManifest | None:
        session = self._provider_session(agent)
        bundle = self._provider_bundle(str(getattr(agent, "cli_type", "")))
        adapter = bundle.artifacts if bundle is not None else None
        if session is not None and adapter is not None:
            stability = adapter.wait_quiescent(
                ArtifactStabilityRequest(session=session, agent_id=str(agent.agent_id))
            )
            if not stability.stable:
                raise RuntimeError(
                    f"provider artifacts are not stable for {agent.agent_id}: "
                    f"{stability.reason or 'unknown'}"
                )
            captured = adapter.capture(
                ArtifactCaptureRequest(
                    session=session,
                    snapshot_root=str(files_root),
                    agent_id=str(agent.agent_id),
                )
            )
            if not captured.manifest.stable:
                raise RuntimeError(f"provider returned an unstable artifact manifest: {agent.agent_id}")
            return captured.manifest

        # COMPAT(legacy-provider-artifact-copy): injected providers without a
        # Provider bundle still use AgentStore.locate_rollout(). Remove after
        # external provider takeover tests register an Artifact adapter.
        rollout = self.store.locate_rollout(str(agent.agent_id))
        if rollout is not None and rollout.exists():
            relpath = rollout.relative_to(self.runtime_root)
            target = files_root / relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rollout, target)
        return None

    def _prepare_provider_artifacts(self, agents: list[Any]) -> None:
        for agent in agents:
            session = self._provider_session(agent)
            bundle = self._provider_bundle(str(getattr(agent, "cli_type", "")))
            adapter = bundle.artifacts if bundle is not None else None
            if session is not None and adapter is not None:
                manifest = adapter.describe(
                    ArtifactDescribeRequest(session=session, agent_id=str(agent.agent_id))
                )
                adapter.prepare_restore(
                    ArtifactRestoreRequest(
                        manifest=manifest,
                        snapshot_root=str(self.runtime_root),
                    )
                )
                continue
            # COMPAT(legacy-provider-artifact-cleanup): no Provider bundle is
            # available, so remove only the path resolved by AgentStore.
            rollout = self.store.locate_rollout(str(agent.agent_id))
            if rollout is not None and rollout.is_file():
                rollout.unlink()

    def _restore_provider_artifacts_or_legacy(
        self,
        snapshot_manifest: dict[str, Any],
        files_root: Path,
    ) -> None:
        artifact_records = snapshot_manifest.get("provider_artifacts")
        if not artifact_records:
            # COMPAT(legacy-snapshot-provider-files): schema v1/v2 and injected
            # provider snapshots stored files under homes/ without a Provider
            # Artifact Manifest. Remove after the supported migration window.
            self._restore_home_files(files_root)
            return
        if not isinstance(artifact_records, list):
            raise RuntimeError("scope snapshot has an invalid provider_artifacts manifest")
        for record in artifact_records:
            if not isinstance(record, dict) or not isinstance(record.get("manifest"), dict):
                raise RuntimeError("scope snapshot has an invalid provider artifact record")
            manifest = _provider_artifact_manifest_from_dict(record["manifest"])
            bundle = self._provider_bundle(manifest.provider_type)
            if bundle is None or bundle.artifacts is None:
                raise RuntimeError(
                    f"provider Artifact adapter is unavailable during restore: {manifest.provider_type}"
                )
            request = ArtifactRestoreRequest(
                manifest=manifest,
                snapshot_root=str(files_root),
            )
            bundle.artifacts.prepare_restore(request)
            result = bundle.artifacts.restore(request)
            if not result.restored:
                raise RuntimeError(
                    f"provider did not restore artifacts: {manifest.provider_type}/{manifest.session_id}"
                )
            bundle.artifacts.rebuild_after_restore(request)

    def _provider_bundle(self, provider_type: str):  # noqa: ANN202
        getter = getattr(self.agent_service, "get_provider_bundle", None)
        return getter(provider_type) if callable(getter) else None

    def _provider_session(self, agent: Any) -> ProviderSessionLocator | None:
        session_id = getattr(agent, "thread_id", None)
        if not session_id:
            return None
        return ProviderSessionLocator(
            provider_type=str(getattr(agent, "cli_type", "")),
            session_id=str(session_id),
            home_id=str(getattr(agent, "home_id", "")),
            created_at=str(getattr(agent, "created_at", "") or utc_now_iso()),
            native_locator={"rollout_relpath": getattr(agent, "rollout_relpath", None)},
        )

    def _read_scope_manifest(self, snapshot_id: str) -> dict[str, Any]:
        return read_json(self._scope_snapshot_dir(snapshot_id) / "snapshot.json")

    def _read_runtime_manifest(self, snapshot_id: str) -> dict[str, Any]:
        return read_json(self._runtime_snapshot_dir(snapshot_id) / "snapshot.json")

    def _snapshot_file_entries(self, files_root: Path) -> list[dict[str, object]]:
        if not files_root.exists():
            return []
        entries: list[dict[str, object]] = []
        for path in sorted(files_root.rglob("*")):
            if not path.is_file():
                continue
            entries.append(
                {
                    "relpath": path.relative_to(files_root).as_posix(),
                    "size": path.stat().st_size,
                    "sha256": self._sha256_file(path),
                }
            )
        return entries

    def _preflight_scope_snapshot(self, snapshot_id: str, *, manifest: dict[str, Any]) -> None:
        entries = manifest.get("files")
        if entries is None:
            return
        if not isinstance(entries, list):
            raise RuntimeError(f"scope snapshot {snapshot_id} has an invalid files manifest")
        files_root = self._scope_snapshot_dir(snapshot_id) / "files"
        for index, raw_entry in enumerate(entries):
            if not isinstance(raw_entry, dict):
                raise RuntimeError(f"scope snapshot {snapshot_id} file entry {index} is invalid")
            relpath = str(raw_entry.get("relpath") or "")
            candidate = Path(relpath)
            if not relpath or candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
                raise RuntimeError(f"scope snapshot {snapshot_id} file entry {index} has an unsafe path")
            path = files_root / candidate
            if not path.is_file():
                raise RuntimeError(f"scope snapshot {snapshot_id} is missing {relpath}")
            expected_size = int(raw_entry.get("size", -1))
            if path.stat().st_size != expected_size:
                raise RuntimeError(f"scope snapshot {snapshot_id} size mismatch for {relpath}")
            expected_sha256 = str(raw_entry.get("sha256") or "")
            if not expected_sha256 or self._sha256_file(path) != expected_sha256:
                raise RuntimeError(f"scope snapshot {snapshot_id} checksum mismatch for {relpath}")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
        pause_controller = self.ark.pause_controller
        if isinstance(pause_controller, RuntimePauseController):
            pause_controller.pause(scope_id)
            return
        if self.agent_service is not None and hasattr(self.agent_service, "pause_runs"):
            self.agent_service.pause_runs(scope_id)

    def _resume(self, scope_id: str | None) -> None:
        pause_controller = self.ark.pause_controller
        if isinstance(pause_controller, RuntimePauseController):
            pause_controller.resume(scope_id)
            return
        if self.agent_service is not None and hasattr(self.agent_service, "resume_runs"):
            self.agent_service.resume_runs(scope_id)

    def _is_paused(self, scope_id: str | None) -> bool:
        pause_controller = self.ark.pause_controller
        if pause_controller is not None and hasattr(pause_controller, "is_paused"):
            return bool(pause_controller.is_paused(scope_id))
        if self.agent_service is not None and hasattr(self.agent_service, "is_paused"):
            return bool(self.agent_service.is_paused(scope_id))
        return False

    def _is_scope_directly_paused(self, scope_id: str) -> bool:
        pause_controller = self.ark.pause_controller
        if pause_controller is not None and hasattr(pause_controller, "is_scope_directly_paused"):
            return bool(pause_controller.is_scope_directly_paused(scope_id))
        return self._is_paused(scope_id)

    def _running_agents(self, scope_id: str | None) -> list[Any]:
        if self.agent_service is not None and hasattr(self.agent_service, "list_running_agents"):
            return list(self.agent_service.list_running_agents(scope_id))
        return self.store.list_agents(scope_id=scope_id, status="running")

    def _running_steps(self, scope_id: str | None) -> list[str]:
        step_service = self.ark.step_service
        if step_service is None or not hasattr(step_service, "list_running_steps"):
            return []
        return list(step_service.list_running_steps(scope_id))

    def _wait_steps(self, step_ids: list[str], *, timeout_s: float | None) -> list[str]:
        step_service = self.ark.step_service
        if step_service is None or not hasattr(step_service, "wait_step"):
            return list(step_ids)
        pending: list[str] = []
        deadline = self._deadline(timeout_s)
        for index, step_id in enumerate(step_ids):
            remaining = self._remaining(deadline)
            if remaining is not None and remaining <= 0:
                pending.extend(step_ids[index:])
                break
            try:
                step_service.wait_step(step_id, timeout_s=remaining)
            except TimeoutError:
                pending.append(step_id)
        still_running = set(self._running_steps(None))
        return pending + [step_id for step_id in step_ids if step_id in still_running and step_id not in set(pending)]

    def _blocked_scope_ids(self, running_agents: list[Any], running_step_ids: list[str]) -> set[str]:
        scope_ids = {str(agent.scope_id) for agent in running_agents}
        step_service = self.ark.step_service
        if step_service is not None and hasattr(step_service, "store"):
            for step_id in running_step_ids:
                try:
                    scope_ids.add(str(step_service.store.get_step(step_id).scope_id))
                except BaseException:
                    pass
        return scope_ids

    def _blocked_scope_ids_from_agent_ids(self, agent_ids: list[str], running_step_ids: list[str]) -> set[str]:
        scope_ids: set[str] = set()
        for agent_id in agent_ids:
            try:
                scope_ids.add(str(self.store.get_agent(agent_id).scope_id))
            except BaseException:
                pass
        return scope_ids | self._blocked_scope_ids([], running_step_ids)

    def _dedupe_scope_ids(self, scope_ids: list[str] | tuple[str, ...]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for scope_id in scope_ids:
            scope_key = str(scope_id)
            if scope_key in seen:
                continue
            seen.add(scope_key)
            deduped.append(scope_key)
        return deduped

    def _running_agents_for_scopes(self, scope_ids: list[str]) -> list[Any]:
        running: list[Any] = []
        seen: set[str] = set()
        for scope_id in scope_ids:
            for agent in self._running_agents(scope_id):
                agent_id = str(agent.agent_id)
                if agent_id in seen:
                    continue
                seen.add(agent_id)
                running.append(agent)
        return running

    def _running_steps_for_scopes(self, scope_ids: list[str]) -> list[str]:
        running: list[str] = []
        seen: set[str] = set()
        for scope_id in scope_ids:
            for step_id in self._running_steps(scope_id):
                if step_id in seen:
                    continue
                seen.add(step_id)
                running.append(step_id)
        return running

    def _wait_selected_agents(self, agent_ids: list[str], timeout_s: float | None) -> Any:
        if self.agent_service is None or not hasattr(self.agent_service, "wait_agents"):
            raise RuntimeError("agent_service with wait_agents is required")
        return self.agent_service.wait_agents(agent_ids, timeout_s=timeout_s)

    def _deadline(self, timeout_s: float | None) -> float | None:
        return None if timeout_s is None else monotonic() + timeout_s

    def _remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, deadline - monotonic())

    def _flow_restorable_error(self, scope_id: str | None) -> BaseException | None:
        flow_service = self.ark.flow_service
        if flow_service is None or not hasattr(flow_service, "assert_restorable_flows"):
            return None
        try:
            flow_service.assert_restorable_flows(scope_id=scope_id)
        except BaseException as exc:
            return exc
        return None

    def _rebuild_flow_indexes(self, scope_id: str | None) -> None:
        flow_service = self.ark.flow_service
        flow_store = getattr(flow_service, "store", None)
        if flow_store is None:
            return
        if scope_id is not None and hasattr(flow_store, "rebuild_scope_index"):
            flow_store.rebuild_scope_index(scope_id)
        if scope_id is None and hasattr(flow_store, "rebuild_all_indexes"):
            flow_store.rebuild_all_indexes()
        elif hasattr(flow_store, "rebuild_global_index"):
            flow_store.rebuild_global_index()

    def _rebuild_scheduler_queues(self) -> None:
        schedule_service = self.ark.schedule_service
        if schedule_service is not None and hasattr(schedule_service, "rebuild_candidate_queues"):
            schedule_service.rebuild_candidate_queues()

    def _wait_scope(self, scope_id: str, timeout_s: float | None) -> Any:
        if self.agent_service is None or not hasattr(self.agent_service, "wait_scope_agents"):
            raise RuntimeError("agent_service with wait_scope_agents is required")
        return self.agent_service.wait_scope_agents(scope_id, timeout_s=timeout_s)

    def _wait_all(self, timeout_s: float | None) -> Any:
        if self.agent_service is None or not hasattr(self.agent_service, "wait_all_active_agents"):
            raise RuntimeError("agent_service with wait_all_active_agents is required")
        return self.agent_service.wait_all_active_agents(timeout_s=timeout_s)


def _provider_artifact_manifest_from_dict(payload: dict[str, Any]) -> ProviderArtifactManifest:
    locator_payload = payload.get("locator")
    locator = None
    if isinstance(locator_payload, dict):
        locator = AgentArtifactLocator(
            provider_type=str(locator_payload["provider_type"]),
            home_id=str(locator_payload["home_id"]),
            session_id=str(locator_payload["session_id"]),
            adapter_version=str(locator_payload["adapter_version"]),
            manifest_relpath=(
                str(locator_payload["manifest_relpath"])
                if locator_payload.get("manifest_relpath") is not None
                else None
            ),
            native_primary_ref=(
                str(locator_payload["native_primary_ref"])
                if locator_payload.get("native_primary_ref") is not None
                else None
            ),
        )
    entries_payload = payload.get("entries") or []
    if not isinstance(entries_payload, list):
        raise RuntimeError("provider artifact entries must be a list")
    entries = tuple(
        ProviderArtifactEntry(
            artifact_id=str(item["artifact_id"]),
            kind=str(item["kind"]),
            authority=str(item["authority"]),
            capture_strategy=str(item["capture_strategy"]),
            native_ref=str(item["native_ref"]) if item.get("native_ref") is not None else None,
            snapshot_relpath=(
                str(item["snapshot_relpath"])
                if item.get("snapshot_relpath") is not None
                else None
            ),
            sha256=str(item["sha256"]) if item.get("sha256") is not None else None,
            size_bytes=int(item["size_bytes"]) if item.get("size_bytes") is not None else None,
            required_for_resume=bool(item.get("required_for_resume", False)),
        )
        for item in entries_payload
        if isinstance(item, dict)
    )
    if len(entries) != len(entries_payload):
        raise RuntimeError("provider artifact entry must be a mapping")
    return ProviderArtifactManifest(
        provider_type=str(payload["provider_type"]),
        home_id=str(payload["home_id"]),
        session_id=str(payload["session_id"]),
        adapter_version=str(payload["adapter_version"]),
        stable=bool(payload["stable"]),
        entries=entries,
        locator=locator,
        warnings=tuple(str(item) for item in payload.get("warnings") or ()),
    )
