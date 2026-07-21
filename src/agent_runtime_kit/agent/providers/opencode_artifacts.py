from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import time
from pathlib import Path

from ..provider_contracts import (
    AgentArtifactLocator,
    ArtifactCaptureRequest,
    ArtifactDescribeRequest,
    ArtifactRestoreRequest,
    ArtifactStabilityRequest,
    ArtifactStabilityResult,
    ProviderArtifactEntry,
    ProviderArtifactManifest,
    ProviderArtifactRestoreResult,
    ProviderArtifactSnapshot,
)
from ..store_utils import utc_now_iso
from .opencode_models import ADAPTER_VERSION, PROVIDER_TYPE, parse_native_locator
from .opencode_runtime import OpenCodeRuntimeRegistry


class OpenCodeArtifactAdapter:
    provider_type = PROVIDER_TYPE

    def __init__(self, *, runtime_root: Path, registry: OpenCodeRuntimeRegistry) -> None:
        self.runtime_root = Path(runtime_root)
        self.registry = registry

    def wait_quiescent(self, request: ArtifactStabilityRequest) -> ArtifactStabilityResult:
        deadline = time.monotonic() + (request.timeout_s or 30)
        while time.monotonic() < deadline:
            try:
                statuses = self.registry.client_for_locator(request.session).session_status()
            except RuntimeError:
                # A stopped process has no in-memory permission/question state and its DB is stable.
                return ArtifactStabilityResult(stable=True, observed_at=utc_now_iso())
            status = statuses.get(request.session.session_id)
            if status is None:
                return ArtifactStabilityResult(stable=True, observed_at=utc_now_iso())
            if isinstance(status, dict) and str(status.get("type") or status.get("status")) == "idle":
                return ArtifactStabilityResult(stable=True, observed_at=utc_now_iso())
            time.sleep(0.1)
        return ArtifactStabilityResult(
            stable=False,
            observed_at=utc_now_iso(),
            reason="OpenCode session did not reach idle; busy/retry/interaction state is not snapshot-safe",
        )

    def describe(self, request: ArtifactDescribeRequest) -> ProviderArtifactManifest:
        native = parse_native_locator(request.session.native_locator)
        database = Path(native.database_path)
        entries: list[ProviderArtifactEntry] = [
            ProviderArtifactEntry(
                artifact_id="opencode-database",
                kind="provider_database",
                authority="provider_native",
                capture_strategy="sqlite_backup",
                native_ref=_native_ref(self.runtime_root, database),
                snapshot_relpath="opencode.db",
                size_bytes=database.stat().st_size if database.is_file() else None,
                required_for_resume=True,
            )
        ]
        runtime = self.runtime_root / native.runtime_relpath
        for name, path in _additional_paths(runtime):
            if path.exists():
                entries.append(
                    ProviderArtifactEntry(
                        artifact_id=f"opencode-{name}",
                        kind=name,
                        authority="provider_native",
                        capture_strategy="copy_tree",
                        native_ref=_native_ref(self.runtime_root, path),
                        snapshot_relpath=name,
                        required_for_resume=name == "tool-output",
                    )
                )
        locator = AgentArtifactLocator(
            provider_type=PROVIDER_TYPE,
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=ADAPTER_VERSION,
            native_primary_ref=_native_ref(self.runtime_root, database),
        )
        return ProviderArtifactManifest(
            provider_type=PROVIDER_TYPE,
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=ADAPTER_VERSION,
            stable=database.is_file(),
            entries=tuple(entries),
            locator=locator,
            warnings=("workspace files are intentionally excluded",),
        )

    def capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot:
        stability = self.wait_quiescent(
            ArtifactStabilityRequest(
                session=request.session,
                timeout_s=30,
                agent_id=request.agent_id,
                execution_context=request.execution_context,
            )
        )
        if not stability.stable:
            raise RuntimeError(stability.reason or "OpenCode artifacts are not stable")
        manifest = self.describe(
            ArtifactDescribeRequest(
                session=request.session,
                agent_id=request.agent_id,
                execution_context=request.execution_context,
            )
        )
        root = Path(request.snapshot_root)
        root.mkdir(parents=True, exist_ok=True)
        captured: list[ProviderArtifactEntry] = []
        for entry in manifest.entries:
            assert entry.native_ref is not None and entry.snapshot_relpath is not None
            source = self.runtime_root / entry.native_ref
            target = root / entry.snapshot_relpath
            if entry.capture_strategy == "sqlite_backup":
                _backup(source, target)
            elif source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
            elif source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            captured.append(
                ProviderArtifactEntry(
                    artifact_id=entry.artifact_id,
                    kind=entry.kind,
                    authority=entry.authority,
                    capture_strategy=entry.capture_strategy,
                    native_ref=entry.native_ref,
                    snapshot_relpath=entry.snapshot_relpath,
                    sha256=_hash_path(target),
                    size_bytes=_size_path(target),
                    required_for_resume=entry.required_for_resume,
                )
            )
        return ProviderArtifactSnapshot(
            manifest=ProviderArtifactManifest(
                provider_type=manifest.provider_type,
                home_id=manifest.home_id,
                session_id=manifest.session_id,
                adapter_version=manifest.adapter_version,
                stable=True,
                entries=tuple(captured),
                locator=manifest.locator,
                warnings=manifest.warnings,
            ),
            captured_at=utc_now_iso(),
            snapshot_root=str(root),
        )

    def prepare_restore(self, request: ArtifactRestoreRequest) -> None:
        for entry in request.manifest.entries:
            if entry.native_ref is None:
                continue
            target = (self.runtime_root / entry.native_ref).resolve()
            if self.runtime_root.resolve() not in target.parents:
                raise ValueError("OpenCode artifact restore target escapes runtime_root")
            parts = Path(entry.native_ref).parts
            if len(parts) >= 4 and parts[:3] == ("providers", PROVIDER_TYPE, "agents"):
                self.registry.close_agent(parts[3])

    def restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult:
        self.prepare_restore(request)
        root = Path(request.snapshot_root)
        warnings: list[str] = []
        for entry in request.manifest.entries:
            if entry.native_ref is None or entry.snapshot_relpath is None:
                continue
            source = root / entry.snapshot_relpath
            target = self.runtime_root / entry.native_ref
            if entry.sha256 and _hash_path(source) != entry.sha256:
                raise RuntimeError(f"OpenCode snapshot checksum mismatch: {entry.artifact_id}")
            if entry.capture_strategy == "sqlite_backup":
                _backup(source, target)
                for suffix in ("-wal", "-shm"):
                    stale = Path(str(target) + suffix)
                    if stale.exists():
                        stale.unlink()
            elif source.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target)
            elif source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            elif entry.required_for_resume:
                raise FileNotFoundError(source)
            else:
                warnings.append(f"optional artifact missing: {entry.artifact_id}")
        return ProviderArtifactRestoreResult(
            restored=True,
            restored_at=utc_now_iso(),
            warnings=tuple(warnings),
        )

    def rebuild_after_restore(self, request: ArtifactRestoreRequest) -> None:
        del request


def _additional_paths(runtime: Path) -> tuple[tuple[str, Path], ...]:
    data = runtime / "xdg-data" / "opencode"
    return (("tool-output", data / "tool-output"), ("plans", data / "plans"))


def _native_ref(runtime_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(runtime_root.resolve()))
    except ValueError as exc:
        raise ValueError("OpenCode artifact path must remain below runtime_root") from exc


def _backup(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with sqlite3.connect(source) as source_conn, sqlite3.connect(temporary) as target_conn:
        source_conn.backup(target_conn)
        result = target_conn.execute("pragma integrity_check").fetchone()
        if result is None or result[0] != "ok":
            raise RuntimeError("OpenCode SQLite artifact failed integrity_check")
    os.replace(temporary, target)


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _size_path(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
