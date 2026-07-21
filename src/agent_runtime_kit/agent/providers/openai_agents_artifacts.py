from __future__ import annotations

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
from .openai_agents_storage import OpenAIAgentsSessionStore, sha256


class OpenAIAgentsArtifactAdapter:
    provider_type = "openai_agents"
    adapter_version = "1"

    def __init__(self, *, runtime_root: Path) -> None:
        self.runtime_root = Path(runtime_root)

    def wait_quiescent(self, request: ArtifactStabilityRequest) -> ArtifactStabilityResult:
        store = self._store(request.session)
        stable = store.path.is_file() and store.is_quiescent()
        return ArtifactStabilityResult(
            stable=stable,
            observed_at=utc_now_iso(),
            reason=None if stable else "SQLite session is missing or active",
        )

    def describe(self, request: ArtifactDescribeRequest) -> ProviderArtifactManifest:
        store = self._store(request.session)
        native_ref = self._native_ref(request.session.home_id, request.session.session_id)
        exists = store.path.is_file()
        entry = ProviderArtifactEntry(
            artifact_id=f"openai-agents-sqlite:{request.session.session_id}",
            kind="session_state",
            authority="ark_owned",
            capture_strategy="sqlite_backup",
            native_ref=native_ref,
            snapshot_relpath=native_ref,
            # The live database may have committed WAL pages that are not reflected
            # in the main file. Authoritative size and digest belong to capture().
            sha256=None,
            size_bytes=None,
            required_for_resume=True,
        )
        return ProviderArtifactManifest(
            provider_type=self.provider_type,
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=self.adapter_version,
            stable=exists and store.is_quiescent(),
            entries=(entry,),
            locator=AgentArtifactLocator(
                provider_type=self.provider_type,
                home_id=request.session.home_id,
                session_id=request.session.session_id,
                adapter_version=self.adapter_version,
                native_primary_ref=native_ref,
            ),
            warnings=() if exists else ("OpenAI Agents session SQLite is missing",),
        )

    def capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot:
        store = self._store(request.session)
        if not store.is_quiescent():
            raise RuntimeError("cannot snapshot an active OpenAI Agents SQLite session")
        snapshot_root = Path(request.snapshot_root)
        native_ref = self._native_ref(request.session.home_id, request.session.session_id)
        target = snapshot_root / native_ref
        store.backup_to(target)
        entry = ProviderArtifactEntry(
            artifact_id=f"openai-agents-sqlite:{request.session.session_id}",
            kind="session_state",
            authority="ark_owned",
            capture_strategy="sqlite_backup",
            native_ref=native_ref,
            snapshot_relpath=native_ref,
            sha256=sha256(target),
            size_bytes=target.stat().st_size,
            required_for_resume=True,
        )
        manifest = ProviderArtifactManifest(
            provider_type=self.provider_type,
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=self.adapter_version,
            stable=True,
            entries=(entry,),
            locator=AgentArtifactLocator(
                provider_type=self.provider_type,
                home_id=request.session.home_id,
                session_id=request.session.session_id,
                adapter_version=self.adapter_version,
                native_primary_ref=native_ref,
            ),
        )
        return ProviderArtifactSnapshot(manifest=manifest, captured_at=utc_now_iso(), snapshot_root=str(snapshot_root))

    def prepare_restore(self, request: ArtifactRestoreRequest) -> None:
        for entry in request.manifest.entries:
            if entry.native_ref is None:
                continue
            target = self._target(entry.native_ref, request)
            for path in (target, Path(str(target) + "-wal"), Path(str(target) + "-shm")):
                if path.is_file():
                    path.unlink()

    def restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult:
        for entry in request.manifest.entries:
            if entry.snapshot_relpath is None or entry.native_ref is None:
                if entry.required_for_resume:
                    raise RuntimeError(f"required OpenAI Agents artifact has no path: {entry.artifact_id}")
                continue
            source = Path(request.snapshot_root) / entry.snapshot_relpath
            if not source.is_file():
                raise RuntimeError(f"OpenAI Agents snapshot artifact is missing: {entry.snapshot_relpath}")
            if entry.sha256 and sha256(source) != entry.sha256:
                raise RuntimeError(f"OpenAI Agents snapshot checksum mismatch: {entry.snapshot_relpath}")
            target = self._target(entry.native_ref, request)
            target.parent.mkdir(parents=True, exist_ok=True)
            session_id = request.manifest.session_id
            home_id = request.target_home_id or request.manifest.home_id
            source_store = OpenAIAgentsSessionStore(source, session_id=session_id, home_id=request.manifest.home_id)
            source_store.backup_to(target)
            target_store = OpenAIAgentsSessionStore(target, session_id=session_id, home_id=home_id)
            target_store.integrity_check()
        return ProviderArtifactRestoreResult(restored=True, restored_at=utc_now_iso())

    def rebuild_after_restore(self, request: ArtifactRestoreRequest) -> None:
        for entry in request.manifest.entries:
            if entry.native_ref is None:
                continue
            target = self._target(entry.native_ref, request)
            for suffix in ("-wal", "-shm"):
                path = Path(str(target) + suffix)
                if path.is_file() and path.stat().st_size == 0:
                    path.unlink()

    def _store(self, session) -> OpenAIAgentsSessionStore:  # noqa: ANN001
        return OpenAIAgentsSessionStore(
            self.runtime_root / self._native_ref(session.home_id, session.session_id),
            session_id=session.session_id,
            home_id=session.home_id,
        )

    def _native_ref(self, home_id: str, session_id: str) -> str:
        return f"homes/{self.provider_type}/{home_id}/sessions/{session_id}.sqlite3"

    def _target(self, native_ref: str, request: ArtifactRestoreRequest) -> Path:
        parts = list(Path(native_ref).parts)
        if len(parts) < 5 or parts[:2] != ["homes", self.provider_type]:
            raise RuntimeError("invalid OpenAI Agents artifact native_ref")
        if parts[2] != request.manifest.home_id:
            raise RuntimeError("OpenAI Agents artifact home does not match manifest")
        if request.target_home_id:
            parts[2] = request.target_home_id
        return self.runtime_root.joinpath(*parts)
