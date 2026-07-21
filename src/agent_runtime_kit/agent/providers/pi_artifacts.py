from __future__ import annotations

import hashlib
import os
import shutil
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
from .pi_session import PI_ADAPTER_VERSION, PiSessionTranscript, find_pi_session


class PiArtifactAdapter:
    provider_type = "pi"
    adapter_version = PI_ADAPTER_VERSION

    def __init__(self, *, runtime_root: Path, active_sessions: object | None = None) -> None:
        self.runtime_root = Path(runtime_root)
        self.active_sessions = active_sessions

    def wait_quiescent(self, request: ArtifactStabilityRequest) -> ArtifactStabilityResult:
        if self._is_active(request.session.session_id):
            return ArtifactStabilityResult(
                stable=False,
                observed_at=utc_now_iso(),
                reason="pi_session_has_active_run_handle",
            )
        path = self._session_path(request.session.home_id, request.session.session_id)
        stable = path is not None and path.is_file()
        return ArtifactStabilityResult(
            stable=stable,
            observed_at=utc_now_iso(),
            reason=None if stable else "pi_session_jsonl_missing",
        )

    def describe(self, request: ArtifactDescribeRequest) -> ProviderArtifactManifest:
        path = self._session_path(request.session.home_id, request.session.session_id)
        entries: tuple[ProviderArtifactEntry, ...] = ()
        relpath = None
        projected: list[ProviderArtifactEntry] = []
        if path is not None and path.is_file():
            PiSessionTranscript.read(path)
            relpath = str(path.relative_to(self.runtime_root))
            projected.append(
                ProviderArtifactEntry(
                    artifact_id=f"pi-session:{request.session.session_id}",
                    kind="session_transcript",
                    authority="provider_native",
                    capture_strategy="copy_file",
                    native_ref=relpath,
                    snapshot_relpath=relpath,
                    sha256=_sha256(path),
                    size_bytes=path.stat().st_size,
                    required_for_resume=True,
                )
            )
        home_manifest = (
            self.runtime_root
            / "homes"
            / "pi"
            / request.session.home_id
            / ".ark"
            / "home_materialization.json"
        )
        if home_manifest.is_file():
            projected.append(
                ProviderArtifactEntry(
                    artifact_id=f"pi-home-manifest:{request.session.home_id}",
                    kind="home_materialization_manifest",
                    authority="ark",
                    capture_strategy="reference_hash",
                    native_ref=str(home_manifest.relative_to(self.runtime_root)),
                    sha256=_sha256(home_manifest),
                    size_bytes=home_manifest.stat().st_size,
                    required_for_resume=False,
                )
            )
        entries = tuple(projected)
        return ProviderArtifactManifest(
            provider_type="pi",
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=self.adapter_version,
            stable=any(item.kind == "session_transcript" for item in entries)
            and not self._is_active(request.session.session_id),
            entries=entries,
            locator=AgentArtifactLocator(
                provider_type="pi",
                home_id=request.session.home_id,
                session_id=request.session.session_id,
                adapter_version=self.adapter_version,
                native_primary_ref=relpath,
            ),
            warnings=(
                ()
                if any(item.kind == "session_transcript" for item in entries)
                else ("Pi session JSONL is missing",)
            ),
        )

    def capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot:
        stability = self.wait_quiescent(
            ArtifactStabilityRequest(
                session=request.session,
                timeout_s=None,
                agent_id=request.agent_id,
                execution_context=request.execution_context,
            )
        )
        if not stability.stable:
            raise RuntimeError(f"Pi session is not quiescent: {stability.reason}")
        manifest = self.describe(
            ArtifactDescribeRequest(
                session=request.session,
                agent_id=request.agent_id,
                execution_context=request.execution_context,
            )
        )
        snapshot_root = Path(request.snapshot_root)
        for entry in manifest.entries:
            if entry.native_ref is None or entry.snapshot_relpath is None:
                continue
            source = self.runtime_root / entry.native_ref
            target = snapshot_root / entry.snapshot_relpath
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return ProviderArtifactSnapshot(
            manifest=manifest,
            captured_at=utc_now_iso(),
            snapshot_root=str(snapshot_root),
        )

    def prepare_restore(self, request: ArtifactRestoreRequest) -> None:
        self._validate_reference_artifacts(request)
        for entry in request.manifest.entries:
            if entry.kind == "home_materialization_manifest":
                continue
            if entry.required_for_resume and (entry.native_ref is None or entry.snapshot_relpath is None):
                raise RuntimeError(f"required Pi artifact has no locator: {entry.artifact_id}")
            if entry.native_ref is not None:
                target = self._target_path(
                    entry.native_ref,
                    target_home_id=request.target_home_id,
                    source_home_id=request.manifest.home_id,
                )
                if target.is_file():
                    target.unlink()

    def restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult:
        snapshot_root = Path(request.snapshot_root)
        self._validate_restore(request, snapshot_root)
        for entry in request.manifest.entries:
            if entry.kind == "home_materialization_manifest":
                continue
            if entry.native_ref is None or entry.snapshot_relpath is None:
                if entry.required_for_resume:
                    raise RuntimeError(f"required Pi artifact has no locator: {entry.artifact_id}")
                continue
            source = self._snapshot_path(snapshot_root, entry.snapshot_relpath)
            target = self._target_path(
                entry.native_ref,
                target_home_id=request.target_home_id,
                source_home_id=request.manifest.home_id,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.ark-restore")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        return ProviderArtifactRestoreResult(restored=True, restored_at=utc_now_iso())

    def rebuild_after_restore(self, request: ArtifactRestoreRequest) -> None:
        del request

    def _validate_reference_artifacts(self, request: ArtifactRestoreRequest) -> None:
        for entry in request.manifest.entries:
            if entry.kind != "home_materialization_manifest":
                continue
            if entry.native_ref is None:
                raise RuntimeError("Pi snapshot has no Home manifest reference")
            current = self._home_manifest_path(
                entry.native_ref,
                target_home_id=request.target_home_id,
                source_home_id=request.manifest.home_id,
            )
            if not current.is_file() or (
                entry.sha256 is not None and _sha256(current) != entry.sha256
            ):
                raise RuntimeError(
                    "Pi Home materialization manifest does not match the captured session"
                )

    def _validate_restore(self, request: ArtifactRestoreRequest, snapshot_root: Path) -> None:
        self._validate_reference_artifacts(request)
        for entry in request.manifest.entries:
            if entry.kind == "home_materialization_manifest":
                continue
            if entry.native_ref is None or entry.snapshot_relpath is None:
                if entry.required_for_resume:
                    raise RuntimeError(f"required Pi artifact has no locator: {entry.artifact_id}")
                continue
            source = self._snapshot_path(snapshot_root, entry.snapshot_relpath)
            if not source.is_file():
                if entry.required_for_resume:
                    raise RuntimeError(f"required Pi artifact is missing: {entry.snapshot_relpath}")
                continue
            if entry.sha256 is not None and _sha256(source) != entry.sha256:
                raise RuntimeError(f"Pi artifact checksum mismatch: {entry.snapshot_relpath}")
            transcript = PiSessionTranscript.read(source)
            if transcript.session_id != request.manifest.session_id:
                raise RuntimeError("Pi artifact session id does not match manifest")
            target = self._target_path(
                entry.native_ref,
                target_home_id=request.target_home_id,
                source_home_id=request.manifest.home_id,
            )
            if target.exists() and _sha256(target) != _sha256(source):
                raise RuntimeError(f"Pi restore target already has different content: {target}")

    def _is_active(self, session_id: str) -> bool:
        if self.active_sessions is None:
            return False
        method = getattr(self.active_sessions, "is_session_active", None)
        return bool(method(session_id)) if callable(method) else False

    def _session_path(self, home_id: str, session_id: str) -> Path | None:
        return find_pi_session(
            self.runtime_root / "homes" / "pi" / home_id / ".pi" / "sessions",
            session_id,
        )

    def _target_path(self, native_ref: str, *, target_home_id: str | None, source_home_id: str) -> Path:
        path = Path(native_ref)
        parts = list(path.parts)
        expected = ["homes", "pi", source_home_id]
        if len(parts) < 6 or parts[:3] != expected or parts[3:5] != [".pi", "sessions"]:
            raise RuntimeError("Pi artifact native_ref is outside the managed session directory")
        if target_home_id is not None:
            parts[2] = target_home_id
        target = self.runtime_root.joinpath(*parts)
        target.resolve().relative_to(self.runtime_root.resolve())
        return target

    def _home_manifest_path(
        self,
        native_ref: str,
        *,
        target_home_id: str | None,
        source_home_id: str,
    ) -> Path:
        path = Path(native_ref)
        parts = list(path.parts)
        expected = ["homes", "pi", source_home_id, ".ark", "home_materialization.json"]
        if parts != expected:
            raise RuntimeError("Pi Home manifest native_ref is outside the managed Home directory")
        if target_home_id is not None:
            parts[2] = target_home_id
        target = self.runtime_root.joinpath(*parts)
        target.resolve().relative_to(self.runtime_root.resolve())
        return target

    @staticmethod
    def _snapshot_path(snapshot_root: Path, relpath: str) -> Path:
        root = Path(snapshot_root).resolve()
        source = (root / relpath).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise RuntimeError("Pi snapshot artifact path escapes the snapshot root") from exc
        return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
