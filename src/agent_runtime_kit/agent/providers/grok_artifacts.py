from __future__ import annotations

import hashlib
import os
import shutil
import uuid
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import quote, unquote

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
    build_provider_payload,
)
from ..store_utils import utc_now_iso
from .grok_home import GROK_ADAPTER_VERSION, GROK_CLI_VERSION


_EXCLUDED_NAMES = {"lock", ".lock", "session.lock"}


class GrokArtifactAdapter:
    provider_type = "grok"
    adapter_version = GROK_ADAPTER_VERSION

    def __init__(self, *, runtime_root: Path, active_sessions: object | None = None) -> None:
        self.runtime_root = Path(runtime_root)
        self.active_sessions = active_sessions

    def wait_quiescent(self, request: ArtifactStabilityRequest) -> ArtifactStabilityResult:
        if not self._runtime_stable(request.session.session_id):
            return ArtifactStabilityResult(
                stable=False,
                observed_at=utc_now_iso(),
                reason="grok_session_has_active_or_unclean_process_group",
            )
        try:
            session_dir = self._session_dir(request.session)
            stable = session_dir.is_dir() and any(self._iter_files(session_dir))
            reason = None if stable else "grok_session_directory_missing_or_empty"
        except BaseException as exc:
            stable = False
            reason = f"{type(exc).__name__}: {exc}"
        return ArtifactStabilityResult(stable=stable, observed_at=utc_now_iso(), reason=reason)

    def describe(self, request: ArtifactDescribeRequest) -> ProviderArtifactManifest:
        session_dir = self._session_dir(request.session)
        projected: list[ProviderArtifactEntry] = []
        if session_dir.is_dir():
            for path in self._iter_files(session_dir):
                native_ref = str(path.relative_to(self.runtime_root))
                relative = str(path.relative_to(session_dir))
                projected.append(
                    ProviderArtifactEntry(
                        artifact_id=f"grok-session:{request.session.session_id}:{relative}",
                        kind="session_file",
                        authority="provider_native",
                        capture_strategy="copy_file",
                        native_ref=native_ref,
                        snapshot_relpath=native_ref,
                        sha256=_sha256(path),
                        size_bytes=path.stat().st_size,
                        required_for_resume=True,
                    )
                )
        home_manifest = self._home_root(request.session.home_id) / ".ark" / "home_materialization.json"
        if home_manifest.is_file():
            projected.append(
                ProviderArtifactEntry(
                    artifact_id=f"grok-home-manifest:{request.session.home_id}",
                    kind="home_materialization_manifest",
                    authority="ark",
                    capture_strategy="reference_hash",
                    native_ref=str(home_manifest.relative_to(self.runtime_root)),
                    sha256=_sha256(home_manifest),
                    size_bytes=home_manifest.stat().st_size,
                    required_for_resume=False,
                )
            )
        native = request.session.native_locator if isinstance(request.session.native_locator, dict) else {}
        session_ref = str(session_dir.relative_to(self.runtime_root))
        has_session = any(item.kind == "session_file" for item in projected)
        return ProviderArtifactManifest(
            provider_type="grok",
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=self.adapter_version,
            stable=has_session and self._runtime_stable(request.session.session_id),
            entries=tuple(projected),
            locator=AgentArtifactLocator(
                provider_type="grok",
                home_id=request.session.home_id,
                session_id=request.session.session_id,
                adapter_version=self.adapter_version,
                native_primary_ref=session_ref,
            ),
            warnings=() if has_session else ("Grok native session directory is missing or empty",),
            provider_payload=build_provider_payload(
                provider_type="grok",
                payload_type="session_artifact_identity",
                data={
                    "workdir": native.get("workdir"),
                    "session_relpath": session_ref,
                },
                adapter_version=GROK_ADAPTER_VERSION,
                sdk_or_cli_version=GROK_CLI_VERSION,
            ),
        )

    def capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot:
        guard = getattr(self.active_sessions, "artifact_boundary", None)
        with guard(request.session.session_id) if callable(guard) else nullcontext():
            return self._capture(request)

    def _capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot:
        stability = self.wait_quiescent(
            ArtifactStabilityRequest(
                session=request.session,
                agent_id=request.agent_id,
                execution_context=request.execution_context,
            )
        )
        if not stability.stable:
            raise RuntimeError(f"Grok session is not quiescent: {stability.reason}")
        manifest = self.describe(
            ArtifactDescribeRequest(
                session=request.session,
                agent_id=request.agent_id,
                execution_context=request.execution_context,
            )
        )
        if not manifest.stable:
            raise RuntimeError("Grok session changed stability during capture")
        snapshot_root = Path(request.snapshot_root)
        for entry in manifest.entries:
            if entry.kind == "home_materialization_manifest":
                continue
            if entry.native_ref is None or entry.snapshot_relpath is None:
                raise RuntimeError(f"Grok session artifact has no path: {entry.artifact_id}")
            source = _safe_join(self.runtime_root, entry.native_ref)
            target = _safe_join(snapshot_root, entry.snapshot_relpath)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return ProviderArtifactSnapshot(
            manifest=manifest,
            captured_at=utc_now_iso(),
            snapshot_root=str(snapshot_root),
        )

    def prepare_restore(self, request: ArtifactRestoreRequest) -> None:
        self._validate_restore_request(request)
        self._validate_snapshot(request)

    def restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult:
        guard = getattr(self.active_sessions, "artifact_boundary", None)
        with guard(request.manifest.session_id) if callable(guard) else nullcontext():
            return self._restore(request)

    def _restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult:
        self._validate_restore_request(request)
        self._validate_snapshot(request)
        session_entries = [item for item in request.manifest.entries if item.kind == "session_file"]
        if not session_entries:
            raise RuntimeError("Grok snapshot has no native session files")
        target_dir = self._manifest_session_dir(request.manifest)
        temporary = target_dir.with_name(f".{target_dir.name}.ark-restore-{uuid.uuid4().hex}")
        backup = target_dir.with_name(f".{target_dir.name}.ark-backup-{uuid.uuid4().hex}")
        temporary.mkdir(parents=True)
        try:
            for entry in session_entries:
                assert entry.snapshot_relpath is not None and entry.native_ref is not None
                source = _safe_join(Path(request.snapshot_root), entry.snapshot_relpath)
                native = _safe_join(self.runtime_root, entry.native_ref)
                relative = native.relative_to(target_dir)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_dir.exists():
                os.replace(target_dir, backup)
            os.replace(temporary, target_dir)
            if backup.exists():
                shutil.rmtree(backup)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            if backup.exists() and not target_dir.exists():
                os.replace(backup, target_dir)
            raise
        return ProviderArtifactRestoreResult(restored=True, restored_at=utc_now_iso())

    def rebuild_after_restore(self, request: ArtifactRestoreRequest) -> None:
        del request

    def _validate_restore_request(self, request: ArtifactRestoreRequest) -> None:
        if request.manifest.provider_type != self.provider_type:
            raise RuntimeError("Grok artifact manifest belongs to another provider")
        if request.target_home_id not in {None, request.manifest.home_id}:
            raise RuntimeError("Grok session restore is limited to the same managed Home")
        if not self._runtime_stable(request.manifest.session_id):
            raise RuntimeError("Grok session restore requires a clean idle process group")
        self._validate_home_reference(request)

    def _validate_home_reference(self, request: ArtifactRestoreRequest) -> None:
        entries = [item for item in request.manifest.entries if item.kind == "home_materialization_manifest"]
        if len(entries) != 1 or entries[0].native_ref is None or entries[0].sha256 is None:
            raise RuntimeError("Grok snapshot has no complete Home materialization dependency")
        current = _safe_join(self.runtime_root, entries[0].native_ref)
        if not current.is_file() or _sha256(current) != entries[0].sha256:
            raise RuntimeError("Grok Home materialization does not match the captured session")

    def _validate_snapshot(self, request: ArtifactRestoreRequest) -> None:
        target_dir = self._manifest_session_dir(request.manifest)
        for entry in request.manifest.entries:
            if entry.kind == "home_materialization_manifest":
                continue
            if entry.native_ref is None or entry.snapshot_relpath is None or entry.sha256 is None:
                raise RuntimeError(f"Grok required artifact has no complete locator: {entry.artifact_id}")
            native = _safe_join(self.runtime_root, entry.native_ref)
            try:
                native.relative_to(target_dir)
            except ValueError as exc:
                raise RuntimeError("Grok artifact is outside the captured session directory") from exc
            source = _safe_join(Path(request.snapshot_root), entry.snapshot_relpath)
            if not source.is_file() or _sha256(source) != entry.sha256:
                raise RuntimeError(f"Grok artifact missing or checksum mismatch: {entry.snapshot_relpath}")

    def _manifest_session_dir(self, manifest: ProviderArtifactManifest) -> Path:
        prefix = self._home_root(manifest.home_id) / ".grok" / "sessions"
        candidates = []
        for entry in manifest.entries:
            if entry.kind != "session_file" or entry.native_ref is None:
                continue
            native = _safe_join(self.runtime_root, entry.native_ref)
            try:
                relative = native.relative_to(prefix)
            except ValueError as exc:
                raise RuntimeError("Grok artifact is outside the managed sessions root") from exc
            if len(relative.parts) < 3 or relative.parts[1] != manifest.session_id:
                raise RuntimeError("Grok artifact path does not match manifest session identity")
            candidates.append(prefix / relative.parts[0] / relative.parts[1])
        if not candidates or any(item != candidates[0] for item in candidates):
            raise RuntimeError("Grok snapshot does not identify exactly one session directory")
        return candidates[0]

    def _session_dir(self, session) -> Path:  # noqa: ANN001
        native = session.native_locator
        if native is None:
            # Legacy failed startups persisted only the session ID. Preserve their
            # artifacts without changing Agent truth or making them resumable.
            root = self._home_root(session.home_id) / ".grok" / "sessions"
            candidates = [path for path in root.glob("*/*") if path.name == session.session_id]
            if len(candidates) != 1:
                raise RuntimeError("Grok legacy session directory is missing or ambiguous")
            candidate = candidates[0]
            workdir = unquote(candidate.parent.name)
            if not Path(workdir).is_absolute() or quote(workdir, safe="") != candidate.parent.name:
                raise RuntimeError("Grok legacy session directory has invalid workdir encoding")
            native = {"session_relpath": str(candidate.relative_to(self.runtime_root))}
        if not isinstance(native, dict) or not isinstance(native.get("session_relpath"), str):
            raise RuntimeError("Grok session locator has no native session path")
        path = _safe_join(self.runtime_root, native["session_relpath"])
        expected = self._home_root(session.home_id) / ".grok" / "sessions"
        try:
            relative = path.relative_to(expected)
        except ValueError as exc:
            raise RuntimeError("Grok session locator is outside its managed Home") from exc
        if len(relative.parts) != 2 or relative.parts[1] != session.session_id:
            raise RuntimeError("Grok session locator does not match Home/workdir/session identity")
        return path

    def _iter_files(self, session_dir: Path):  # noqa: ANN201
        for path in sorted(session_dir.rglob("*")):
            if path.is_symlink():
                raise RuntimeError("Grok session artifacts must not contain symlinks")
            if not path.is_file() or path.name in _EXCLUDED_NAMES or path.name.endswith(".lock"):
                continue
            yield path

    def _runtime_stable(self, session_id: str) -> bool:
        if self.active_sessions is None:
            return True
        method = getattr(self.active_sessions, "is_session_stable", None)
        if callable(method):
            return bool(method(session_id))
        active = getattr(self.active_sessions, "is_session_active", None)
        return not bool(active(session_id)) if callable(active) else True

    def _home_root(self, home_id: str) -> Path:
        return self.runtime_root / "homes" / "grok" / home_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_join(root: Path, relpath: str) -> Path:
    root = Path(root).resolve()
    target = (root / relpath).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"path escapes allowed root: {relpath}")
    return target
