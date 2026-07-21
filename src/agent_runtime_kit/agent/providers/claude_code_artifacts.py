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
from ..store_utils import read_json, utc_now_iso
from .claude_code import ClaudeCodeProvider
from .claude_code_home import _manifest_hash
from .claude_code_normalization import find_session_file, read_transcript


class ClaudeCodeArtifactAdapter:
    provider_type = "claude_code"
    adapter_version = "1"

    def __init__(self, *, runtime_root: Path, provider: ClaudeCodeProvider) -> None:
        self.runtime_root = Path(runtime_root)
        self.provider = provider

    def wait_quiescent(self, request: ArtifactStabilityRequest) -> ArtifactStabilityResult:
        try:
            path = self._transcript(request.session.home_id, request.session.session_id)
            stable = path is not None and path.is_file() and bool(read_transcript(path))
            reason = None if stable else "Claude session transcript is missing or empty"
        except BaseException as exc:
            stable = False
            reason = f"{type(exc).__name__}: {str(exc)}"
        return ArtifactStabilityResult(stable=stable, observed_at=utc_now_iso(), reason=reason)

    def describe(self, request: ArtifactDescribeRequest) -> ProviderArtifactManifest:
        home_root = self._home_root(request.session.home_id)
        transcript = self._transcript(request.session.home_id, request.session.session_id)
        home_manifest = home_root / ".ark" / "home_materialization.json"
        entries: list[ProviderArtifactEntry] = []
        warnings: list[str] = []
        native_primary_ref = None
        if transcript is not None and transcript.is_file():
            native_ref = str(transcript.relative_to(self.runtime_root))
            native_primary_ref = str(transcript.relative_to(home_root / ".claude"))
            entries.append(
                ProviderArtifactEntry(
                    artifact_id=f"claude-transcript:{request.session.session_id}",
                    kind="session_transcript",
                    authority="provider_native",
                    capture_strategy="copy_file",
                    native_ref=native_ref,
                    snapshot_relpath=native_ref,
                    sha256=_sha256(transcript),
                    size_bytes=transcript.stat().st_size,
                    required_for_resume=True,
                )
            )
        else:
            warnings.append("Claude transcript is missing; session cannot be resumed")
        if home_manifest.is_file() and _valid_home_manifest(home_manifest):
            entries.append(
                ProviderArtifactEntry(
                    artifact_id=f"claude-home-manifest:{request.session.home_id}",
                    kind="home_materialization_manifest",
                    authority="external",
                    capture_strategy="external_reference",
                    native_ref=str(home_manifest.relative_to(self.runtime_root)),
                    snapshot_relpath=None,
                    sha256=_sha256(home_manifest),
                    size_bytes=home_manifest.stat().st_size,
                    required_for_resume=True,
                )
            )
        else:
            warnings.append("Claude Home materialization manifest is missing or invalid")
        stable = len(entries) == 2
        return ProviderArtifactManifest(
            provider_type=self.provider_type,
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=self.adapter_version,
            stable=stable,
            entries=tuple(entries),
            locator=AgentArtifactLocator(
                provider_type=self.provider_type,
                home_id=request.session.home_id,
                session_id=request.session.session_id,
                adapter_version=self.adapter_version,
                native_primary_ref=native_primary_ref,
            ),
            warnings=tuple(warnings),
        )

    def capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot:
        manifest = self.describe(
            ArtifactDescribeRequest(
                session=request.session,
                agent_id=request.agent_id,
                execution_context=request.execution_context,
            )
        )
        if not manifest.stable:
            raise RuntimeError("Claude artifacts are not stable for snapshot")
        snapshot_root = Path(request.snapshot_root)
        for entry in manifest.entries:
            if entry.kind != "session_transcript":
                continue
            if entry.native_ref is None or entry.snapshot_relpath is None:
                raise RuntimeError("Claude transcript entry has no copy locator")
            source = _safe_join(self.runtime_root, entry.native_ref)
            target = _safe_join(snapshot_root, entry.snapshot_relpath)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            shutil.copy2(source, temporary)
            if entry.sha256 is not None and _sha256(temporary) != entry.sha256:
                temporary.unlink(missing_ok=True)
                raise RuntimeError("Claude transcript changed while snapshotting")
            os.replace(temporary, target)
        return ProviderArtifactSnapshot(
            manifest=manifest,
            captured_at=utc_now_iso(),
            snapshot_root=str(snapshot_root),
        )

    def prepare_restore(self, request: ArtifactRestoreRequest) -> None:
        if Path(request.snapshot_root).resolve() == self.runtime_root.resolve():
            # SnapshotManager first calls prepare_restore with a manifest
            # described from the live Agent as a legacy cleanup phase. It has
            # not exposed the snapshot's external Home dependency yet, so
            # deleting here would make a later dependency mismatch destructive.
            # The snapshot-backed prepare call below replaces the exact target.
            return
        # Validate the external Home dependency before removing any live
        # provider-native artifact. Restore must fail closed without turning a
        # Home mismatch into destructive partial preparation.
        self._validate_home_dependency(request)
        for entry in request.manifest.entries:
            if entry.kind != "session_transcript" or entry.native_ref is None:
                continue
            target = self._target_path(
                entry.native_ref,
                request.target_home_id,
                request.manifest.home_id,
            )
            target.unlink(missing_ok=True)

    def restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult:
        self._validate_home_dependency(request)
        snapshot_root = Path(request.snapshot_root)
        for entry in request.manifest.entries:
            if entry.kind != "session_transcript":
                continue
            if entry.native_ref is None or entry.snapshot_relpath is None:
                raise RuntimeError("required Claude transcript has no copy locator")
            source = _safe_join(snapshot_root, entry.snapshot_relpath)
            if not source.is_file():
                raise RuntimeError(f"required Claude artifact is missing: {entry.snapshot_relpath}")
            if entry.sha256 is not None and _sha256(source) != entry.sha256:
                raise RuntimeError(f"Claude artifact checksum mismatch: {entry.snapshot_relpath}")
            target = self._target_path(
                entry.native_ref,
                request.target_home_id,
                request.manifest.home_id,
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.restore-{os.getpid()}")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            read_transcript(target)
        return ProviderArtifactRestoreResult(restored=True, restored_at=utc_now_iso())

    def rebuild_after_restore(self, request: ArtifactRestoreRequest) -> None:
        home_id = request.target_home_id or request.manifest.home_id
        path = self._transcript(home_id, request.manifest.session_id)
        if path is None or not read_transcript(path):
            raise RuntimeError("restored Claude transcript cannot be queried")

    def _validate_home_dependency(self, request: ArtifactRestoreRequest) -> None:
        entries = [
            entry for entry in request.manifest.entries if entry.kind == "home_materialization_manifest"
        ]
        if len(entries) != 1:
            raise RuntimeError("Claude snapshot has no unique Home manifest dependency")
        entry = entries[0]
        if entry.native_ref is None or entry.sha256 is None:
            raise RuntimeError("Claude Home dependency is incomplete")
        native_ref = entry.native_ref
        if request.target_home_id is not None:
            native_ref = _replace_home_id(
                native_ref,
                request.manifest.home_id,
                request.target_home_id,
            )
        path = _safe_join(self.runtime_root, native_ref)
        if not path.is_file() or _sha256(path) != entry.sha256 or not _valid_home_manifest(path):
            raise RuntimeError("Claude Home materialization does not match the snapshot dependency")

    def _transcript(self, home_id: str, session_id: str) -> Path | None:
        return find_session_file(self._home_root(home_id), session_id)

    def _home_root(self, home_id: str) -> Path:
        return self.runtime_root / "homes" / self.provider_type / home_id

    def _target_path(
        self,
        native_ref: str,
        target_home_id: str | None,
        source_home_id: str,
    ) -> Path:
        ref = (
            _replace_home_id(native_ref, source_home_id, target_home_id)
            if target_home_id is not None
            else native_ref
        )
        return _safe_join(self.runtime_root, ref)


def _replace_home_id(native_ref: str, source_home_id: str, target_home_id: str) -> str:
    parts = list(Path(native_ref).parts)
    if len(parts) < 3 or parts[:2] != ["homes", "claude_code"] or parts[2] != source_home_id:
        raise RuntimeError("Claude artifact native_ref does not match manifest home")
    parts[2] = target_home_id
    return str(Path(*parts))


def _valid_home_manifest(path: Path) -> bool:
    try:
        payload = read_json(path)
    except Exception:
        return False
    declared = str(payload.get("manifest_hash", ""))
    return bool(declared) and _manifest_hash(payload) == declared


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_join(root: Path, relpath: str) -> Path:
    root = Path(root).resolve()
    target = (root / relpath).resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"path escapes allowed root: {relpath}")
    return target
