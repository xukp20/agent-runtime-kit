from __future__ import annotations

import hashlib
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
from .codex import CodexProvider


class CodexArtifactAdapter:
    """Own Codex session artifact discovery, capture, restore, and cache cleanup."""

    provider_type = "codex"
    adapter_version = "1"

    def __init__(self, *, runtime_root: Path, provider: CodexProvider) -> None:
        self.runtime_root = Path(runtime_root)
        self.provider = provider

    def wait_quiescent(self, request: ArtifactStabilityRequest) -> ArtifactStabilityResult:
        path, _ = self._rollout(request.session)
        return ArtifactStabilityResult(
            stable=path is not None and path.is_file(),
            observed_at=utc_now_iso(),
            reason=None if path is not None and path.is_file() else "rollout_missing",
        )

    def describe(self, request: ArtifactDescribeRequest) -> ProviderArtifactManifest:
        path, relpath = self._rollout(request.session)
        entries: tuple[ProviderArtifactEntry, ...] = ()
        warnings: tuple[str, ...] = ()
        if path is not None and relpath is not None and path.is_file():
            runtime_relpath = str(path.relative_to(self.runtime_root))
            entries = (
                ProviderArtifactEntry(
                    artifact_id=f"codex-rollout:{request.session.session_id}",
                    kind="session_transcript",
                    authority="provider_native",
                    capture_strategy="copy_file",
                    native_ref=runtime_relpath,
                    snapshot_relpath=runtime_relpath,
                    sha256=_sha256(path),
                    size_bytes=path.stat().st_size,
                    required_for_resume=True,
                ),
            )
        else:
            warnings = ("Codex rollout is missing; session cannot be resumed from this snapshot",)
        return ProviderArtifactManifest(
            provider_type="codex",
            home_id=request.session.home_id,
            session_id=request.session.session_id,
            adapter_version=self.adapter_version,
            stable=bool(entries),
            entries=entries,
            locator=AgentArtifactLocator(
                provider_type="codex",
                home_id=request.session.home_id,
                session_id=request.session.session_id,
                adapter_version=self.adapter_version,
                native_primary_ref=relpath,
            ),
            warnings=warnings,
        )

    def capture(self, request: ArtifactCaptureRequest) -> ProviderArtifactSnapshot:
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
        for entry in request.manifest.entries:
            if entry.native_ref is None:
                continue
            target = self._target_path(entry.native_ref, request.target_home_id, request.manifest.home_id)
            if target.is_file():
                target.unlink()
        self._discard_rebuildable_state(request.target_home_id or request.manifest.home_id)

    def restore(self, request: ArtifactRestoreRequest) -> ProviderArtifactRestoreResult:
        snapshot_root = Path(request.snapshot_root)
        for entry in request.manifest.entries:
            if entry.native_ref is None or entry.snapshot_relpath is None:
                if entry.required_for_resume:
                    raise RuntimeError(f"required Codex artifact has no copy locator: {entry.artifact_id}")
                continue
            source = snapshot_root / entry.snapshot_relpath
            if not source.is_file():
                if entry.required_for_resume:
                    raise RuntimeError(f"required Codex artifact is missing: {entry.snapshot_relpath}")
                continue
            if entry.sha256 is not None and _sha256(source) != entry.sha256:
                raise RuntimeError(f"Codex artifact checksum mismatch: {entry.snapshot_relpath}")
            target = self._target_path(entry.native_ref, request.target_home_id, request.manifest.home_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        return ProviderArtifactRestoreResult(restored=True, restored_at=utc_now_iso())

    def rebuild_after_restore(self, request: ArtifactRestoreRequest) -> None:
        self._discard_rebuildable_state(request.target_home_id or request.manifest.home_id)

    def _rollout(self, session) -> tuple[Path | None, str | None]:  # noqa: ANN001
        native = session.native_locator if isinstance(session.native_locator, dict) else {}
        relpath = native.get("rollout_relpath")
        home_root = self.runtime_root / "homes" / "codex" / session.home_id
        if not relpath:
            relpath = self.provider.find_rollout_relpath(
                home_root=home_root,
                thread_id=session.session_id,
            )
        if not relpath:
            return None, None
        return home_root / ".codex" / str(relpath), str(relpath)

    def _target_path(self, native_ref: str, target_home_id: str | None, source_home_id: str) -> Path:
        path = Path(native_ref)
        parts = list(path.parts)
        if target_home_id is not None and len(parts) >= 3 and parts[:2] == ["homes", "codex"]:
            if parts[2] != source_home_id:
                raise RuntimeError("Codex artifact native_ref does not match manifest home")
            parts[2] = target_home_id
        return self.runtime_root.joinpath(*parts)

    def _discard_rebuildable_state(self, home_id: str) -> None:
        codex_root = self.runtime_root / "homes" / "codex" / home_id / ".codex"
        if not codex_root.exists():
            return
        for path in codex_root.glob("state_5.sqlite*"):
            if path.is_file():
                path.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
