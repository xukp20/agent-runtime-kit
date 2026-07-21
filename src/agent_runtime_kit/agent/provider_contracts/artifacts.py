from __future__ import annotations

from dataclasses import dataclass, field

from .homes import ProviderExecutionContext
from .identities import ProviderPayload
from .locators import AgentArtifactLocator, ProviderSessionLocator


@dataclass(frozen=True)
class ProviderArtifactEntry:
    artifact_id: str
    kind: str
    authority: str
    capture_strategy: str
    native_ref: str | None = None
    snapshot_relpath: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    required_for_resume: bool = False
    provider_payload: ProviderPayload | None = None


@dataclass(frozen=True)
class ProviderArtifactManifest:
    provider_type: str
    home_id: str
    session_id: str
    adapter_version: str
    stable: bool
    entries: tuple[ProviderArtifactEntry, ...]
    locator: AgentArtifactLocator | None = None
    warnings: tuple[str, ...] = ()
    provider_payload: ProviderPayload | None = None


@dataclass(frozen=True)
class ArtifactStabilityRequest:
    session: ProviderSessionLocator
    timeout_s: float | None = None
    agent_id: str | None = None
    execution_context: ProviderExecutionContext | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ArtifactStabilityResult:
    stable: bool
    observed_at: str
    reason: str | None = None


@dataclass(frozen=True)
class ArtifactDescribeRequest:
    session: ProviderSessionLocator
    agent_id: str | None = None
    execution_context: ProviderExecutionContext | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ArtifactCaptureRequest:
    session: ProviderSessionLocator
    snapshot_root: str
    agent_id: str | None = None
    execution_context: ProviderExecutionContext | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ProviderArtifactSnapshot:
    manifest: ProviderArtifactManifest
    captured_at: str
    snapshot_root: str


@dataclass(frozen=True)
class ArtifactRestoreRequest:
    manifest: ProviderArtifactManifest
    snapshot_root: str
    target_home_id: str | None = None
    execution_context: ProviderExecutionContext | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class ProviderArtifactRestoreResult:
    restored: bool
    restored_at: str
    warnings: tuple[str, ...] = ()
    provider_payload: ProviderPayload | None = None
