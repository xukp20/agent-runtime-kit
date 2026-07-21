from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .capabilities import ProviderCapabilities
from .identities import ModelBackendIdentity, ProviderPayload


@dataclass(frozen=True)
class BaseConfigSource:
    path: str | None = None
    text: str | None = None
    mapping: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        selected = sum(value is not None for value in (self.path, self.text, self.mapping))
        if selected > 1:
            raise ValueError("base config must use exactly one source kind")


@dataclass(frozen=True)
class ProviderHomeSpec:
    provider_type: str
    home_id: str
    base_config: BaseConfigSource | None = None
    config_overrides: Mapping[str, object] = field(default_factory=dict)
    model_config: ModelBackendIdentity | None = None
    instructions: tuple[object, ...] = ()
    skills: tuple[object, ...] = ()
    mcp_servers: tuple[object, ...] = ()
    tools: tuple[object, ...] = ()
    extensions: tuple[object, ...] = ()
    auth_refs: tuple[str, ...] = ()
    fixed_env: Mapping[str, str] = field(default_factory=dict, repr=False)
    fixed_env_refs: Mapping[str, str] = field(default_factory=dict)
    required_env: tuple[str, ...] = ()
    workdir_policy: object | None = None
    provider_options: object | None = None

    def __post_init__(self) -> None:
        if not self.provider_type.strip() or not self.home_id.strip():
            raise ValueError("provider_type and home_id must not be empty")


@dataclass(frozen=True)
class HomeValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class HomeMaterializedFile:
    relpath: str
    sha256: str
    source_fingerprint: str | None = None
    secret: bool = False


@dataclass(frozen=True)
class HomeMaterializationResult:
    provider_type: str
    home_id: str
    renderer_version: str
    manifest_schema_version: int
    manifest_hash: str
    generated_files: tuple[HomeMaterializedFile, ...] = ()
    source_resource_hashes: Mapping[str, str] = field(default_factory=dict)
    required_env: tuple[str, ...] = ()
    auth_refs: tuple[str, ...] = ()
    resolved_defaults: ModelBackendIdentity | None = None
    warnings: tuple[str, ...] = ()
    effective_capabilities: ProviderCapabilities | None = None
    provider_payload: ProviderPayload | None = None


@dataclass(frozen=True)
class HomeInitializationResult:
    initialized: bool
    marker_ref: str | None = None
    warnings: tuple[str, ...] = ()
    provider_payload: ProviderPayload | None = None


@dataclass(frozen=True)
class ProviderExecutionContext:
    provider_type: str
    home_id: str
    home_root: Path
    process_environment: Mapping[str, str] = field(repr=False)
    materialization_manifest: HomeMaterializationResult | None = None
    workdir: str | None = None
    resolved_defaults: ModelBackendIdentity | None = None
    resource_handles: tuple[object, ...] = field(default_factory=tuple, repr=False, compare=False)
    runtime_payload: object | None = field(default=None, repr=False, compare=False)
