from __future__ import annotations

from dataclasses import dataclass

from .identities import ModelBackendIdentity


@dataclass(frozen=True)
class ProviderSessionLocator:
    provider_type: str
    session_id: str
    home_id: str
    created_at: str
    backend_identity: ModelBackendIdentity | None = None
    native_locator: object | None = None

    def __post_init__(self) -> None:
        if not self.provider_type.strip():
            raise ValueError("provider_type must not be empty")
        if not self.session_id.strip():
            raise ValueError("session_id must not be empty")
        if not self.home_id.strip():
            raise ValueError("home_id must not be empty")


@dataclass(frozen=True)
class ProviderTurnLocator:
    session: ProviderSessionLocator
    turn_id: str
    request_ids: tuple[str, ...] = ()
    sequence: int | None = None

    def __post_init__(self) -> None:
        if not self.turn_id.strip():
            raise ValueError("turn_id must not be empty")
        if self.sequence is not None and self.sequence < 0:
            raise ValueError("sequence must not be negative")


@dataclass(frozen=True)
class AgentArtifactLocator:
    provider_type: str
    home_id: str
    session_id: str
    adapter_version: str
    manifest_relpath: str | None = None
    native_primary_ref: str | None = None

    def __post_init__(self) -> None:
        for name in ("provider_type", "home_id", "session_id", "adapter_version"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")
