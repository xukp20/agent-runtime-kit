from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPayload:
    provider_type: str
    payload_type: str
    payload_schema_version: int = 1
    adapter_version: str | None = None
    sdk_or_cli_version: str | None = None
    sanitized_data: object | None = None
    truncated: bool = False

    def __post_init__(self) -> None:
        if not self.provider_type.strip():
            raise ValueError("provider_type must not be empty")
        if not self.payload_type.strip():
            raise ValueError("payload_type must not be empty")
        if self.payload_schema_version < 1:
            raise ValueError("payload_schema_version must be positive")


@dataclass(frozen=True)
class ModelBackendIdentity:
    api_provider: str
    api_mode: str
    endpoint_id: str | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    model_version: str | None = None
    service_tier: str | None = None
    reasoning_effort: str | None = None
    tokenizer_id: str | None = None
    model_config_hash: str | None = None
    provider_payload: ProviderPayload | None = None

    def __post_init__(self) -> None:
        if not self.api_provider.strip():
            raise ValueError("api_provider must not be empty")
        if not self.api_mode.strip():
            raise ValueError("api_mode must not be empty")

    @property
    def effective_model(self) -> str | None:
        return self.resolved_model or self.requested_model

    @property
    def backend_key(self) -> str:
        endpoint = f":{self.endpoint_id}" if self.endpoint_id else ""
        return f"{self.api_provider}:{self.api_mode}{endpoint}"
