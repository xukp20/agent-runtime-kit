from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum

from .identities import ProviderPayload


_SECRET_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
)

_SECRET_TOKEN_KEYS = {
    "token",
    "access_token",
    "auth_token",
    "bearer_token",
    "id_token",
    "refresh_token",
}


def sanitize_provider_data(value: object, *, max_string_chars: int = 4096) -> tuple[object, bool]:
    """Return a bounded, JSON-shaped payload with secret-looking values removed."""

    truncated = False

    def visit(item: object) -> object:
        nonlocal truncated
        if isinstance(item, Enum):
            return visit(item.value)
        if is_dataclass(item) and not isinstance(item, type):
            return visit(asdict(item))
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            return visit(model_dump(mode="json", by_alias=False))
        if isinstance(item, Mapping):
            result: dict[str, object] = {}
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                lowered = key.lower()
                if lowered in _SECRET_TOKEN_KEYS or any(fragment in lowered for fragment in _SECRET_FRAGMENTS):
                    result[key] = "[REDACTED]"
                else:
                    result[key] = visit(raw_value)
            return result
        if isinstance(item, str):
            if len(item) > max_string_chars:
                truncated = True
                return item[:max_string_chars] + "…"
            return item
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [visit(child) for child in item]
        if item is None or isinstance(item, (bool, int, float)):
            return item
        truncated = True
        return repr(item)[:max_string_chars]

    return visit(value), truncated


def build_provider_payload(
    *,
    provider_type: str,
    payload_type: str,
    data: object,
    adapter_version: str | None = None,
    sdk_or_cli_version: str | None = None,
    payload_schema_version: int = 1,
    max_string_chars: int = 4096,
) -> ProviderPayload:
    sanitized, truncated = sanitize_provider_data(data, max_string_chars=max_string_chars)
    return ProviderPayload(
        provider_type=provider_type,
        payload_type=payload_type,
        payload_schema_version=payload_schema_version,
        adapter_version=adapter_version,
        sdk_or_cli_version=sdk_or_cli_version,
        sanitized_data=sanitized,
        truncated=truncated,
    )
