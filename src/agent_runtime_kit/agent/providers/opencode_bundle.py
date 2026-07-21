from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Mapping

from ..provider_contracts import (
    AgentProviderBundle,
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    ModelBackendIdentity,
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderExecutionKind,
    ProviderHomeKind,
)
from .opencode_artifacts import OpenCodeArtifactAdapter
from .opencode_context import OpenCodeContextAdapter
from .opencode_home import OpenCodeHomeRenderer
from .opencode_models import ADAPTER_VERSION, PROVIDER_TYPE
from .opencode_query import OpenCodeQueryAdapter
from .opencode_runtime import OpenCodeRuntimeAdapter, OpenCodeRuntimeRegistry


class OpenCodeCapabilityResolver:
    provider_type = PROVIDER_TYPE

    def __init__(self, base: ProviderCapabilities) -> None:
        self.base = base

    def resolve_capabilities(
        self,
        home: object,
        model_backend: ModelBackendIdentity | None = None,
    ) -> ProviderCapabilities:
        backend = model_backend or _home_backend(home)
        home_id = str(getattr(home, "home_id", "")) or None
        backend_key = backend.backend_key if backend is not None else None
        supports = {
            key: replace(
                value,
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
            )
            for key, value in self.base.supports.items()
            if key not in _MODEL_CAPABILITIES
        }
        for capability, mode in (
            (CapabilityKey.MODEL_RESPONSES, "responses"),
            (CapabilityKey.MODEL_CHAT_COMPLETIONS, "chat_completions"),
        ):
            available = backend is not None and backend.api_mode == mode
            supports[capability] = CapabilitySupport(
                capability=capability,
                status=CapabilityStatus.NATIVE if available else CapabilityStatus.UNSUPPORTED,
                available=available,
                reason=None if available else f"effective OpenCode backend does not use {mode}",
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
                evidence_version="opencode-home-backend-v1",
            )
        known = backend is not None and backend.api_mode in {"responses", "chat_completions"}
        supports[CapabilityKey.MODEL_OTHER_API] = CapabilitySupport(
            capability=CapabilityKey.MODEL_OTHER_API,
            status=CapabilityStatus.UNSUPPORTED if known else CapabilityStatus.UNVERIFIED,
            available=False,
            reason="effective backend API mode is known" if known else "backend API mode is not verified",
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
            evidence_version="opencode-home-backend-v1",
        )
        compact_available = backend is not None and backend.effective_model is not None
        supports[CapabilityKey.CONTROL_COMPACT] = CapabilitySupport(
            capability=CapabilityKey.CONTROL_COMPACT,
            status=CapabilityStatus.NATIVE if compact_available else CapabilityStatus.UNVERIFIED,
            available=compact_available,
            reason=None if compact_available else "OpenCode summarize needs a resolved model backend",
            limitations=("uses OpenCode model-backed summarize, not Responses native compact",),
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
            evidence_version="opencode-1.18.4",
        )
        return ProviderCapabilities(
            provider_type=PROVIDER_TYPE,
            supports=supports,
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
        )


def build_opencode_provider_bundle(
    *,
    runtime_root: Path,
    binary_path: str | Path = "opencode",
) -> AgentProviderBundle:
    native = {
        CapabilityKey.HOME_BASE_CONFIG,
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_RAW_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_INSTRUCTIONS,
        CapabilityKey.HOME_SKILLS,
        CapabilityKey.HOME_MCP,
        CapabilityKey.SESSION_CREATE,
        CapabilityKey.SESSION_RESUME,
        CapabilityKey.SESSION_READ,
        CapabilityKey.RUN_STREAM,
        CapabilityKey.RUN_INTERRUPT,
        CapabilityKey.CONTROL_APPROVAL_RESPONSE,
        CapabilityKey.CONTROL_INPUT_RESPONSE,
        CapabilityKey.CONTROL_FORK,
        CapabilityKey.QUERY_TURNS,
        CapabilityKey.QUERY_EVENTS,
        CapabilityKey.QUERY_CONTENT,
        CapabilityKey.QUERY_TOOL_CALLS,
        CapabilityKey.QUERY_REQUEST_USAGE,
        CapabilityKey.QUERY_SESSION_USAGE,
        CapabilityKey.QUERY_CONTEXT_USAGE,
    }
    adaptable = {
        CapabilityKey.RUN_WAIT_TERMINAL: ("ARK confirms armed + turn evidence + live idle",),
        CapabilityKey.ARTIFACT_SNAPSHOT: ("SQLite online backup plus referenced data files",),
        CapabilityKey.ARTIFACT_RESTORE: ("same runtime root and Agent path are required",),
        CapabilityKey.ARTIFACT_CACHE_REBUILD: ("WAL/SHM are discarded and rebuilt",),
    }
    supports = {
        key: CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            evidence_version="opencode-1.18.4",
        )
        for key in native
    }
    supports.update(
        {
            key: CapabilitySupport(
                capability=key,
                status=CapabilityStatus.ADAPTABLE,
                available=True,
                limitations=limitations,
                evidence_version="opencode-adapter-v1",
            )
            for key, limitations in adaptable.items()
        }
    )
    supports[CapabilityKey.CONTROL_FORK_FROM_TURN] = CapabilitySupport(
        capability=CapabilityKey.CONTROL_FORK_FROM_TURN,
        status=CapabilityStatus.UNSUPPORTED,
        available=False,
        reason="first version only forks the latest complete session",
        evidence_version="opencode-adapter-v1",
    )
    supports[CapabilityKey.ARTIFACT_IN_FLIGHT_STATE] = CapabilitySupport(
        capability=CapabilityKey.ARTIFACT_IN_FLIGHT_STATE,
        status=CapabilityStatus.UNSUPPORTED,
        available=False,
        reason="OpenCode permission/question state is process memory and not snapshot-safe",
        evidence_version="opencode-1.18.4",
    )
    supports[CapabilityKey.ARTIFACT_OFFLINE_QUERY] = CapabilitySupport(
        capability=CapabilityKey.ARTIFACT_OFFLINE_QUERY,
        status=CapabilityStatus.UNSUPPORTED,
        available=False,
        reason="first version queries through a live isolated OpenCode server",
        evidence_version="opencode-adapter-v1",
    )
    capabilities = ProviderCapabilities(provider_type=PROVIDER_TYPE, supports=supports)
    registry = OpenCodeRuntimeRegistry(runtime_root, binary_path=binary_path)
    runtime = OpenCodeRuntimeAdapter(registry)
    query = OpenCodeQueryAdapter(registry.client_for_locator)
    return AgentProviderBundle(
        descriptor=ProviderDescriptor(
            provider_type=PROVIDER_TYPE,
            display_name="OpenCode",
            adapter_version=ADAPTER_VERSION,
            execution_kind=ProviderExecutionKind.SUBPROCESS_RPC,
            home_kind=ProviderHomeKind.NATIVE,
            sdk_or_cli_name="opencode",
            sdk_or_cli_version="1.18.4",
            supported_api_modes=("responses", "chat_completions"),
            static_capabilities=capabilities,
        ),
        runtime=runtime,
        home_renderer=OpenCodeHomeRenderer(runtime_root=runtime_root),
        capability_resolver=OpenCodeCapabilityResolver(capabilities),
        query=query,
        context=OpenCodeContextAdapter(registry=registry, query=query),
        artifacts=OpenCodeArtifactAdapter(runtime_root=runtime_root, registry=registry),
    )


_MODEL_CAPABILITIES = {
    CapabilityKey.MODEL_RESPONSES,
    CapabilityKey.MODEL_CHAT_COMPLETIONS,
    CapabilityKey.MODEL_OTHER_API,
}


def _home_backend(home: object) -> ModelBackendIdentity | None:
    value = getattr(home, "resolved_defaults", None)
    if not isinstance(value, Mapping):
        return None
    api_provider = value.get("api_provider")
    api_mode = value.get("api_mode")
    if not isinstance(api_provider, str) or not isinstance(api_mode, str):
        return None
    return ModelBackendIdentity(
        api_provider=api_provider,
        api_mode=api_mode,
        endpoint_id=value.get("endpoint_id"),
        requested_model=value.get("requested_model"),
        resolved_model=value.get("resolved_model"),
        model_version=value.get("model_version"),
    )
