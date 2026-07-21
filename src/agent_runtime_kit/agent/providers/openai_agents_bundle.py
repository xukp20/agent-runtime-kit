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
from .openai_agents import OpenAIAgentsResourceRegistry
from .openai_agents_artifacts import OpenAIAgentsArtifactAdapter
from .openai_agents_context import OpenAIAgentsContextAdapter
from .openai_agents_home import OpenAIAgentsHomeRenderer
from .openai_agents_query import OpenAIAgentsQueryAdapter
from .openai_agents_runtime import OpenAIAgentsRuntimeAdapter


class OpenAIAgentsCapabilityResolver:
    provider_type = "openai_agents"

    def __init__(self, base: ProviderCapabilities) -> None:
        self.base = base

    def resolve_capabilities(self, home: object, model_backend: ModelBackendIdentity | None = None) -> ProviderCapabilities:
        identity = model_backend or _home_identity(home)
        home_id = str(getattr(home, "home_id", "")) or None
        backend = identity.backend_key if identity else None
        values = {
            key: replace(item, resolved_for_home_id=home_id, resolved_for_backend=backend)
            for key, item in self.base.supports.items()
        }
        for capability, mode in (
            (CapabilityKey.MODEL_RESPONSES, "responses"),
            (CapabilityKey.MODEL_CHAT_COMPLETIONS, "chat_completions"),
        ):
            available = identity is not None and identity.api_mode == mode
            values[capability] = CapabilitySupport(
                capability=capability,
                status=CapabilityStatus.NATIVE if available else CapabilityStatus.UNSUPPORTED,
                available=available,
                reason=None if available else f"effective backend does not use {mode}",
                resolved_for_home_id=home_id,
                resolved_for_backend=backend,
                evidence_version="openai-agents-0.18.3-model-wrapper-v1",
            )
        compact_mode = _home_provider_value(home, "compaction_mode")
        compact = identity is not None and identity.api_mode == "responses" and compact_mode == "input_history"
        values[CapabilityKey.CONTROL_COMPACT] = CapabilitySupport(
            capability=CapabilityKey.CONTROL_COMPACT,
            status=CapabilityStatus.ADAPTABLE if compact else CapabilityStatus.UNSUPPORTED,
            available=compact,
            reason=None if compact else "requires verified Responses input-history compaction",
            limitations=("previous_response_id compaction is unsupported",) if compact else (),
            resolved_for_home_id=home_id,
            resolved_for_backend=backend,
            evidence_version="openai-agents-0.18.3-input-history-v1",
        )
        return ProviderCapabilities(
            provider_type=self.provider_type,
            supports=values,
            resolved_for_home_id=home_id,
            resolved_for_backend=backend,
        )


def build_openai_agents_provider_bundle(*, runtime_root: Path, registry: OpenAIAgentsResourceRegistry) -> AgentProviderBundle:
    native = {
        CapabilityKey.SESSION_CREATE, CapabilityKey.SESSION_RESUME,
        CapabilityKey.RUN_STREAM,
    }
    ark_owned = {
        CapabilityKey.HOME_BASE_CONFIG, CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_RAW_OVERRIDES, CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS, CapabilityKey.HOME_INSTRUCTIONS,
        CapabilityKey.HOME_SKILLS, CapabilityKey.HOME_MCP,
        CapabilityKey.SESSION_READ, CapabilityKey.SESSION_LIST,
        CapabilityKey.RUN_WAIT_TERMINAL, CapabilityKey.RUN_INTERRUPT,
        CapabilityKey.RUN_CANCEL, CapabilityKey.CONTROL_FORK,
        CapabilityKey.CONTROL_APPROVAL_RESPONSE,
        CapabilityKey.QUERY_TURNS, CapabilityKey.QUERY_EVENTS,
        CapabilityKey.QUERY_CONTENT, CapabilityKey.QUERY_TOOL_CALLS,
        CapabilityKey.QUERY_REQUEST_USAGE, CapabilityKey.QUERY_SESSION_USAGE,
        CapabilityKey.QUERY_CONTEXT_USAGE, CapabilityKey.ARTIFACT_OFFLINE_QUERY,
        CapabilityKey.ARTIFACT_SNAPSHOT, CapabilityKey.ARTIFACT_RESTORE,
    }
    supports = {
        key: CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            evidence_version="openai-agents-0.18.3-v1",
        )
        for key in native
    }
    supports.update(
        {
            key: CapabilitySupport(
                capability=key,
                status=CapabilityStatus.ARK_OWNED,
                available=True,
                evidence_version="ark-openai-agents-adapter-v1",
            )
            for key in ark_owned
        }
    )
    supports[CapabilityKey.CONTROL_APPROVAL_RESPONSE] = CapabilitySupport(
        capability=CapabilityKey.CONTROL_APPROVAL_RESPONSE,
        status=CapabilityStatus.ARK_OWNED,
        available=True,
        limitations=(
            "durable provider control is available directly; the current AgentService "
            "does not yet expose a provider-neutral approval-response method",
        ),
        evidence_version="ark-openai-agents-run-state-v1",
    )
    capabilities = ProviderCapabilities(provider_type="openai_agents", supports=supports)
    return AgentProviderBundle(
        descriptor=ProviderDescriptor(
            provider_type="openai_agents",
            display_name="OpenAI Agents Python SDK",
            adapter_version="1",
            execution_kind=ProviderExecutionKind.SDK,
            home_kind=ProviderHomeKind.ARK_OWNED,
            sdk_or_cli_name="openai-agents",
            sdk_or_cli_version="0.18.3",
            supported_api_modes=("responses", "chat_completions"),
            static_capabilities=capabilities,
        ),
        runtime=OpenAIAgentsRuntimeAdapter(runtime_root=runtime_root, registry=registry),
        home_renderer=OpenAIAgentsHomeRenderer(runtime_root=runtime_root, registry=registry),
        capability_resolver=OpenAIAgentsCapabilityResolver(capabilities),
        query=OpenAIAgentsQueryAdapter(runtime_root=runtime_root),
        context=OpenAIAgentsContextAdapter(),
        artifacts=OpenAIAgentsArtifactAdapter(runtime_root=runtime_root),
    )


def _home_identity(home: object) -> ModelBackendIdentity | None:
    raw = getattr(home, "resolved_defaults", None)
    if not isinstance(raw, Mapping):
        return None
    return ModelBackendIdentity(
        api_provider=str(raw["api_provider"]),
        api_mode=str(raw["api_mode"]),
        endpoint_id=str(raw["endpoint_id"]) if raw.get("endpoint_id") is not None else None,
        requested_model=str(raw["requested_model"]) if raw.get("requested_model") is not None else None,
        resolved_model=str(raw["resolved_model"]) if raw.get("resolved_model") is not None else None,
    )


def _home_provider_value(home: object, key: str) -> object | None:
    payload = getattr(home, "provider_payload", None)
    if isinstance(payload, Mapping):
        data = payload.get("sanitized_data", payload)
        if isinstance(data, Mapping):
            return data.get(key)
    return None
