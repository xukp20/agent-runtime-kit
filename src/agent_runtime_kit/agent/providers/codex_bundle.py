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
from .codex import CodexProvider
from .codex_artifacts import CodexArtifactAdapter
from .codex_context_adapter import CodexContextAdapter
from .codex_home import CodexHomeRenderer
from .codex_query import CodexQueryAdapter
from .codex_runtime import CodexRuntimeAdapter


class CodexCapabilityResolver:
    provider_type = "codex"

    def __init__(self, base: ProviderCapabilities) -> None:
        self.base = base

    def resolve_capabilities(
        self,
        home: object,
        model_backend: ModelBackendIdentity | None = None,
    ) -> ProviderCapabilities:
        backend = model_backend or _home_model_backend(home)
        home_id = str(getattr(home, "home_id", "")) or None
        backend_key = backend.backend_key
        supports = {
            key: replace(
                support,
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
            )
            for key, support in self.base.supports.items()
            if key
            not in {
                CapabilityKey.MODEL_RESPONSES,
                CapabilityKey.MODEL_CHAT_COMPLETIONS,
                CapabilityKey.MODEL_OTHER_API,
            }
        }
        model_modes = {
            CapabilityKey.MODEL_RESPONSES: "responses",
            CapabilityKey.MODEL_CHAT_COMPLETIONS: "chat_completions",
        }
        for capability, api_mode in model_modes.items():
            available = backend.api_mode == api_mode
            supports[capability] = CapabilitySupport(
                capability=capability,
                status=CapabilityStatus.NATIVE if available else CapabilityStatus.UNSUPPORTED,
                available=available,
                reason=None if available else f"effective backend uses {backend.api_mode}",
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
                evidence_version="codex-home-backend-v1",
            )
        known_mode = backend.api_mode in set(model_modes.values())
        supports[CapabilityKey.MODEL_OTHER_API] = CapabilitySupport(
            capability=CapabilityKey.MODEL_OTHER_API,
            status=CapabilityStatus.UNSUPPORTED if known_mode else CapabilityStatus.UNVERIFIED,
            available=False,
            reason=(
                f"effective backend uses verified Codex API mode {backend.api_mode}"
                if known_mode
                else f"Codex adapter has not verified API mode {backend.api_mode}"
            ),
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
            evidence_version="codex-home-backend-v1",
        )
        return ProviderCapabilities(
            provider_type="codex",
            supports=supports,
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
        )


def _home_model_backend(home: object) -> ModelBackendIdentity:
    payload = getattr(home, "resolved_defaults", None)
    if isinstance(payload, Mapping):
        api_provider = payload.get("api_provider")
        api_mode = payload.get("api_mode")
        if isinstance(api_provider, str) and isinstance(api_mode, str):
            return ModelBackendIdentity(
                api_provider=api_provider,
                api_mode=api_mode,
                endpoint_id=payload.get("endpoint_id"),
                requested_model=payload.get("requested_model"),
                resolved_model=payload.get("resolved_model"),
                model_version=payload.get("model_version"),
                service_tier=payload.get("service_tier"),
                reasoning_effort=payload.get("reasoning_effort"),
                tokenizer_id=payload.get("tokenizer_id"),
                model_config_hash=payload.get("model_config_hash"),
            )
    return ModelBackendIdentity(api_provider="openai", api_mode="responses")


def build_codex_provider_bundle(
    provider: CodexProvider,
    *,
    runtime_root: Path,
) -> AgentProviderBundle:
    native = {
        CapabilityKey.HOME_BASE_CONFIG,
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_RAW_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_SKILLS,
        CapabilityKey.HOME_MCP,
        CapabilityKey.SESSION_CREATE,
        CapabilityKey.SESSION_RESUME,
        CapabilityKey.SESSION_READ,
        CapabilityKey.RUN_STREAM,
        CapabilityKey.CONTROL_FORK,
        CapabilityKey.MODEL_RESPONSES,
    }
    adaptable: dict[CapabilityKey, tuple[str, ...]] = {
        CapabilityKey.SESSION_LIST: (
            "projected from idle Codex rollout files under one isolated Home",
        ),
        CapabilityKey.RUN_WAIT_TERMINAL: (
            "ARK waits for a terminal SDK event before reporting completion",
        ),
        CapabilityKey.RUN_INTERRUPT: (
            "ARK resolves the active-turn race and confirms terminal interruption",
        ),
        CapabilityKey.QUERY_TURNS: ("projected from Codex rollout JSONL",),
        CapabilityKey.QUERY_EVENTS: ("projected from Codex rollout JSONL",),
        CapabilityKey.QUERY_CONTENT: ("projected from Codex rollout JSONL",),
        CapabilityKey.QUERY_TOOL_CALLS: ("projected from Codex rollout JSONL",),
        CapabilityKey.QUERY_SESSION_USAGE: (
            "aggregated only from complete turn-level usage available in rollout JSONL",
        ),
        CapabilityKey.QUERY_CONTEXT_USAGE: (
            "ARK reconciles Codex token events into latest non-cumulative context usage",
        ),
        CapabilityKey.CONTROL_COMPACT: (
            "ARK adds an idle/terminal barrier and maintenance reconciliation",
        ),
        CapabilityKey.ARTIFACT_OFFLINE_QUERY: ("ARK parses Codex rollout JSONL",),
        CapabilityKey.ARTIFACT_SNAPSHOT: (
            "ARK copies the stable single-session rollout selected by the Codex adapter",
        ),
        CapabilityKey.ARTIFACT_RESTORE: (
            "ARK restores the selected rollout and invalidates rebuildable Codex caches",
        ),
        CapabilityKey.ARTIFACT_CACHE_REBUILD: (
            "Codex rebuilds discarded state indexes from restored rollout data",
        ),
    }
    supports = {
        key: CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            evidence_version="codex-sdk-turn-handle-v1",
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
                evidence_version="codex-adapter-v1",
            )
            for key, limitations in adaptable.items()
        }
    )
    supports[CapabilityKey.QUERY_REQUEST_USAGE] = CapabilitySupport(
        capability=CapabilityKey.QUERY_REQUEST_USAGE,
        status=CapabilityStatus.UNSUPPORTED,
        available=False,
        reason=(
            "current Codex rollout evidence exposes turn/session aggregates but no "
            "authoritative per-request usage entries"
        ),
        evidence_version="codex-adapter-v1",
    )
    capabilities = ProviderCapabilities(
        provider_type="codex",
        supports=supports,
    )
    return AgentProviderBundle(
        descriptor=ProviderDescriptor(
            provider_type="codex",
            display_name="OpenAI Codex",
            adapter_version="1",
            execution_kind=ProviderExecutionKind.SDK,
            home_kind=ProviderHomeKind.NATIVE,
            sdk_or_cli_name="openai_codex",
            supported_api_modes=("responses", "chat_completions"),
            static_capabilities=capabilities,
        ),
        runtime=CodexRuntimeAdapter(provider),
        home_renderer=CodexHomeRenderer(runtime_root=runtime_root, provider=provider),
        capability_resolver=CodexCapabilityResolver(capabilities),
        query=CodexQueryAdapter(runtime_root=runtime_root, provider=provider),
        context=CodexContextAdapter(provider),
        artifacts=CodexArtifactAdapter(runtime_root=runtime_root, provider=provider),
    )
