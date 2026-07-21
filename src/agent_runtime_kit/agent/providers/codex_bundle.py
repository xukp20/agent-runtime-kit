from __future__ import annotations

from pathlib import Path

from ..provider_contracts import (
    AgentProviderBundle,
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    ProviderCapabilities,
    ProviderDescriptor,
    ProviderExecutionKind,
    ProviderHomeKind,
    ProviderRunHandle,
    ProviderTurnResult,
)
from .codex import CodexProvider
from .codex_artifacts import CodexArtifactAdapter
from .codex_context_adapter import CodexContextAdapter
from .codex_home import CodexHomeRenderer
from .codex_query import CodexQueryAdapter
from .codex_runtime import CodexProviderRunHandle, CodexRuntimeAdapter


class CodexCompatibilityBridge:
    provider_type = "codex"

    def completion_turn_result(
        self,
        handle: ProviderRunHandle,
        result: ProviderTurnResult,
    ) -> object:
        del result
        if not isinstance(handle, CodexProviderRunHandle):
            raise TypeError("Codex compatibility bridge received a non-Codex run handle")
        legacy = handle.legacy_turn_result
        if legacy is None:
            raise RuntimeError("Codex run completed without a legacy turn result")
        return legacy


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
            supported_api_modes=("responses",),
            static_capabilities=capabilities,
        ),
        runtime=CodexRuntimeAdapter(provider),
        home_renderer=CodexHomeRenderer(runtime_root=runtime_root, provider=provider),
        query=CodexQueryAdapter(runtime_root=runtime_root, provider=provider),
        context=CodexContextAdapter(provider),
        artifacts=CodexArtifactAdapter(runtime_root=runtime_root, provider=provider),
        compatibility=CodexCompatibilityBridge(),
    )
