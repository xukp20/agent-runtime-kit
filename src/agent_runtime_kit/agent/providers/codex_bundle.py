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
        CapabilityKey.RUN_WAIT_TERMINAL,
        CapabilityKey.RUN_INTERRUPT,
        CapabilityKey.QUERY_TURNS,
        CapabilityKey.QUERY_EVENTS,
        CapabilityKey.QUERY_CONTENT,
        CapabilityKey.QUERY_TOOL_CALLS,
        CapabilityKey.QUERY_REQUEST_USAGE,
        CapabilityKey.QUERY_SESSION_USAGE,
        CapabilityKey.ARTIFACT_OFFLINE_QUERY,
        CapabilityKey.MODEL_RESPONSES,
    }
    capabilities = ProviderCapabilities(
        provider_type="codex",
        supports={
            key: CapabilitySupport(
                capability=key,
                status=CapabilityStatus.NATIVE,
                available=True,
                evidence_version="codex-sdk-turn-handle-v1",
            )
            for key in native
        },
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
        compatibility=CodexCompatibilityBridge(),
    )
