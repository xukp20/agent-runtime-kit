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
from .pi_artifacts import PiArtifactAdapter
from .pi_context import PiContextAdapter
from .pi_home import PiHomeRenderer
from .pi_query import PiQueryAdapter
from .pi_runtime import PiRuntimeAdapter
from .pi_session import PI_CLI_VERSION


class PiCapabilityResolver:
    provider_type = "pi"

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
                support,
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
            )
            for key, support in self.base.supports.items()
            if key not in {
                CapabilityKey.MODEL_RESPONSES,
                CapabilityKey.MODEL_CHAT_COMPLETIONS,
                CapabilityKey.MODEL_OTHER_API,
            }
        }
        for key, mode in (
            (CapabilityKey.MODEL_RESPONSES, "responses"),
            (CapabilityKey.MODEL_CHAT_COMPLETIONS, "chat_completions"),
        ):
            available = backend is not None and backend.api_mode == mode
            supports[key] = CapabilitySupport(
                capability=key,
                status=CapabilityStatus.NATIVE if available else CapabilityStatus.UNSUPPORTED,
                available=available,
                reason=None if available else f"effective Pi backend does not use {mode}",
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
                evidence_version="pi-home-backend-v1",
            )
        messages = backend is not None and backend.api_mode == "messages"
        known = backend is not None and backend.api_mode in {
            "responses",
            "chat_completions",
            "messages",
        }
        supports[CapabilityKey.MODEL_OTHER_API] = CapabilitySupport(
            capability=CapabilityKey.MODEL_OTHER_API,
            status=(
                CapabilityStatus.NATIVE
                if messages
                else CapabilityStatus.UNSUPPORTED
                if known
                else CapabilityStatus.UNVERIFIED
            ),
            available=messages,
            reason=(
                None
                if messages
                else "effective Pi backend is unresolved or uses a non-verified API mode"
            ),
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
            evidence_version="pi-home-backend-v1",
        )
        return ProviderCapabilities(
            provider_type="pi",
            supports=supports,
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
        )


def build_pi_provider_bundle(*, runtime_root: Path) -> AgentProviderBundle:
    native = {
        CapabilityKey.HOME_BASE_CONFIG,
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_RAW_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_INSTRUCTIONS,
        CapabilityKey.HOME_SKILLS,
        CapabilityKey.HOME_EXTENSIONS,
        CapabilityKey.SESSION_CREATE,
        CapabilityKey.SESSION_RESUME,
        CapabilityKey.SESSION_READ,
        CapabilityKey.RUN_STREAM,
        CapabilityKey.RUN_WAIT_TERMINAL,
        CapabilityKey.RUN_INTERRUPT,
        CapabilityKey.RUN_STEER,
        CapabilityKey.RUN_FOLLOW_UP,
        CapabilityKey.CONTROL_FORK,
        CapabilityKey.CONTROL_FORK_FROM_TURN,
        CapabilityKey.CONTROL_COMPACT,
    }
    adaptable = {
        CapabilityKey.SESSION_LIST: ("projected from one isolated Pi Home session directory",),
        CapabilityKey.QUERY_TURNS: ("projected from Pi v3 session JSONL active branch",),
        CapabilityKey.QUERY_EVENTS: ("projected from Pi RPC events and session entries",),
        CapabilityKey.QUERY_CONTENT: ("projected from Pi v3 message entries",),
        CapabilityKey.QUERY_TOOL_CALLS: ("paired from assistant toolCall and toolResult entries",),
        CapabilityKey.QUERY_REQUEST_USAGE: ("projected from persisted assistant/summary usage",),
        CapabilityKey.QUERY_SESSION_USAGE: ("aggregated from all persisted Pi usage entries",),
        CapabilityKey.QUERY_CONTEXT_USAGE: ("Pi native stats include an estimated trailing context projection",),
        CapabilityKey.ARTIFACT_OFFLINE_QUERY: ("ARK parses Pi v3 JSONL without model access",),
        CapabilityKey.ARTIFACT_SNAPSHOT: ("ARK copies one idle single-session JSONL",),
        CapabilityKey.ARTIFACT_RESTORE: ("ARK restores the required session JSONL by stable session id",),
    }
    supports = {
        key: CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            limitations=(
                ("Pi compact is agent-owned history summarization, not backend responses.compact",)
                if key is CapabilityKey.CONTROL_COMPACT
                else ()
            ),
            evidence_version="pi-rpc-0.80.10",
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
                evidence_version="pi-adapter-v1",
            )
            for key, limitations in adaptable.items()
        }
    )
    supports[CapabilityKey.HOME_MCP] = CapabilitySupport(
        capability=CapabilityKey.HOME_MCP,
        status=CapabilityStatus.ARK_OWNED,
        available=True,
        requirements=("prepared @modelcontextprotocol/sdk runtime",),
        limitations=(
            "Pi 0.80.10 cannot unregister a projected MCP tool mid-session; removed tools fail closed until the next session",
        ),
        evidence_version="pi-mcp-bridge-v1",
    )
    for key, reason in (
        (CapabilityKey.ARTIFACT_IN_FLIGHT_STATE, "Pi snapshot requires an idle session"),
        (CapabilityKey.CONTROL_APPROVAL_RESPONSE, "Pi extension approval response is not connected in v1"),
        (CapabilityKey.CONTROL_INPUT_RESPONSE, "Pi extension UI input response is not connected in v1"),
    ):
        supports[key] = CapabilitySupport(
            capability=key,
            status=CapabilityStatus.UNSUPPORTED,
            available=False,
            reason=reason,
            evidence_version="pi-adapter-v1",
        )
    capabilities = ProviderCapabilities(provider_type="pi", supports=supports)
    runtime = PiRuntimeAdapter(runtime_root=runtime_root)
    return AgentProviderBundle(
        descriptor=ProviderDescriptor(
            provider_type="pi",
            display_name="Pi Coding Agent",
            adapter_version="1",
            execution_kind=ProviderExecutionKind.SUBPROCESS_RPC,
            home_kind=ProviderHomeKind.NATIVE,
            sdk_or_cli_name="@earendil-works/pi-coding-agent",
            sdk_or_cli_version=PI_CLI_VERSION,
            supported_api_modes=("responses", "chat_completions", "messages"),
            static_capabilities=capabilities,
        ),
        runtime=runtime,
        home_renderer=PiHomeRenderer(runtime_root=runtime_root),
        capability_resolver=PiCapabilityResolver(capabilities),
        query=PiQueryAdapter(runtime_root=runtime_root),
        context=PiContextAdapter(),
        artifacts=PiArtifactAdapter(runtime_root=runtime_root, active_sessions=runtime),
    )


def _home_backend(home: object) -> ModelBackendIdentity | None:
    payload = getattr(home, "resolved_defaults", None)
    if not isinstance(payload, Mapping):
        return None
    provider = payload.get("api_provider")
    mode = payload.get("api_mode")
    if not isinstance(provider, str) or not isinstance(mode, str):
        return None
    return ModelBackendIdentity(
        api_provider=provider,
        api_mode=mode,
        endpoint_id=payload.get("endpoint_id"),
        requested_model=payload.get("requested_model"),
        resolved_model=payload.get("resolved_model"),
        model_config_hash=payload.get("model_config_hash"),
    )
