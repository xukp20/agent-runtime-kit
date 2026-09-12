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
)
from .grok_artifacts import GrokArtifactAdapter
from .grok_home import GROK_CLI_VERSION, GrokHomeRenderer
from .grok_runtime import GrokRuntimeAdapter


def build_grok_provider_bundle(
    *,
    runtime_root: Path,
    binary_path: str | Path = "/root/.grok/bin/grok",
) -> AgentProviderBundle:
    native = {
        CapabilityKey.HOME_TYPED_OVERRIDES,
        CapabilityKey.HOME_ENV,
        CapabilityKey.HOME_AUTH_REFS,
        CapabilityKey.HOME_INSTRUCTIONS,
        CapabilityKey.HOME_MCP,
        CapabilityKey.SESSION_CREATE,
        CapabilityKey.SESSION_RESUME,
        CapabilityKey.RUN_STREAM,
        CapabilityKey.RUN_WAIT_TERMINAL,
        CapabilityKey.RUN_INTERRUPT,
        CapabilityKey.RUN_CANCEL,
        CapabilityKey.MODEL_OTHER_API,
    }
    supports = {
        key: CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            limitations=(
                ("MCP tools are discovered through Grok search_tool/use_tool rather than commands/list",)
                if key is CapabilityKey.HOME_MCP
                else ()
            ),
            evidence_version="grok-1.0.30",
        )
        for key in native
    }
    for key, reason in (
        (CapabilityKey.HOME_BASE_CONFIG, "raw Grok base configuration is outside the curated Home boundary"),
        (CapabilityKey.HOME_RAW_OVERRIDES, "raw Grok configuration overrides are unsupported"),
        (CapabilityKey.HOME_SKILLS, "Grok skill discovery and inheritance are disabled"),
        (CapabilityKey.HOME_EXTENSIONS, "Grok extensions and plugins are disabled"),
        (CapabilityKey.RUN_STEER, "Grok ACP v1 adapter does not support live steering"),
        (CapabilityKey.RUN_FOLLOW_UP, "Grok ACP v1 adapter does not support follow-up injection"),
        (CapabilityKey.CONTROL_FORK, "Grok session fork is unsupported"),
        (CapabilityKey.CONTROL_FORK_FROM_TURN, "Grok session fork is unsupported"),
        (CapabilityKey.CONTROL_COMPACT, "Grok compact is unsupported"),
        (CapabilityKey.CONTROL_APPROVAL_RESPONSE, "Grok permissions are resolved non-interactively"),
        (CapabilityKey.CONTROL_INPUT_RESPONSE, "Grok interactive input is unsupported"),
        (CapabilityKey.ARTIFACT_IN_FLIGHT_STATE, "Grok snapshot requires a clean idle process group"),
        (CapabilityKey.MODEL_RESPONSES, "Grok CLI does not expose this backend as an ARK Responses mode"),
        (CapabilityKey.MODEL_CHAT_COMPLETIONS, "Grok CLI does not expose this backend as an ARK Chat Completions mode"),
    ):
        supports[key] = CapabilitySupport(
            capability=key,
            status=CapabilityStatus.UNSUPPORTED,
            available=False,
            reason=reason,
            evidence_version="grok-adapter-v1",
        )
    for key, limitation in (
        (CapabilityKey.ARTIFACT_SNAPSHOT, "captures one complete idle native session directory"),
        (CapabilityKey.ARTIFACT_RESTORE, "restore is limited to the same Home, cwd, and session identity"),
    ):
        supports[key] = CapabilitySupport(
            capability=key,
            status=CapabilityStatus.ADAPTABLE,
            available=True,
            limitations=(limitation, "workspace files are not captured or restored"),
            evidence_version="grok-adapter-v1",
        )
    capabilities = ProviderCapabilities(provider_type="grok", supports=supports)
    runtime = GrokRuntimeAdapter(runtime_root=runtime_root)
    return AgentProviderBundle(
        descriptor=ProviderDescriptor(
            provider_type="grok",
            display_name="Grok Build",
            adapter_version="1",
            execution_kind=ProviderExecutionKind.SUBPROCESS_RPC,
            home_kind=ProviderHomeKind.NATIVE,
            sdk_or_cli_name="grok",
            sdk_or_cli_version=GROK_CLI_VERSION,
            supported_api_modes=("other",),
            static_capabilities=capabilities,
        ),
        runtime=runtime,
        home_renderer=GrokHomeRenderer(runtime_root=runtime_root, binary_path=binary_path),
        artifacts=GrokArtifactAdapter(runtime_root=runtime_root, active_sessions=runtime),
    )
