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
from ..store_utils import read_json
from .claude_code import ClaudeCodeProvider
from .claude_code_artifacts import ClaudeCodeArtifactAdapter
from .claude_code_context import ClaudeCodeContextAdapter, _version_tuple
from .claude_code_home import ClaudeCodeHomeRenderer
from .claude_code_query import ClaudeCodeQueryAdapter
from .claude_code_runtime import ClaudeCodeRuntimeAdapter


class ClaudeCodeCapabilityResolver:
    provider_type = "claude_code"

    def __init__(self, *, runtime_root: Path, base: ProviderCapabilities) -> None:
        self.runtime_root = Path(runtime_root)
        self.base = base

    def resolve_capabilities(
        self,
        home: object,
        model_backend: ModelBackendIdentity | None = None,
    ) -> ProviderCapabilities:
        home_id = str(getattr(home, "home_id", "")) or None
        backend = model_backend or _home_backend(home)
        backend_key = backend.backend_key
        supports = {
            key: replace(
                value,
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
            )
            for key, value in self.base.supports.items()
        }
        runtime_config, marker = self._home_evidence(home)
        actual_cli = _version_tuple(marker.get("cli_version")) if marker else None
        minimum_cli = _version_tuple(runtime_config.get("minimum_context_cli_version"))
        version_verified = (
            actual_cli is not None and minimum_cli is not None and actual_cli >= minimum_cli
        )
        for key in (CapabilityKey.QUERY_CONTEXT_USAGE, CapabilityKey.CONTROL_COMPACT):
            supports[key] = CapabilitySupport(
                capability=key,
                status=CapabilityStatus.ADAPTABLE if version_verified else CapabilityStatus.UNVERIFIED,
                available=version_verified,
                reason=(
                    None
                    if version_verified
                    else "Claude CLI context/compact control requires verified CLI >= 2.1.216"
                ),
                limitations=(
                    "compact completion requires terminal Result and a new transcript compact_boundary",
                )
                if key is CapabilityKey.CONTROL_COMPACT
                else (),
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
                evidence_version="claude-cli-2.1.216",
            )
        checkpointing = bool(runtime_config.get("enable_file_checkpointing", False))
        for key in (CapabilityKey.ARTIFACT_SNAPSHOT, CapabilityKey.ARTIFACT_RESTORE):
            supports[key] = CapabilitySupport(
                capability=key,
                status=CapabilityStatus.UNSUPPORTED if checkpointing else CapabilityStatus.ADAPTABLE,
                available=not checkpointing,
                reason=(
                    "Claude file checkpoint artifacts are not mapped by the v1 adapter"
                    if checkpointing
                    else None
                ),
                limitations=(
                    "captures one idle session transcript and validates the external Home manifest",
                ),
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
                evidence_version="claude-jsonl-snapshot-v1",
            )
        for capability, mode in (
            (CapabilityKey.MODEL_RESPONSES, "responses"),
            (CapabilityKey.MODEL_CHAT_COMPLETIONS, "chat_completions"),
        ):
            supports[capability] = CapabilitySupport(
                capability=capability,
                status=CapabilityStatus.UNSUPPORTED,
                available=False,
                reason=f"Claude Code backend uses {backend.api_mode}, not {mode}",
                resolved_for_home_id=home_id,
                resolved_for_backend=backend_key,
            )
        other_available = backend.api_mode == "anthropic_messages"
        supports[CapabilityKey.MODEL_OTHER_API] = CapabilitySupport(
            capability=CapabilityKey.MODEL_OTHER_API,
            status=CapabilityStatus.NATIVE if other_available else CapabilityStatus.UNVERIFIED,
            available=other_available,
            reason=None if other_available else f"unverified Claude API mode: {backend.api_mode}",
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
            evidence_version="claude-anthropic-messages-v1",
        )
        return ProviderCapabilities(
            provider_type=self.provider_type,
            supports=supports,
            resolved_for_home_id=home_id,
            resolved_for_backend=backend_key,
        )

    def _home_evidence(self, home: object) -> tuple[dict[str, object], dict[str, object]]:
        relpath = getattr(home, "home_relpath", None)
        if not relpath:
            return {}, {}
        root = self.runtime_root / str(relpath)
        config_path = root / ".ark" / "claude_code_home.json"
        marker_path = root / ".ark" / "claude_home_initialized.json"
        config = read_json(config_path) if config_path.is_file() else {}
        marker = read_json(marker_path) if marker_path.is_file() else {}
        return config, marker


def build_claude_code_provider_bundle(
    provider: ClaudeCodeProvider,
    *,
    runtime_root: Path,
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
        CapabilityKey.CONTROL_FORK,
    }
    adaptable = {
        CapabilityKey.SESSION_LIST: ("projected from isolated Claude session JSONL files",),
        CapabilityKey.RUN_WAIT_TERMINAL: ("requires terminal ResultMessage",),
        CapabilityKey.QUERY_TURNS: ("projected from Claude session JSONL",),
        CapabilityKey.QUERY_EVENTS: ("projected from Claude session JSONL",),
        CapabilityKey.QUERY_CONTENT: ("projected from Claude session JSONL",),
        CapabilityKey.QUERY_TOOL_CALLS: ("projected from Claude session JSONL",),
        CapabilityKey.QUERY_REQUEST_USAGE: ("deduplicated by Claude assistant message id",),
        CapabilityKey.QUERY_SESSION_USAGE: ("aggregated only where token semantics are complete",),
        CapabilityKey.ARTIFACT_OFFLINE_QUERY: ("ARK parses Claude session JSONL",),
        CapabilityKey.ARTIFACT_CACHE_REBUILD: ("no v1 cache is authoritative",),
    }
    supports = {
        key: CapabilitySupport(
            capability=key,
            status=CapabilityStatus.NATIVE,
            available=True,
            evidence_version="claude-agent-sdk-0.2.124",
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
                evidence_version="claude-code-adapter-v1",
            )
            for key, limitations in adaptable.items()
        }
    )
    for key, reason in {
        CapabilityKey.CONTROL_APPROVAL_RESPONSE: "approval response persistence is not implemented",
        CapabilityKey.CONTROL_INPUT_RESPONSE: "deferred input response is not implemented",
        CapabilityKey.ARTIFACT_IN_FLIGHT_STATE: "Claude v1 snapshots require an idle session",
        CapabilityKey.RUN_CANCEL: "Claude v1 exposes interrupt but no distinct cancel contract",
    }.items():
        supports[key] = CapabilitySupport(
            capability=key,
            status=CapabilityStatus.UNSUPPORTED,
            available=False,
            reason=reason,
        )
    base = ProviderCapabilities(provider_type="claude_code", supports=supports)
    return AgentProviderBundle(
        descriptor=ProviderDescriptor(
            provider_type="claude_code",
            display_name="Claude Code",
            adapter_version="1",
            execution_kind=ProviderExecutionKind.SUBPROCESS_RPC,
            home_kind=ProviderHomeKind.NATIVE,
            sdk_or_cli_name="claude-agent-sdk/claude-code",
            sdk_or_cli_version="0.2.124/2.1.216-verified",
            supported_api_modes=("anthropic_messages",),
            static_capabilities=base,
        ),
        runtime=ClaudeCodeRuntimeAdapter(provider),
        home_renderer=ClaudeCodeHomeRenderer(runtime_root=runtime_root, provider=provider),
        capability_resolver=ClaudeCodeCapabilityResolver(runtime_root=runtime_root, base=base),
        query=ClaudeCodeQueryAdapter(runtime_root=runtime_root, provider=provider),
        context=ClaudeCodeContextAdapter(provider),
        artifacts=ClaudeCodeArtifactAdapter(runtime_root=runtime_root, provider=provider),
    )


def _home_backend(home: object) -> ModelBackendIdentity:
    value = getattr(home, "resolved_defaults", None)
    if isinstance(value, Mapping) and value.get("api_provider") and value.get("api_mode"):
        return ModelBackendIdentity(
            api_provider=str(value["api_provider"]),
            api_mode=str(value["api_mode"]),
            endpoint_id=value.get("endpoint_id"),
            requested_model=value.get("requested_model"),
            resolved_model=value.get("resolved_model"),
            model_version=value.get("model_version"),
            service_tier=value.get("service_tier"),
            reasoning_effort=value.get("reasoning_effort"),
            tokenizer_id=value.get("tokenizer_id"),
            model_config_hash=value.get("model_config_hash"),
        )
    return ModelBackendIdentity(api_provider="anthropic", api_mode="anthropic_messages")
