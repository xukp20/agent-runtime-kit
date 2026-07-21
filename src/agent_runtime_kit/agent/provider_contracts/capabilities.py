from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ProviderExecutionKind(str, Enum):
    SDK = "sdk"
    SUBPROCESS_RPC = "subprocess_rpc"
    PYTHON_LIBRARY = "python_library"
    EXTERNAL = "external"


class ProviderHomeKind(str, Enum):
    NATIVE = "native"
    ARK_OWNED = "ark_owned"
    EXTERNAL = "external"


class CapabilityStatus(str, Enum):
    NATIVE = "native"
    ADAPTABLE = "adaptable"
    ARK_OWNED = "ark_owned"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


class CapabilityKey(str, Enum):
    HOME_BASE_CONFIG = "home.base_config"
    HOME_TYPED_OVERRIDES = "home.typed_overrides"
    HOME_RAW_OVERRIDES = "home.raw_overrides"
    HOME_ENV = "home.env"
    HOME_AUTH_REFS = "home.auth_refs"
    HOME_INSTRUCTIONS = "home.instructions"
    HOME_SKILLS = "home.skills"
    HOME_MCP = "home.mcp"
    HOME_EXTENSIONS = "home.extensions"
    SESSION_CREATE = "session.create"
    SESSION_RESUME = "session.resume"
    SESSION_READ = "session.read"
    SESSION_LIST = "session.list"
    SESSION_ARCHIVE = "session.archive"
    SESSION_CLOSE = "session.close"
    RUN_STREAM = "run.stream"
    RUN_WAIT_TERMINAL = "run.wait_terminal"
    RUN_INTERRUPT = "run.interrupt"
    RUN_CANCEL = "run.cancel"
    RUN_STEER = "run.steer"
    RUN_FOLLOW_UP = "run.follow_up"
    CONTROL_FORK = "control.fork"
    CONTROL_FORK_FROM_TURN = "control.fork_from_turn"
    CONTROL_COMPACT = "control.compact"
    CONTROL_APPROVAL_RESPONSE = "control.approval_response"
    CONTROL_INPUT_RESPONSE = "control.input_response"
    CONTROL_HANDOFF = "control.handoff"
    QUERY_TURNS = "query.turns"
    QUERY_EVENTS = "query.events"
    QUERY_CONTENT = "query.content"
    QUERY_TOOL_CALLS = "query.tool_calls"
    QUERY_REQUEST_USAGE = "query.request_usage"
    QUERY_SESSION_USAGE = "query.session_usage"
    QUERY_CONTEXT_USAGE = "query.context_usage"
    ARTIFACT_OFFLINE_QUERY = "artifact.offline_query"
    ARTIFACT_SNAPSHOT = "artifact.snapshot"
    ARTIFACT_RESTORE = "artifact.restore"
    ARTIFACT_CACHE_REBUILD = "artifact.cache_rebuild"
    ARTIFACT_IN_FLIGHT_STATE = "artifact.in_flight_state"
    MODEL_RESPONSES = "model.responses"
    MODEL_CHAT_COMPLETIONS = "model.chat_completions"
    MODEL_OTHER_API = "model.other_api"


@dataclass(frozen=True)
class CapabilitySupport:
    capability: CapabilityKey
    status: CapabilityStatus
    available: bool
    reason: str | None = None
    requirements: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    resolved_for_home_id: str | None = None
    resolved_for_backend: str | None = None
    evidence_version: str | None = None

    def __post_init__(self) -> None:
        if self.status in {CapabilityStatus.UNSUPPORTED, CapabilityStatus.UNVERIFIED} and self.available:
            raise ValueError(f"{self.status.value} capability cannot be available")
        if self.status in {
            CapabilityStatus.NATIVE,
            CapabilityStatus.ADAPTABLE,
            CapabilityStatus.ARK_OWNED,
        } and not self.available:
            raise ValueError(f"{self.status.value} capability must be available")


@dataclass(frozen=True)
class ProviderCapabilities:
    provider_type: str
    supports: Mapping[CapabilityKey, CapabilitySupport] = field(default_factory=dict)
    resolved_for_home_id: str | None = None
    resolved_for_backend: str | None = None

    def __post_init__(self) -> None:
        if not self.provider_type.strip():
            raise ValueError("provider_type must not be empty")
        normalized: dict[CapabilityKey, CapabilitySupport] = {}
        for raw_key, support in self.supports.items():
            key = raw_key if isinstance(raw_key, CapabilityKey) else CapabilityKey(str(raw_key))
            if support.capability != key:
                raise ValueError(f"capability key mismatch: {key.value} != {support.capability.value}")
            normalized[key] = support
        object.__setattr__(self, "supports", normalized)

    def get(self, capability: CapabilityKey) -> CapabilitySupport:
        support = self.supports.get(capability)
        if support is not None:
            return support
        return CapabilitySupport(
            capability=capability,
            status=CapabilityStatus.UNSUPPORTED,
            available=False,
            reason="provider did not declare this capability",
            resolved_for_home_id=self.resolved_for_home_id,
            resolved_for_backend=self.resolved_for_backend,
        )

    def available(self, capability: CapabilityKey) -> bool:
        return self.get(capability).available

    def require(self, capability: CapabilityKey) -> CapabilitySupport:
        support = self.get(capability)
        if not support.available:
            reason = f": {support.reason}" if support.reason else ""
            raise ProviderCapabilityUnavailable(
                f"provider {self.provider_type} does not support {capability.value}{reason}"
            )
        return support


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_type: str
    display_name: str
    adapter_version: str
    execution_kind: ProviderExecutionKind
    home_kind: ProviderHomeKind
    sdk_or_cli_name: str | None = None
    sdk_or_cli_version: str | None = None
    supported_api_modes: tuple[str, ...] = ()
    static_capabilities: ProviderCapabilities | None = None
    provider_payload: object | None = None

    def __post_init__(self) -> None:
        provider_type = self.provider_type.strip()
        if not provider_type:
            raise ValueError("provider_type must not be empty")
        if not self.display_name.strip():
            raise ValueError("display_name must not be empty")
        if not self.adapter_version.strip():
            raise ValueError("adapter_version must not be empty")
        if self.static_capabilities is not None and self.static_capabilities.provider_type != provider_type:
            raise ValueError("descriptor and static capabilities provider_type differ")
        object.__setattr__(self, "provider_type", provider_type)


class ProviderCapabilityUnavailable(RuntimeError):
    pass
