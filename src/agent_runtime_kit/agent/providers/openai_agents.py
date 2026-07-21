from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Callable, Mapping

from ..provider_contracts import AgentProviderBundle, ModelBackendIdentity


@dataclass(frozen=True)
class OpenAIAgentsHomeOptions:
    """Provider-private, serializable configuration for one Agents SDK Home."""

    agent_factory_ref: str
    agent_factory_version: str = "1"
    resource_fingerprint: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    base_url_env: str | None = None
    session_backend: str = "sqlite"
    store: bool = False
    compaction_mode: str = "unsupported"
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    model_settings: Mapping[str, object] = field(default_factory=dict)
    tracing_disabled: bool = True

    def __post_init__(self) -> None:
        if not self.agent_factory_ref.strip():
            raise ValueError("agent_factory_ref must not be empty")
        if not self.agent_factory_version.strip():
            raise ValueError("agent_factory_version must not be empty")
        if not self.api_key_env.strip():
            raise ValueError("api_key_env must not be empty")
        if self.session_backend != "sqlite":
            raise ValueError("OpenAI Agents provider currently requires session_backend='sqlite'")
        if self.compaction_mode not in {"unsupported", "input_history"}:
            raise ValueError("compaction_mode must be unsupported or input_history")
        for name in ("context_window_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class OpenAIAgentsRunOptions:
    """Optional per-run provider values that never enter the neutral contract."""

    context: object | None = field(default=None, repr=False, compare=False)
    hooks: object | None = field(default=None, repr=False, compare=False)
    run_config: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class OpenAIAgentsControlOptions:
    """Provider-private context required to resume durable RunState after restart."""

    execution_context: object = field(repr=False, compare=False)
    agent_id: str
    scope_id: str
    agent_type: str
    home_id: str


@dataclass(frozen=True)
class OpenAIAgentsBuildContext:
    home_id: str
    home_root: Path
    workdir: str | None
    model: object
    model_identity: ModelBackendIdentity
    instructions: str
    skills_root: Path
    mcp_servers: tuple[object, ...]
    provider_config: Mapping[str, object]


@dataclass(frozen=True)
class _ResourceFactory:
    factory: Callable[[OpenAIAgentsBuildContext], object]
    version: str
    fingerprint: str | None


class OpenAIAgentsResourceRegistry:
    """Application-owned registry for non-serializable Agent graphs and tools."""

    def __init__(self) -> None:
        self._factories: dict[str, _ResourceFactory] = {}
        self._lock = RLock()

    def register_agent_factory(
        self,
        ref: str,
        factory: Callable[[OpenAIAgentsBuildContext], object],
        *,
        version: str = "1",
        fingerprint: str | None = None,
    ) -> None:
        key = _resource_ref(ref)
        if not callable(factory):
            raise TypeError("agent factory must be callable")
        if not version.strip():
            raise ValueError("agent factory version must not be empty")
        with self._lock:
            if key in self._factories:
                raise ValueError(f"duplicate OpenAI Agents resource ref: {key}")
            self._factories[key] = _ResourceFactory(factory, version, fingerprint)

    def resolve_agent_factory(
        self,
        ref: str,
        *,
        version: str,
        fingerprint: str | None,
    ) -> Callable[[OpenAIAgentsBuildContext], object]:
        key = _resource_ref(ref)
        with self._lock:
            try:
                resource = self._factories[key]
            except KeyError as exc:
                raise KeyError(f"unregistered OpenAI Agents resource ref: {key}") from exc
        if resource.version != version:
            raise RuntimeError(
                f"OpenAI Agents resource version mismatch for {key}: "
                f"expected {version}, got {resource.version}"
            )
        if fingerprint is not None and resource.fingerprint != fingerprint:
            raise RuntimeError(f"OpenAI Agents resource fingerprint mismatch for {key}")
        return resource.factory

    def validate_ref(self, ref: str) -> str:
        return _resource_ref(ref)


class OpenAIAgentsProvider:
    provider_type = "openai_agents"

    def __init__(self, *, registry: OpenAIAgentsResourceRegistry | None = None) -> None:
        self.registry = registry or OpenAIAgentsResourceRegistry()

    def build_bundle(self, *, runtime_root: Path) -> AgentProviderBundle:
        from .openai_agents_bundle import build_openai_agents_provider_bundle

        return build_openai_agents_provider_bundle(
            runtime_root=runtime_root,
            registry=self.registry,
        )

    def build_provider_bundle(self, *, runtime_root: Path) -> AgentProviderBundle:
        """Compatibility with AgentService's provider self-bundle bootstrap."""

        return self.build_bundle(runtime_root=runtime_root)


def _resource_ref(value: str) -> str:
    ref = str(value).strip()
    if not ref or ref in {".", ".."} or "\x00" in ref:
        raise ValueError("invalid OpenAI Agents resource ref")
    return ref
