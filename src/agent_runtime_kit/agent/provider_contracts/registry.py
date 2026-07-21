from __future__ import annotations

from dataclasses import dataclass

from .capabilities import ProviderCapabilityUnavailable, ProviderCapabilities, ProviderDescriptor
from .protocols import (
    ProviderArtifactAdapter,
    ProviderCapabilityResolver,
    ProviderContextAdapter,
    ProviderHomeRenderer,
    ProviderQueryAdapter,
    ProviderRuntimeAdapter,
)


@dataclass(frozen=True)
class AgentProviderBundle:
    descriptor: ProviderDescriptor
    runtime: ProviderRuntimeAdapter
    home_renderer: ProviderHomeRenderer
    capability_resolver: ProviderCapabilityResolver | None = None
    query: ProviderQueryAdapter | None = None
    context: ProviderContextAdapter | None = None
    artifacts: ProviderArtifactAdapter | None = None

    @property
    def provider_type(self) -> str:
        return self.descriptor.provider_type

    def resolve_capabilities(self, home: object, model_backend: object | None = None) -> ProviderCapabilities:
        if self.capability_resolver is not None:
            return self.capability_resolver.resolve_capabilities(home, model_backend)  # type: ignore[arg-type]
        if self.descriptor.static_capabilities is None:
            raise ProviderCapabilityUnavailable(
                f"provider {self.provider_type} has no capability resolver or static capabilities"
            )
        return self.descriptor.static_capabilities


class ProviderRegistry:
    def __init__(self, bundles: tuple[AgentProviderBundle, ...] = ()) -> None:
        self._bundles: dict[str, AgentProviderBundle] = {}
        for bundle in bundles:
            self.register(bundle)

    def register(self, bundle: AgentProviderBundle) -> None:
        provider_type = bundle.provider_type.strip()
        if not provider_type:
            raise ValueError("provider_type must not be empty")
        if provider_type in self._bundles:
            raise ValueError(f"duplicate provider_type: {provider_type}")
        self._validate_bundle(bundle)
        self._bundles[provider_type] = bundle

    def replace(self, bundle: AgentProviderBundle) -> None:
        self._validate_bundle(bundle)
        self._bundles[bundle.provider_type] = bundle

    def get(self, provider_type: str) -> AgentProviderBundle:
        key = provider_type.strip()
        try:
            return self._bundles[key]
        except KeyError as exc:
            raise KeyError(f"unknown provider_type: {key}") from exc

    def list(self) -> tuple[AgentProviderBundle, ...]:
        return tuple(self._bundles[key] for key in sorted(self._bundles))

    def __contains__(self, provider_type: object) -> bool:
        return isinstance(provider_type, str) and provider_type in self._bundles

    @staticmethod
    def _validate_bundle(bundle: AgentProviderBundle) -> None:
        provider_type = bundle.provider_type
        for name, adapter in (
            ("runtime", bundle.runtime),
            ("home_renderer", bundle.home_renderer),
            ("query", bundle.query),
            ("context", bundle.context),
            ("artifacts", bundle.artifacts),
        ):
            if adapter is None:
                continue
            adapter_provider_type = getattr(adapter, "provider_type", None)
            if adapter_provider_type != provider_type:
                raise ValueError(
                    f"{name} provider_type mismatch: expected {provider_type}, got {adapter_provider_type}"
                )
