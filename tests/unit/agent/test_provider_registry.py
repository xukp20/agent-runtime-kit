from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    AgentProviderBundle,
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    HomeInitializationResult,
    HomeMaterializationResult,
    HomeValidationResult,
    ProviderCapabilities,
    ProviderCapabilityUnavailable,
    ProviderDescriptor,
    ProviderExecutionKind,
    ProviderHomeKind,
    ProviderRegistry,
)


class _Adapter:
    def __init__(self, provider_type: str) -> None:
        self.provider_type = provider_type


class _Resolver:
    provider_type = "fake"

    def resolve_capabilities(self, home, model_backend=None):  # noqa: ANN001, ANN201
        api_mode = getattr(model_backend, "api_mode", None)
        available = api_mode == "responses"
        return ProviderCapabilities(
            provider_type="fake",
            resolved_for_home_id=home.home_id,
            resolved_for_backend=api_mode,
            supports={
                CapabilityKey.CONTROL_COMPACT: CapabilitySupport(
                    capability=CapabilityKey.CONTROL_COMPACT,
                    status=CapabilityStatus.NATIVE if available else CapabilityStatus.UNSUPPORTED,
                    available=available,
                    reason=None if available else "backend has no compact endpoint",
                )
            },
        )


@dataclass
class _Home:
    provider_type: str = "fake"
    home_id: str = "home"


def _descriptor(provider_type: str = "fake", *, static: bool = True) -> ProviderDescriptor:
    capabilities = None
    if static:
        capabilities = ProviderCapabilities(
            provider_type=provider_type,
            supports={
                CapabilityKey.SESSION_CREATE: CapabilitySupport(
                    capability=CapabilityKey.SESSION_CREATE,
                    status=CapabilityStatus.NATIVE,
                    available=True,
                )
            },
        )
    return ProviderDescriptor(
        provider_type=provider_type,
        display_name="Fake",
        adapter_version="1",
        execution_kind=ProviderExecutionKind.SDK,
        home_kind=ProviderHomeKind.ARK_OWNED,
        static_capabilities=capabilities,
    )


def _bundle(provider_type: str = "fake") -> AgentProviderBundle:
    return AgentProviderBundle(
        descriptor=_descriptor(provider_type),
        runtime=_Adapter(provider_type),  # type: ignore[arg-type]
        home_renderer=_Adapter(provider_type),  # type: ignore[arg-type]
    )


def test_registry_rejects_duplicate_and_adapter_identity_mismatch() -> None:
    registry = ProviderRegistry((_bundle(),))
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(_bundle())

    mismatched = AgentProviderBundle(
        descriptor=_descriptor(),
        runtime=_Adapter("other"),  # type: ignore[arg-type]
        home_renderer=_Adapter("fake"),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="runtime provider_type mismatch"):
        ProviderRegistry((mismatched,))


def test_registry_lists_deterministically_and_replace_is_explicit() -> None:
    registry = ProviderRegistry((_bundle("zeta"), _bundle("alpha")))
    assert [item.provider_type for item in registry.list()] == ["alpha", "zeta"]

    replacement = _bundle("alpha")
    registry.replace(replacement)
    assert registry.get("alpha") is replacement


def test_bundle_resolves_dynamic_backend_capability() -> None:
    from agent_runtime_kit.agent.provider_contracts import ModelBackendIdentity

    bundle = AgentProviderBundle(
        descriptor=_descriptor(static=False),
        runtime=_Adapter("fake"),  # type: ignore[arg-type]
        home_renderer=_Adapter("fake"),  # type: ignore[arg-type]
        capability_resolver=_Resolver(),
    )

    responses = bundle.resolve_capabilities(
        _Home(),
        ModelBackendIdentity(api_provider="openai", api_mode="responses"),
    )
    chat = bundle.resolve_capabilities(
        _Home(),
        ModelBackendIdentity(api_provider="deepseek", api_mode="chat_completions"),
    )

    assert responses.available(CapabilityKey.CONTROL_COMPACT)
    assert not chat.available(CapabilityKey.CONTROL_COMPACT)


def test_bundle_without_static_or_dynamic_capabilities_fails_closed() -> None:
    bundle = AgentProviderBundle(
        descriptor=_descriptor(static=False),
        runtime=_Adapter("fake"),  # type: ignore[arg-type]
        home_renderer=_Adapter("fake"),  # type: ignore[arg-type]
    )
    with pytest.raises(ProviderCapabilityUnavailable):
        bundle.resolve_capabilities(_Home())
