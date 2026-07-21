from __future__ import annotations

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    CapabilityKey,
    CapabilityStatus,
    CapabilitySupport,
    ProviderCapabilities,
    ProviderCapabilityUnavailable,
    ProviderDescriptor,
    ProviderExecutionKind,
    ProviderHomeKind,
)


def test_capabilities_do_not_treat_missing_as_available() -> None:
    capabilities = ProviderCapabilities(
        provider_type="example",
        supports={
            CapabilityKey.RUN_INTERRUPT: CapabilitySupport(
                capability=CapabilityKey.RUN_INTERRUPT,
                status=CapabilityStatus.ADAPTABLE,
                available=True,
                limitations=("requires terminal barrier",),
            )
        },
    )

    assert capabilities.require(CapabilityKey.RUN_INTERRUPT).available is True
    missing = capabilities.get(CapabilityKey.CONTROL_COMPACT)
    assert missing.status is CapabilityStatus.UNSUPPORTED
    assert missing.available is False
    with pytest.raises(ProviderCapabilityUnavailable):
        capabilities.require(CapabilityKey.CONTROL_COMPACT)


@pytest.mark.parametrize(
    ("status", "available"),
    [
        (CapabilityStatus.NATIVE, False),
        (CapabilityStatus.ADAPTABLE, False),
        (CapabilityStatus.ARK_OWNED, False),
        (CapabilityStatus.UNSUPPORTED, True),
        (CapabilityStatus.UNVERIFIED, True),
    ],
)
def test_capability_status_and_availability_cannot_disagree(
    status: CapabilityStatus,
    available: bool,
) -> None:
    with pytest.raises(ValueError):
        CapabilitySupport(
            capability=CapabilityKey.SESSION_CREATE,
            status=status,
            available=available,
        )


def test_descriptor_keeps_provider_separate_from_backend_and_model() -> None:
    descriptor = ProviderDescriptor(
        provider_type="openai_agents",
        display_name="OpenAI Agents",
        adapter_version="1",
        execution_kind=ProviderExecutionKind.PYTHON_LIBRARY,
        home_kind=ProviderHomeKind.ARK_OWNED,
        supported_api_modes=("responses", "chat_completions"),
    )

    assert descriptor.provider_type == "openai_agents"
    assert descriptor.supported_api_modes == ("responses", "chat_completions")
