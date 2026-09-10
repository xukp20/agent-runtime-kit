from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import ProviderHomeSpec
from agent_runtime_kit.agent.models import (
    AgentProviderTurnFailed,
    AgentProviderUnavailable,
    MissingProviderEnvError,
)
from agent_runtime_kit.agent.provider_contracts import ProviderRegistry, ProviderRunState
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from tests.integration.agent.provider_contract_harness import make_contract_fake_bundle


class _FailingAgentType(AgentType):
    agent_type = "failing-agent"
    provider_type = "contract_fake"
    default_home_id = "contract-home"
    start_prompt_template = "work"

    def __init__(self) -> None:
        self.completion_calls = 0

    def check_completion(self, ctx):  # noqa: ANN001, ANN201
        self.completion_calls += 1
        return super().check_completion(ctx)


def _service(tmp_path: Path) -> tuple[AgentService, _FailingAgentType, str]:
    bundle = make_contract_fake_bundle(initial_state=ProviderRunState.FAILED)
    agent_type = _FailingAgentType()
    registry = AgentTypeRegistry()
    registry.register(agent_type)
    service = AgentService(
        tmp_path / ".agent_runtime",
        agent_types=registry,
        provider_registry=ProviderRegistry((bundle,)),
    )
    service.home_service.create_home(
        ProviderHomeSpec(provider_type="contract_fake", home_id="contract-home")
    )
    agent = service.create_agent("scope", agent_type.agent_type)
    return service, agent_type, agent.agent_id


def test_failed_provider_turn_skips_completion_checker_and_exposes_typed_failure(
    tmp_path: Path,
) -> None:
    service, agent_type, agent_id = _service(tmp_path)

    service.start_agent(agent_id)
    with pytest.raises(AgentProviderTurnFailed) as caught:
        service.wait_agent(agent_id, timeout_s=2)

    assert agent_type.completion_calls == 0
    assert caught.value.provider_type == "contract_fake"
    assert caught.value.provider_error_type == "fake_failure"
    assert caught.value.run_id == f"run-{agent_id}"
    persisted = service.get_agent(agent_id)
    assert persisted.status == "idle"
    assert persisted.session_locator is not None
    assert persisted.latest_turn_locator is not None


def test_cached_failed_provider_turn_still_raises_typed_failure(tmp_path: Path) -> None:
    service, _, agent_id = _service(tmp_path)
    service.start_agent(agent_id)
    with pytest.raises(AgentProviderTurnFailed):
        service.wait_agent(agent_id, timeout_s=2)
    with pytest.raises(AgentProviderTurnFailed):
        service.wait_agent(agent_id, timeout_s=2)


def test_known_provider_startup_failure_maps_to_typed_unavailable(tmp_path: Path) -> None:
    service, _, agent_id = _service(tmp_path)
    runtime = service.get_provider_bundle("contract_fake").runtime

    def unavailable(_request):  # noqa: ANN001, ANN202
        raise MissingProviderEnvError("TOKEN")

    runtime.start = unavailable
    service.start_agent(agent_id)

    with pytest.raises(AgentProviderUnavailable) as caught:
        service.wait_agent(agent_id, timeout_s=2)
    assert caught.value.provider_error_type == "MissingProviderEnvError"
    assert caught.value.retryable is False


def test_new_start_clears_prior_failed_result_and_caches_startup_unavailable_after_cleanup(
    tmp_path: Path,
) -> None:
    service, _, agent_id = _service(tmp_path)
    service.start_agent(agent_id)
    with pytest.raises(AgentProviderTurnFailed):
        service.wait_agent(agent_id, timeout_s=2)

    runtime = service.get_provider_bundle("contract_fake").runtime

    def unavailable(_request):  # noqa: ANN001, ANN202
        raise MissingProviderEnvError("TOKEN")

    runtime.start = unavailable
    service.start_agent(agent_id)
    with pytest.raises(AgentProviderUnavailable):
        service.wait_agent(agent_id, timeout_s=2)
    assert agent_id not in service._active
    with pytest.raises(AgentProviderUnavailable):
        service.wait_agent(agent_id, timeout_s=2)


def test_unknown_provider_startup_exception_is_not_reclassified(tmp_path: Path) -> None:
    service, _, agent_id = _service(tmp_path)
    runtime = service.get_provider_bundle("contract_fake").runtime

    def broken(_request):  # noqa: ANN001, ANN202
        raise TypeError("assembly bug")

    runtime.start = broken
    service.start_agent(agent_id)

    with pytest.raises(TypeError, match="assembly bug"):
        service.wait_agent(agent_id, timeout_s=2)
