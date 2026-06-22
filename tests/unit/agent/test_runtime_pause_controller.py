from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.models import AgentPausedError
from agent_runtime_kit.agent.service import AgentRunPauseController, AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.runtime import ARKServices, RuntimePausedError, RuntimePauseController


class PauseAgentType(AgentType):
    agent_type = "pause_worker"
    start_prompt_template = "start"


def test_runtime_pause_controller_global_and_scope_semantics() -> None:
    controller = RuntimePauseController()

    assert controller.is_paused() is False
    assert controller.is_paused("scope-a") is False

    controller.pause("scope-a")
    assert controller.is_paused() is False
    assert controller.is_paused("scope-a") is True
    assert controller.is_scope_directly_paused("scope-a") is True
    assert controller.is_scope_directly_paused("scope-b") is False
    assert controller.is_paused("scope-b") is False
    with pytest.raises(RuntimePausedError):
        controller.assert_can_start("scope-a")

    controller.pause(None)
    assert controller.is_paused() is True
    assert controller.is_paused("scope-b") is True
    assert controller.is_scope_directly_paused("scope-b") is False

    controller.resume("scope-a")
    assert controller.is_paused("scope-a") is True
    assert controller.is_scope_directly_paused("scope-a") is False
    controller.resume(None)
    assert controller.is_paused("scope-a") is False
    assert controller.is_paused("scope-b") is False


def test_agent_run_pause_controller_is_runtime_alias() -> None:
    assert AgentRunPauseController is RuntimePauseController


def test_agent_service_reuses_shared_runtime_pause_controller(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(PauseAgentType())
    controller = RuntimePauseController()
    ark = ARKServices(pause_controller=controller)
    service = AgentService(
        runtime_root,
        agent_types=registry,
        providers={"codex": object()},
        ark_services=ark,
    )
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="pause_worker"))
    agent = service.create_agent("scope-a", "pause_worker")

    assert service.pause_controller is controller
    assert ark.pause_controller is controller

    controller.pause("scope-a")
    with pytest.raises(AgentPausedError):
        service.start_agent(agent.agent_id)
    controller.resume("scope-a")
    assert service.is_paused("scope-a") is False


def test_agent_service_start_paused_sets_shared_global_pause(tmp_path: Path) -> None:
    registry = AgentTypeRegistry()
    registry.register(PauseAgentType())
    ark = ARKServices()

    service = AgentService(
        tmp_path / ".agent_runtime",
        agent_types=registry,
        providers={"codex": object()},
        ark_services=ark,
        start_paused=True,
    )

    assert isinstance(service.pause_controller, RuntimePauseController)
    assert ark.pause_controller is service.pause_controller
    assert service.is_paused(None) is True
