from pathlib import Path
from threading import Thread

import pytest

from agent_runtime_kit.agent.homes import ProviderHomeSpec
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


def test_runtime_pause_controller_bypass_current_thread_is_nested_and_non_destructive() -> None:
    controller = RuntimePauseController(global_paused=True)
    controller.pause("scope-a")

    assert controller.is_paused() is True
    assert controller.is_paused("scope-a") is True
    assert controller.is_scope_directly_paused("scope-a") is True

    with controller.bypass_current_thread():
        assert controller.is_paused() is False
        assert controller.is_paused("scope-a") is False
        assert controller.is_scope_directly_paused("scope-a") is True
        controller.assert_can_start("scope-a")
        with controller.bypass_current_thread():
            assert controller.is_paused("scope-a") is False
            controller.assert_can_start("scope-a")
        assert controller.is_paused("scope-a") is False

    assert controller.is_paused() is True
    assert controller.is_paused("scope-a") is True
    assert controller.is_scope_directly_paused("scope-a") is True
    with pytest.raises(RuntimePausedError):
        controller.assert_can_start("scope-a")


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
        ark_services=ark,
    )
    service.home_service.create_home(ProviderHomeSpec(provider_type="codex", home_id="pause_worker"))
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
        ark_services=ark,
        start_paused=True,
    )

    assert isinstance(service.pause_controller, RuntimePauseController)
    assert ark.pause_controller is service.pause_controller
    assert service.is_paused(None) is True


def test_recovery_agent_boundary_serializes_close_and_paused_start_fork(tmp_path: Path) -> None:
    registry = AgentTypeRegistry()
    registry.register(PauseAgentType())
    controller = RuntimePauseController()
    service = AgentService(
        tmp_path / ".agent_runtime",
        agent_types=registry,
        ark_services=ARKServices(pause_controller=controller),
    )
    service.home_service.create_home(ProviderHomeSpec(provider_type="codex", home_id="pause_worker"))
    agent = service.create_agent("scope-a", "pause_worker")
    controller.pause("scope-a")
    outcomes: list[str] = []

    def record(name, operation):  # noqa: ANN001
        try:
            operation()
        except Exception as exc:  # noqa: BLE001
            outcomes.append(f"{name}:{type(exc).__name__}")
        else:
            outcomes.append(f"{name}:ok")

    with controller.hold_paused("scope-a"), service.hold_agent_boundary():
        threads = [
            Thread(target=record, args=("close", lambda: service.close_agent(agent.agent_id))),
            Thread(target=record, args=("start", lambda: service.start_agent(agent.agent_id))),
            Thread(target=record, args=("fork", lambda: service.fork_agent(agent.agent_id))),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(0.05)
            assert thread.is_alive()

    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert "close:ok" in outcomes
    assert "start:AgentPausedError" in outcomes
    assert "fork:AgentPausedError" in outcomes
