from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.models import AgentIncompleteError, AgentPausedError, CompletionDecision
from agent_runtime_kit.agent.service import AgentCompletionContext, AgentService, AgentType, AgentTypeRegistry

from .fakes import FakeProvider, FakeTurnResult


class BasicAgentType(AgentType):
    agent_type = "worker"
    developer_instructions_template = "Developer instructions for {{item}}."
    start_prompt_template = "Start {{item}}."
    continue_prompt_template = "Continue {{item}} because {{reason}}."


class OneContinueAgentType(BasicAgentType):
    def check_completion(self, ctx: AgentCompletionContext) -> CompletionDecision:
        if ctx.auto_continue_count == 0:
            return CompletionDecision(complete=False, reason="first turn incomplete")
        return CompletionDecision(complete=True, reason="done")

    def max_auto_continue_turns(self, ctx: AgentCompletionContext | None) -> int:
        return 1


class NeverCompleteAgentType(BasicAgentType):
    def check_completion(self, ctx: AgentCompletionContext) -> CompletionDecision:
        return CompletionDecision(complete=False, reason="still incomplete")


def test_agent_service_runs_agent_and_reads_latest_result(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("repo:node", "worker")

    service.start_agent(agent.agent_id, variables={"item": "lemma"})
    result = service.wait_agent(agent.agent_id)

    assert isinstance(result, FakeTurnResult)
    assert result.prompt == "Start lemma."
    assert result.developer_instructions == "Developer instructions for lemma."
    restored_agent = service.get_agent(agent.agent_id)
    assert restored_agent.status == "idle"
    assert restored_agent.thread_id == "thread-1"
    assert restored_agent.rollout_relpath == "sessions/fake/rollout-thread-1.jsonl"
    assert provider.calls[0]["env_home"] == str(runtime_root / "homes" / "codex" / "worker")
    assert provider.calls[0]["home_id"] == "worker"
    assert provider.calls[0]["agent_id"] == agent.agent_id
    assert service.read_latest_turn_result(agent.agent_id).id == result.id


def test_agent_service_auto_continues_until_completion(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, OneContinueAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope", "worker")

    result = service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "goal"}).agent_id)

    assert isinstance(result, FakeTurnResult)
    assert result.prompt == "Continue goal because first turn incomplete."
    assert len(provider.calls) == 2
    restored_agent = service.get_agent(agent.agent_id)
    assert restored_agent.last_completion is not None
    assert restored_agent.last_completion.status == "complete"
    assert len(service.list_turns(agent.agent_id)) == 2


def test_agent_service_persists_incomplete_result_for_late_wait(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, _provider = _make_service(runtime_root, NeverCompleteAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope", "worker")

    service.start_agent(agent.agent_id, variables={"item": "goal"})
    with pytest.raises(AgentIncompleteError):
        service.wait_agent(agent.agent_id)
    with pytest.raises(AgentIncompleteError):
        service.wait_agent(agent.agent_id)

    restored_agent = service.get_agent(agent.agent_id)
    assert restored_agent.status == "idle"
    assert restored_agent.last_completion is not None
    assert restored_agent.last_completion.status == "incomplete"


def test_agent_service_pause_blocks_new_runs(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, _provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope", "worker")

    service.pause_runs("scope")
    with pytest.raises(AgentPausedError):
        service.start_agent(agent.agent_id, variables={"item": "goal"})
    service.resume_runs("scope")

    service.start_agent(agent.agent_id, variables={"item": "goal"})
    assert service.wait_agent(agent.agent_id).prompt == "Start goal."


def test_agent_service_forks_finished_agent_thread(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, _provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope-a", "worker")
    service.start_agent(agent.agent_id, variables={"item": "goal"})
    service.wait_agent(agent.agent_id)

    forked = service.fork_agent(agent.agent_id, target_scope_id="scope-b")

    assert forked.scope_id == "scope-b"
    assert forked.agent_type == "worker"
    assert forked.home_id == "worker"
    assert forked.fork_source_agent_id == agent.agent_id
    assert forked.fork_source_thread_id == "thread-1"
    assert forked.thread_id == "thread-2"
    assert (runtime_root / "homes" / "codex" / "worker" / ".codex" / forked.rollout_relpath).exists()


def test_agent_service_runs_same_home_agents_concurrently(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, BasicAgentType(), run_delay_s=0.1)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent_a = service.create_agent("scope", "worker")
    agent_b = service.create_agent("scope", "worker")

    service.start_agent(agent_a.agent_id, variables={"item": "a"})
    service.start_agent(agent_b.agent_id, variables={"item": "b"})
    waited = service.wait_agents([agent_a.agent_id, agent_b.agent_id], timeout_s=5)

    assert waited.clean
    assert provider.max_active_by_home["worker"] == 2
    assert service.get_agent(agent_a.agent_id).status == "idle"
    assert service.get_agent(agent_b.agent_id).status == "idle"
    assert service.is_stable()


def test_agent_service_interrupt_and_close_delegate_to_provider(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope", "worker")

    assert service.interrupt_agent(agent.agent_id) is False
    assert provider.interrupt_calls == [agent.agent_id]
    assert service.close_provider_home("codex", "worker") is True
    assert provider.close_home_calls == [{"home_id": "worker", "force": False}]
    service.close(force_provider_homes=True)
    assert provider.close_all_calls == [{"force": True}]


def _make_service(
    runtime_root: Path,
    agent_type: AgentType,
    *,
    run_delay_s: float = 0.0,
) -> tuple[AgentService, FakeProvider]:
    registry = AgentTypeRegistry()
    registry.register(agent_type)
    provider = FakeProvider(runtime_root, run_delay_s=run_delay_s)
    service = AgentService(runtime_root, agent_types=registry, providers={"codex": provider})
    return service, provider
