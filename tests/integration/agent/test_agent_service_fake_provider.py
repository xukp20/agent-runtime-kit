from pathlib import Path
from time import monotonic

import pytest

from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.models import AgentIncompleteError, AgentPausedError, CompletionDecision
from agent_runtime_kit.agent.report_policy import AgentTraceReportPolicy, TraceReportPersistence
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
    report_paths = service.get_default_trace_report_paths(agent.agent_id)
    assert Path(report_paths.latest_json_path).exists()
    assert not (Path(report_paths.reports_root) / "turns").exists()
    assert service.read_default_trace_report(agent.agent_id)["latest_turn"]["turn_id"] == result.id
    assert provider.calls[0]["env_home"] == str(runtime_root / "homes" / "codex" / "worker")
    assert provider.calls[0]["home_id"] == "worker"
    assert provider.calls[0]["agent_id"] == agent.agent_id
    assert service.read_latest_turn_result(agent.agent_id).id == result.id


@pytest.mark.parametrize(
    ("persistence", "expect_latest", "expect_turns"),
    [
        (TraceReportPersistence.DISABLED, False, False),
        (TraceReportPersistence.LATEST_ONLY, True, False),
        (TraceReportPersistence.LATEST_AND_TURNS, True, True),
    ],
)
def test_agent_service_applies_automatic_trace_report_persistence(
    tmp_path: Path,
    persistence: TraceReportPersistence,
    expect_latest: bool,
    expect_turns: bool,
) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(BasicAgentType())
    provider = FakeProvider(runtime_root)
    service = AgentService(
        runtime_root,
        agent_types=registry,
        providers={"codex": provider},
        trace_report_policy=AgentTraceReportPolicy(persistence=persistence),
    )
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("repo:node", "worker")

    service.start_agent(agent.agent_id, variables={"item": "lemma"})
    result = service.wait_agent(agent.agent_id)

    paths = service.get_default_trace_report_paths(agent.agent_id)
    assert Path(paths.latest_json_path).exists() is expect_latest
    assert (Path(paths.reports_root) / "turns" / f"{result.id}.json").exists() is expect_turns
    explicit_path = runtime_root / "explicit.json"
    service.export_trace_report(agent.agent_id, output_path=explicit_path)
    assert explicit_path.exists()


def test_agent_service_supports_run_level_template_overrides_on_resume(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("repo:node", "worker")

    first = service.wait_agent(
        service.start_agent(
            agent.agent_id,
            variables={"item": "alpha"},
            developer_instructions_template_override="Override developer {{item}}.",
            start_prompt_template_override="Override start {{item}}.",
        ).agent_id
    )
    restored = service.get_agent(agent.agent_id)
    thread_id = restored.thread_id

    second = service.wait_agent(
        service.start_agent(
            agent.agent_id,
            variables={"item": "beta"},
            developer_instructions_template_override="Second developer {{item}}.",
            start_prompt_template_override="Second start {{item}}.",
        ).agent_id
    )

    assert isinstance(first, FakeTurnResult)
    assert first.prompt == "Override start alpha."
    assert first.developer_instructions == "Override developer alpha."
    assert isinstance(second, FakeTurnResult)
    assert second.thread_id == thread_id
    assert second.prompt == "Second start beta."
    assert second.developer_instructions == "Second developer beta."
    assert provider.calls[0]["thread_id"] == provider.calls[1]["thread_id"]
    assert provider.calls[0]["overwrite_developer_instructions"] is True
    assert provider.calls[1]["overwrite_developer_instructions"] is True


def test_agent_service_passes_run_level_env_on_each_start(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("repo:node", "worker")

    service.wait_agent(
        service.start_agent(
            agent.agent_id,
            variables={"item": "alpha"},
            env={"ARK_STEP_ID": "step-a", "ARK_RUN_TOKEN": "token-a"},
        ).agent_id
    )
    service.wait_agent(
        service.start_agent(
            agent.agent_id,
            variables={"item": "beta"},
            env={"ARK_STEP_ID": "step-b", "ARK_RUN_TOKEN": "token-b"},
        ).agent_id
    )

    assert provider.calls[0]["env"]["ARK_STEP_ID"] == "step-a"
    assert provider.calls[0]["env"]["ARK_RUN_TOKEN"] == "token-a"
    assert provider.calls[1]["env"]["ARK_STEP_ID"] == "step-b"
    assert provider.calls[1]["env"]["ARK_RUN_TOKEN"] == "token-b"
    assert provider.calls[0]["env"]["CODEX_HOME"] == str(runtime_root / "homes" / "codex" / "worker" / ".codex")
    assert provider.calls[1]["env"]["CODEX_HOME"] == str(runtime_root / "homes" / "codex" / "worker" / ".codex")


def test_agent_service_create_home_can_initialize_provider_home(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, BasicAgentType())

    service.create_home(
        HomeCreateSpec(cli_type="codex", home_id="worker"),
        env={"ARK_BOOT_TOKEN": "boot"},
        workdir=str(tmp_path / "work"),
    )

    assert provider.ensure_home_initialized_calls == [
        {
            "home_id": "worker",
            "home_root": str(runtime_root / "homes" / "codex" / "worker"),
            "env": provider.ensure_home_initialized_calls[0]["env"],
            "workdir": str(tmp_path / "work"),
        }
    ]
    env = provider.ensure_home_initialized_calls[0]["env"]
    assert env["ARK_BOOT_TOKEN"] == "boot"
    assert env["HOME"] == str(runtime_root / "homes" / "codex" / "worker")
    assert env["CODEX_HOME"] == str(runtime_root / "homes" / "codex" / "worker" / ".codex")


def test_agent_service_prompt_direct_text_still_beats_start_template_override(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, _provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("repo:node", "worker")

    result = service.wait_agent(
        service.start_agent(
            agent.agent_id,
            variables={"item": "alpha"},
            prompt="Direct prompt.",
            start_prompt_template_override="Override start {{item}}.",
        ).agent_id
    )

    assert isinstance(result, FakeTurnResult)
    assert result.prompt == "Direct prompt."
    assert _provider.calls[0]["overwrite_developer_instructions"] is False


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


def test_agent_service_supports_run_level_continue_template_override(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, OneContinueAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope", "worker")

    result = service.wait_agent(
        service.start_agent(
            agent.agent_id,
            variables={"item": "goal"},
            continue_prompt_template_override="Override continue {{item}} after {{reason}}.",
        ).agent_id
    )

    assert isinstance(result, FakeTurnResult)
    assert result.prompt == "Override continue goal after first turn incomplete."
    assert provider.calls[1]["prompt"] == "Override continue goal after first turn incomplete."


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


def test_wait_agents_uses_one_shared_timeout_budget(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, _provider = _make_service(runtime_root, BasicAgentType(), run_delay_s=0.25)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent_a = service.create_agent("scope", "worker")
    agent_b = service.create_agent("scope", "worker")
    service.start_agent(agent_a.agent_id, variables={"item": "a"})
    service.start_agent(agent_b.agent_id, variables={"item": "b"})

    started = monotonic()
    waited = service.wait_agents([agent_a.agent_id, agent_b.agent_id], timeout_s=0.05)
    elapsed = monotonic() - started

    assert waited.timeout is True
    assert set(waited.pending) == {agent_a.agent_id, agent_b.agent_id}
    assert elapsed < 0.2
    assert service.wait_agents([agent_a.agent_id, agent_b.agent_id], timeout_s=2).clean


def test_reconcile_stale_running_agents_repairs_only_inactive_scope(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, _provider = _make_service(runtime_root, BasicAgentType(), run_delay_s=0.2)
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    stale_a = service.create_agent("scope-a", "worker")
    stale_b = service.create_agent("scope-b", "worker")
    active = service.create_agent("scope-a", "worker")
    service.store.patch_agent(stale_a.agent_id, status="running")
    service.store.patch_agent(stale_b.agent_id, status="running")
    service.start_agent(active.agent_id, variables={"item": "active"})

    repaired_scope_a = service.reconcile_stale_running_agents(scope_id="scope-a")

    assert repaired_scope_a == [stale_a.agent_id]
    assert service.get_agent(stale_a.agent_id).status == "idle"
    assert service.get_agent(stale_b.agent_id).status == "running"
    assert service.get_agent(active.agent_id).status == "running"
    assert service.wait_agent(active.agent_id, timeout_s=2).prompt == "Start active."


def test_agent_service_interrupt_and_close_delegate_to_provider(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    service, provider = _make_service(runtime_root, BasicAgentType())
    service.home_service.create_home(HomeCreateSpec(cli_type="codex", home_id="worker"))
    agent = service.create_agent("scope", "worker")

    assert service.interrupt_agent(agent.agent_id) is False
    assert provider.interrupt_calls == [agent.agent_id]
    service.close()
    assert provider.close_calls == 1


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
