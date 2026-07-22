from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime_kit.agent.homes import ProviderHomeSpec
from agent_runtime_kit.agent.models import AgentContextCompactionEvidenceError
from agent_runtime_kit.agent.provider_contracts import ProviderRegistry, ProviderRunState
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.providers.codex_bundle import build_codex_provider_bundle
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry


class FakeCodex:
    created: list["FakeCodex"] = []
    account_calls = 0
    account_delay_s = 0.0

    def __init__(self, config=None) -> None:
        self.config = config
        self.closed = False
        self._client = FakeClient()
        FakeCodex.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True

    def account(self):
        import time

        if FakeCodex.account_delay_s:
            time.sleep(FakeCodex.account_delay_s)
        FakeCodex.account_calls += 1
        env = dict(self.config.get("env") or {})
        codex_home = Path(env["CODEX_HOME"])
        codex_home.mkdir(parents=True, exist_ok=True)
        for name in CodexProvider.REQUIRED_STATE_DATABASES:
            (codex_home / name).write_text("", encoding="utf-8")
        return SimpleNamespace(account_id="fake")

    def thread_start(self, **kwargs):
        self._client.thread_start_calls.append(dict(kwargs))
        return FakeHighLevelThread("thread-started")

    def thread_resume(self, thread_id: str, **kwargs):
        self._client.high_level_resume_calls.append({"thread_id": thread_id, **kwargs})
        return FakeHighLevelThread(thread_id)

    def thread_fork(self, thread_id: str, **kwargs):
        self._client.thread_fork_calls.append({"thread_id": thread_id, **kwargs})
        return SimpleNamespace(id=f"{thread_id}-forked")


class FakeHighLevelThread:
    run_started: threading.Event | None = None
    run_release: threading.Event | None = None
    compact_callback = None
    status_type = "idle"
    last_turn_handle = None

    def __init__(self, thread_id: str) -> None:
        self.id = thread_id

    def run(self, prompt: str, **kwargs):
        if self.run_started is not None:
            self.run_started.set()
        if self.run_release is not None:
            assert self.run_release.wait(timeout=5)
        return SimpleNamespace(id=f"turn-{prompt}", prompt=prompt, run_kwargs=kwargs)

    def turn(self, prompt: str, **kwargs):
        handle = FakeHighLevelTurnHandle(self.id, prompt, kwargs)
        type(self).last_turn_handle = handle
        return handle

    def read(self, include_turns: bool = True):
        turns = [SimpleNamespace(id="turn-read", status="completed", items=[])] if include_turns else []
        return SimpleNamespace(
            thread=SimpleNamespace(
                id=self.id,
                turns=turns,
                status=SimpleNamespace(root=SimpleNamespace(type=self.status_type)),
            )
        )

    def compact(self):
        callback = type(self).compact_callback
        if callback is not None:
            callback()
        return SimpleNamespace()


class FakeHighLevelTurnHandle:
    def __init__(self, thread_id: str, prompt: str, run_kwargs: dict[str, object]) -> None:
        self.thread_id = thread_id
        self.prompt = prompt
        self.run_kwargs = run_kwargs
        self.id = f"turn-{prompt}"
        self.interrupted = False

    def run(self):
        if FakeHighLevelThread.run_started is not None:
            FakeHighLevelThread.run_started.set()
        if FakeHighLevelThread.run_release is not None:
            assert FakeHighLevelThread.run_release.wait(timeout=5)
        return SimpleNamespace(id=self.id, prompt=self.prompt, run_kwargs=self.run_kwargs)

    def interrupt(self):
        self.interrupted = True
        if FakeHighLevelThread.run_release is not None:
            FakeHighLevelThread.run_release.set()
        return SimpleNamespace(accepted=True)


class FakeClient:
    def __init__(self) -> None:
        self.thread_start_calls: list[dict[str, object]] = []
        self.thread_fork_calls: list[dict[str, object]] = []
        self.thread_resume_calls: list[dict[str, object]] = []
        self.high_level_resume_calls: list[dict[str, object]] = []
        self.turn_start_calls: list[dict[str, object]] = []

    def thread_resume(self, thread_id: str, params: dict[str, object] | None = None):
        self.thread_resume_calls.append({"thread_id": thread_id, "params": params})
        return SimpleNamespace(thread=SimpleNamespace(id=thread_id), model="fake-model")

    def turn_start(
        self,
        thread_id: str,
        input_items: str,
        params: dict[str, object] | None = None,
    ):
        self.turn_start_calls.append(
            {"thread_id": thread_id, "input_items": input_items, "params": params}
        )
        return SimpleNamespace(turn=SimpleNamespace(id="turn-1"))


class FakeThread:
    def __init__(self, client: FakeClient, thread_id: str) -> None:
        self._client = client
        self.id = thread_id


class FakeTurnHandle:
    def __init__(self, client: FakeClient, thread_id: str, turn_id: str) -> None:
        self._client = client
        self.thread_id = thread_id
        self.id = turn_id

    def run(self):
        return SimpleNamespace(id=self.id, thread_id=self.thread_id, final_response="done")


class FakeSdk:
    Codex = FakeCodex
    Thread = FakeThread
    TurnHandle = FakeTurnHandle

    @staticmethod
    def CodexConfig(**kwargs):
        return dict(kwargs)


class LiveLocatorAgentType(AgentType):
    agent_type = "live-locator"
    developer_instructions_template = "Observe {{item}}."
    start_prompt_template = "Start {{item}}."


def test_codex_provider_initializes_home_once_then_uses_per_run_codex(tmp_path: Path) -> None:
    _reset_fake_codex()
    provider = _provider()
    home_root = tmp_path / "home"
    started_thread_ids: list[str] = []

    first = provider.start_thread(
        home_id="worker",
        home_root=home_root,
        env={"CODEX_HOME": str(home_root / ".codex"), "ARK_RUN_TOKEN": "first"},
        workdir=str(tmp_path / "work-a"),
        prompt="first",
        developer_instructions="developer first",
        agent_id="agent-a",
        on_thread_started=started_thread_ids.append,
    )
    second = provider.start_thread(
        home_id="worker",
        home_root=home_root,
        env={"CODEX_HOME": str(home_root / ".codex"), "ARK_RUN_TOKEN": "second"},
        workdir=str(tmp_path / "work-b"),
        prompt="second",
        developer_instructions="developer second",
        agent_id="agent-a",
    )

    assert first.thread_id == "thread-started"
    assert started_thread_ids == ["thread-started"]
    assert second.thread_id == "thread-started"
    assert FakeCodex.account_calls == 1
    assert len(FakeCodex.created) == 3
    assert all(codex.closed for codex in FakeCodex.created)
    init_codex, first_run_codex, second_run_codex = FakeCodex.created
    assert init_codex.config["env"]["ARK_RUN_TOKEN"] == "first"
    assert first_run_codex.config["env"]["ARK_RUN_TOKEN"] == "first"
    assert second_run_codex.config["env"]["ARK_RUN_TOKEN"] == "second"
    assert first_run_codex.config["cwd"] == str(tmp_path / "work-a")
    assert second_run_codex.config["cwd"] == str(tmp_path / "work-b")
    assert first_run_codex._client.thread_start_calls[0]["approval_mode"] == "deny_all"
    assert second_run_codex._client.thread_start_calls[0]["approval_mode"] == "deny_all"
    assert (home_root / ".ark" / "codex_home_initialized.json").exists()


def test_codex_provider_concurrent_home_initialization_is_locked(tmp_path: Path) -> None:
    _reset_fake_codex()
    FakeCodex.account_delay_s = 0.05
    provider = _provider()
    home_root = tmp_path / "home"

    def initialize() -> None:
        provider.ensure_home_initialized(
            home_id="worker",
            home_root=home_root,
            env={"CODEX_HOME": str(home_root / ".codex")},
            workdir=None,
        )

    threads = [threading.Thread(target=initialize) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert FakeCodex.account_calls == 1
    assert len(FakeCodex.created) == 1
    assert (home_root / ".ark" / "codex_home_initialized.json").exists()


def test_codex_provider_tracks_active_agents_without_home_cache(tmp_path: Path) -> None:
    provider = _provider()

    provider._begin_agent_run(home_id="worker", agent_id="agent-a")
    provider._begin_agent_run(home_id="worker", agent_id="agent-b")
    provider._begin_agent_run(home_id="other", agent_id="agent-c")

    assert provider.list_active_agents("worker") == ["agent-a", "agent-b"]
    assert provider.list_active_agents() == ["agent-a", "agent-b", "agent-c"]

    provider._finish_agent_run("agent-b")
    assert provider.list_active_agents("worker") == ["agent-a"]


def test_codex_provider_interrupts_the_live_turn_handle(tmp_path: Path) -> None:
    _reset_fake_codex()
    started = threading.Event()
    release = threading.Event()
    FakeHighLevelThread.run_started = started
    FakeHighLevelThread.run_release = release
    provider = _provider()
    home_root = tmp_path / "home"
    error: list[BaseException] = []

    def run() -> None:
        try:
            provider.start_thread(
                home_id="worker",
                home_root=home_root,
                env={"CODEX_HOME": str(home_root / ".codex")},
                workdir=str(tmp_path),
                prompt="interruptible",
                developer_instructions=None,
                agent_id="agent-a",
            )
        except BaseException as exc:  # pragma: no cover - assertion aid
            error.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(timeout=5)

    assert provider.interrupt_agent("agent-a") is True
    worker.join(timeout=5)

    assert not error
    assert not worker.is_alive()
    assert FakeHighLevelThread.last_turn_handle.interrupted is True
    assert provider.list_active_agents() == []


def test_agent_service_persists_new_thread_locator_while_first_turn_is_running(
    tmp_path: Path,
) -> None:
    _reset_fake_codex()
    started = threading.Event()
    release = threading.Event()
    FakeHighLevelThread.run_started = started
    FakeHighLevelThread.run_release = release
    provider = _provider()
    registry = AgentTypeRegistry()
    registry.register(LiveLocatorAgentType())
    service = AgentService(
        tmp_path / "runtime",
        agent_types=registry,
        provider_registry=ProviderRegistry(
            (build_codex_provider_bundle(provider, runtime_root=tmp_path / "runtime"),)
        ),
    )
    service.home_service.create_home(ProviderHomeSpec(provider_type="codex", home_id="live-locator"))
    agent = service.create_agent("scope", "live-locator")

    service.start_agent(agent.agent_id, variables={"item": "state"})
    assert started.wait(timeout=5)

    running = service.get_agent(agent.agent_id)
    assert running.status == "running"
    assert running.session_locator is not None
    assert running.session_locator.session_id == "thread-started"
    assert provider.list_active_agents("live-locator") == [agent.agent_id]

    release.set()
    result = service.wait_agent(agent.agent_id)
    assert result.status is ProviderRunState.COMPLETED
    assert result.session_locator.session_id == "thread-started"
    assert result.turn_locator is not None
    assert result.turn_locator.turn_id == "turn-Start state."
    assert result.completion is not None
    assert result.completion.status == "complete"
    persisted = service.get_agent(agent.agent_id)
    assert persisted.schema_version == 3
    assert persisted.provider_type == "codex"
    assert persisted.session_locator == result.session_locator
    assert persisted.latest_turn_locator == result.turn_locator
    assert persisted.artifact_locator == result.provider_result.artifact_locator
    assert service.provider_registry.get("codex").provider_type == "codex"
    assert provider.list_active_agents("live-locator") == []


def test_agent_service_reseals_provider_declared_home_initialization_changes(
    tmp_path: Path,
) -> None:
    _reset_fake_codex()
    runtime_root = tmp_path / "runtime"
    provider = _provider()
    registry = AgentTypeRegistry()
    registry.register(LiveLocatorAgentType())
    service = AgentService(
        runtime_root,
        agent_types=registry,
        provider_registry=ProviderRegistry(
            (build_codex_provider_bundle(provider, runtime_root=runtime_root),)
        ),
    )
    home = service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="codex",
            home_id="live-locator",
            config_overrides={"model": "gpt-before-initialization"},
        )
    )
    home_root = service.home_service.resolve_home_root("codex", "live-locator")
    config_path = home_root / ".codex" / "config.toml"
    initial_manifest_hash = home.materialization_manifest_hash
    initialize_home = provider.ensure_home_initialized

    def initialize_with_managed_change(**kwargs):  # noqa: ANN003, ANN202
        marker_path = Path(kwargs["home_root"]) / ".ark" / "codex_home_initialized.json"
        if not marker_path.exists():
            config_path.write_text(
                config_path.read_text(encoding="utf-8")
                + "\n# provider initialization mutation\n",
                encoding="utf-8",
            )
        return initialize_home(**kwargs)

    provider.ensure_home_initialized = initialize_with_managed_change  # type: ignore[method-assign]
    agent = service.create_agent("scope", "live-locator")

    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "first"}).agent_id)
    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "second"}).agent_id)

    context = service.home_service.build_execution_context("codex", "live-locator")
    refreshed = service.home_service.get_home("codex", "live-locator")
    assert refreshed.materialization_manifest_hash != initial_manifest_hash
    assert context.materialization_manifest is not None
    assert context.materialization_manifest.manifest_hash == refreshed.materialization_manifest_hash

    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "# unsealed tampering\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="materialized file hash mismatch"):
        service.home_service.build_execution_context("codex", "live-locator")


def test_agent_service_commits_session_start_home_only_for_new_session(
    tmp_path: Path,
) -> None:
    _reset_fake_codex()
    runtime_root = tmp_path / "runtime"
    provider = _provider()
    registry = AgentTypeRegistry()
    registry.register(LiveLocatorAgentType())
    service = AgentService(
        runtime_root,
        agent_types=registry,
        provider_registry=ProviderRegistry(
            (build_codex_provider_bundle(provider, runtime_root=runtime_root),)
        ),
    )
    service.home_service.create_home(
        ProviderHomeSpec(provider_type="codex", home_id="live-locator")
    )
    commits: list[tuple[str, str, str]] = []
    commit = service.home_service.commit_provider_lifecycle_materialization

    def record_commit(
        provider_type: str,
        home_id: str,
        *,
        lifecycle: str,
    ):
        commits.append((provider_type, home_id, lifecycle))
        return commit(provider_type, home_id, lifecycle=lifecycle)

    service.home_service.commit_provider_lifecycle_materialization = record_commit  # type: ignore[method-assign]
    agent = service.create_agent("scope", "live-locator")

    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "first"}).agent_id)
    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "second"}).agent_id)

    assert commits == [("codex", "live-locator", "session_start")]


def test_agent_service_interrupt_waits_until_codex_turn_is_terminal(tmp_path: Path) -> None:
    _reset_fake_codex()
    started = threading.Event()
    release = threading.Event()
    FakeHighLevelThread.run_started = started
    FakeHighLevelThread.run_release = release
    provider = _provider()
    registry = AgentTypeRegistry()
    registry.register(LiveLocatorAgentType())
    service = AgentService(
        tmp_path / "runtime",
        agent_types=registry,
        provider_registry=ProviderRegistry(
            (build_codex_provider_bundle(provider, runtime_root=tmp_path / "runtime"),)
        ),
    )
    service.home_service.create_home(ProviderHomeSpec(provider_type="codex", home_id="live-locator"))
    agent = service.create_agent("scope", "live-locator")
    service.start_agent(agent.agent_id, variables={"item": "state"})
    assert started.wait(timeout=5)

    assert service.interrupt_agent(agent.agent_id, timeout_s=5) is True

    assert service.get_agent(agent.agent_id).status == "idle"
    assert provider.list_active_agents() == []
    assert FakeHighLevelThread.last_turn_handle.interrupted is True


def test_codex_provider_resume_instruction_overwrite_uses_fresh_run_codex(
    tmp_path: Path,
) -> None:
    _reset_fake_codex()
    provider = _provider()
    home_root = tmp_path / "home"

    result = provider.resume_thread(
        home_id="worker",
        home_root=home_root,
        env={"CODEX_HOME": str(home_root / ".codex"), "ARK_RUN_TOKEN": "resume"},
        thread_id="thread-existing",
        workdir=str(tmp_path),
        prompt="next prompt",
        developer_instructions="new developer instruction",
        agent_id="agent-a",
        overwrite_developer_instructions=True,
    )

    assert result.thread_id == "thread-existing"
    assert result.turn_result.id == "turn-1"
    assert FakeCodex.account_calls == 1
    assert len(FakeCodex.created) == 2
    run_codex = FakeCodex.created[1]
    assert run_codex.config["env"]["ARK_RUN_TOKEN"] == "resume"
    client = run_codex._client
    assert client.high_level_resume_calls == []
    assert client.thread_resume_calls == [
        {
            "thread_id": "thread-existing",
            "params": {"cwd": str(tmp_path), "model": None, "config": None},
        }
    ]
    assert client.turn_start_calls[0]["thread_id"] == "thread-existing"
    assert client.turn_start_calls[0]["input_items"] == "next prompt"
    assert client.turn_start_calls[0]["params"] == {
        "cwd": str(tmp_path),
        "model": "fake-model",
        "collaborationMode": {
            "mode": "default",
            "settings": {
                "model": "fake-model",
                "developer_instructions": "new developer instruction",
            },
        },
    }


def test_codex_provider_resume_without_instruction_overwrite_uses_default_resume(
    tmp_path: Path,
) -> None:
    _reset_fake_codex()
    provider = _provider()
    home_root = tmp_path / "home"

    provider.resume_thread(
        home_id="worker",
        home_root=home_root,
        env={"CODEX_HOME": str(home_root / ".codex"), "ARK_RUN_TOKEN": "default"},
        thread_id="thread-existing",
        workdir=str(tmp_path),
        prompt="next prompt",
        developer_instructions="developer instruction",
        agent_id="agent-a",
    )

    assert FakeCodex.account_calls == 1
    assert len(FakeCodex.created) == 2
    run_codex = FakeCodex.created[1]
    client = run_codex._client
    assert run_codex.config["env"]["ARK_RUN_TOKEN"] == "default"
    assert client.high_level_resume_calls == [
        {
            "thread_id": "thread-existing",
            "cwd": str(tmp_path),
            "developer_instructions": "developer instruction",
            "model": None,
            "config": None,
        }
    ]
    assert client.thread_resume_calls == []
    assert client.turn_start_calls == []


def test_codex_provider_compact_waits_for_rollout_evidence_and_idle(tmp_path: Path) -> None:
    _reset_fake_codex()
    provider = _provider()
    home_root = tmp_path / "home"
    rollout = home_root / ".codex" / "sessions" / "rollout-thread-existing.jsonl"
    _append_rollout(rollout, _token_count(total_tokens=80))

    def compact() -> None:
        _append_rollout(rollout, {"type": "compacted", "payload": {"message": "summary"}})
        _append_rollout(rollout, {"type": "event_msg", "payload": {"type": "context_compacted"}})
        _append_rollout(rollout, _token_count(total_tokens=20))

    FakeHighLevelThread.compact_callback = compact
    started: list[tuple[dict[str, object], str | None]] = []

    result = provider.compact_thread(
        home_id="worker",
        home_root=home_root,
        env={"CODEX_HOME": str(home_root / ".codex")},
        thread_id="thread-existing",
        workdir=str(tmp_path),
        agent_id="agent-a",
        timeout_s=1,
        on_compaction_started=lambda baseline, operation_id: started.append((baseline, operation_id)),
    )

    assert result.usage_after is not None
    assert result.usage_after.used_tokens == 20
    assert len(started) == 1
    assert started[0][0]["token_count_count"] == 1
    assert provider.list_active_agents() == []


def test_codex_provider_compact_rejects_non_idle_thread(tmp_path: Path) -> None:
    _reset_fake_codex()
    provider = _provider()
    home_root = tmp_path / "home"
    rollout = home_root / ".codex" / "sessions" / "rollout-thread-existing.jsonl"
    _append_rollout(rollout, _token_count(total_tokens=80))
    FakeHighLevelThread.status_type = "active"

    with pytest.raises(AgentContextCompactionEvidenceError, match="not idle"):
        provider.compact_thread(
            home_id="worker",
            home_root=home_root,
            env={"CODEX_HOME": str(home_root / ".codex")},
            thread_id="thread-existing",
            workdir=str(tmp_path),
            agent_id="agent-a",
            timeout_s=1,
        )


def _provider() -> CodexProvider:
    provider = CodexProvider()
    provider._sdk = lambda: FakeSdk  # type: ignore[method-assign]
    return provider


def _reset_fake_codex() -> None:
    FakeCodex.created.clear()
    FakeCodex.account_calls = 0
    FakeCodex.account_delay_s = 0.0
    FakeHighLevelThread.run_started = None
    FakeHighLevelThread.run_release = None
    FakeHighLevelThread.compact_callback = None
    FakeHighLevelThread.status_type = "idle"
    FakeHighLevelThread.last_turn_handle = None


def _append_rollout(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _token_count(*, total_tokens: int) -> dict[str, object]:
    return {
        "type": "event_msg",
        "payload": {
            "type": "token_count",
            "info": {
                "last_token_usage": {"total_tokens": total_tokens},
                "model_context_window": 100,
            },
        },
    }
