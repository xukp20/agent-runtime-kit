from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

from agent_runtime_kit.agent.providers.codex import CodexProvider


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
    def __init__(self, thread_id: str) -> None:
        self.id = thread_id

    def run(self, prompt: str, **kwargs):
        return SimpleNamespace(id=f"turn-{prompt}", prompt=prompt, run_kwargs=kwargs)

    def read(self, include_turns: bool = True):
        turns = [SimpleNamespace(id="turn-read", status="completed", items=[])] if include_turns else []
        return SimpleNamespace(thread=SimpleNamespace(id=self.id, turns=turns))


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


def test_codex_provider_initializes_home_once_then_uses_per_run_codex(tmp_path: Path) -> None:
    _reset_fake_codex()
    provider = _provider()
    home_root = tmp_path / "home"

    first = provider.start_thread(
        home_id="worker",
        home_root=home_root,
        env={"CODEX_HOME": str(home_root / ".codex"), "ARK_RUN_TOKEN": "first"},
        workdir=str(tmp_path / "work-a"),
        prompt="first",
        developer_instructions="developer first",
        agent_id="agent-a",
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


def _provider() -> CodexProvider:
    provider = CodexProvider()
    provider._sdk = lambda: FakeSdk  # type: ignore[method-assign]
    return provider


def _reset_fake_codex() -> None:
    FakeCodex.created.clear()
    FakeCodex.account_calls = 0
    FakeCodex.account_delay_s = 0.0
