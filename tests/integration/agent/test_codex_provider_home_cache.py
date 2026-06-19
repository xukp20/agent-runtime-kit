from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime_kit.agent.providers.codex import CodexProvider


class FakeCodex:
    created: list["FakeCodex"] = []

    def __init__(self, config=None) -> None:
        self.config = config
        self.closed = False
        self._client = FakeClient()
        FakeCodex.created.append(self)

    def close(self) -> None:
        self.closed = True

    def thread_resume(self, thread_id: str, **kwargs):
        self._client.high_level_resume_calls.append({"thread_id": thread_id, **kwargs})
        return SimpleNamespace(
            id=thread_id,
            run=lambda prompt, **run_kwargs: SimpleNamespace(
                id="high-level-turn",
                prompt=prompt,
                run_kwargs=run_kwargs,
            ),
        )


class FakeClient:
    def __init__(self) -> None:
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


def test_codex_provider_reuses_same_home_object(tmp_path: Path) -> None:
    FakeCodex.created.clear()
    provider = _provider()
    home_root = tmp_path / "home"

    first = provider._get_or_start_home(home_id="worker", home_root=home_root, env={}, workdir=None)
    second = provider._get_or_start_home(home_id="worker", home_root=home_root, env={}, workdir=None)

    assert first is second
    assert len(FakeCodex.created) == 1


def test_codex_provider_rejects_same_home_id_at_different_root(tmp_path: Path) -> None:
    provider = _provider()
    provider._get_or_start_home(home_id="worker", home_root=tmp_path / "a", env={}, workdir=None)

    with pytest.raises(ValueError):
        provider._get_or_start_home(home_id="worker", home_root=tmp_path / "b", env={}, workdir=None)


def test_codex_provider_tracks_active_agents_and_close_rules(tmp_path: Path) -> None:
    provider = _provider()
    home = provider._get_or_start_home(home_id="worker", home_root=tmp_path / "home", env={}, workdir=None)

    provider._begin_agent_run(home, "agent-a")
    assert provider.list_active_agents("worker") == ["agent-a"]
    assert provider.close_home("worker") is False
    assert "worker" in provider._homes

    provider._finish_agent_run("agent-a")
    assert provider.list_active_agents("worker") == []
    assert provider.close_home("worker") is True
    assert "worker" not in provider._homes


def test_codex_provider_evicts_only_idle_homes(tmp_path: Path) -> None:
    provider = _provider(max_idle_homes=1)
    active_home = provider._get_or_start_home(home_id="active", home_root=tmp_path / "active", env={}, workdir=None)
    provider._begin_agent_run(active_home, "agent-a")
    provider._get_or_start_home(home_id="idle", home_root=tmp_path / "idle", env={}, workdir=None)

    assert "active" in provider._homes
    assert len(provider._homes) == 1

    provider._finish_agent_run("agent-a")

    assert len(provider._homes) <= 1
    assert all(not state.active_agent_ids for state in provider._homes.values())


def test_codex_provider_close_all_force_closes_active(tmp_path: Path) -> None:
    provider = _provider()
    home = provider._get_or_start_home(home_id="worker", home_root=tmp_path / "home", env={}, workdir=None)
    provider._begin_agent_run(home, "agent-a")

    provider.close_all(force=True)

    assert provider._homes == {}
    assert provider._agent_runs == {}


def test_codex_provider_resume_instruction_overwrite_uses_turn_collaboration_mode(
    tmp_path: Path,
) -> None:
    FakeCodex.created.clear()
    provider = _provider()

    result = provider.resume_thread(
        home_id="worker",
        home_root=tmp_path / "home",
        env={},
        thread_id="thread-existing",
        workdir=str(tmp_path),
        prompt="next prompt",
        developer_instructions="new developer instruction",
        agent_id="agent-a",
        overwrite_developer_instructions=True,
    )

    codex = FakeCodex.created[0]
    client = codex._client
    assert result.thread_id == "thread-existing"
    assert result.turn_result.id == "turn-1"
    assert client.high_level_resume_calls == []
    assert client.thread_resume_calls == [
        {
            "thread_id": "thread-existing",
            "params": {"cwd": str(tmp_path), "model": None, "config": None},
        }
    ]
    assert client.turn_start_calls[0]["thread_id"] == "thread-existing"
    assert client.turn_start_calls[0]["input_items"] == "next prompt"
    params = client.turn_start_calls[0]["params"]
    assert params == {
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
    FakeCodex.created.clear()
    provider = _provider()

    provider.resume_thread(
        home_id="worker",
        home_root=tmp_path / "home",
        env={},
        thread_id="thread-existing",
        workdir=str(tmp_path),
        prompt="next prompt",
        developer_instructions="developer instruction",
        agent_id="agent-a",
    )

    codex = FakeCodex.created[0]
    client = codex._client
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


def _provider(*, max_idle_homes: int = 8) -> CodexProvider:
    provider = CodexProvider(max_idle_homes=max_idle_homes)
    provider._sdk = lambda: FakeSdk  # type: ignore[method-assign]
    return provider
