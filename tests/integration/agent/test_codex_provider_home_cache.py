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
        FakeCodex.created.append(self)

    def close(self) -> None:
        self.closed = True


class FakeSdk:
    Codex = FakeCodex

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


def _provider(*, max_idle_homes: int = 8) -> CodexProvider:
    provider = CodexProvider(max_idle_homes=max_idle_homes)
    provider._sdk = lambda: FakeSdk  # type: ignore[method-assign]
    return provider
