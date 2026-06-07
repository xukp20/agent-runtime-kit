from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeCreateSpec, HomeService, build_provider_env
from agent_runtime_kit.agent.models import MissingProviderEnvError


def test_create_codex_home_copies_config_auth_and_skills(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    auth = tmp_path / "auth.json"
    skill = tmp_path / "skill"
    config.write_text("model = 'test'\n", encoding="utf-8")
    auth.write_text('{"token": "x"}\n', encoding="utf-8")
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    service = HomeService(tmp_path / ".agent_runtime")
    record = service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="node_worker",
            base_config_path=config,
            auth_json_path=auth,
            skill_paths={"lean": skill},
            fixed_env={"FIXED": "1"},
            required_env={"SECRET"},
        )
    )

    home_root = service.resolve_home_root("codex", "node_worker")
    assert record.home_relpath == "homes/codex/node_worker"
    assert (home_root / ".codex" / "config.toml").read_text(encoding="utf-8") == "model = 'test'\n"
    assert (home_root / ".codex" / "auth.json").read_text(encoding="utf-8") == '{"token": "x"}\n'
    assert (home_root / ".agents" / "skills" / "lean" / "SKILL.md").exists()


def test_build_provider_env_sets_fake_home_and_codex_home(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")
    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            fixed_env={"TOKEN": "home"},
            required_env={"TOKEN", "EXTRA"},
        )
    )
    home = service.get_home("codex", "planner")
    home_root = service.resolve_home_root("codex", "planner")

    env = build_provider_env(
        home=home,
        home_root=home_root,
        base_env={"EXTRA": "base", "TOKEN": "base-token"},
        run_env={"TOKEN": "run"},
    )

    assert env["HOME"] == str(home_root)
    assert env["CODEX_HOME"] == str(home_root / ".codex")
    assert env["TOKEN"] == "run"
    assert env["EXTRA"] == "base"


def test_build_provider_env_reports_missing_required_env(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")
    service.create_home(HomeCreateSpec(cli_type="codex", home_id="planner", required_env={"TOKEN"}))
    home = service.get_home("codex", "planner")

    with pytest.raises(MissingProviderEnvError) as exc_info:
        build_provider_env(home=home, home_root=service.resolve_home_root("codex", "planner"), base_env={})

    assert exc_info.value.name == "TOKEN"
