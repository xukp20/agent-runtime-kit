from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeCreateSpec, HomeService, McpServerSpec, build_provider_env
from agent_runtime_kit.agent.models import MissingProviderEnvError
from agent_runtime_kit.agent.skills import SkillSpec


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


def test_create_codex_home_writes_skill_specs(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")

    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            skill_specs={
                "mathlib-recon": SkillSpec(
                    name="mathlib-recon",
                    description="Use when searching Mathlib.",
                    body="Search Mathlib carefully.",
                    files={"references/search.md": "Use exact names.\n"},
                )
            },
        )
    )

    home_root = service.resolve_home_root("codex", "planner")
    skill_root = home_root / ".agents" / "skills" / "mathlib-recon"
    assert 'name: "mathlib-recon"' in (skill_root / "SKILL.md").read_text(encoding="utf-8")
    assert (skill_root / "references" / "search.md").read_text(encoding="utf-8") == "Use exact names.\n"


def test_create_codex_home_renders_mcp_servers_into_config(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("model = 'test'\n", encoding="utf-8")
    service = HomeService(tmp_path / ".agent_runtime")

    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            base_config_path=config,
            mcp_servers=[
                McpServerSpec(
                    name="ark_identity",
                    transport="http",
                    url="http://127.0.0.1:8765/mcp",
                    startup_timeout_sec=10,
                    tool_timeout_sec=20,
                    required=True,
                    enabled_tools=["read_identity"],
                    http_headers={"X-Static": "static"},
                    env_http_headers={"X-Ark-Step-Id": "ARK_STEP_ID"},
                )
            ],
        )
    )

    home_root = service.resolve_home_root("codex", "planner")
    rendered = (home_root / ".codex" / "config.toml").read_text(encoding="utf-8")
    assert "model = 'test'" in rendered
    assert "[mcp_servers.ark_identity]" in rendered
    assert 'url = "http://127.0.0.1:8765/mcp"' in rendered
    assert "startup_timeout_sec = 10" in rendered
    assert "tool_timeout_sec = 20" in rendered
    assert "required = true" in rendered
    assert 'enabled_tools = ["read_identity"]' in rendered
    assert "[mcp_servers.ark_identity.http_headers]" in rendered
    assert 'X-Static = "static"' in rendered
    assert "[mcp_servers.ark_identity.env_http_headers]" in rendered
    assert 'X-Ark-Step-Id = "ARK_STEP_ID"' in rendered


def test_create_codex_home_supports_path_and_spec_skills_together(tmp_path: Path) -> None:
    path_skill = tmp_path / "path_skill"
    path_skill.mkdir()
    (path_skill / "SKILL.md").write_text("# Path skill\n", encoding="utf-8")
    service = HomeService(tmp_path / ".agent_runtime")

    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            skill_paths={"path-skill": path_skill},
            skill_specs={
                "spec-skill": SkillSpec(
                    name="spec-skill",
                    description="Spec skill.",
                    body="Spec body.",
                )
            },
        )
    )

    home_root = service.resolve_home_root("codex", "planner")
    assert (home_root / ".agents" / "skills" / "path-skill" / "SKILL.md").read_text(
        encoding="utf-8"
    ) == "# Path skill\n"
    assert "Spec body." in (
        home_root / ".agents" / "skills" / "spec-skill" / "SKILL.md"
    ).read_text(encoding="utf-8")


def test_create_codex_home_rejects_duplicate_path_and_spec_skill_names(tmp_path: Path) -> None:
    path_skill = tmp_path / "path_skill"
    path_skill.mkdir()
    (path_skill / "SKILL.md").write_text("# Path skill\n", encoding="utf-8")
    service = HomeService(tmp_path / ".agent_runtime")

    with pytest.raises(ValueError, match="duplicate skill names"):
        service.create_home(
            HomeCreateSpec(
                cli_type="codex",
                home_id="planner",
                skill_paths={"demo": path_skill},
                skill_specs={"demo": SkillSpec(name="demo", description="Demo.", body="Demo.")},
            )
        )


def test_create_codex_home_rejects_skill_spec_key_mismatch(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")

    with pytest.raises(ValueError, match="must match SkillSpec.name"):
        service.create_home(
            HomeCreateSpec(
                cli_type="codex",
                home_id="planner",
                skill_specs={
                    "wrong": SkillSpec(name="demo", description="Demo.", body="Demo."),
                },
            )
        )


def test_create_codex_home_replaces_existing_spec_skill_directory(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")

    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            skill_specs={"demo": SkillSpec(name="demo", description="Demo.", body="Old.")},
        )
    )
    home_root = service.resolve_home_root("codex", "planner")
    extra = home_root / ".agents" / "skills" / "demo" / "old.txt"
    extra.write_text("old", encoding="utf-8")

    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            skill_specs={"demo": SkillSpec(name="demo", description="Demo.", body="New.")},
        )
    )

    skill_root = home_root / ".agents" / "skills" / "demo"
    assert not extra.exists()
    assert "New." in (skill_root / "SKILL.md").read_text(encoding="utf-8")


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
