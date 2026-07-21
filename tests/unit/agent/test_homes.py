import json
import sqlite3
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import (
    HomeCreateSpec,
    HomeService,
    McpServerSpec,
    ModelConfigOverrides,
    build_provider_env,
)
from agent_runtime_kit.agent.models import MissingProviderEnvError
from agent_runtime_kit.agent.provider_contracts import (
    BaseConfigSource,
    HomeMaterializationResult,
    HomeValidationResult,
    ProviderHomeSpec,
)
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


def test_create_codex_home_projects_model_overrides_before_tables(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        "model = 'terra'\nmodel_reasoning_effort = 'medium'\n\n[features]\napps = false\n",
        encoding="utf-8",
    )
    service = HomeService(tmp_path / ".agent_runtime")
    spec = HomeCreateSpec(
        cli_type="codex",
        home_id="planner",
        base_config_path=config,
        model_config_overrides=ModelConfigOverrides(model="sol", reasoning_effort="high"),
    )

    service.create_home(spec)
    service.create_home(spec)

    rendered = (
        service.resolve_home_root("codex", "planner") / ".codex" / "config.toml"
    ).read_text(encoding="utf-8")
    assert rendered.count('model = "sol"') == 1
    assert rendered.count('model_reasoning_effort = "high"') == 1
    assert "terra" not in rendered
    assert "medium" not in rendered
    assert rendered.index('model = "sol"') < rendered.index("[features]")


def test_create_codex_home_supports_partial_model_override(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("model = 'terra'\nmodel_reasoning_effort = 'high'\n", encoding="utf-8")
    service = HomeService(tmp_path / ".agent_runtime")

    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            base_config_path=config,
            model_config_overrides=ModelConfigOverrides(model="sol"),
        )
    )

    rendered = (
        service.resolve_home_root("codex", "planner") / ".codex" / "config.toml"
    ).read_text(encoding="utf-8")
    assert 'model = "sol"' in rendered
    assert "model_reasoning_effort = 'high'" in rendered


def test_model_overrides_reject_empty_values_and_unsupported_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model override must be a non-empty string"):
        ModelConfigOverrides(model=" ")

    service = HomeService(tmp_path / ".agent_runtime")
    with pytest.raises(ValueError, match="not supported for provider"):
        service.create_home(
            HomeCreateSpec(
                cli_type="future-provider",
                home_id="planner",
                model_config_overrides=ModelConfigOverrides(model="example"),
            )
        )


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


def test_codex_home_records_versioned_materialization_manifest(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")
    spec = HomeCreateSpec(
        cli_type="codex",
        home_id="planner",
        config_overrides={"model": "gpt-example", "model_reasoning_effort": "high"},
    )

    first = service.create_home(spec)
    second = service.create_home(spec)
    manifest_path = service.runtime_root / str(second.materialization_manifest_ref)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert first.provider_type == "codex"
    assert first.cli_type == "codex"
    assert first.schema_version == 2
    assert first.materialization_manifest_hash == second.materialization_manifest_hash
    assert manifest["manifest_hash"] == second.materialization_manifest_hash
    assert "home.base_config" in second.capability_snapshot["supports"]
    assert second.resolved_defaults == {
        "api_provider": "openai",
        "api_mode": "responses",
        "endpoint_id": None,
        "requested_model": "gpt-example",
        "resolved_model": None,
        "model_version": None,
        "service_tier": None,
        "reasoning_effort": "high",
        "tokenizer_id": None,
        "model_config_hash": None,
        "provider_payload": None,
    }


def test_provider_home_spec_uses_registered_renderer_without_service_type_branch(tmp_path: Path) -> None:
    class DemoRenderer:
        provider_type = "demo"

        def validate(self, spec):  # noqa: ANN001, ANN201
            return HomeValidationResult(valid=True)

        def materialize(self, spec, home_root):  # noqa: ANN001, ANN201
            (home_root / "demo.txt").write_text("demo", encoding="utf-8")
            return HomeMaterializationResult(
                provider_type="demo",
                home_id=spec.home_id,
                renderer_version="demo-1",
                manifest_schema_version=1,
                manifest_hash="demo-hash",
            )

    service = HomeService(
        tmp_path / ".agent_runtime",
        renderers={"demo": DemoRenderer()},
    )
    record = service.create_home(
        ProviderHomeSpec(
            provider_type="demo",
            home_id="worker",
            base_config=BaseConfigSource(text="demo"),
        )
    )

    assert record.provider_type == "demo"
    assert record.materialization_manifest_hash == "demo-hash"
    assert (service.resolve_home_root("demo", "worker") / "demo.txt").read_text() == "demo"


def test_home_store_additively_reads_legacy_sqlite_rows(tmp_path: Path) -> None:
    runtime_root = tmp_path / ".agent_runtime"
    homes_root = runtime_root / "homes"
    homes_root.mkdir(parents=True)
    with sqlite3.connect(homes_root / "index.sqlite") as conn:
        conn.execute(
            """
            create table homes(
              cli_type text not null,
              home_id text not null,
              home_relpath text not null,
              status text not null,
              created_at text not null,
              updated_at text not null,
              fixed_env_json text not null default '{}',
              required_env_csv text not null default '',
              primary key(cli_type, home_id)
            )
            """
        )
        conn.execute(
            "insert into homes values (?, ?, ?, ?, ?, ?, ?, ?)",
            ("codex", "legacy", "homes/codex/legacy", "active", "old", "old", "{}", ""),
        )

    record = HomeService(runtime_root).get_home("codex", "legacy")

    assert record.schema_version == 1
    assert record.provider_type == "codex"
    assert record.materialization_manifest_hash is None


def test_codex_execution_context_checks_manifest_and_managed_file_hashes(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")
    service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id="planner",
            config_overrides={"model": "gpt-example"},
            fixed_env={"HOME_VALUE": "home"},
            required_env={"REQUIRED"},
        )
    )

    context = service.build_execution_context(
        "codex",
        "planner",
        run_env={"REQUIRED": "yes", "HOME_VALUE": "run"},
        workdir=str(tmp_path),
    )
    assert context.provider_type == "codex"
    assert context.process_environment["HOME_VALUE"] == "run"
    assert context.process_environment["CODEX_HOME"].endswith("/.codex")

    config_path = service.resolve_home_root("codex", "planner") / ".codex" / "config.toml"
    config_path.write_text('model = "tampered"\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="materialized file hash mismatch"):
        service.build_execution_context("codex", "planner", run_env={"REQUIRED": "yes"})
