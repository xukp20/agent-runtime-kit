from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import (
    MCP_RESULT_PROFILE_ENV,
    HomeService,
    McpServerSpec,
)
from agent_runtime_kit.agent.provider_contracts import (
    BaseConfigSource,
    ModelBackendIdentity,
    ProviderHomeSpec,
)
from agent_runtime_kit.agent.providers.claude_code_home import (
    ClaudeCodeHomeOptions,
    ClaudeCodeHomeRenderer,
)
from agent_runtime_kit.agent.providers.claude_code import ClaudeCodeProvider
from agent_runtime_kit.agent.providers.claude_code_bundle import build_claude_code_provider_bundle
from agent_runtime_kit.agent.skills import SkillSpec


def _service(tmp_path: Path) -> HomeService:
    return HomeService(
        tmp_path,
        renderers={
            "claude_code": ClaudeCodeHomeRenderer(runtime_root=tmp_path),
        },
    )


def test_claude_home_materializes_settings_skills_mcp_and_isolated_env(tmp_path: Path) -> None:
    service = _service(tmp_path)
    spec = ProviderHomeSpec(
        provider_type="claude_code",
        home_id="worker",
        base_config=BaseConfigSource(mapping={"permissions": {"allow": ["Read"]}}),
        config_overrides={"permissions": {"deny": ["WebFetch"]}},
        model_config=ModelBackendIdentity(
            api_provider="deepseek",
            api_mode="anthropic_messages",
            requested_model="deepseek-chat",
        ),
        instructions=("Base instruction",),
        skills=(SkillSpec(name="lean", description="Lean help", body="Use Lean."),),
        mcp_servers=(
            McpServerSpec(
                name="probe",
                transport="stdio",
                command="python",
                args=["server.py"],
                required=True,
                env_vars=["MCP_TOKEN"],
                result_profile="content_only",
            ),
        ),
        required_env=("MCP_TOKEN",),
        provider_options=ClaudeCodeHomeOptions(
            cli_path="/opt/claude",
            setting_sources=("user",),
            tools=("Read",),
        ),
    )

    home = service.create_home(spec)
    root = service.resolve_home_root("claude_code", "worker")
    settings = json.loads((root / ".claude" / "settings.json").read_text())
    runtime = json.loads((root / ".ark" / "claude_code_home.json").read_text())

    assert settings == {
        "permissions": {"allow": ["Read"], "deny": ["WebFetch"]},
    }
    assert (root / ".claude" / "skills" / "lean" / "SKILL.md").is_file()
    assert runtime["system_prompt"] == "Base instruction"
    assert runtime["skills"] == ["lean"]
    assert home.resolved_defaults["api_provider"] == "deepseek"

    context = service.build_execution_context(
        "claude_code",
        "worker",
        run_env={"MCP_TOKEN": "runtime-secret"},
        workdir="/workspace",
    )
    assert context.process_environment["CLAUDE_CONFIG_DIR"] == str(root / ".claude")
    assert context.process_environment["HOME"] == str(root)
    assert context.runtime_payload["mcp_servers_resolved"]["probe"]["env"]["MCP_TOKEN"] == "runtime-secret"
    assert (
        context.runtime_payload["mcp_servers_resolved"]["probe"]["env"][
            MCP_RESULT_PROFILE_ENV
        ]
        == "content_only"
    )
    manifest_text = (root / ".ark" / "home_materialization.json").read_text()
    assert "runtime-secret" not in manifest_text


def test_claude_home_rejects_file_checkpointing_and_unmappable_mcp(tmp_path: Path) -> None:
    renderer = ClaudeCodeHomeRenderer(runtime_root=tmp_path)
    checkpoint = ProviderHomeSpec(
        provider_type="claude_code",
        home_id="checkpoint",
        provider_options=ClaudeCodeHomeOptions(enable_file_checkpointing=True),
    )
    assert not renderer.validate(checkpoint).valid

    unmappable = ProviderHomeSpec(
        provider_type="claude_code",
        home_id="mcp",
        mcp_servers=(
            McpServerSpec(name="bad", transport="stdio", command="server", cwd="/tmp"),
        ),
    )
    assert not renderer.validate(unmappable).valid


def test_claude_home_detects_unsealed_file_mutation(tmp_path: Path) -> None:
    service = _service(tmp_path)
    service.create_home(
        ProviderHomeSpec(
            provider_type="claude_code",
            home_id="worker",
            base_config=BaseConfigSource(mapping={"theme": "dark"}),
        )
    )
    root = service.resolve_home_root("claude_code", "worker")
    (root / ".claude" / "settings.json").write_text('{"theme":"light"}\n')

    with pytest.raises(RuntimeError, match="file hash mismatch"):
        service.build_execution_context("claude_code", "worker")


def test_claude_capabilities_require_verified_cli_for_context_control(tmp_path: Path) -> None:
    provider = ClaudeCodeProvider(runtime_root=tmp_path, sdk_loader=lambda: object())
    bundle = build_claude_code_provider_bundle(provider, runtime_root=tmp_path)
    service = HomeService(tmp_path, renderers={"claude_code": bundle.home_renderer})
    home = service.create_home(
        ProviderHomeSpec(provider_type="claude_code", home_id="worker")
    )

    unresolved = bundle.resolve_capabilities(home)
    assert not unresolved.get("control.compact").available
    marker = (
        service.resolve_home_root("claude_code", "worker")
        / ".ark"
        / "claude_home_initialized.json"
    )
    marker.write_text(json.dumps({"cli_version": "2.1.216"}) + "\n", encoding="utf-8")

    resolved = bundle.resolve_capabilities(home)

    assert resolved.get("control.compact").available
    assert resolved.get("query.context_usage").available
    assert resolved.get("model.other_api").available
    assert not resolved.get("model.responses").available
    assert not resolved.get("model.chat_completions").available
