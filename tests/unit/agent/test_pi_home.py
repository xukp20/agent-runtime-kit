from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeService, McpServerSpec
from agent_runtime_kit.agent.provider_contracts import (
    BaseConfigSource,
    CapabilityKey,
    ProviderHomeSpec,
)
from agent_runtime_kit.agent.providers.pi_bundle import build_pi_provider_bundle
from agent_runtime_kit.agent.providers.pi_home import PiHomeOptions
from agent_runtime_kit.agent.providers.pi_runtime import build_pi_command
from agent_runtime_kit.agent.instructions import TextFragment
from agent_runtime_kit.agent.skills import SkillSpec


def test_pi_home_materializes_isolated_config_resources_and_backend(tmp_path: Path) -> None:
    cli = tmp_path / "cli.js"
    cli.write_text("// pi", encoding="utf-8")
    auth = tmp_path / "auth.json"
    auth.write_text('{"openai-codex":{"access":"secret"}}\n', encoding="utf-8")
    bundle = build_pi_provider_bundle(runtime_root=tmp_path / "runtime")
    service = HomeService(
        tmp_path / "runtime",
        renderers={"pi": bundle.home_renderer},
    )

    home = service.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="demo",
            base_config=BaseConfigSource(
                mapping={"defaultProvider": "old", "defaultModel": "old-model"}
            ),
            config_overrides={"defaultProvider": "beeapi"},
            skills=(SkillSpec(name="lean", description="Lean", body="Use Lean."),),
            instructions=(TextFragment(key="base", text="Always verify the proof."),),
            required_env=("BEEAPI_API_KEY",),
            fixed_env={"ARK_FIXED": "yes"},
            provider_options=PiHomeOptions(
                auth_json_path=auth,
                pi_cli_path=cli,
                node_executable="/usr/bin/node",
                settings={"defaultModel": "gpt-5.4"},
                models={
                    "providers": {
                        "beeapi": {
                            "api": "openai-responses",
                            "baseUrl": "https://beeapi.example/v1",
                        }
                    }
                },
            ),
        )
    )

    root = service.resolve_home_root("pi", "demo")
    settings = json.loads((root / ".pi" / "settings.json").read_text(encoding="utf-8"))
    assert settings["defaultProvider"] == "beeapi"
    assert settings["defaultModel"] == "gpt-5.4"
    assert (root / ".pi" / "skills" / "lean" / "SKILL.md").is_file()
    assert (root / ".ark" / "pi_instructions.md").read_text(encoding="utf-8") == (
        "Always verify the proof.\n"
    )
    assert (root / ".pi" / "auth.json").read_text(encoding="utf-8") == auth.read_text(encoding="utf-8")
    assert home.resolved_defaults is not None
    assert home.resolved_defaults["api_mode"] == "responses"

    context = service.build_execution_context(
        "pi",
        "demo",
        run_env={"BEEAPI_API_KEY": "runtime-secret"},
        workdir=str(tmp_path),
    )
    assert context.process_environment["PI_CODING_AGENT_DIR"] == str(root / ".pi")
    assert context.process_environment["BEEAPI_API_KEY"] == "runtime-secret"
    assert context.resolved_defaults is not None
    assert context.resolved_defaults.effective_model == "gpt-5.4"
    command = build_pi_command(context, model=context.resolved_defaults, session_id="session-id")
    assert "--append-system-prompt" in command
    assert "Always verify the proof.\n" in command
    capabilities = bundle.resolve_capabilities(home, context.resolved_defaults)
    assert capabilities.available(CapabilityKey.MODEL_RESPONSES)
    assert not capabilities.available(CapabilityKey.MODEL_CHAT_COMPLETIONS)


def test_pi_home_mcp_bridge_manifest_uses_env_references_and_packaged_extension(tmp_path: Path) -> None:
    cli = tmp_path / "cli.js"
    cli.write_text("// pi", encoding="utf-8")
    mcp_runtime = tmp_path / "mcp-runtime"
    (mcp_runtime / "node_modules" / "@modelcontextprotocol" / "sdk").mkdir(parents=True)
    bundle = build_pi_provider_bundle(runtime_root=tmp_path / "runtime")
    service = HomeService(tmp_path / "runtime", renderers={"pi": bundle.home_renderer})
    service.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="mcp",
            mcp_servers=(
                McpServerSpec(
                    name="demo",
                    transport="http",
                    url="https://mcp.example/rpc",
                    required=True,
                    bearer_token_env_var="MCP_TOKEN",
                ),
            ),
            provider_options=PiHomeOptions(
                pi_cli_path=cli,
                node_executable="/usr/bin/node",
                mcp_runtime_root=mcp_runtime,
            ),
        )
    )
    root = service.resolve_home_root("pi", "mcp")
    manifest = json.loads((root / ".ark" / "pi_mcp_manifest.json").read_text(encoding="utf-8"))
    assert manifest["servers"][0]["bearer_token_env_var"] == "MCP_TOKEN"
    assert "token-value" not in json.dumps(manifest)
    assert (root / ".pi" / "extensions" / "ark_pi_mcp_bridge.mjs").is_file()
    context = service.build_execution_context("pi", "mcp", run_env={"MCP_TOKEN": "token-value"})
    assert context.process_environment["ARK_PI_MCP_RUNTIME_ROOT"] == str(mcp_runtime)


def test_pi_home_initialization_fails_closed_without_mcp_sdk_runtime(tmp_path: Path) -> None:
    cli = tmp_path / "cli.js"
    cli.write_text("// pi", encoding="utf-8")
    bundle = build_pi_provider_bundle(runtime_root=tmp_path / "runtime")
    service = HomeService(tmp_path / "runtime", renderers={"pi": bundle.home_renderer})
    home = service.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="mcp-missing-runtime",
            mcp_servers=(McpServerSpec(name="demo", command="demo", transport="stdio"),),
            provider_options=PiHomeOptions(
                pi_cli_path=cli,
                node_executable="/usr/bin/node",
            ),
        )
    )
    context = service.build_execution_context("pi", "mcp-missing-runtime")

    with pytest.raises(RuntimeError, match="MCP bridge requires"):
        bundle.home_renderer.initialize(home, context)


def test_pi_home_rejects_managed_cli_overrides(tmp_path: Path) -> None:
    bundle = build_pi_provider_bundle(runtime_root=tmp_path / "runtime")
    service = HomeService(tmp_path / "runtime", renderers={"pi": bundle.home_renderer})
    with pytest.raises(ValueError, match="ARK-managed"):
        service.create_home(
            ProviderHomeSpec(
                provider_type="pi",
                home_id="invalid",
                provider_options=PiHomeOptions(extra_cli_args=("--session", "other")),
            )
        )


def test_pi_home_resolves_messages_backend_as_other_native_api(tmp_path: Path) -> None:
    cli = tmp_path / "cli.js"
    cli.write_text("// pi", encoding="utf-8")
    bundle = build_pi_provider_bundle(runtime_root=tmp_path / "runtime")
    service = HomeService(tmp_path / "runtime", renderers={"pi": bundle.home_renderer})
    home = service.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="messages",
            provider_options=PiHomeOptions(
                pi_cli_path=cli,
                node_executable="/usr/bin/node",
                settings={"defaultProvider": "anthropic", "defaultModel": "claude"},
                models={
                    "providers": {
                        "anthropic": {
                            "api": "anthropic-messages",
                            "baseUrl": "https://example.invalid",
                        }
                    }
                },
            ),
        )
    )

    capabilities = bundle.resolve_capabilities(home)
    assert capabilities.available(CapabilityKey.MODEL_OTHER_API)
    assert not capabilities.available(CapabilityKey.MODEL_RESPONSES)
    assert not capabilities.available(CapabilityKey.MODEL_CHAT_COMPLETIONS)
