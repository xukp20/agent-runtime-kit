from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeService, McpServerSpec
from agent_runtime_kit.agent.instructions import TextFragment
from agent_runtime_kit.agent.provider_contracts import ProviderHomeSpec
from agent_runtime_kit.agent.providers import GrokHomeOptions, build_grok_provider_bundle
from agent_runtime_kit.agent.providers.grok_runtime import build_grok_command


def test_grok_home_materializes_isolated_profile_auth_and_instructions(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n", encoding="utf-8")
    binary = tmp_path / "grok"
    binary.write_text("fixture", encoding="utf-8")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path / "runtime", binary_path=binary)
    service = HomeService(tmp_path / "runtime", renderers={"grok": bundle.home_renderer})
    service.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id="demo",
            instructions=(TextFragment(key="base", text="HOME_INSTRUCTION_MARKER"),),
            provider_options=GrokHomeOptions(auth_json_path=auth),
        )
    )

    root = service.resolve_home_root("grok", "demo")
    assert (root / ".grok" / "auth.json").is_symlink()
    assert (root / ".grok" / "auth.json").resolve() == auth.resolve()
    profile = (root / ".ark" / "grok-profile.md").read_text(encoding="utf-8")
    assert "HOME_INSTRUCTION_MARKER" in profile
    for name in ("read_file", "list_dir", "grep"):
        assert f"GrokBuild:{name}" in profile
    context = service.build_execution_context("grok", "demo", workdir=str(tmp_path))
    assert context.process_environment["HOME"] == str(root / ".ark" / "os_home")
    assert context.process_environment["GROK_HOME"] == str(root / ".grok")
    assert context.process_environment["GROK_DISABLE_AUTOUPDATER"] == "1"


def test_grok_home_model_and_reasoning_survive_execution_context_into_command(tmp_path: Path) -> None:
    binary = tmp_path / "grok"
    binary.write_text("fixture", encoding="utf-8")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path / "runtime", binary_path=binary)
    service = HomeService(tmp_path / "runtime", renderers={"grok": bundle.home_renderer})
    service.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id="model",
            provider_options=GrokHomeOptions(
                auth_json_path=None,
                model="grok-4.6",
                reasoning_effort="high",
            ),
        )
    )

    context = service.build_execution_context("grok", "model", workdir=str(tmp_path))
    assert context.resolved_defaults is not None
    assert context.resolved_defaults.requested_model == "grok-4.6"
    assert context.resolved_defaults.reasoning_effort == "high"
    command = build_grok_command(context, model=context.resolved_defaults)
    assert command[command.index("--model") + 1] == "grok-4.6"
    assert command[command.index("--reasoning-effort") + 1] == "high"


def test_grok_home_tool_none_empty_and_ambiguous_semantics(tmp_path: Path) -> None:
    binary = tmp_path / "grok"
    binary.write_text("fixture", encoding="utf-8")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path / "runtime", binary_path=binary)
    service = HomeService(tmp_path / "runtime", renderers={"grok": bundle.home_renderer})

    with pytest.raises(ValueError, match="explicitly empty"):
        service.create_home(
            ProviderHomeSpec(
                provider_type="grok",
                home_id="empty",
                provider_options=GrokHomeOptions(auth_json_path=None, tools=()),
            )
        )
    with pytest.raises(ValueError, match="either spec.tools"):
        service.create_home(
            ProviderHomeSpec(
                provider_type="grok",
                home_id="ambiguous",
                tools=("read_file",),
                provider_options=GrokHomeOptions(auth_json_path=None, tools=("grep",)),
            )
        )


def test_grok_home_renders_only_native_stdio_http_mcp_and_hidden_tools(tmp_path: Path) -> None:
    binary = tmp_path / "grok"
    binary.write_text("fixture", encoding="utf-8")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path / "runtime", binary_path=binary)
    service = HomeService(tmp_path / "runtime", renderers={"grok": bundle.home_renderer})
    service.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id="mcp",
            mcp_servers=(
                McpServerSpec(name="local", transport="stdio", command="server", args=["--stdio"]),
                McpServerSpec(name="remote", transport="http", url="https://example.invalid/mcp"),
            ),
            provider_options=GrokHomeOptions(auth_json_path=None, tools=("read_file",)),
        )
    )
    root = service.resolve_home_root("grok", "mcp")
    config = (root / ".grok" / "config.toml").read_text(encoding="utf-8")
    assert "[managed_mcps]" in config and "gateway_tools_enabled = false" in config
    assert "[mcp_servers.local]" in config and "[mcp_servers.remote]" in config
    runtime = json.loads((root / ".ark" / "grok_runtime.json").read_text(encoding="utf-8"))
    assert runtime["tools"] == ["read_file", "search_tool", "use_tool"]

    with pytest.raises(ValueError, match="SSE MCP transport is unsupported"):
        service.create_home(
            ProviderHomeSpec(
                provider_type="grok",
                home_id="sse",
                mcp_servers=(McpServerSpec(name="legacy", transport="sse", url="https://x/sse"),),
                provider_options=GrokHomeOptions(auth_json_path=None),
            )
        )


def test_grok_home_session_start_commit_accepts_only_verified_marketplace_metadata(tmp_path: Path) -> None:
    binary = tmp_path / "grok"
    binary.write_text("fixture", encoding="utf-8")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path / "runtime", binary_path=binary)
    service = HomeService(tmp_path / "runtime", renderers={"grok": bundle.home_renderer})
    service.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id="demo",
            provider_options=GrokHomeOptions(auth_json_path=None),
        )
    )
    root = service.resolve_home_root("grok", "demo")
    config = root / ".grok" / "config.toml"
    config.write_text(
        config.read_text(encoding="utf-8")
        + '\n[marketplace]\ndefault_skills_installs_purged = true\n'
        + 'official_marketplace_auto_installed = true\n\n[[marketplace.sources]]\n'
        + 'name = "xAI Official"\ngit = "https://github.com/xai-org/plugin-marketplace.git"\n',
        encoding="utf-8",
    )
    service.commit_provider_lifecycle_materialization("grok", "demo", lifecycle="session_start")
    (root / ".grok" / "sessions" / "native-data").mkdir(parents=True)
    service.build_execution_context("grok", "demo", workdir=str(tmp_path))
