from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeService, McpServerSpec
from agent_runtime_kit.agent.instructions import TextFragment
from agent_runtime_kit.agent.provider_contracts import ProviderHomeSpec
from agent_runtime_kit.agent.providers import GrokHomeOptions, build_grok_provider_bundle
from agent_runtime_kit.agent.providers.grok_runtime import build_grok_command
from agent_runtime_kit.agent.skills import SkillSpec
from agent_runtime_kit.agent.providers.grok_home import validate_grok_workdir


def test_standard_mcp_dynamic_maps_and_skill_query(tmp_path: Path) -> None:
    binary = tmp_path / "grok"
    binary.write_text("fixture")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path, binary_path=binary)
    service = HomeService(tmp_path, renderers={"grok": bundle.home_renderer})
    service.create_home(ProviderHomeSpec(
        provider_type="grok", home_id="dynamic",
        skills=(SkillSpec(name="receipt", description="receipt", body="Read me"),),
        mcp_servers=(
            McpServerSpec(name="http", url="http://localhost/mcp", env_http_headers={"x-step": "ARK_STEP_ID"}, result_profile="content_only"),
            McpServerSpec(name="stdio", transport="stdio", command="fixture", env_vars=["ARK_STEP_ID"], result_profile="content_only"),
        ),
        provider_options=GrokHomeOptions(auth_json_path=None),
    ))
    root = service.resolve_home_root("grok", "dynamic")
    config = tomllib.loads((root / ".grok/config.toml").read_text())
    assert config["mcp_servers"]["http"]["headers"] == {"x-step": "${ARK_STEP_ID:-}", "x-ark-mcp-result-profile": "content_only"}
    assert config["mcp_servers"]["stdio"]["env"] == {"ARK_STEP_ID": "${ARK_STEP_ID:-}", "ARK_MCP_RESULT_PROFILE": "content_only"}
    assert service.get_skill_paths("grok", "dynamic") == {"receipt": root / ".grok/skills/receipt"}
    for value in ("step-a", "step-b"):
        ctx = service.build_execution_context("grok", "dynamic", run_env={"ARK_STEP_ID": value})
        assert ctx.process_environment["ARK_STEP_ID"] == value
    for server in (
        McpServerSpec(name="bad", url="http://localhost/mcp", http_headers={"X-Step": "fixed"}, env_http_headers={"x-step": "ARK_STEP_ID"}),
        McpServerSpec(name="bad", command="fixture", transport="stdio", env={"ID": "fixed"}, env_vars=["ID"]),
    ):
        validation = bundle.home_renderer.validate(ProviderHomeSpec(provider_type="grok", home_id="bad", mcp_servers=(server,), provider_options=GrokHomeOptions(auth_json_path=None)))
        assert not validation.valid
        assert "conflicting" in str(validation.errors)


def test_old_skill_home_remains_valid_and_query_rejects_escape(tmp_path: Path) -> None:
    binary = tmp_path / "grok"
    binary.write_text("fixture")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path, binary_path=binary)
    service = HomeService(tmp_path, renderers={"grok": bundle.home_renderer})
    service.create_home(ProviderHomeSpec(provider_type="grok", home_id="legacy",
        skills=(SkillSpec(name="receipt", description="receipt", body="Read me"),),
        provider_options=GrokHomeOptions(auth_json_path=None)))
    root = service.resolve_home_root("grok", "legacy")
    (root / ".grok/skills").rename(root / ".ark/grok-skills")
    service.seal_home_materialization("grok", "legacy")
    service.build_execution_context("grok", "legacy")
    assert service.get_skill_paths("grok", "legacy")["receipt"] == root / ".ark/grok-skills/receipt"
    manifest = root / ".ark/home_materialization.json"
    payload = json.loads(manifest.read_text())
    payload["generated_files"].append({"relpath": "../../escape/SKILL.md", "sha256": "unused"})
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        service.get_skill_paths("grok", "legacy")


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
def test_managed_skills_are_sealed_and_only_explicit_specs_are_accepted(tmp_path: Path) -> None:
    binary = tmp_path / "grok"
    binary.write_text("fixture")
    bundle = build_grok_provider_bundle(runtime_root=tmp_path / "runtime", binary_path=binary)
    service = HomeService(tmp_path / "runtime", renderers={"grok": bundle.home_renderer})
    skill = SkillSpec(name="receipt", description="receipt", body="Read resource", files={"resource.txt": "original"})
    options = GrokHomeOptions(auth_json_path=None)
    for skills in (("unknown",), (skill, skill)):
        with pytest.raises(ValueError):
            service.create_home(ProviderHomeSpec(provider_type="grok", home_id="invalid", skills=skills, provider_options=options))
    service.create_home(ProviderHomeSpec(provider_type="grok", home_id="skills", skills=(skill,), provider_options=options))
    root = service.resolve_home_root("grok", "skills")
    service.build_execution_context("grok", "skills")
    resource = root / ".grok/skills/receipt/resource.txt"
    resource.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        service.build_execution_context("grok", "skills")
    resource.write_text("original")
    (resource.parent / "extra.txt").write_text("unsealed")
    with pytest.raises(RuntimeError, match="file set"):
        service.build_execution_context("grok", "skills")


def test_managed_skill_mode_rejects_project_skill_discovery(tmp_path: Path) -> None:
    (tmp_path / ".grok/skills").mkdir(parents=True)
    validate_grok_workdir(tmp_path)
    with pytest.raises(RuntimeError, match="configuration"):
        validate_grok_workdir(tmp_path, managed_skills=True)
