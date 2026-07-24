from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import (
    MCP_RESULT_PROFILE_ENV,
    MCP_RESULT_PROFILE_HTTP_HEADER,
    HomeService,
    HomeStore,
    McpServerSpec,
)
from agent_runtime_kit.agent.provider_contracts import BaseConfigSource, ProviderHomeSpec
from agent_runtime_kit.agent.providers.codex_home import CodexHomeOptions


def test_provider_home_spec_materializes_schema_v3_codex_home(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime")
    record = service.create_home(
        ProviderHomeSpec(
            provider_type="codex",
            home_id="worker",
            base_config=BaseConfigSource(text='model = "gpt-5"\n'),
            config_overrides={"model_reasoning_effort": "high"},
            provider_options=CodexHomeOptions(),
            required_env=("TOKEN",),
        )
    )

    root = service.resolve_home_root("codex", "worker")

    assert record.schema_version == 3
    assert record.provider_type == "codex"
    assert "model_reasoning_effort = \"high\"" in (root / ".codex" / "config.toml").read_text()
    assert service.get_home("codex", "worker").required_env == {"TOKEN"}


def test_home_service_requires_registered_provider_renderer(tmp_path: Path) -> None:
    service = HomeService(tmp_path / ".agent_runtime", renderers={})

    with pytest.raises(ValueError, match="no Home renderer"):
        service.create_home(ProviderHomeSpec(provider_type="unknown", home_id="worker"))


def test_mcp_server_result_profile_projects_to_fixed_transport_configuration() -> None:
    http = McpServerSpec(
        name="http",
        transport="http",
        url="https://mcp.example/rpc",
        result_profile="content_only",
    )
    stdio = McpServerSpec(
        name="stdio",
        transport="stdio",
        command="mcp-server",
        result_profile="dual",
    )

    assert http.result_profile == "content_only"
    assert http.http_headers[MCP_RESULT_PROFILE_HTTP_HEADER] == "content_only"
    assert MCP_RESULT_PROFILE_ENV not in http.env
    assert stdio.result_profile == "dual"
    assert stdio.env[MCP_RESULT_PROFILE_ENV] == "dual"
    assert MCP_RESULT_PROFILE_HTTP_HEADER not in stdio.http_headers


def test_mcp_server_result_profile_rejects_invalid_or_conflicting_configuration() -> None:
    with pytest.raises(ValueError, match="unsupported MCP result_profile"):
        McpServerSpec(name="bad", result_profile="summary")
    with pytest.raises(ValueError, match="conflicts with result_profile"):
        McpServerSpec(
            name="bad-header",
            url="https://mcp.example/rpc",
            result_profile="content_only",
            http_headers={"X-Ark-Mcp-Result-Profile": "dual"},
        )
    with pytest.raises(ValueError, match="conflicts with result_profile"):
        McpServerSpec(
            name="bad-env",
            transport="stdio",
            command="mcp-server",
            result_profile="content_only",
            env={MCP_RESULT_PROFILE_ENV: "dual"},
        )


def test_home_store_does_not_migrate_pre_v3_sql_schema(tmp_path: Path) -> None:
    root = tmp_path / ".agent_runtime"
    store = HomeStore(root)
    record = HomeService(root).create_home(
        ProviderHomeSpec(provider_type="codex", home_id="worker")
    )
    assert store.get_home("codex", "worker") == record
