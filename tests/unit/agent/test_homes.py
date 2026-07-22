from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeService, HomeStore
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


def test_home_store_does_not_migrate_pre_v3_sql_schema(tmp_path: Path) -> None:
    root = tmp_path / ".agent_runtime"
    store = HomeStore(root)
    record = HomeService(root).create_home(
        ProviderHomeSpec(provider_type="codex", home_id="worker")
    )
    assert store.get_home("codex", "worker") == record
