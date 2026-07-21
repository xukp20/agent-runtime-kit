from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeRecord
from agent_runtime_kit.agent.provider_contracts import (
    BaseConfigSource,
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
    ModelBackendIdentity,
    ProviderContextCompactionRequest,
    ProviderForkRequest,
    ProviderHomeSpec,
    ProviderRunOptions,
    ProviderRunRequest,
    ProviderTurnQuery,
)
from agent_runtime_kit.agent.providers.opencode_artifacts import OpenCodeArtifactAdapter
from agent_runtime_kit.agent.providers.opencode_context import OpenCodeContextAdapter
from agent_runtime_kit.agent.providers.opencode_home import OpenCodeHomeRenderer
from agent_runtime_kit.agent.providers.opencode_models import OpenCodeHomeOptions
from agent_runtime_kit.agent.providers.opencode_models import OpenCodeRunOptions
from agent_runtime_kit.agent.providers.opencode_query import OpenCodeQueryAdapter
from agent_runtime_kit.agent.providers.opencode_runtime import (
    OpenCodeRuntimeAdapter,
    OpenCodeRuntimeRegistry,
)


pytestmark = pytest.mark.real


def test_real_opencode_server_health_session_and_isolated_database(tmp_path: Path) -> None:
    binary = os.environ.get("ARK_OPENCODE_TEST_BINARY")
    if not binary:
        pytest.skip("set ARK_OPENCODE_TEST_BINARY to an OpenCode 1.18.4 executable")
    runtime_root = tmp_path / "runtime"
    home_root = runtime_root / "homes" / "opencode" / "real"
    renderer = OpenCodeHomeRenderer(runtime_root=runtime_root)
    materialization = renderer.materialize(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="real",
            base_config=BaseConfigSource(
                mapping={"model": "deepseek/deepseek-chat", "snapshot": True}
            ),
            provider_options=OpenCodeHomeOptions(binary_path=binary),
        ),
        home_root,
    )
    record = HomeRecord(
        cli_type="opencode",
        home_id="real",
        home_relpath="homes/opencode/real",
        materialization_manifest_hash=materialization.manifest_hash,
    )
    context = renderer.build_execution_context(record, run_env={}, workdir=str(tmp_path))
    registry = OpenCodeRuntimeRegistry(runtime_root, binary_path=binary)
    request = ProviderRunRequest(
        agent_id="agent-real",
        scope_id="scope-real",
        agent_type="build",
        provider_type="opencode",
        home_id="real",
        prompt="not submitted",
        workdir=str(tmp_path),
        model_overrides=ModelBackendIdentity(
            api_provider="deepseek",
            api_mode="chat_completions",
            requested_model="deepseek-chat",
        ),
        execution_context=context,
    )
    try:
        server = registry.ensure(request)
        health = server.client.health()
        assert health.get("healthy") is True
        assert health.get("version") == "1.18.4"
        session = server.client.create_session()
        assert str(session.get("id", "")).startswith("ses_")
        assert server.client.session_status() == {}
        assert server.database_path.is_file()
        assert str(server.database_path).startswith(str(runtime_root / "providers" / "opencode"))
        assert server.directory == str(tmp_path.resolve())
    finally:
        registry.close()


def test_real_opencode_deepseek_run_and_query(tmp_path: Path) -> None:
    binary = os.environ.get("ARK_OPENCODE_TEST_BINARY")
    key = os.environ.get("ARK_OPENCODE_REAL_DEEPSEEK_KEY")
    if os.environ.get("ARK_OPENCODE_RUN_REAL_MODELS") != "1" or not binary or not key:
        pytest.skip("enable the gated OpenCode DeepSeek real test")
    runtime_root = tmp_path / "runtime"
    home_root = runtime_root / "homes" / "opencode" / "deepseek"
    renderer = OpenCodeHomeRenderer(runtime_root=runtime_root)
    materialization = renderer.materialize(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="deepseek",
            base_config=BaseConfigSource(
                mapping={
                    "model": "deepseek/deepseek-chat",
                    "provider": {
                        "deepseek": {
                            "npm": "@ai-sdk/openai-compatible",
                            "options": {
                                "baseURL": "https://api.deepseek.com/v1",
                                "apiKey": "{env:DEEPSEEK_API_KEY}",
                            },
                            "models": {"deepseek-chat": {"name": "DeepSeek Chat"}},
                        }
                    },
                }
            ),
            provider_options=OpenCodeHomeOptions(binary_path=binary),
        ),
        home_root,
    )
    record = HomeRecord(
        cli_type="opencode",
        home_id="deepseek",
        home_relpath="homes/opencode/deepseek",
        materialization_manifest_hash=materialization.manifest_hash,
    )
    context = renderer.build_execution_context(
        record,
        run_env={"DEEPSEEK_API_KEY": key},
        workdir=str(tmp_path),
    )
    registry = OpenCodeRuntimeRegistry(runtime_root, binary_path=binary)
    runtime = OpenCodeRuntimeAdapter(registry)
    request = ProviderRunRequest(
        agent_id="agent-deepseek",
        scope_id="scope-real",
        agent_type="worker",
        provider_type="opencode",
        home_id="deepseek",
        prompt="Reply with exactly OPENCODE_OK and no other text.",
        workdir=str(tmp_path),
        model_overrides=ModelBackendIdentity(
            api_provider="deepseek",
            api_mode="chat_completions",
            requested_model="deepseek-chat",
        ),
        run_options=ProviderRunOptions(timeout_s=120),
        provider_options=OpenCodeRunOptions(
            provider_id="deepseek",
            model_id="deepseek-chat",
            tools={"bash": False, "edit": False, "write": False},
        ),
        execution_context=context,
    )
    try:
        result = runtime.start(request).wait_terminal(130)
        assert result.status.value == "completed"
        assert result.session_locator.native_locator is not None
        assert result.turn_usage is not None
        assert "OPENCODE_OK" in (result.final_text or "")
        assert result.artifact_locator is not None

        query = OpenCodeQueryAdapter(registry.client_for_locator)
        artifacts = OpenCodeArtifactAdapter(runtime_root=runtime_root, registry=registry)
        snapshot = artifacts.capture(
            ArtifactCaptureRequest(
                session=result.session_locator,
                snapshot_root=str(tmp_path / "snapshot"),
                agent_id="agent-deepseek",
            )
        )
        second = runtime.resume(
            replace(
                request,
                prompt="Reply with exactly SECOND_TURN and no other text.",
                session_locator=result.session_locator,
            )
        ).wait_terminal(130)
        assert "SECOND_TURN" in (second.final_text or "")

        restored = artifacts.restore(
            ArtifactRestoreRequest(
                manifest=snapshot.manifest,
                snapshot_root=snapshot.snapshot_root,
            )
        )
        assert restored.restored
        third = runtime.resume(
            replace(
                request,
                prompt="Reply with exactly RESTORED_OK and no other text.",
                session_locator=result.session_locator,
            )
        ).wait_terminal(130)
        assert "RESTORED_OK" in (third.final_text or "")
        turns = query.list_turns(ProviderTurnQuery(session=third.session_locator)).items
        texts = [turn.result.final_text for turn in turns if turn.result is not None]
        assert any(text and "OPENCODE_OK" in text for text in texts)
        assert any(text and "RESTORED_OK" in text for text in texts)
        assert not any(text and "SECOND_TURN" in text for text in texts)

        context_adapter = OpenCodeContextAdapter(registry=registry, query=query)
        compact = context_adapter.compact(
            ProviderContextCompactionRequest(
                session=third.session_locator,
                trigger="real_test",
                timeout_s=120,
                agent_id="agent-deepseek",
            )
        )
        assert compact.status == "completed"

        forked = runtime.fork(
            ProviderForkRequest(
                source_agent_id="agent-deepseek",
                source_session=third.session_locator,
                target_agent_id="agent-deepseek-fork",
                target_scope_id="scope-fork",
                target_home_id="deepseek",
            )
        )
        assert forked.workspace_isolated is False
        fork_result = runtime.resume(
            replace(
                request,
                agent_id="agent-deepseek-fork",
                scope_id="scope-fork",
                prompt="Reply with exactly FORK_OK and no other text.",
                session_locator=forked.target_session,
            )
        ).wait_terminal(130)
        assert "FORK_OK" in (fork_result.final_text or "")
        assert fork_result.session_locator.native_locator != third.session_locator.native_locator
    finally:
        runtime.close()


def test_real_opencode_beeapi_responses_run(tmp_path: Path) -> None:
    binary = os.environ.get("ARK_OPENCODE_TEST_BINARY")
    key = os.environ.get("BEEAPI_API_KEY")
    if os.environ.get("ARK_OPENCODE_RUN_REAL_MODELS") != "1" or not binary or not key:
        pytest.skip("enable the gated OpenCode BeeAPI Responses real test")
    runtime_root = tmp_path / "runtime"
    home_root = runtime_root / "homes" / "opencode" / "beeapi"
    renderer = OpenCodeHomeRenderer(runtime_root=runtime_root)
    materialization = renderer.materialize(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="beeapi",
            base_config=BaseConfigSource(
                mapping={
                    "model": "beeapi-responses/gpt-5.4",
                    "small_model": "beeapi-responses/gpt-5.4",
                    "provider": {
                        "beeapi-responses": {
                            "npm": "@ai-sdk/openai",
                            "name": "BeeAPI Responses",
                            "options": {
                                "baseURL": "https://beeapi.ai/v1",
                                "apiKey": "{env:BEEAPI_API_KEY}",
                            },
                            "models": {
                                "gpt-5.4": {
                                    "name": "GPT-5.4 (BeeAPI Responses)",
                                    "limit": {"context": 1050000, "output": 128000},
                                }
                            },
                        }
                    },
                }
            ),
            provider_options=OpenCodeHomeOptions(binary_path=binary),
        ),
        home_root,
    )
    record = HomeRecord(
        cli_type="opencode",
        home_id="beeapi",
        home_relpath="homes/opencode/beeapi",
        materialization_manifest_hash=materialization.manifest_hash,
    )
    context = renderer.build_execution_context(
        record, run_env={"BEEAPI_API_KEY": key}, workdir=str(tmp_path)
    )
    registry = OpenCodeRuntimeRegistry(runtime_root, binary_path=binary)
    runtime = OpenCodeRuntimeAdapter(registry)
    request = ProviderRunRequest(
        agent_id="agent-beeapi",
        scope_id="scope-real",
        agent_type="worker",
        provider_type="opencode",
        home_id="beeapi",
        prompt="Reply with exactly RESPONSES_OK and no other text.",
        workdir=str(tmp_path),
        model_overrides=ModelBackendIdentity(
            api_provider="beeapi-responses",
            api_mode="responses",
            requested_model="gpt-5.4",
        ),
        run_options=ProviderRunOptions(timeout_s=180),
        provider_options=OpenCodeRunOptions(
            provider_id="beeapi-responses",
            model_id="gpt-5.4",
            tools={"bash": False, "edit": False, "write": False},
        ),
        execution_context=context,
    )
    try:
        result = runtime.start(request).wait_terminal(190)
        assert result.status.value == "completed"
        assert "RESPONSES_OK" in (result.final_text or "")
        assert result.turn_usage is not None
        assert result.turn_usage.requests
        assert result.turn_usage.requests[-1].model_identity.api_provider == "beeapi-responses"
        assert result.turn_usage.requests[-1].model_identity.api_mode == "responses"
    finally:
        runtime.close()
