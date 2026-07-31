from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeRecord
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.provider_contracts import (
    BaseConfigSource,
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
    ModelBackendIdentity,
    ProviderContextCompactionRequest,
    ProviderContextQuery,
    ProviderForkRequest,
    ProviderHomeSpec,
    ProviderRunOptions,
    ProviderRunRequest,
    ProviderRegistry,
    ProviderSessionLocator,
    ProviderTurnQuery,
)
from agent_runtime_kit.agent.providers.opencode_artifacts import OpenCodeArtifactAdapter
from agent_runtime_kit.agent.providers.opencode_bundle import build_opencode_provider_bundle
from agent_runtime_kit.agent.providers.opencode_context import OpenCodeContextAdapter
from agent_runtime_kit.agent.providers.opencode_home import OpenCodeHomeRenderer
from agent_runtime_kit.agent.providers.opencode_models import (
    OpenCodeHomeOptions,
    OpenCodeNativeLocator,
    OpenCodeRunOptions,
)
from agent_runtime_kit.agent.providers.opencode_query import OpenCodeQueryAdapter
from agent_runtime_kit.agent.providers.opencode_runtime import (
    OpenCodeRuntimeAdapter,
    OpenCodeRuntimeRegistry,
)


pytestmark = pytest.mark.real


class _OpenCodeRealAgentType(AgentType):
    agent_type = "OpenCodeRealAgent"


def test_real_opencode_server_health_session_and_isolated_database(tmp_path: Path) -> None:
    binary = os.environ.get("ARK_OPENCODE_TEST_BINARY")
    if not binary:
        pytest.skip("set ARK_OPENCODE_TEST_BINARY to an OpenCode 1.18.4 executable")
    runtime_root = tmp_path / "runtime"
    home_root = runtime_root / "homes" / "opencode" / "real"
    renderer = OpenCodeHomeRenderer(runtime_root=runtime_root)
    auth_source = tmp_path / "auth.json"
    auth_source.write_text('{"opencode":{"type":"api","key":"isolated-test-key"}}\n')
    materialization = renderer.materialize(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="real",
            base_config=BaseConfigSource(
                mapping={"model": "deepseek/deepseek-chat", "snapshot": True}
            ),
            provider_options=OpenCodeHomeOptions(
                binary_path=binary,
                auth_json_path=auth_source,
            ),
        ),
        home_root,
    )
    record = HomeRecord(
        provider_type="opencode",
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
        isolated_auth = server.runtime_root / "xdg-data" / "opencode" / "auth.json"
        assert isolated_auth.read_text() == auth_source.read_text()
        assert isolated_auth.stat().st_mode & 0o777 == 0o600
        assert str(server.database_path).startswith(str(runtime_root / "providers" / "opencode"))
        assert server.directory == str(tmp_path.resolve())
    finally:
        registry.close()


def test_real_opencode_service_bootstraps_query_context_and_fork_after_restart(
    tmp_path: Path,
) -> None:
    binary = os.environ.get("ARK_OPENCODE_TEST_BINARY")
    if not binary:
        pytest.skip("set ARK_OPENCODE_TEST_BINARY to an OpenCode 1.18.4 executable")
    runtime_root = tmp_path / "runtime"
    agent_types = AgentTypeRegistry()
    agent_types.register(_OpenCodeRealAgentType())

    first_bundle = build_opencode_provider_bundle(
        runtime_root=runtime_root,
        binary_path=binary,
    )
    first_service = AgentService(
        runtime_root,
        agent_types=agent_types,
        provider_registry=ProviderRegistry((first_bundle,)),
    )
    first_service.create_home(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="restart-home",
            base_config=BaseConfigSource(
                mapping={"model": "opencode-go/deepseek-v4-flash"}
            ),
            provider_options=OpenCodeHomeOptions(binary_path=binary),
        )
    )
    agent = first_service.create_agent(
        "scope-restart",
        "OpenCodeRealAgent",
        provider_type="opencode",
        home_id="restart-home",
    )
    first_registry = first_bundle.session_access
    assert isinstance(first_registry, OpenCodeRuntimeRegistry)
    execution_context = first_service.home_service.build_execution_context(
        "opencode",
        "restart-home",
        run_env={},
        workdir=str(tmp_path),
    )
    server = first_registry.ensure(
        ProviderRunRequest(
            agent_id=agent.agent_id,
            scope_id=agent.scope_id,
            agent_type=agent.agent_type,
            provider_type="opencode",
            home_id="restart-home",
            prompt="not submitted",
            workdir=str(tmp_path),
            execution_context=execution_context,
        )
    )
    session_id = str(server.client.create_session()["id"])
    session = ProviderSessionLocator(
        provider_type="opencode",
        session_id=session_id,
        home_id="restart-home",
        created_at="2026-08-01T00:00:00Z",
        backend_identity=ModelBackendIdentity(
            api_provider="opencode-go",
            api_mode="chat_completions",
            requested_model="deepseek-v4-flash",
        ),
        native_locator=OpenCodeNativeLocator(
            agent_id=agent.agent_id,
            directory=str(tmp_path.resolve()),
            database_path=str(server.database_path),
            runtime_relpath=str(server.runtime_root.relative_to(runtime_root)),
        ).as_dict(),
    )
    first_service.store.update_session_locators(
        agent.agent_id,
        session_locator=session,
    )
    first_service.close()

    second_bundle = build_opencode_provider_bundle(
        runtime_root=runtime_root,
        binary_path=binary,
    )
    second_service = AgentService(
        runtime_root,
        agent_types=agent_types,
        provider_registry=ProviderRegistry((second_bundle,)),
    )
    second_registry = second_bundle.session_access
    assert isinstance(second_registry, OpenCodeRuntimeRegistry)
    try:
        assert second_registry.server_for_agent(agent.agent_id) is None
        preflight = second_service.compact_agent_if_needed(
            agent.agent_id,
            threshold=0.80,
            timeout_s=30,
        )
        assert preflight.status.value == "skipped"
        assert preflight.reason == "OpenCode did not report latest input usage"
        assert second_registry.server_for_agent(agent.agent_id) is not None
        assert second_service.query_turns(agent.agent_id).items == ()
        usage = second_service.inspect_agent_context(agent.agent_id)
        assert usage.session_id == session_id
        assert usage.available is False
        read_bootstrap_server = second_registry.server_for_agent(agent.agent_id)
        assert read_bootstrap_server is not None
        turn_context = second_service.home_service.build_execution_context(
            "opencode",
            "restart-home",
            run_env={
                "ARK_STEP_ID": "step-callback",
                "ARK_FLOW_ID": "flow-content",
                "ARK_AGENT_ID": agent.agent_id,
            },
            workdir=str(tmp_path),
        )
        second_registry.prepare_session_access(
            session,
            agent_id=agent.agent_id,
            execution_context=turn_context,
        )
        turn_server = second_registry.server_for_agent(agent.agent_id)
        assert turn_server is not None
        assert turn_server is not read_bootstrap_server
        assert read_bootstrap_server.process.poll() is not None
        assert turn_server.process.poll() is None
        assert second_service.query_turns(agent.agent_id).items == ()
        assert second_registry.server_for_agent(agent.agent_id) is turn_server
    finally:
        second_service.close()

    third_bundle = build_opencode_provider_bundle(
        runtime_root=runtime_root,
        binary_path=binary,
    )
    third_service = AgentService(
        runtime_root,
        agent_types=agent_types,
        provider_registry=ProviderRegistry((third_bundle,)),
    )
    third_registry = third_bundle.session_access
    assert isinstance(third_registry, OpenCodeRuntimeRegistry)
    try:
        assert third_registry.server_for_agent(agent.agent_id) is None
        forked = third_service.fork_agent(agent.agent_id)
        assert forked.session_locator is not None
        assert forked.session_locator.session_id != session_id
        assert third_registry.server_for_agent(agent.agent_id) is not None
    finally:
        third_service.close()


def test_real_opencode_account_auth_run(tmp_path: Path) -> None:
    binary = os.environ.get("ARK_OPENCODE_TEST_BINARY")
    auth_path = os.environ.get("ARK_OPENCODE_ACCOUNT_AUTH_JSON")
    model = os.environ.get("ARK_OPENCODE_ACCOUNT_MODEL", "opencode-go/deepseek-v4-flash")
    if (
        os.environ.get("ARK_OPENCODE_RUN_REAL_MODELS") != "1"
        or not binary
        or not auth_path
    ):
        pytest.skip("enable the gated OpenCode account-auth real test")
    provider_id, model_id = model.split("/", 1)
    runtime_root = tmp_path / "runtime"
    home_root = runtime_root / "homes" / "opencode" / "account"
    renderer = OpenCodeHomeRenderer(runtime_root=runtime_root)
    materialization = renderer.materialize(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="account",
            base_config=BaseConfigSource(mapping={"model": model}),
            provider_options=OpenCodeHomeOptions(
                binary_path=binary,
                auth_json_path=Path(auth_path),
            ),
        ),
        home_root,
    )
    record = HomeRecord(
        provider_type="opencode",
        home_id="account",
        home_relpath="homes/opencode/account",
        materialization_manifest_hash=materialization.manifest_hash,
    )
    context = renderer.build_execution_context(record, run_env={}, workdir=str(tmp_path))
    registry = OpenCodeRuntimeRegistry(runtime_root, binary_path=binary)
    runtime = OpenCodeRuntimeAdapter(registry)
    active_runtime = runtime
    request = ProviderRunRequest(
        agent_id="agent-account",
        scope_id="scope-real",
        agent_type="worker",
        provider_type="opencode",
        home_id="account",
        prompt="Reply with exactly OPENCODE_ACCOUNT_OK and no other text.",
        workdir=str(tmp_path),
        model_overrides=ModelBackendIdentity(
            api_provider=provider_id,
            api_mode="chat_completions",
            requested_model=model_id,
        ),
        run_options=ProviderRunOptions(timeout_s=180),
        provider_options=OpenCodeRunOptions(
            provider_id=provider_id,
            model_id=model_id,
            tools={"bash": False, "edit": False, "write": False},
        ),
        execution_context=context,
    )
    try:
        result = runtime.start(request).wait_terminal(190)
        assert result.status.value == "completed", result.error
        assert "OPENCODE_ACCOUNT_OK" in (result.final_text or "")
        assert result.turn_usage is not None
        isolated_auth = (
            runtime_root
            / "providers"
            / "opencode"
            / "agents"
            / "agent-account"
            / "xdg-data"
            / "opencode"
            / "auth.json"
        )
        assert isolated_auth.is_file()
        assert isolated_auth.stat().st_mode & 0o777 == 0o600

        artifacts = OpenCodeArtifactAdapter(runtime_root=runtime_root, registry=registry)
        snapshot = artifacts.capture(
            ArtifactCaptureRequest(
                session=result.session_locator,
                snapshot_root=str(tmp_path / "snapshot"),
                agent_id="agent-account",
            )
        )
        assert all(entry.snapshot_relpath != "auth.json" for entry in snapshot.manifest.entries)

        runtime.close()
        restarted_registry = OpenCodeRuntimeRegistry(runtime_root, binary_path=binary)
        restarted_runtime = OpenCodeRuntimeAdapter(restarted_registry)
        active_runtime = restarted_runtime
        restarted_query = OpenCodeQueryAdapter(restarted_registry.client_for_locator)
        restarted_context = OpenCodeContextAdapter(
            registry=restarted_registry,
            query=restarted_query,
        )
        usage = restarted_context.inspect(
            ProviderContextQuery(
                session=result.session_locator,
                agent_id="agent-account",
                execution_context=context,
            )
        )
        assert usage.session_id == result.session_locator.session_id
        assert restarted_registry.server_for_agent("agent-account") is not None
        compact = restarted_context.compact(
            ProviderContextCompactionRequest(
                session=result.session_locator,
                trigger="account_auth_restart_test",
                timeout_s=180,
                agent_id="agent-account",
                execution_context=context,
            )
        )
        assert compact.status == "completed"
        assert restarted_query.list_turns(
            ProviderTurnQuery(session=result.session_locator)
        ).items

        second = restarted_runtime.resume(
            replace(
                request,
                prompt="Reply with exactly ACCOUNT_SECOND and no other text.",
                session_locator=result.session_locator,
            )
        ).wait_terminal(190)
        assert "ACCOUNT_SECOND" in (second.final_text or "")

        restored_artifacts = OpenCodeArtifactAdapter(
            runtime_root=runtime_root,
            registry=restarted_registry,
        )
        restored = restored_artifacts.restore(
            ArtifactRestoreRequest(
                manifest=snapshot.manifest,
                snapshot_root=snapshot.snapshot_root,
            )
        )
        assert restored.restored
        third = restarted_runtime.resume(
            replace(
                request,
                prompt="Reply with exactly ACCOUNT_RESTORED and no other text.",
                session_locator=result.session_locator,
            )
        ).wait_terminal(190)
        assert "ACCOUNT_RESTORED" in (third.final_text or "")
        turns = restarted_query.list_turns(
            ProviderTurnQuery(session=third.session_locator)
        ).items
        texts = [turn.result.final_text for turn in turns if turn.result is not None]
        assert any(text and "OPENCODE_ACCOUNT_OK" in text for text in texts)
        assert any(text and "ACCOUNT_RESTORED" in text for text in texts)
        assert not any(text and "ACCOUNT_SECOND" in text for text in texts)

    finally:
        active_runtime.close()


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
        provider_type="opencode",
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
        provider_type="opencode",
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
        query = OpenCodeQueryAdapter(registry.client_for_locator)
        context_usage = OpenCodeContextAdapter(registry=registry, query=query).inspect(
            ProviderContextQuery(session=result.session_locator)
        )
        assert context_usage.context_window_tokens == 1_050_000
        assert context_usage.max_output_tokens == 128_000
        assert context_usage.used_tokens is not None
    finally:
        runtime.close()


def test_real_opencode_first_turn_through_agent_service(tmp_path: Path) -> None:
    binary = os.environ.get("ARK_OPENCODE_TEST_BINARY")
    key = os.environ.get("ARK_OPENCODE_REAL_DEEPSEEK_KEY")
    if os.environ.get("ARK_OPENCODE_RUN_REAL_MODELS") != "1" or not binary or not key:
        pytest.skip("enable the gated OpenCode AgentService real test")
    runtime_root = tmp_path / "runtime"
    agent_types = AgentTypeRegistry()
    agent_types.register(_OpenCodeRealAgentType())
    registry = ProviderRegistry(
        (build_opencode_provider_bundle(runtime_root=runtime_root, binary_path=binary),)
    )
    service = AgentService(
        runtime_root,
        agent_types=agent_types,
        provider_registry=registry,
    )
    service.create_home(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="service-home",
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
        )
    )
    agent = service.create_agent(
        "scope-service", "OpenCodeRealAgent", provider_type="opencode", home_id="service-home"
    )
    try:
        service.start_agent(
            agent.agent_id,
            prompt="Reply with exactly SERVICE_OK and no other text.",
            env={"DEEPSEEK_API_KEY": key},
            workdir=str(tmp_path),
        )
        result = service.wait_agent(agent.agent_id, timeout_s=130)
        assert result.provider_type == "opencode"
        assert "SERVICE_OK" in (result.final_text or "")
        stored = service.get_agent(agent.agent_id)
        assert stored.session_locator is not None
        assert stored.session_locator.native_locator is not None
    finally:
        service.close()
