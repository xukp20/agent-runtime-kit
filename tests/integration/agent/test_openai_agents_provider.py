from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator

import pytest

pytest.importorskip("agents")

from agents import Agent, ModelResponse, Usage, function_tool
from agents.agent_output import AgentOutputSchemaBase
from agents.handoffs import Handoff
from agents.items import TResponseInputItem, TResponseOutputItem, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models.interface import Model, ModelTracing
from agents.tool import Tool
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from agent_runtime_kit.agent.provider_contracts import (
    ModelBackendIdentity,
    ProviderHomeSpec,
    ProviderControlAction,
    ProviderControlRequest,
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
    ProviderContextQuery,
    ProviderContextCompactionRequest,
    ProviderRegistry,
    AgentTurnResult,
    ProviderForkRequest,
    ProviderRunRequest,
    ProviderRunState,
    ProviderSessionQuery,
)
from agent_runtime_kit.agent.providers import (
    OpenAIAgentsHomeOptions,
    OpenAIAgentsControlOptions,
    OpenAIAgentsProvider,
)
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.homes import McpServerSpec


class OpenAIAgentsTestType(AgentType):
    agent_type = "openai-worker"
    start_prompt_template = "Run {{item}}."


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[], logprobs=[])],
        status="completed",
    )


class ScriptedModel(Model):
    def __init__(self, outputs: list[list[TResponseOutputItem]]) -> None:
        self.outputs = list(outputs)

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
    ) -> ModelResponse:
        return ModelResponse(
            output=self.outputs.pop(0),
            usage=Usage(requests=1, input_tokens=11, output_tokens=7, total_tokens=18),
            response_id="response-1",
            request_id="request-1",
        )

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        output = self.outputs.pop(0)
        response = Response(
            id="response-1",
            created_at=123,
            model="test-model",
            object="response",
            output=output,
            tool_choice="none",
            tools=[],
            top_p=None,
            parallel_tool_calls=False,
            usage=ResponseUsage(
                input_tokens=11,
                output_tokens=7,
                total_tokens=18,
                input_tokens_details=InputTokensDetails.model_validate(
                    {"cached_tokens": 0, "cache_write_tokens": 0}
                ),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            ),
        )
        for index, item in enumerate(output):
            yield ResponseOutputItemDoneEvent(
                type="response.output_item.done",
                item=item,
                output_index=index,
                sequence_number=index,
            )
        yield ResponseCompletedEvent(
            type="response.completed",
            response=response,
            sequence_number=len(output),
        )


class DummyClient:
    async def close(self) -> None:
        return None


class SlowModel(Model):
    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        await asyncio.sleep(30)
        return ModelResponse(output=[_message("late")], usage=Usage(), response_id="late")

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[TResponseStreamEvent]:
        await asyncio.sleep(30)
        if False:
            yield {}  # type: ignore[misc]


@function_tool(needs_approval=True)
def approval_tool(value: str) -> str:
    """Return the approved marker."""

    return f"approved:{value}"


def test_openai_agents_home_runtime_query_and_usage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIAgentsProvider()
    provider.registry.register_agent_factory(
        "tests/basic",
        lambda ctx: Agent(name="ark-test", instructions=ctx.instructions, model=ctx.model),
    )
    bundle = provider.build_bundle(runtime_root=tmp_path)
    home_root = tmp_path / "homes" / "openai_agents" / "home-1"
    identity = ModelBackendIdentity(
        api_provider="test_backend",
        api_mode="responses",
        requested_model="gpt-test",
    )
    spec = ProviderHomeSpec(
        provider_type="openai_agents",
        home_id="home-1",
        model_config=identity,
        instructions=("Follow the deterministic test.",),
        required_env=("TEST_OPENAI_KEY",),
        provider_options=OpenAIAgentsHomeOptions(
            agent_factory_ref="tests/basic",
            api_key_env="TEST_OPENAI_KEY",
        ),
    )
    materialized = bundle.home_renderer.materialize(spec, home_root)
    assert materialized.resolved_defaults == identity
    assert "secret-value" not in (home_root / "provider.json").read_text()
    home = SimpleNamespace(
        home_id="home-1",
        home_relpath="homes/openai_agents/home-1",
        fixed_env={"TEST_OPENAI_KEY": "secret-value"},
        materialization_manifest_hash=materialized.manifest_hash,
    )
    execution = bundle.home_renderer.build_execution_context(home, run_env=None, workdir=str(tmp_path))
    bundle.home_renderer.initialize(home, execution)

    model = ScriptedModel([[_message("ARK_OPENAI_OK")]])
    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_runtime._build_model",
        lambda request, config: (DummyClient(), model),
    )
    handle = bundle.runtime.start(
        ProviderRunRequest(
            agent_id="agent-1",
            scope_id="scope-1",
            agent_type="test",
            provider_type="openai_agents",
            home_id="home-1",
            prompt="hello",
            execution_context=execution,
        )
    )
    result = handle.wait_terminal(10)
    assert result.status is ProviderRunState.COMPLETED
    assert result.final_text == "ARK_OPENAI_OK"
    assert result.request_usages[0].token_usage.input_tokens == 11
    assert result.request_usages[0].model_identity.api_provider == "test_backend"
    assert result.artifact_locator is not None

    session = bundle.query.read_session(ProviderSessionQuery(locator=result.session_locator))
    assert session.turns[0].result is not None
    assert session.turns[0].result.final_text == "ARK_OPENAI_OK"
    assert session.usage is not None
    assert session.usage.token_usage.total_tokens == 18


def test_openai_agents_interrupt_waits_for_terminal_barrier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, bundle, execution = _prepared_provider(tmp_path, factory=lambda ctx: Agent(name="slow", model=ctx.model))
    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_runtime._build_model",
        lambda request, config: (DummyClient(), SlowModel()),
    )
    handle = bundle.runtime.start(_request(execution))
    control = handle.interrupt(10)
    assert control.accepted
    assert control.terminal_confirmed
    assert handle.wait_terminal(1).status is ProviderRunState.INTERRUPTED


def test_openai_agents_durable_approval_and_sqlite_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider, bundle, execution = _prepared_provider(
        tmp_path,
        factory=lambda ctx: Agent(
            name="approval",
            instructions=ctx.instructions,
            model=ctx.model,
            tools=[approval_tool],
        ),
    )
    from openai.types.responses import ResponseFunctionToolCall

    model = ScriptedModel(
        [
            [
                ResponseFunctionToolCall(
                    id="item-call-1",
                    call_id="call-1",
                    type="function_call",
                    name="approval_tool",
                    arguments='{"value":"ARK"}',
                )
            ],
            [_message("ARK_APPROVAL_RESUMED")],
        ]
    )
    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_runtime._build_model",
        lambda request, config: (DummyClient(), model),
    )
    handle = bundle.runtime.start(_request(execution))
    waiting = handle.wait_terminal(10)
    assert waiting.status is ProviderRunState.NEEDS_INPUT
    assert waiting.tool_calls[0].status == "needs_approval"

    handle.close()
    restarted_bundle = provider.build_bundle(runtime_root=tmp_path)
    control = restarted_bundle.runtime.control(
        ProviderControlRequest(
            action=ProviderControlAction.RESPOND_APPROVAL,
            requested_at="2026-07-21T10:00:00Z",
            session_id=waiting.session_locator.session_id,
            content={"approval_id": "call-1", "decision": "approve"},
            options={"timeout_s": 10},
            provider_options=OpenAIAgentsControlOptions(
                execution_context=execution,
                agent_id="agent-1",
                scope_id="scope-1",
                agent_type="test",
                home_id="home-1",
            ),
        )
    )
    assert control.accepted
    assert control.terminal_confirmed
    resumed_session = restarted_bundle.query.read_session(
        ProviderSessionQuery(locator=waiting.session_locator)
    )
    resumed = resumed_session.turns[-1].result
    assert resumed is not None
    assert resumed.status is ProviderRunState.COMPLETED
    assert resumed.final_text == "ARK_APPROVAL_RESUMED"

    context = bundle.context.inspect(
        ProviderContextQuery(session=resumed.session_locator, execution_context=execution)
    )
    assert context.available
    assert context.used_tokens == 11

    snapshot_root = tmp_path / "snapshot"
    snapshot = restarted_bundle.artifacts.capture(
        ArtifactCaptureRequest(
            session=resumed.session_locator,
            snapshot_root=str(snapshot_root),
            execution_context=execution,
        )
    )
    assert snapshot.manifest.entries[0].capture_strategy == "sqlite_backup"
    db_path = execution.home_root / "sessions" / f"{resumed.session_locator.session_id}.sqlite3"
    restarted_bundle.artifacts.prepare_restore(
        ArtifactRestoreRequest(
            manifest=snapshot.manifest,
            snapshot_root=str(snapshot_root),
            execution_context=execution,
        )
    )
    assert not db_path.exists()
    restored = restarted_bundle.artifacts.restore(
        ArtifactRestoreRequest(
            manifest=snapshot.manifest,
            snapshot_root=str(snapshot_root),
            execution_context=execution,
        )
    )
    assert restored.restored
    session = restarted_bundle.query.read_session(ProviderSessionQuery(locator=resumed.session_locator))
    assert len(session.turns) == 2
    assert session.turns[-1].result.final_text == "ARK_APPROVAL_RESUMED"  # type: ignore[union-attr]


def test_openai_agents_compact_is_backend_gated_and_records_normalized_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle, execution = _prepared_provider(
        tmp_path,
        factory=lambda ctx: Agent(name="compact", model=ctx.model),
        compaction_mode="input_history",
    )
    model = ScriptedModel([[_message("before compact")]])
    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_runtime._build_model",
        lambda request, config: (DummyClient(), model),
    )
    result = bundle.runtime.start(_request(execution)).wait_terminal(10)

    async def fake_compact(path, session_id, model_name, config, env):  # noqa: ANN001, ANN202
        from agents.memory import SQLiteSession

        session = SQLiteSession(session_id, db_path=path)
        try:
            await session.clear_session()
            await session.add_items(
                [{"type": "compaction_summary", "encrypted_content": "opaque"}]
            )
        finally:
            session.close()
        return ["compaction_summary"]

    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_context._compact",
        fake_compact,
    )
    started: list[tuple[dict[str, object], str | None]] = []
    compacted = bundle.context.compact(
        ProviderContextCompactionRequest(
            session=result.session_locator,
            trigger="manual",
            execution_context=execution,
            on_started=lambda baseline, operation_id: started.append((baseline, operation_id)),
        )
    )
    assert compacted.status == "compacted"
    assert started and started[0][0]["item_count"] > 0
    assert compacted.provider_payload is not None
    assert compacted.provider_payload.sanitized_data["normalized_type"] == "compaction_summary"  # type: ignore[index]

    _, chat_bundle, chat_execution = _prepared_provider(
        tmp_path / "chat",
        factory=lambda ctx: Agent(name="chat", model=ctx.model),
        api_mode="chat_completions",
        compaction_mode="input_history",
    )
    chat_started: list[object] = []
    chat_session = result.session_locator.__class__(
        provider_type="openai_agents",
        session_id="chat-session",
        home_id="home-1",
        created_at=result.session_locator.created_at,
        backend_identity=chat_execution.resolved_defaults,
    )
    with pytest.raises(RuntimeError, match="verified Responses"):
        chat_bundle.context.compact(
            ProviderContextCompactionRequest(
                session=chat_session,
                trigger="manual",
                execution_context=chat_execution,
                on_started=lambda baseline, operation_id: chat_started.append(baseline),
            )
        )
    assert chat_started == []


def test_agent_service_runs_openai_agents_through_provider_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = OpenAIAgentsProvider()
    provider.registry.register_agent_factory(
        "tests/service",
        lambda ctx: Agent(name="service-agent", instructions=ctx.instructions, model=ctx.model),
    )
    bundle = provider.build_bundle(runtime_root=tmp_path)
    model = ScriptedModel([[_message("ARK_SERVICE_OK")]])
    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_runtime._build_model",
        lambda request, config: (DummyClient(), model),
    )
    types = AgentTypeRegistry()
    types.register(OpenAIAgentsTestType())
    service = AgentService(
        tmp_path,
        agent_types=types,
        provider_registry=ProviderRegistry((bundle,)),
    )
    service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="openai_agents",
            home_id="openai-worker",
            model_config=ModelBackendIdentity(
                api_provider="test_backend",
                api_mode="responses",
                requested_model="gpt-test",
            ),
            required_env=("TEST_OPENAI_KEY",),
            provider_options=OpenAIAgentsHomeOptions(
                agent_factory_ref="tests/service",
                api_key_env="TEST_OPENAI_KEY",
            ),
        )
    )
    agent = service.create_agent(
        "scope-1",
        "openai-worker",
        provider_type="openai_agents",
    )
    service.start_agent(
        agent.agent_id,
        variables={"item": "provider-neutral"},
        env={"TEST_OPENAI_KEY": "secret-value"},
    )
    result = service.wait_agent(agent.agent_id)
    assert isinstance(result, AgentTurnResult)
    assert result.final_text == "ARK_SERVICE_OK"
    restored = service.get_agent(agent.agent_id)
    assert restored.provider_type == "openai_agents"
    assert restored.session_locator == result.session_locator
    assert restored.artifact_locator is not None

    snapshots = AgentSnapshotService(tmp_path, store=service.store, agent_service=service)
    captured = snapshots.create_scope_snapshot("scope-1")
    assert captured.snapshot_id is not None
    manifest_path = tmp_path / str(captured.snapshot_relpath) / "snapshot.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = manifest["provider_artifacts"][0]["manifest"]["entries"][0]
    assert artifact["capture_strategy"] == "sqlite_backup"
    db_path = (
        tmp_path
        / "homes"
        / "openai_agents"
        / "openai-worker"
        / "sessions"
        / f"{restored.session_locator.session_id}.sqlite3"
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("delete from ark_turn")
    assert service.query_turns(agent.agent_id).items == ()
    restored_snapshot = snapshots.restore_scope_snapshot(captured.snapshot_id)
    assert restored_snapshot.status == "created"
    assert len(service.query_turns(agent.agent_id).items) == 1


def test_openai_agents_fork_copies_idle_session_without_workspace_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bundle, execution = _prepared_provider(
        tmp_path,
        factory=lambda ctx: Agent(name="fork", model=ctx.model),
    )
    model = ScriptedModel([[_message("SOURCE")], [_message("TARGET")]])
    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_runtime._build_model",
        lambda request, config: (DummyClient(), model),
    )
    source = bundle.runtime.start(_request(execution)).wait_terminal(10)
    forked = bundle.runtime.fork(
        ProviderForkRequest(
            source_agent_id="source-agent",
            source_session=source.session_locator,
            target_agent_id="target-agent",
            target_scope_id="scope-1",
            target_home_id="home-1",
            execution_context=execution,
        )
    )
    assert forked.status == "forked"
    assert forked.fork_mode == "session_only"
    assert not forked.workspace_isolated
    assert forked.target_session.session_id != source.session_locator.session_id
    target = bundle.runtime.resume(
        ProviderRunRequest(
            agent_id="target-agent",
            scope_id="scope-1",
            agent_type="test",
            provider_type="openai_agents",
            home_id="home-1",
            prompt="continue target",
            session_locator=forked.target_session,
            execution_context=execution,
        )
    ).wait_terminal(10)
    assert target.final_text == "TARGET"
    source_view = bundle.query.read_session(ProviderSessionQuery(locator=source.session_locator))
    target_view = bundle.query.read_session(ProviderSessionQuery(locator=forked.target_session))
    assert len(source_view.turns) == 1
    assert len(target_view.turns) == 2


def test_openai_agents_maps_stdio_mcp_and_owns_connection_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = Path(__file__).parents[2] / "fixtures" / "openai_agents_mcp_server.py"
    _, bundle, execution = _prepared_provider(
        tmp_path,
        factory=lambda ctx: Agent(name="mcp", model=ctx.model),
        mcp_servers=(
            McpServerSpec(
                name="ark-mcp-test",
                transport="stdio",
                command=sys.executable,
                args=[str(fixture)],
                required=True,
            ),
        ),
    )
    from openai.types.responses import ResponseFunctionToolCall

    model = ScriptedModel(
        [
            [
                ResponseFunctionToolCall(
                    id="item-mcp-call",
                    call_id="mcp-call-1",
                    type="function_call",
                    name="echo_marker",
                    arguments='{"value":"SDK"}',
                )
            ],
            [_message("ARK_MCP_DONE")],
        ]
    )
    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.openai_agents_runtime._build_model",
        lambda request, config: (DummyClient(), model),
    )
    result = bundle.runtime.start(_request(execution)).wait_terminal(20)
    assert result.status is ProviderRunState.COMPLETED
    assert result.final_text == "ARK_MCP_DONE"
    assert result.tool_calls
    assert result.tool_calls[0].tool_kind == "mcp"
    assert result.tool_calls[0].server_name == "ark-mcp-test"
    assert result.tool_calls[0].result is not None


def _prepared_provider(
    tmp_path: Path,
    *,
    factory,
    api_mode: str = "responses",
    compaction_mode: str = "unsupported",
    mcp_servers: tuple[McpServerSpec, ...] = (),
):  # noqa: ANN001, ANN202
    provider = OpenAIAgentsProvider()
    provider.registry.register_agent_factory("tests/factory", factory)
    bundle = provider.build_bundle(runtime_root=tmp_path)
    identity = ModelBackendIdentity(
        api_provider="test_backend",
        api_mode=api_mode,
        requested_model="gpt-test",
    )
    spec = ProviderHomeSpec(
        provider_type="openai_agents",
        home_id="home-1",
        model_config=identity,
        required_env=("TEST_OPENAI_KEY",),
        mcp_servers=mcp_servers,
        provider_options=OpenAIAgentsHomeOptions(
            agent_factory_ref="tests/factory",
            api_key_env="TEST_OPENAI_KEY",
            context_window_tokens=128_000,
            compaction_mode=compaction_mode,
        ),
    )
    home_root = tmp_path / "homes" / "openai_agents" / "home-1"
    materialized = bundle.home_renderer.materialize(spec, home_root)
    home = SimpleNamespace(
        home_id="home-1",
        home_relpath="homes/openai_agents/home-1",
        fixed_env={"TEST_OPENAI_KEY": "secret-value"},
        materialization_manifest_hash=materialized.manifest_hash,
    )
    execution = bundle.home_renderer.build_execution_context(home, run_env=None, workdir=str(tmp_path))
    return provider, bundle, execution


def _request(execution) -> ProviderRunRequest:  # noqa: ANN001
    return ProviderRunRequest(
        agent_id="agent-1",
        scope_id="scope-1",
        agent_type="test",
        provider_type="openai_agents",
        home_id="home-1",
        prompt="hello",
        execution_context=execution,
    )
