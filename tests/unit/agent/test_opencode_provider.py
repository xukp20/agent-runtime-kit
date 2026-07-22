from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeRecord, McpServerSpec
from agent_runtime_kit.agent.provider_contracts import (
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
    BaseConfigSource,
    CapabilityKey,
    ModelBackendIdentity,
    ProviderContextQuery,
    ProviderHomeSpec,
    ProviderControlAction,
    ProviderControlRequest,
    ProviderExecutionContext,
    ProviderRunOptions,
    ProviderRunRequest,
    ProviderSessionLocator,
)
from agent_runtime_kit.agent.providers.opencode_artifacts import OpenCodeArtifactAdapter
from agent_runtime_kit.agent.providers.opencode_bundle import build_opencode_provider_bundle
from agent_runtime_kit.agent.providers.opencode_home import OpenCodeHomeRenderer
from agent_runtime_kit.agent.providers.opencode_models import OpenCodeHomeOptions
from agent_runtime_kit.agent.providers.opencode_query import OpenCodeQueryAdapter, project_turns
from agent_runtime_kit.agent.providers.opencode_runtime import _message_id
from agent_runtime_kit.agent.providers.opencode_client import OpenCodeSseEvent, _safe_body
from agent_runtime_kit.agent.providers.opencode_context import OpenCodeContextAdapter, _model_limits
from agent_runtime_kit.agent.providers.opencode_runtime import OpenCodeProviderRunHandle
from agent_runtime_kit.agent.skills import SkillSpec


class _StoppedRegistry:
    def client_for_locator(self, locator):  # noqa: ANN001, ANN201
        del locator
        raise RuntimeError("not running")

    def close_agent(self, agent_id: str) -> None:
        del agent_id


def test_opencode_home_materializes_resources_and_isolated_context(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    home_root = runtime_root / "homes" / "opencode" / "main"
    renderer = OpenCodeHomeRenderer(runtime_root=runtime_root)
    result = renderer.materialize(
        ProviderHomeSpec(
            provider_type="opencode",
            home_id="main",
            base_config=BaseConfigSource(
                text='''{
                  // JSONC is accepted
                  "model": "deepseek/deepseek-chat",
                  "provider": {"deepseek": {"npm": "@ai-sdk/openai-compatible"}},
                  "snapshot": true,
                  "plugin": ["unsafe-plugin"],
                }'''
            ),
            config_overrides={"provider": {"deepseek": {"options": {"baseURL": "https://example.test"}}}},
            instructions=("Keep changes surgical.",),
            skills=(SkillSpec(name="proof", description="Proof helper", body="Use Lean."),),
            mcp_servers=(
                McpServerSpec(
                    name="lean",
                    command="python",
                    args=["-m", "lean_mcp"],
                    env_vars=["LEAN_TOKEN"],
                ),
            ),
            fixed_env={"SAFE_SETTING": "yes"},
            fixed_env_refs={"MAPPED_TOKEN": "SOURCE_TOKEN"},
            required_env=("LEAN_TOKEN",),
            tools=({"bash": False, "mcp_lean_*": True},),
            provider_options=OpenCodeHomeOptions(
                binary_path="/opt/opencode",
                agent_files={"reviewer": "---\ndescription: Review changes\n---\nReview."},
                command_files={"check": "Run the focused checks."},
            ),
        ),
        home_root,
    )
    config = json.loads((home_root / "opencode.json").read_text())
    assert config["snapshot"] is False
    assert config["share"] == "disabled"
    assert config["plugin"] == []
    assert config["provider"]["deepseek"]["options"]["baseURL"] == "https://example.test"
    assert config["mcp"]["lean"]["command"] == ["python", "-m", "lean_mcp"]
    assert config["mcp"]["lean"]["environment"]["LEAN_TOKEN"] == "{env:LEAN_TOKEN}"
    assert config["tools"] == {"bash": False, "mcp_lean_*": True}
    assert (home_root / "AGENTS.md").read_text() == "Keep changes surgical.\n"
    assert (home_root / "skills" / "proof" / "SKILL.md").is_file()
    assert (home_root / "agents" / "reviewer.md").is_file()
    assert (home_root / "commands" / "check.md").is_file()
    assert result.resolved_defaults == ModelBackendIdentity(
        api_provider="deepseek",
        api_mode="chat_completions",
        requested_model="deepseek-chat",
        resolved_model="deepseek-chat",
    )
    record = HomeRecord(
        provider_type="opencode",
        home_id="main",
        home_relpath="homes/opencode/main",
        materialization_manifest_hash=result.manifest_hash,
        fixed_env={"SAFE_SETTING": "yes"},
        required_env={"LEAN_TOKEN"},
    )
    with pytest.raises(Exception, match="LEAN_TOKEN|SOURCE_TOKEN"):
        renderer.build_execution_context(record, run_env={}, workdir=str(tmp_path))
    context = renderer.build_execution_context(
        record,
        run_env={
            "LEAN_TOKEN": "secret",
            "SOURCE_TOKEN": "source-secret",
            "OPENCODE_CONFIG": "/tmp/bypass.json",
            "OPENCODE_CONFIG_CONTENT": '{"share": "auto"}',
            "OPENCODE_DISABLE_PROJECT_CONFIG": "0",
            "OPENCODE_PURE": "0",
        },
        workdir=str(tmp_path),
    )
    assert context.process_environment["ARK_OPENCODE_BINARY"] == "/opt/opencode"
    assert context.process_environment["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert context.process_environment["OPENCODE_PURE"] == "1"
    assert "OPENCODE_CONFIG" not in context.process_environment
    assert "OPENCODE_CONFIG_CONTENT" not in context.process_environment
    assert context.process_environment["MAPPED_TOKEN"] == "source-secret"
    assert '"LEAN_TOKEN": "secret"' not in (
        home_root / ".ark" / "home_materialization.json"
    ).read_text()


def test_opencode_query_projects_usage_tools_and_cost() -> None:
    session = ProviderSessionLocator(
        provider_type="opencode",
        session_id="ses_1",
        home_id="main",
        created_at="2026-07-21T00:00:00Z",
    )
    messages = [
        {
            "info": {"id": "msg_user", "role": "user", "time": {"created": 1000}},
            "parts": [{"id": "p_user", "type": "text", "text": "hello"}],
        },
        {
            "info": {
                "id": "msg_assistant",
                "role": "assistant",
                "parentID": "msg_user",
                "providerID": "beeapi-responses",
                "modelID": "gpt-5.4",
                "apiMode": "responses",
                "finish": "stop",
                "cost": 0.125,
                "tokens": {
                    "input": 10,
                    "output": 4,
                    "reasoning": 2,
                    "cache": {"read": 3, "write": 1},
                },
                "time": {"created": 1100, "completed": 1200},
            },
            "parts": [
                {"id": "p_text", "type": "text", "text": "done"},
                {
                    "id": "p_tool",
                    "type": "tool",
                    "callID": "call_1",
                    "tool": "bash",
                    "state": {"status": "completed", "input": {"cmd": "pwd"}, "output": "/tmp"},
                },
            ],
        },
    ]
    turns = project_turns(session, messages)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.result is not None
    assert turn.result.final_text == "done"
    assert turn.tool_calls[0].call_id == "call_1"
    request = turn.usage.requests[0]
    assert request.model_identity.api_mode == "responses"
    assert request.token_usage.cache_read_input_tokens == 3
    assert request.token_usage.total_tokens is None
    assert request.reported_cost is not None
    assert request.reported_cost.total_cost == "0.125"


def test_opencode_artifact_uses_sqlite_backup_and_restores(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    agent_runtime = runtime_root / "providers" / "opencode" / "agents" / "agent-1"
    database = agent_runtime / "opencode.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as conn:
        conn.execute("create table sessions(id text primary key, value text)")
        conn.execute("insert into sessions values ('ses_1', 'before')")
    tool_output = agent_runtime / "xdg-data" / "opencode" / "tool-output"
    tool_output.mkdir(parents=True)
    (tool_output / "large.txt").write_text("result")
    locator = ProviderSessionLocator(
        provider_type="opencode",
        session_id="ses_1",
        home_id="main",
        created_at="2026-07-21T00:00:00Z",
        native_locator={
            "agent_id": "agent-1",
            "directory": str(tmp_path),
            "database_path": str(database),
            "runtime_relpath": "providers/opencode/agents/agent-1",
        },
    )
    adapter = OpenCodeArtifactAdapter(runtime_root=runtime_root, registry=_StoppedRegistry())
    snapshot = adapter.capture(
        ArtifactCaptureRequest(session=locator, snapshot_root=str(tmp_path / "snapshot"))
    )
    with sqlite3.connect(database) as conn:
        conn.execute("update sessions set value='after'")
    (tool_output / "large.txt").write_text("changed")
    restored = adapter.restore(
        ArtifactRestoreRequest(manifest=snapshot.manifest, snapshot_root=snapshot.snapshot_root)
    )
    assert restored.restored is True
    with sqlite3.connect(database) as conn:
        assert conn.execute("select value from sessions").fetchone()[0] == "before"
        assert conn.execute("pragma integrity_check").fetchone()[0] == "ok"
    assert (tool_output / "large.txt").read_text() == "result"


def test_opencode_bundle_resolves_backend_capabilities(tmp_path: Path) -> None:
    bundle = build_opencode_provider_bundle(runtime_root=tmp_path)
    assert bundle.descriptor.execution_kind.value == "subprocess_rpc"
    home = HomeRecord(
        provider_type="opencode",
        home_id="main",
        home_relpath="homes/opencode/main",
        resolved_defaults={
            "api_provider": "beeapi-responses",
            "api_mode": "responses",
            "requested_model": "gpt-5.4",
        },
    )
    capabilities = bundle.resolve_capabilities(home)
    assert capabilities.available(CapabilityKey.MODEL_RESPONSES)
    assert not capabilities.available(CapabilityKey.MODEL_CHAT_COMPLETIONS)
    assert capabilities.available(CapabilityKey.CONTROL_COMPACT)
    assert not capabilities.available(CapabilityKey.CONTROL_FORK_FROM_TURN)
    bundle.runtime.close()


def test_opencode_message_ids_match_native_shape_and_are_ascending() -> None:
    first = _message_id()
    second = _message_id()
    assert len(first) == len("msg_") + 26
    assert first.startswith("msg_")
    assert first < second
    int(first[4:16], 16)


def test_opencode_home_rejects_inline_secrets(tmp_path: Path) -> None:
    renderer = OpenCodeHomeRenderer(runtime_root=tmp_path)
    with pytest.raises(ValueError, match="env reference"):
        renderer.materialize(
            ProviderHomeSpec(
                provider_type="opencode",
                home_id="unsafe",
                base_config=BaseConfigSource(
                    mapping={"provider": {"custom": {"options": {"apiKey": "inline-secret"}}}}
                ),
            ),
            tmp_path / "homes" / "opencode" / "unsafe",
        )


def test_opencode_error_body_redacts_credentials() -> None:
    value = _safe_body('{"apiKey":"sk-test-secret", "Authorization":"Bearer private"}')
    assert "sk-test-secret" not in value
    assert "Bearer private" not in value


def test_opencode_context_resolves_live_model_limits_without_provider_secrets() -> None:
    identity = ModelBackendIdentity(
        api_provider="beeapi-responses",
        api_mode="responses",
        requested_model="gpt-5.4",
    )
    provider_payload = {
        "all": [
            {
                "id": "beeapi-responses",
                "key": "must-not-be-retained",
                "options": {"apiKey": "also-must-not-be-retained"},
                    "models": {
                        "gpt-5.4": {
                            "limit": {
                                "context": 1_050_000,
                                "input": 922_000,
                                "output": 128_000,
                            }
                        }
                    },
            }
        ]
    }

    class _Client:
        def list_messages(self, session_id: str) -> list[object]:
            assert session_id == "ses_limits"
            return [
                {
                    "info": {"id": "msg_user", "role": "user", "time": {"created": 1000}},
                    "parts": [{"id": "p_user", "type": "text", "text": "hello"}],
                },
                {
                    "info": {
                        "id": "msg_assistant",
                        "role": "assistant",
                        "parentID": "msg_user",
                        "providerID": "beeapi-responses",
                        "modelID": "gpt-5.4",
                        "finish": "stop",
                        "tokens": {"input": 100, "cache": {"read": 20}},
                        "time": {"created": 1100, "completed": 1200},
                    },
                    "parts": [{"id": "p_text", "type": "text", "text": "done"}],
                },
            ]

        def list_providers(self) -> dict[str, object]:
            return provider_payload

    client = _Client()

    class _Registry:
        def client_for_locator(self, locator):  # noqa: ANN001, ANN201
            del locator
            return client

    session = ProviderSessionLocator(
        provider_type="opencode",
        session_id="ses_limits",
        home_id="main",
        created_at="2026-07-21T00:00:00Z",
        backend_identity=identity,
    )
    usage = OpenCodeContextAdapter(
        registry=_Registry(),
        query=OpenCodeQueryAdapter(lambda locator: client),
    ).inspect(
        ProviderContextQuery(session=session)
    )
    assert usage.context_window_tokens == 1_050_000
    assert usage.effective_context_window_tokens == 922_000
    assert usage.max_output_tokens == 128_000
    assert usage.used_tokens == 120
    assert usage.remaining_tokens == 921_880
    assert usage.provider_payload is not None
    assert usage.provider_payload.sanitized_data == {
        "provider_id": "beeapi-responses",
        "model_id": "gpt-5.4",
        "limit": {"context": 1_050_000, "input": 922_000, "output": 128_000},
    }
    assert "must-not-be-retained" not in repr(usage.provider_payload)
    assert _model_limits(provider_payload, replace(identity, requested_model="unknown")) == {}


class _InteractionClient:
    def __init__(self) -> None:
        self.session_id = "ses_interaction"
        self.turn_id: str | None = None
        self.prompted = threading.Event()
        self.answered = threading.Event()

    def create_session(self):  # noqa: ANN201
        return {"id": self.session_id}

    def list_messages(self, session_id: str) -> list[object]:
        assert session_id == self.session_id
        if self.turn_id is None:
            return []
        values: list[object] = [
            {
                "info": {"id": self.turn_id, "role": "user", "time": {"created": 1000}},
                "parts": [{"type": "text", "text": "approve"}],
            }
        ]
        if self.answered.is_set():
            values.append(
                {
                    "info": {
                        "id": "msg_answer",
                        "role": "assistant",
                        "parentID": self.turn_id,
                        "providerID": "deepseek",
                        "modelID": "deepseek-chat",
                        "finish": "stop",
                        "time": {"created": 1100, "completed": 1200},
                    },
                    "parts": [{"id": "part_answer", "type": "text", "text": "approved"}],
                }
            )
        return values

    def prompt_async(self, session_id: str, payload) -> None:  # noqa: ANN001
        assert session_id == self.session_id
        self.turn_id = str(payload["messageID"])
        self.prompted.set()

    def session_status(self):  # noqa: ANN201
        return {} if self.answered.is_set() else {self.session_id: {"type": "busy"}}

    def iter_events(self, stop):  # noqa: ANN001, ANN201
        yield OpenCodeSseEvent(
            event="message", data={"type": "server.connected", "properties": {}}
        )
        assert self.prompted.wait(2)
        yield OpenCodeSseEvent(
            event="message",
            data={
                "type": "permission.asked",
                "properties": {"id": "per_1", "sessionID": self.session_id},
            },
        )
        while not stop.wait(0.01):
            pass

    def reply_permission(self, permission_id: str, payload) -> None:  # noqa: ANN001
        assert permission_id == "per_1"
        assert payload == {"reply": "once"}
        self.answered.set()


@dataclass
class _InteractionServer:
    client: _InteractionClient
    directory: str
    database_path: Path
    runtime_root: Path


class _InteractionRegistry:
    def __init__(self, tmp_path: Path, client: _InteractionClient | None = None) -> None:
        self.runtime_root = tmp_path / "runtime"
        runtime = self.runtime_root / "providers" / "opencode" / "agents" / "agent-1"
        runtime.mkdir(parents=True)
        self.server = _InteractionServer(
            client=client or _InteractionClient(),
            directory=str(tmp_path),
            database_path=runtime / "opencode.db",
            runtime_root=runtime,
        )

    def ensure(self, request):  # noqa: ANN001, ANN201
        del request
        return self.server

    def server_for_agent(self, agent_id: str):  # noqa: ANN201
        return self.server if agent_id == "agent-1" else None


def test_opencode_run_handle_keeps_interaction_alive(tmp_path: Path) -> None:
    registry = _InteractionRegistry(tmp_path)
    context = ProviderExecutionContext(
        provider_type="opencode",
        home_id="main",
        home_root=tmp_path / "home",
        process_environment={},
        resolved_defaults=ModelBackendIdentity(
            api_provider="deepseek", api_mode="chat_completions", requested_model="deepseek-chat"
        ),
    )
    request = ProviderRunRequest(
        agent_id="agent-1",
        scope_id="scope-1",
        agent_type="worker",
        provider_type="opencode",
        home_id="main",
        prompt="approve",
        workdir=str(tmp_path),
        run_options=ProviderRunOptions(timeout_s=5),
        execution_context=context,
    )
    handle = OpenCodeProviderRunHandle(registry, request, resume=False)
    needs_input = handle.wait_terminal(3)
    assert needs_input.status.value == "needs_input"
    control = handle.control(
        ProviderControlRequest(
            action=ProviderControlAction.RESPOND_APPROVAL,
            requested_at="2026-07-21T00:00:00Z",
            run_id=handle.run_id,
            content="once",
        )
    )
    assert control.accepted
    result = handle.wait_terminal(3)
    assert result.status.value == "completed"
    assert result.final_text == "approved"
    handle.close()


class _AbortClient(_InteractionClient):
    def __init__(self) -> None:
        super().__init__()
        self.aborted = threading.Event()

    def session_status(self):  # noqa: ANN201
        return {} if self.aborted.is_set() else {self.session_id: {"type": "busy"}}

    def iter_events(self, stop):  # noqa: ANN001, ANN201
        yield OpenCodeSseEvent(
            event="message", data={"type": "server.connected", "properties": {}}
        )
        while not stop.wait(0.01):
            pass

    def abort(self, session_id: str):  # noqa: ANN201
        assert session_id == self.session_id
        self.aborted.set()
        return True


def test_opencode_interrupt_waits_for_idle_confirmation(tmp_path: Path) -> None:
    client = _AbortClient()
    registry = _InteractionRegistry(tmp_path, client=client)
    context = ProviderExecutionContext(
        provider_type="opencode",
        home_id="main",
        home_root=tmp_path / "home",
        process_environment={},
        resolved_defaults=ModelBackendIdentity(
            api_provider="deepseek", api_mode="chat_completions", requested_model="deepseek-chat"
        ),
    )
    request = ProviderRunRequest(
        agent_id="agent-1",
        scope_id="scope-1",
        agent_type="worker",
        provider_type="opencode",
        home_id="main",
        prompt="long run",
        workdir=str(tmp_path),
        run_options=ProviderRunOptions(timeout_s=5),
        execution_context=context,
    )
    handle = OpenCodeProviderRunHandle(registry, request, resume=False)
    assert client.prompted.wait(2)
    result = handle.interrupt(2)
    assert result.accepted
    assert result.terminal_confirmed
    assert result.resulting_state is not None
    assert result.resulting_state.value == "interrupted"
    assert handle.wait_terminal(1).status.value == "interrupted"
    handle.close()
