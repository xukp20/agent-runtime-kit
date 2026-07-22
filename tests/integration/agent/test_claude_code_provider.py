from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_runtime_kit.agent.provider_contracts import (
    ModelBackendIdentity,
    ProviderHomeSpec,
    ProviderRegistry,
    ProviderRunState,
)
from agent_runtime_kit.agent.providers.claude_code import ClaudeCodeProvider
from agent_runtime_kit.agent.providers.claude_code_bundle import build_claude_code_provider_bundle
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService


class ClaudeWorker(AgentType):
    agent_type = "claude-worker"
    developer_instructions_template = "Use the configured tools."
    start_prompt_template = "Solve {{item}}."
    continue_prompt_template = "Continue {{item}}."


class ClaudeAgentOptions:
    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class AssistantMessage:
    def __init__(self, text: str, message_id: str) -> None:
        self.content = [TextBlock(text)]
        self.message_id = message_id
        self.model = "deepseek-chat"
        self.stop_reason = "end_turn"
        self.error = None
        self.usage = {"input_tokens": 5, "output_tokens": 2}


class ResultMessage:
    subtype = "success"
    is_error = False
    result = "done"
    structured_output = None
    total_cost_usd = None
    num_turns = 1
    stop_reason = "end_turn"
    api_error_status = None
    usage = None
    model_usage = None
    deferred_tool_use = None
    errors = None


class FakeClaudeSDKClient:
    block_until_interrupt = False

    def __init__(self, *, options: ClaudeAgentOptions) -> None:
        self.options = options
        self._messages: list[object] = []
        self._interrupted = asyncio.Event()

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def query(self, prompt: str) -> None:
        session_id = str(self.options.resume or self.options.session_id)
        transcript = _transcript_path(self.options, session_id)
        transcript.parent.mkdir(parents=True, exist_ok=True)
        if prompt.startswith("/compact"):
            _append_jsonl(
                transcript,
                {
                    "type": "system",
                    "subtype": "compact_boundary",
                    "uuid": str(uuid.uuid4()),
                    "timestamp": "2026-07-21T00:00:02Z",
                },
            )
            self._messages = [ResultMessage()]
            return
        turn_id = str(uuid.uuid4())
        message_id = f"message-{turn_id}"
        _append_jsonl(
            transcript,
            {
                "type": "user",
                "uuid": turn_id,
                "timestamp": "2026-07-21T00:00:00Z",
                "message": {"content": prompt},
            },
        )
        _append_jsonl(
            transcript,
            {
                "type": "assistant",
                "uuid": str(uuid.uuid4()),
                "timestamp": "2026-07-21T00:00:01Z",
                "message": {
                    "id": message_id,
                    "model": "deepseek-chat",
                    "content": [{"type": "text", "text": "done"}],
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                    "stop_reason": "end_turn",
                },
            },
        )
        self._messages = [AssistantMessage("done", message_id), ResultMessage()]

    async def receive_response(self):  # noqa: ANN201
        if self.block_until_interrupt:
            await self._interrupted.wait()
        for message in self._messages:
            yield message

    async def interrupt(self) -> None:
        self._interrupted.set()

    async def get_context_usage(self) -> dict[str, object]:
        return {
            "totalTokens": 7,
            "maxTokens": 100,
            "rawMaxTokens": 128,
            "model": "deepseek-chat",
            "isAutoCompactEnabled": True,
            "categories": [{"name": "messages", "tokens": 7}],
        }


def _sdk(*, blocking: bool = False) -> object:
    client = type(
        "ClaudeSDKClient",
        (FakeClaudeSDKClient,),
        {"block_until_interrupt": blocking},
    )
    return SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=client)


def _service(tmp_path: Path, *, blocking: bool = False) -> tuple[AgentService, ClaudeCodeProvider]:
    runtime_root = tmp_path / ".agent_runtime"
    registry = AgentTypeRegistry()
    registry.register(ClaudeWorker())
    provider = ClaudeCodeProvider(runtime_root=runtime_root, sdk_loader=lambda: _sdk(blocking=blocking))
    service = AgentService(
        runtime_root,
        agent_types=registry,
        provider_registry=ProviderRegistry(
            (build_claude_code_provider_bundle(provider, runtime_root=runtime_root),)
        ),
    )
    service.create_home(
        ProviderHomeSpec(
            provider_type="claude_code",
            home_id="worker",
            model_config=ModelBackendIdentity(
                api_provider="deepseek",
                api_mode="anthropic_messages",
                requested_model="deepseek-chat",
            ),
        ),
        initialize_provider_home=False,
    )
    return service, provider


def test_claude_provider_runs_resumes_queries_compacts_and_restores_snapshot(
    tmp_path: Path,
) -> None:
    service, _provider = _service(tmp_path)
    agent = service.create_agent("scope-a", "claude-worker", provider_type="claude_code", home_id="worker")

    first = service.wait_agent(
        service.start_agent(agent.agent_id, variables={"item": "first"}).agent_id
    )
    second = service.wait_agent(
        service.start_agent(agent.agent_id, variables={"item": "second"}).agent_id
    )

    restored_agent = service.get_agent(agent.agent_id)
    assert first.status is ProviderRunState.COMPLETED
    assert second.status is ProviderRunState.COMPLETED
    assert first.session_locator.session_id == second.session_locator.session_id
    assert second.session_locator.backend_identity is not None
    assert second.session_locator.backend_identity.api_provider == "deepseek"
    assert restored_agent.session_locator == second.session_locator
    assert len(service.query_turns(agent.agent_id).items) == 2
    assert service.query_turn(agent.agent_id, latest=True).result.final_text == "done"
    assert service.query_usage(agent.agent_id, latest=True).request_count == 1

    marker = (
        service.runtime_root
        / "homes"
        / "claude_code"
        / "worker"
        / ".ark"
        / "claude_home_initialized.json"
    )
    marker.write_text(
        json.dumps({"provider_type": "claude_code", "cli_version": "2.1.216"}) + "\n",
        encoding="utf-8",
    )
    usage = service.inspect_agent_context(agent.agent_id)
    assert usage.available
    assert usage.used_tokens == 7
    compacted = service.compact_agent(agent.agent_id)
    assert compacted.status.value == "compacted"

    snapshot_service = AgentSnapshotService(
        service.runtime_root,
        store=service.store,
        agent_service=service,
    )
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.status == "created"
    transcript = _transcript_path_for_agent(service, agent.agent_id)
    expected = transcript.read_text(encoding="utf-8")
    transcript.write_text("corrupted\n", encoding="utf-8")

    restored = snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert restored.status == "created"
    assert transcript.read_text(encoding="utf-8") == expected
    assert service.query_turn(agent.agent_id, latest=True).result.final_text == "done"


def test_claude_provider_interrupt_reaches_terminal_result(tmp_path: Path) -> None:
    service, _provider = _service(tmp_path, blocking=True)
    agent = service.create_agent("scope-a", "claude-worker", provider_type="claude_code", home_id="worker")
    service.start_agent(agent.agent_id, variables={"item": "interrupt"})
    deadline = time.monotonic() + 2
    while service.get_agent(agent.agent_id).status != "running" and time.monotonic() < deadline:
        time.sleep(0.01)

    assert service.interrupt_agent(agent.agent_id, timeout_s=2)
    result = service.wait_agent(agent.agent_id, timeout_s=2).provider_result

    assert result.status is ProviderRunState.INTERRUPTED
    assert service.get_agent(agent.agent_id).status == "idle"


def test_claude_snapshot_home_mismatch_does_not_remove_live_transcript(tmp_path: Path) -> None:
    service, _provider = _service(tmp_path)
    agent = service.create_agent("scope-a", "claude-worker", provider_type="claude_code", home_id="worker")
    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "snapshot"}).agent_id)
    snapshot_service = AgentSnapshotService(
        service.runtime_root,
        store=service.store,
        agent_service=service,
    )
    snapshot = snapshot_service.create_scope_snapshot("scope-a")
    assert snapshot.snapshot_id is not None
    transcript = _transcript_path_for_agent(service, agent.agent_id)
    transcript.write_text("live-sentinel\n", encoding="utf-8")
    home_manifest = (
        service.runtime_root
        / "homes"
        / "claude_code"
        / "worker"
        / ".ark"
        / "home_materialization.json"
    )
    home_manifest.write_text(home_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Home materialization does not match"):
        snapshot_service.restore_scope_snapshot(snapshot.snapshot_id)

    assert transcript.read_text(encoding="utf-8") == "live-sentinel\n"


def test_claude_provider_fork_is_session_only(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    service, _provider = _service(tmp_path)
    agent = service.create_agent("scope-a", "claude-worker", provider_type="claude_code", home_id="worker")
    service.wait_agent(service.start_agent(agent.agent_id, variables={"item": "fork"}).agent_id)
    target_session_id = str(uuid.uuid4())

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        source = _transcript_path_for_agent(service, agent.agent_id)
        target = source.with_name(f"{target_session_id}.jsonl")
        shutil.copy2(source, target)
        return subprocess.CompletedProcess(command, 0, json.dumps({"session_id": target_session_id}), "")

    monkeypatch.setattr(
        "agent_runtime_kit.agent.providers.claude_code_runtime.subprocess.run",
        fake_run,
    )

    forked = service.fork_agent(agent.agent_id, target_scope_id="scope-b")

    assert forked.session_locator.session_id == target_session_id
    assert forked.fork_info.fork_mode == "session_only"
    assert not forked.fork_info.workspace_isolated


def _transcript_path(options: ClaudeAgentOptions, session_id: str) -> Path:
    return Path(str(options.env["CLAUDE_CONFIG_DIR"])) / "projects" / "test" / f"{session_id}.jsonl"


def _transcript_path_for_agent(service: AgentService, agent_id: str) -> Path:
    agent = service.get_agent(agent_id)
    return (
        service.runtime_root
        / "homes"
        / "claude_code"
        / agent.home_id
        / ".claude"
        / str(agent.artifact_locator.native_primary_ref)
    )


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
