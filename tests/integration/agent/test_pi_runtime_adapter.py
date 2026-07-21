from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeService, McpServerSpec
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.snapshots import AgentSnapshotService
from agent_runtime_kit.agent.provider_contracts import (
    ModelBackendIdentity,
    ProviderContextCompactionRequest,
    ProviderContextQuery,
    ProviderForkRequest,
    ProviderHomeSpec,
    ProviderRegistry,
    ProviderRunRequest,
    ProviderRunState,
)
from agent_runtime_kit.agent.providers.pi_bundle import build_pi_provider_bundle
from agent_runtime_kit.agent.providers.pi_home import PiHomeOptions


NODE = Path("/root/.npm/_npx/992a19d7d9bf36d4/node_modules/node/bin/node")
PI_CLI = Path(
    "/root/code/worktrees/agent-runtime-kit-provider-research/data/provider_research/"
    "pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js"
)
MCP_RUNTIME = Path(
    "/root/code/worktrees/agent-runtime-kit-provider-research/data/provider_research/pi-runtime"
)
FIXTURES = Path(__file__).parents[2] / "fixtures"


class PiIntegrationAgentType(AgentType):
    agent_type = "pi-worker"
    developer_instructions_template = "Use the Pi test provider."
    start_prompt_template = "Run the Pi provider integration probe."


pytestmark = pytest.mark.skipif(
    not NODE.is_file() or not PI_CLI.is_file(),
    reason="fixed Pi 0.80.10 research runtime is unavailable",
)


def _request(context, *, prompt: str, session=None, provider: str = "ark-probe", model: str = "faux-1"):  # noqa: ANN001, ANN202
    return ProviderRunRequest(
        agent_id="agent-1",
        scope_id="scope-1",
        agent_type="worker",
        provider_type="pi",
        home_id="demo",
        prompt=prompt,
        session_locator=session,
        workdir=context.workdir,
        model_overrides=ModelBackendIdentity(
            api_provider=provider,
            api_mode="ark-probe-api",
            requested_model=model,
        ),
        execution_context=context,
    )


def test_pi_fixed_rpc_runtime_wait_resume_fork_and_snapshot_locator(tmp_path: Path) -> None:
    bundle = build_pi_provider_bundle(runtime_root=tmp_path)
    homes = HomeService(tmp_path, renderers={"pi": bundle.home_renderer})
    home = homes.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="demo",
            provider_options=PiHomeOptions(
                node_executable=str(NODE),
                pi_cli_path=PI_CLI,
                extension_paths=(FIXTURES / "pi_faux_provider.ts",),
                offline=True,
                tools=("read", "write"),
                settings={
                    "defaultProvider": "ark-probe",
                    "defaultModel": "faux-1",
                    "compaction": {"enabled": True, "reserveTokens": 100, "keepRecentTokens": 1},
                },
            ),
        )
    )
    context = homes.build_execution_context("pi", "demo", workdir=str(tmp_path))
    bundle.home_renderer.initialize(home, context)

    first = bundle.runtime.start(_request(context, prompt="run")).wait_terminal(60)
    assert first.status is ProviderRunState.COMPLETED
    assert first.final_text == "ARK_PI_PROVIDER_DONE"
    assert first.tool_calls[0].tool_name == "write"
    assert first.artifact_locator is not None
    assert (tmp_path / str(first.artifact_locator.native_primary_ref)).is_file()

    second = bundle.runtime.resume(
        _request(context, prompt="resume", session=first.session_locator)
    ).wait_terminal(60)
    assert second.session_locator.session_id == first.session_locator.session_id
    assert second.turn_locator is not None
    assert second.turn_locator.turn_id != first.turn_locator.turn_id

    session_root = context.home_root / ".pi" / "sessions"
    before_historical_fork = set(session_root.glob("*.jsonl"))
    historical = bundle.runtime.fork(
        ProviderForkRequest(
            source_agent_id="agent-1",
            source_session=second.session_locator,
            source_turn=first.turn_locator,
            target_agent_id="agent-historical",
            target_scope_id="scope-1",
            target_home_id="demo",
            execution_context=context,
        )
    )
    after_historical_fork = set(session_root.glob("*.jsonl"))
    assert historical.target_session.session_id != second.session_locator.session_id
    assert len(after_historical_fork - before_historical_fork) == 1

    assert bundle.context is not None
    before = bundle.context.inspect(
        ProviderContextQuery(session=second.session_locator, execution_context=context)
    )
    assert before.available
    compacted = bundle.context.compact(
        ProviderContextCompactionRequest(
            session=second.session_locator,
            trigger="test",
            timeout_s=60,
            execution_context=context,
        )
    )
    assert compacted.status == "compacted"
    assert compacted.usage_after is not None
    assert not compacted.usage_after.available
    assert compacted.usage_after.stale

    forked = bundle.runtime.fork(
        ProviderForkRequest(
            source_agent_id="agent-1",
            source_session=second.session_locator,
            target_agent_id="agent-2",
            target_scope_id="scope-1",
            target_home_id="demo",
            execution_context=context,
        )
    )
    assert forked.target_session.session_id != second.session_locator.session_id
    assert forked.fork_mode == "session_only"
    assert not forked.workspace_isolated


@pytest.mark.skipif(
    not (MCP_RUNTIME / "node_modules" / "@modelcontextprotocol" / "sdk").exists(),
    reason="fixed MCP SDK runtime is unavailable",
)
def test_pi_ark_owned_mcp_bridge_calls_stdio_server(tmp_path: Path) -> None:
    bundle = build_pi_provider_bundle(runtime_root=tmp_path)
    homes = HomeService(tmp_path, renderers={"pi": bundle.home_renderer})
    home = homes.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="demo",
            mcp_servers=(
                McpServerSpec(
                    name="demo",
                    transport="stdio",
                    command=sys.executable,
                    args=[str(FIXTURES / "pi_mcp_stdio_server.py")],
                    required=True,
                    tool_timeout_sec=20,
                ),
            ),
            provider_options=PiHomeOptions(
                node_executable=str(NODE),
                pi_cli_path=PI_CLI,
                mcp_runtime_root=MCP_RUNTIME,
                extension_paths=(FIXTURES / "pi_faux_mcp_provider.ts",),
                offline=True,
                tools=("mcp__demo__echo",),
                settings={"defaultProvider": "ark-mcp-probe", "defaultModel": "faux-mcp"},
            ),
        )
    )
    context = homes.build_execution_context("pi", "demo", workdir=str(tmp_path))
    bundle.home_renderer.initialize(home, context)
    result = bundle.runtime.start(
        _request(
            context,
            prompt="call MCP",
            provider="ark-mcp-probe",
            model="faux-mcp",
        )
    ).wait_terminal(60)

    assert result.status is ProviderRunState.COMPLETED
    assert result.final_text == "ARK_PI_MCP_DONE"
    assert result.tool_calls[0].tool_name == "mcp__demo__echo"
    assert "ARK_MCP_SERVER:ARK_PI_MCP_OK" in str(result.tool_calls[0].result)


@pytest.mark.skipif(
    not (MCP_RUNTIME / "node_modules" / "@modelcontextprotocol" / "sdk").exists(),
    reason="fixed MCP SDK runtime is unavailable",
)
def test_pi_ark_owned_mcp_bridge_calls_streamable_http_server(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    env = dict(os.environ)
    env["ARK_PI_MCP_HTTP_PORT"] = str(port)
    server = subprocess.Popen(
        [sys.executable, str(FIXTURES / "pi_mcp_http_server.py")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                if server.poll() is not None:
                    pytest.fail(f"HTTP MCP fixture exited with code {server.returncode}")
                time.sleep(0.05)
        else:
            pytest.fail("HTTP MCP fixture did not become ready")

        bundle = build_pi_provider_bundle(runtime_root=tmp_path)
        homes = HomeService(tmp_path, renderers={"pi": bundle.home_renderer})
        home = homes.create_home(
            ProviderHomeSpec(
                provider_type="pi",
                home_id="demo",
                mcp_servers=(
                    McpServerSpec(
                        name="demo",
                        transport="http",
                        url=f"http://127.0.0.1:{port}/mcp",
                        required=True,
                        tool_timeout_sec=20,
                    ),
                ),
                provider_options=PiHomeOptions(
                    node_executable=str(NODE),
                    pi_cli_path=PI_CLI,
                    mcp_runtime_root=MCP_RUNTIME,
                    extension_paths=(FIXTURES / "pi_faux_mcp_provider.ts",),
                    offline=True,
                    tools=("mcp__demo__echo",),
                    settings={"defaultProvider": "ark-mcp-probe", "defaultModel": "faux-mcp"},
                ),
            )
        )
        context = homes.build_execution_context("pi", "demo", workdir=str(tmp_path))
        bundle.home_renderer.initialize(home, context)
        result = bundle.runtime.start(
            _request(
                context,
                prompt="call HTTP MCP",
                provider="ark-mcp-probe",
                model="faux-mcp",
            )
        ).wait_terminal(60)

        assert result.status is ProviderRunState.COMPLETED
        assert result.final_text == "ARK_PI_MCP_DONE"
        assert result.tool_calls[0].tool_name == "mcp__demo__echo"
        assert "ARK_MCP_SERVER:ARK_PI_MCP_OK" in str(result.tool_calls[0].result)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def test_pi_runtime_abort_waits_for_settled_terminal(tmp_path: Path) -> None:
    bundle = build_pi_provider_bundle(runtime_root=tmp_path)
    homes = HomeService(tmp_path, renderers={"pi": bundle.home_renderer})
    homes.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="demo",
            provider_options=PiHomeOptions(
                node_executable=str(NODE),
                pi_cli_path=PI_CLI,
                extension_paths=(FIXTURES / "pi_faux_slow_provider.ts",),
                offline=True,
                settings={"defaultProvider": "ark-slow-probe", "defaultModel": "faux-slow"},
            ),
        )
    )
    context = homes.build_execution_context("pi", "demo", workdir=str(tmp_path))
    handle = bundle.runtime.start(
        _request(
            context,
            prompt="start slow response",
            provider="ark-slow-probe",
            model="faux-slow",
        )
    )
    deadline = time.monotonic() + 15
    while handle.session_locator() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    interrupted = handle.interrupt(30)
    result = handle.wait_terminal(30)

    assert interrupted.accepted
    assert interrupted.terminal_confirmed
    assert result.status is ProviderRunState.INTERRUPTED
    assert any(event.kind == "terminal.settled" for event in handle.drain_events().events)


def test_pi_agent_service_snapshot_restore_and_standard_query(tmp_path: Path) -> None:
    bundle = build_pi_provider_bundle(runtime_root=tmp_path)
    types = AgentTypeRegistry()
    types.register(PiIntegrationAgentType())
    service = AgentService(
        tmp_path,
        agent_types=types,
        provider_registry=ProviderRegistry((bundle,)),
    )
    service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id="pi-worker",
            provider_options=PiHomeOptions(
                node_executable=str(NODE),
                pi_cli_path=PI_CLI,
                extension_paths=(FIXTURES / "pi_faux_provider.ts",),
                offline=True,
                tools=("read", "write"),
                settings={"defaultProvider": "ark-probe", "defaultModel": "faux-1"},
            ),
        )
    )
    service.ensure_provider_home_initialized("pi", "pi-worker", workdir=str(tmp_path))
    agent = service.create_agent("scope-pi", "pi-worker", provider_type="pi")
    service.start_agent(agent.agent_id, workdir=str(tmp_path))
    result = service.wait_agent(agent.agent_id, 60)
    assert result.final_text == "ARK_PI_PROVIDER_DONE"
    persisted = service.get_agent(agent.agent_id)
    assert persisted.session_locator == result.session_locator
    usage = service.query_usage(agent.agent_id, latest=True)
    assert usage.request_count == 2

    assert persisted.artifact_locator is not None
    session_path = tmp_path / str(persisted.artifact_locator.native_primary_ref)
    original = session_path.read_bytes()
    snapshots = AgentSnapshotService(tmp_path, store=service.store, agent_service=service)
    snapshot = snapshots.create_scope_snapshot("scope-pi")
    session_path.write_text("corrupted\n", encoding="utf-8")
    restored = snapshots.restore_scope_snapshot(str(snapshot.snapshot_id))

    assert restored.status == "created"
    assert session_path.read_bytes() == original
    assert service.query_usage(agent.agent_id, latest=True).request_count == 2
    restored_session_id = service.get_agent(agent.agent_id).session_locator.session_id
    service.pause_controller.resume("scope-pi")
    service.start_agent(agent.agent_id, workdir=str(tmp_path))
    resumed = service.wait_agent(agent.agent_id, 60)
    assert resumed.session_locator.session_id == restored_session_id
    assert resumed.turn_locator is not None
    assert resumed.turn_locator.turn_id != result.turn_locator.turn_id
