from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("agents")

from agents import Agent, function_tool

from agent_runtime_kit.agent.provider_contracts import (
    CapabilityKey,
    ModelBackendIdentity,
    ProviderContextCompactionRequest,
    ProviderHomeSpec,
    ProviderRunRequest,
    ProviderRunState,
)
from agent_runtime_kit.agent.providers import OpenAIAgentsHomeOptions, OpenAIAgentsProvider


pytestmark = [pytest.mark.real, pytest.mark.real_openai_agents]


@function_tool
def ark_live_echo(value: str) -> str:
    """Return an exact marker for the ARK live provider test."""

    return f"ARK_LIVE_TOOL:{value}"


def _enabled() -> None:
    if os.environ.get("ARK_RUN_REAL_OPENAI_AGENTS") != "1":
        pytest.skip("set ARK_RUN_REAL_OPENAI_AGENTS=1 to run live OpenAI Agents tests")


def _bundle(
    tmp_path: Path,
    *,
    identity: ModelBackendIdentity,
    api_key_env: str,
    base_url: str,
    compaction_mode: str = "unsupported",
):  # noqa: ANN202
    provider = OpenAIAgentsProvider()
    provider.registry.register_agent_factory(
        "real/basic",
        lambda ctx: Agent(
            name="ark-openai-agents-real",
            instructions=ctx.instructions,
            model=ctx.model,
            tools=[ark_live_echo],
        ),
    )
    bundle = provider.build_bundle(runtime_root=tmp_path)
    home_root = tmp_path / "homes" / "openai_agents" / "real"
    spec = ProviderHomeSpec(
        provider_type="openai_agents",
        home_id="real",
        model_config=identity,
        instructions=("Follow the user exactly. Keep the final answer short.",),
        required_env=(api_key_env,),
        provider_options=OpenAIAgentsHomeOptions(
            agent_factory_ref="real/basic",
            api_key_env=api_key_env,
            base_url=base_url,
            store=False,
            compaction_mode=compaction_mode,
        ),
    )
    materialized = bundle.home_renderer.materialize(spec, home_root)
    home = SimpleNamespace(
        home_id="real",
        home_relpath="homes/openai_agents/real",
        fixed_env={},
        materialization_manifest_hash=materialized.manifest_hash,
    )
    execution = bundle.home_renderer.build_execution_context(home, run_env=None, workdir=str(tmp_path))
    return bundle, execution


def _run(bundle, execution, prompt: str, session=None):  # noqa: ANN001, ANN202
    request = ProviderRunRequest(
        agent_id="real-agent",
        scope_id="real-scope",
        agent_type="real",
        provider_type="openai_agents",
        home_id="real",
        prompt=prompt,
        session_locator=session,
        execution_context=execution,
    )
    handle = bundle.runtime.resume(request) if session is not None else bundle.runtime.start(request)
    return handle.wait_terminal(240)


def test_deepseek_chat_sqlite_tool_usage_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _enabled()
    settings = json.loads(Path("/root/.claude/settings.json").read_text())
    configured = settings.get("env") or {}
    token = configured.get("ANTHROPIC_AUTH_TOKEN")
    anthropic_base = configured.get("ANTHROPIC_BASE_URL")
    if not token or not anthropic_base:
        pytest.skip("DeepSeek credentials are unavailable")
    monkeypatch.setenv("ARK_DEEPSEEK_API_KEY", token)
    base_url = str(anthropic_base).removesuffix("/anthropic").rstrip("/")
    bundle, execution = _bundle(
        tmp_path,
        identity=ModelBackendIdentity(
            api_provider="deepseek",
            api_mode="chat_completions",
            endpoint_id="deepseek-openai-compatible",
            requested_model="deepseek-v4-flash",
        ),
        api_key_env="ARK_DEEPSEEK_API_KEY",
        base_url=base_url,
    )
    first = _run(
        bundle,
        execution,
        "Call ark_live_echo exactly once with value DEEPSEEK, then reply with the tool result.",
    )
    assert first.status is ProviderRunState.COMPLETED
    assert first.final_text
    assert first.tool_calls
    assert first.turn_usage is not None and first.turn_usage.token_usage.total_tokens
    assert first.session_locator.backend_identity.api_mode == "chat_completions"  # type: ignore[union-attr]
    second = _run(
        bundle,
        execution,
        "Reply exactly ARK_DEEPSEEK_RESUMED without tools.",
        first.session_locator,
    )
    assert second.status is ProviderRunState.COMPLETED
    assert "ARK_DEEPSEEK_RESUMED" in (second.final_text or "")
    capabilities = bundle.resolve_capabilities(
        SimpleNamespace(
            home_id="real",
            resolved_defaults={
                "api_provider": "deepseek",
                "api_mode": "chat_completions",
                "requested_model": "deepseek-v4-flash",
            },
            provider_payload=None,
        )
    )
    assert not capabilities.available(CapabilityKey.CONTROL_COMPACT)


def test_beeapi_responses_tool_usage_cancel_compact_and_replay(tmp_path: Path) -> None:
    _enabled()
    if not os.environ.get("BEEAPI_API_KEY"):
        pytest.skip("BEEAPI_API_KEY is unavailable")
    bundle, execution = _bundle(
        tmp_path,
        identity=ModelBackendIdentity(
            api_provider="beeapi",
            api_mode="responses",
            endpoint_id="beeapi-responses",
            requested_model="gpt-5.4",
        ),
        api_key_env="BEEAPI_API_KEY",
        base_url="https://beeapi.ai/v1",
        compaction_mode="input_history",
    )
    first = _run(
        bundle,
        execution,
        "Call ark_live_echo exactly once with value BEEAPI. Then remember marker ARK_BEE_MEMORY_7429 and reply briefly.",
    )
    assert first.status is ProviderRunState.COMPLETED
    assert first.tool_calls
    assert first.turn_usage is not None and first.turn_usage.token_usage.total_tokens
    assert first.session_locator.backend_identity.api_mode == "responses"  # type: ignore[union-attr]

    cancel_request = ProviderRunRequest(
        agent_id="cancel-agent",
        scope_id="real-scope",
        agent_type="real",
        provider_type="openai_agents",
        home_id="real",
        prompt="Write a very long detailed essay with at least 5000 words.",
        execution_context=execution,
    )
    cancel_handle = bundle.runtime.start(cancel_request)
    cancelled = cancel_handle.interrupt(60)
    assert cancelled.accepted and cancelled.terminal_confirmed
    assert cancel_handle.wait_terminal(1).status is ProviderRunState.INTERRUPTED

    started: list[object] = []
    compacted = bundle.context.compact(
        ProviderContextCompactionRequest(
            session=first.session_locator,
            trigger="real_test",
            timeout_s=240,
            execution_context=execution,
            on_started=lambda baseline, operation_id: started.append((baseline, operation_id)),
        )
    )
    assert compacted.status == "compacted"
    assert started
    payload = compacted.provider_payload.sanitized_data  # type: ignore[union-attr]
    assert payload["normalized_type"] == "compaction_summary"
    assert "compaction_summary" in payload["raw_item_types"]

    replay = _run(
        bundle,
        execution,
        "What exact ARK_BEE_MEMORY marker were you told to remember? Reply with only it.",
        first.session_locator,
    )
    assert replay.status is ProviderRunState.COMPLETED
    assert "ARK_BEE_MEMORY_7429" in (replay.final_text or "")
