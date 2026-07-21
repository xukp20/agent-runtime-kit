from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeService
from agent_runtime_kit.agent.provider_contracts import (
    ModelBackendIdentity,
    ProviderHomeSpec,
    ProviderRunOptions,
    ProviderRunRequest,
    ProviderRunState,
)
from agent_runtime_kit.agent.providers.pi_bundle import build_pi_provider_bundle
from agent_runtime_kit.agent.providers.pi_home import PiHomeOptions


pytestmark = [pytest.mark.real, pytest.mark.real_pi]


def _runtime() -> tuple[str, Path]:
    if os.getenv("ARK_RUN_REAL_PI") != "1":
        pytest.skip("set ARK_RUN_REAL_PI=1 to enable real Pi tests")
    node = os.getenv(
        "ARK_PI_NODE_EXECUTABLE",
        "/root/.npm/_npx/992a19d7d9bf36d4/node_modules/node/bin/node",
    )
    cli = Path(
        os.getenv(
            "ARK_PI_CLI_PATH",
            "/root/code/worktrees/agent-runtime-kit-provider-research/data/provider_research/"
            "pi-runtime/node_modules/@earendil-works/pi-coding-agent/dist/cli.js",
        )
    )
    if not Path(node).is_file() or not cli.is_file():
        pytest.skip("fixed Pi 0.80.10 runtime is unavailable")
    return node, cli


def _run(
    tmp_path: Path,
    *,
    profile: str,
    provider: str,
    api_mode: str,
    model: str,
    marker: str,
    auth: Path | None = None,
    models: dict[str, object] | None = None,
    required_env: tuple[str, ...] = (),
) -> object:
    node, cli = _runtime()
    bundle = build_pi_provider_bundle(runtime_root=tmp_path)
    homes = HomeService(tmp_path, renderers={"pi": bundle.home_renderer})
    home = homes.create_home(
        ProviderHomeSpec(
            provider_type="pi",
            home_id=profile,
            required_env=required_env,
            provider_options=PiHomeOptions(
                node_executable=node,
                pi_cli_path=cli,
                auth_json_path=auth,
                models=models,
                settings={"defaultProvider": provider, "defaultModel": model},
            ),
        )
    )
    context = homes.build_execution_context("pi", profile, workdir=str(tmp_path))
    bundle.home_renderer.initialize(home, context)
    result = bundle.runtime.start(
        ProviderRunRequest(
            agent_id=f"real-{profile}",
            scope_id="real-pi",
            agent_type="real-pi",
            provider_type="pi",
            home_id=profile,
            prompt=f"Reply with exactly {marker} and no other text.",
            workdir=str(tmp_path),
            model_overrides=ModelBackendIdentity(
                api_provider=provider,
                api_mode=api_mode,
                requested_model=model,
            ),
            run_options=ProviderRunOptions(timeout_s=180),
            execution_context=context,
        )
    ).wait_terminal(200)
    assert result.status is ProviderRunState.COMPLETED
    assert result.final_text == marker
    assert result.turn_usage is not None
    assert result.turn_usage.request_count
    return result


def test_pi_openai_codex_oauth_real(tmp_path: Path) -> None:
    auth = Path(os.getenv("ARK_PI_OAUTH_AUTH_JSON", "/root/.pi/ark-provider-tests/auth.json"))
    if not auth.is_file():
        pytest.skip("Pi OAuth auth.json is unavailable")
    result = _run(
        tmp_path,
        profile="oauth",
        provider="openai-codex",
        api_mode="responses",
        model=os.getenv("ARK_PI_OAUTH_MODEL", "gpt-5.5"),
        marker="ARK_PI_OAUTH_OK",
        auth=auth,
    )
    assert result.request_usages[0].model_identity.api_mode == "responses"


def test_pi_deepseek_chat_completions_real(tmp_path: Path) -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is unavailable")
    result = _run(
        tmp_path,
        profile="deepseek",
        provider="deepseek",
        api_mode="chat_completions",
        model=os.getenv("ARK_PI_DEEPSEEK_MODEL", "deepseek-v4-flash"),
        marker="ARK_PI_DEEPSEEK_OK",
        required_env=("DEEPSEEK_API_KEY",),
    )
    assert result.request_usages[0].model_identity.api_mode == "chat_completions"


def test_pi_beeapi_responses_real(tmp_path: Path) -> None:
    if not os.getenv("BEEAPI_API_KEY"):
        pytest.skip("BEEAPI_API_KEY is unavailable")
    model = os.getenv("ARK_PI_BEEAPI_MODEL", "gpt-5.4")
    provider = "beeapi-responses"
    models = {
        "providers": {
            provider: {
                "baseUrl": os.getenv("ARK_PI_BEEAPI_BASE_URL", "https://beeapi.ai/v1"),
                "api": "openai-responses",
                "apiKey": "$BEEAPI_API_KEY",
                "models": [
                    {
                        "id": model,
                        "name": "BeeAPI Responses",
                        "reasoning": True,
                        "input": ["text"],
                        "contextWindow": 1_050_000,
                        "maxTokens": 128_000,
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    }
                ],
            }
        }
    }
    result = _run(
        tmp_path,
        profile="beeapi",
        provider=provider,
        api_mode="responses",
        model=model,
        marker="ARK_PI_BEEAPI_OK",
        models=models,
        required_env=("BEEAPI_API_KEY",),
    )
    assert result.request_usages[0].response_id
    assert result.request_usages[0].model_identity.api_mode == "responses"

