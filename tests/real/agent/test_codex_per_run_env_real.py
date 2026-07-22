from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import ProviderHomeSpec, McpServerSpec
from agent_runtime_kit.agent.models import CompletionDecision
from agent_runtime_kit.agent.provider_contracts import BaseConfigSource, ProviderRegistry
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.providers.codex_home import CodexHomeOptions
from agent_runtime_kit.agent.service import AgentCompletionContext, AgentService, AgentType, AgentTypeRegistry


pytestmark = pytest.mark.real_codex


class RealCodexMcpEnvAgentType(AgentType):
    agent_type = "mcp_env_worker"
    developer_instructions_template = (
        "You are running an agent-runtime-kit real MCP environment smoke test. "
        "When asked, call the ark_identity MCP tool exactly once and reply with exactly the tool result."
    )
    start_prompt_template = (
        "Call the ark_identity MCP tool now. Reply with exactly the returned text and no extra text."
    )
    continue_prompt_template = start_prompt_template

    def check_completion(self, ctx: AgentCompletionContext) -> CompletionDecision:
        if not ctx.turn_result.run_id:
            return CompletionDecision(complete=False, reason="turn result has no run id")
        return CompletionDecision(complete=True)


def test_real_codex_same_home_per_run_env_reaches_stdio_mcp(tmp_path: Path) -> None:
    _ensure_real_codex_enabled()
    runtime_root = tmp_path / "project" / ".agent_runtime"
    marker_path = tmp_path / "ark_identity_marker.json"
    server_path = _write_env_mcp_server(tmp_path / "ark_identity_mcp.py")
    service = _service(runtime_root)
    try:
        service.create_home(
            ProviderHomeSpec(
                provider_type="codex",
                home_id="mcp_env_worker",
                base_config=BaseConfigSource(path=str(_config_dir() / "config.toml")),
                provider_options=CodexHomeOptions(auth_json_path=_config_dir() / "auth.json"),
                mcp_servers=(
                    McpServerSpec(
                        name="ark_identity",
                        command=sys.executable,
                        args=[str(server_path), str(marker_path)],
                        env_vars=["ARK_STEP_ID", "ARK_RUN_TOKEN"],
                        required=True,
                    ),
                ),
            ),
            initialize_provider_home=True,
        )
        agent = service.create_agent("repo-a:mcp-env", "mcp_env_worker")

        service.wait_agent(
            service.start_agent(
                agent.agent_id,
                variables={},
                env={"ARK_STEP_ID": "step-one", "ARK_RUN_TOKEN": "token-one"},
            ).agent_id,
            timeout_s=600,
        )
        first_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert first_marker["step_id"] == "step-one"
        assert first_marker["run_token"] == "token-one"

        service.wait_agent(
            service.start_agent(
                agent.agent_id,
                variables={},
                env={"ARK_STEP_ID": "step-two", "ARK_RUN_TOKEN": "token-two"},
            ).agent_id,
            timeout_s=600,
        )
        second_marker = json.loads(marker_path.read_text(encoding="utf-8"))
        assert second_marker["step_id"] == "step-two"
        assert second_marker["run_token"] == "token-two"
    finally:
        service.close()


def _write_env_mcp_server(server_path: Path) -> Path:
    server_path.write_text(
        "from __future__ import annotations\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "\n"
        "marker_path = Path(sys.argv[1])\n"
        "mcp = FastMCP('ark-identity')\n"
        "\n"
        "@mcp.tool()\n"
        "def ark_identity() -> str:\n"
        "    payload = {\n"
        "        'step_id': os.environ.get('ARK_STEP_ID'),\n"
        "        'run_token': os.environ.get('ARK_RUN_TOKEN'),\n"
        "    }\n"
        "    marker_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')\n"
        "    return f\"ARK_IDENTITY::{payload['step_id']}::{payload['run_token']}\"\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    mcp.run(transport='stdio')\n",
        encoding="utf-8",
    )
    return server_path


def _service(runtime_root: Path) -> AgentService:
    registry = AgentTypeRegistry()
    registry.register(RealCodexMcpEnvAgentType())
    provider = CodexProvider(
        runtime_root=runtime_root,
        codex_bin=os.environ.get("ARK_CODEX_BIN") or shutil.which("codex"),
        sdk_python_root=_sdk_python_root(),
        model=os.environ.get("ARK_REAL_CODEX_MODEL"),
    )
    return AgentService(
        runtime_root,
        agent_types=registry,
        provider_registry=ProviderRegistry((provider.build_provider_bundle(runtime_root=runtime_root),)),
    )


def _ensure_real_codex_enabled() -> None:
    if os.environ.get("ARK_RUN_REAL_CODEX") != "1":
        pytest.skip("set ARK_RUN_REAL_CODEX=1 to run real Codex SDK tests")
    if shutil.which("codex") is None and not os.environ.get("ARK_CODEX_BIN"):
        pytest.skip("codex binary is not available")
    sdk_root = _sdk_python_root()
    if importlib.util.find_spec("openai_codex") is None and sdk_root is None:
        pytest.skip("openai_codex is not installed and no local SDK root is available")


def _sdk_python_root() -> Path | None:
    value = os.environ.get("ARK_CODEX_SDK_PYTHON_ROOT")
    root = Path(value) if value else Path("/root/code/tools/codex/sdk/python")
    if not root.exists():
        return None
    src = root / "src"
    if not (src / "openai_codex").exists():
        pytest.skip(f"invalid Codex SDK Python root: {root}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def _config_dir() -> Path:
    path = Path(os.environ.get("ARK_CODEX_CONFIG_DIR", "data/configs/codex"))
    if not path.exists():
        pytest.skip(f"Codex config dir does not exist: {path}")
    if not (path / "config.toml").exists() or not (path / "auth.json").exists():
        pytest.skip(f"Codex config dir must contain config.toml and auth.json: {path}")
    return path
