from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

import pytest

from agent_runtime_kit.agent.homes import HomeCreateSpec
from agent_runtime_kit.agent.models import CompletionDecision
from agent_runtime_kit.agent.providers.codex import CodexProvider
from agent_runtime_kit.agent.service import AgentCompletionContext, AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.skills import SkillSpec


pytestmark = pytest.mark.real_codex


class RealCodexSkillAgentType(AgentType):
    agent_type = "skill_worker"
    developer_instructions_template = (
        "You are running an agent-runtime-kit real skill smoke test. "
        "If the user asks for the ARK developer instruction sentinel, reply exactly: "
        "ARK_DEV_INSTRUCTIONS_SEEN. Keep all answers short."
    )
    start_prompt_template = "{{prompt}}"
    continue_prompt_template = "{{prompt}}"

    def check_completion(self, ctx: AgentCompletionContext) -> CompletionDecision:
        if not getattr(ctx.turn_result, "id", None):
            return CompletionDecision(complete=False, reason="turn result has no id")
        return CompletionDecision(complete=True)


def test_real_codex_developer_instructions_and_skills(tmp_path: Path) -> None:
    runtime_root = tmp_path / "project" / ".agent_runtime"
    service = _service(runtime_root)
    try:
        _create_codex_home_with_skills(service, tmp_path, "skill_worker")
        agent = service.create_agent("repo-a:skill-node", "skill_worker")

        _run_and_assert_latest_text(
            service,
            agent.agent_id,
            prompt="Reply with the ARK developer instruction sentinel and no extra text.",
            expected="ARK_DEV_INSTRUCTIONS_SEEN",
        )
        _run_and_assert_latest_text(
            service,
            agent.agent_id,
            prompt="Use $ark-path-skill and reply only with the sentinel from that skill.",
            expected="ARK_PATH_SKILL_SEEN",
        )
        _run_and_assert_latest_text(
            service,
            agent.agent_id,
            prompt="Use $ark-spec-skill and reply only with the sentinel from that skill.",
            expected="ARK_SPEC_SKILL_SEEN",
        )
    finally:
        service.close(force_provider_homes=True)


def _run_and_assert_latest_text(service: AgentService, agent_id: str, *, prompt: str, expected: str) -> None:
    service.wait_agent(
        service.start_agent(agent_id, variables={"prompt": prompt}).agent_id,
        timeout_s=600,
    )
    latest = service.read_latest_turn_result(agent_id)
    text = getattr(latest, "final_response", None)
    assert isinstance(text, str), f"latest turn has no final response: {latest!r}"
    assert expected in text


def _service(runtime_root: Path) -> AgentService:
    _ensure_real_codex_enabled()
    registry = AgentTypeRegistry()
    registry.register(RealCodexSkillAgentType())
    provider = CodexProvider(
        runtime_root=runtime_root,
        codex_bin=os.environ.get("ARK_CODEX_BIN") or shutil.which("codex"),
        sdk_python_root=_sdk_python_root(),
        model=os.environ.get("ARK_REAL_CODEX_MODEL"),
    )
    return AgentService(runtime_root, agent_types=registry, providers={"codex": provider})


def _create_codex_home_with_skills(service: AgentService, tmp_path: Path, home_id: str) -> None:
    config_dir = _config_dir()
    path_skill = tmp_path / "ark-path-skill"
    path_skill.mkdir()
    (path_skill / "SKILL.md").write_text(
        """---
name: ark-path-skill
description: Use this skill when asked for the ARK path skill sentinel.
---

The sentinel is ARK_PATH_SKILL_SEEN.
""",
        encoding="utf-8",
    )

    service.home_service.create_home(
        HomeCreateSpec(
            cli_type="codex",
            home_id=home_id,
            base_config_path=config_dir / "config.toml",
            auth_json_path=config_dir / "auth.json",
            skill_paths={"ark-path-skill": path_skill},
            skill_specs={
                "ark-spec-skill": SkillSpec(
                    name="ark-spec-skill",
                    description="Use this skill when asked for the ARK spec skill sentinel.",
                    body="The sentinel is ARK_SPEC_SKILL_SEEN.",
                )
            },
        )
    )


def _ensure_real_codex_enabled() -> None:
    if os.environ.get("ARK_RUN_REAL_CODEX") != "1":
        pytest.skip("set ARK_RUN_REAL_CODEX=1 to run real Codex SDK tests")
    if shutil.which("codex") is None and not os.environ.get("ARK_CODEX_BIN"):
        pytest.skip("codex binary is not available")
    sdk_root = _sdk_python_root()
    if importlib.util.find_spec("openai_codex") is None and sdk_root is None:
        pytest.skip("openai_codex is not installed and ARK_CODEX_SDK_PYTHON_ROOT is not set")


def _sdk_python_root() -> Path | None:
    value = os.environ.get("ARK_CODEX_SDK_PYTHON_ROOT")
    if not value:
        return None
    root = Path(value)
    src = root / "src"
    if not (src / "openai_codex").exists():
        pytest.skip(f"invalid ARK_CODEX_SDK_PYTHON_ROOT: {root}")
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

