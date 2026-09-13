"""Opt-in real model acceptance for Grok extended capabilities."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

if os.environ.get("ARK_RUN_REAL_GROK") != "1":
    raise SystemExit("Set ARK_RUN_REAL_GROK=1 to authorize real model calls.")

from agent_runtime_kit.agent.models import to_jsonable
from agent_runtime_kit.agent.provider_contracts import (
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
    ProviderHomeSpec,
    ProviderRegistry,
    ProviderControlAction,
    ProviderControlRequest,
    ProviderRunRequest,
)
from agent_runtime_kit.agent.providers.grok_bundle import build_grok_provider_bundle
from agent_runtime_kit.agent.providers.grok_home import GrokHomeOptions
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.skills import SkillSpec

ROOT = Path(tempfile.mkdtemp(prefix="ark-grok-extended-acceptance-"))
WORK = ROOT / "workspace"
WORK.mkdir()
REPORT = {"root": str(ROOT), "checks": {}}
print("ROOT", ROOT, flush=True)


def save():
    (ROOT / "acceptance.json").write_text(json.dumps(REPORT, indent=2))


class Worker(AgentType):
    agent_type = "grok-extended-validation"
    provider_type = "grok"
    start_prompt_template = "{{task}}"


registry = AgentTypeRegistry()
registry.register(Worker())
bundle = build_grok_provider_bundle(runtime_root=ROOT / "runtime")
service = AgentService(
    ROOT / "runtime",
    agent_types=registry,
    provider_registry=ProviderRegistry((bundle,)),
)


def home(name, skills=(), tools=None):
    service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id=name,
            skills=skills,
            provider_options=GrokHomeOptions(
                model="grok-4.6", reasoning_effort="low", tools=tools
            ),
        )
    )
    return service.create_agent("validation", Worker.agent_type, home_id=name)


def turn(agent, name, prompt):
    service.start_agent(agent.agent_id, prompt=prompt, workdir=str(WORK))
    value = service.wait_agent(agent.agent_id, timeout_s=180).provider_result
    REPORT["checks"][name] = to_jsonable(value)
    save()
    print(name, value.status, repr(value.final_text), flush=True)
    assert value.status.value == "completed", value.error
    return value


def capture(session, name):
    snapshot = bundle.artifacts.capture(
        ArtifactCaptureRequest(
            session=session,
            snapshot_root=str(ROOT / name),
        )
    )
    REPORT["checks"][name] = to_jsonable(snapshot)
    save()
    return snapshot


def restore(session, snapshot):
    request = ArtifactRestoreRequest(
        manifest=snapshot.manifest, snapshot_root=snapshot.snapshot_root
    )
    bundle.artifacts.prepare_restore(request)
    bundle.artifacts.restore(request)


try:
    mode = sys.argv[1]
    if mode == "skills":
        skill = SkillSpec(
            name="ark-receipt",
            description="Use when asked for an ARK receipt.",
            body=(
                "For an ARK receipt, read resources/token.txt relative to this "
                "skill directory using read_file, then reply ARK_RECEIPT: followed "
                "by its exact contents. Do not guess the token."
            ),
            files={"resources/token.txt": "NECTAR_90317"},
        )
        agent = home("skills", (skill,))
        result = turn(
            agent, "skill", "Use the ark-receipt skill to give an ARK receipt."
        )
        assert (
            "ARK_RECEIPT:" in result.final_text and "NECTAR_90317" in result.final_text
        )
        assert result.tool_calls
        result = turn(
            agent,
            "skill_resume",
            "Give another ARK receipt using the skill and read its resource again.",
        )
        assert "NECTAR_90317" in result.final_text and result.tool_calls
    elif mode == "fork":
        parent = home("fork")
        turn(
            parent,
            "parent_initial",
            "Remember my code PARENT_18473. Reply SAVED; no tools.",
        )
        child = service.fork_agent(parent.agent_id)
        REPORT["child"] = to_jsonable(child)
        result = turn(
            child, "child_inherited", "What is my code? Reply only it; no tools."
        )
        assert "PARENT_18473" in result.final_text
        snapshot = capture(result.session_locator, "child_snapshot")
        turn(
            child,
            "child_change",
            "My code is now CHILD_69742. Reply UPDATED; no tools.",
        )
        result = turn(
            parent, "parent_independent", "What is my code? Reply only it; no tools."
        )
        assert (
            "PARENT_18473" in result.final_text
            and "CHILD_69742" not in result.final_text
        )
        result = turn(
            child, "child_independent", "What is my code? Reply only it; no tools."
        )
        assert "CHILD_69742" in result.final_text
        restore(result.session_locator, snapshot)
        result = turn(
            child, "child_restored", "What is my code? Reply only it; no tools."
        )
        assert "PARENT_18473" in result.final_text
    elif mode == "compact":
        agent = home("compact")
        filler = "\n".join(
            f"Archive entry {i}: completed routine task with no pending action."
            for i in range(250)
        )
        result = turn(
            agent,
            "before_compact",
            "Remember important code AMBER_52186. The following archive is low priority.\n"
            + filler
            + "\nReply SAVED; no tools.",
        )
        snapshot = capture(result.session_locator, "pre_compact_snapshot")
        compact = service.compact_agent(
            agent.agent_id, timeout_s=180, workdir=str(WORK)
        )
        REPORT["compaction"] = to_jsonable(compact)
        save()
        assert compact.status.value == "compacted", compact
        result = turn(
            agent,
            "after_compact",
            "What important code did I ask you to remember? Reply only it; no tools.",
        )
        assert "AMBER_52186" in result.final_text
        restore(result.session_locator, snapshot)
        result = turn(
            agent,
            "restored_pre_compact",
            "What important code did I ask you to remember? Reply only it; no tools.",
        )
        assert "AMBER_52186" in result.final_text
    elif mode == "steer":
        agent = home("steer", tools=("read_file", "run_terminal_cmd"))
        initial = turn(agent, "initial", "Reply READY; no tools.")
        context = service.home_service.build_execution_context(
            "grok", "steer", workdir=str(WORK)
        )
        handle = bundle.runtime.resume(
            ProviderRunRequest(
                agent_id=agent.agent_id,
                scope_id=agent.scope_id,
                agent_type=agent.agent_type,
                provider_type="grok",
                home_id="steer",
                session_locator=initial.session_locator,
                workdir=str(WORK),
                execution_context=context,
                prompt="Use run_terminal_cmd to run: echo ready > steer_started; sleep 15 . Wait for it in foreground, then reply ORIGINAL. Do not run in background.",
            )
        )
        deadline = time.monotonic() + 120
        while not (WORK / "steer_started").exists() and time.monotonic() < deadline:
            if handle.poll_state().terminal:
                raise AssertionError("Turn ended before steer test synchronization")
            time.sleep(0.1)
        assert (WORK / "steer_started").exists()
        receipt = handle.control(
            ProviderControlRequest(
                action=ProviderControlAction.STEER,
                requested_at="2026-09-13T00:00:00Z",
                run_id=handle.run_id,
                content="Change the final reply to STEERED_86319, not ORIGINAL.",
            )
        )
        REPORT["steer_receipt"] = to_jsonable(receipt)
        save()
        assert receipt.accepted and not receipt.terminal_confirmed, receipt
        result = handle.wait_terminal(timeout_s=180)
        REPORT["checks"]["steered"] = to_jsonable(result)
        assert result.status.value == "completed"
        assert "STEERED_86319" in result.final_text
        assert result.session_locator.session_id == initial.session_locator.session_id
    else:
        raise ValueError("Expected skills, fork, compact or steer")
    REPORT["passed"] = True
except BaseException as exc:
    REPORT["passed"] = False
    REPORT["error"] = repr(exc)
    raise
finally:
    save()
    bundle.runtime.close()
    print("REPORT", ROOT / "acceptance.json", flush=True)
