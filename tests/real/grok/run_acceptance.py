"""Real ARK/Grok acceptance in fresh temporary workspaces; model calls authorized."""

import json
import os
from pathlib import Path
import sys
import tempfile
import time

if os.environ.get("ARK_RUN_REAL_GROK") != "1":
    raise SystemExit("Set ARK_RUN_REAL_GROK=1 to authorize real Grok model calls.")

from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.homes import McpServerSpec
from agent_runtime_kit.agent.provider_contracts import (
    ProviderHomeSpec,
    ProviderRegistry,
    ArtifactCaptureRequest,
    ArtifactRestoreRequest,
)
from agent_runtime_kit.agent.providers.grok_bundle import build_grok_provider_bundle
from agent_runtime_kit.agent.providers.grok_home import GrokHomeOptions
from agent_runtime_kit.agent.models import to_jsonable
from agent_runtime_kit.agent.instructions import TextFragment
from agent_runtime_kit.agent.snapshots import AgentSnapshotService

ROOT = Path(tempfile.mkdtemp(prefix="ark-grok-acceptance-"))
WORK = ROOT / "workspace"
WORK.mkdir()
REPORT = {"root": str(ROOT), "checks": {}}
print("ROOT", ROOT, flush=True)


def save():
    (ROOT / "acceptance.json").write_text(json.dumps(REPORT, indent=2, default=str))


class Worker(AgentType):
    agent_type = "grok-validation"
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


def home(name, tools=None, servers=(), instructions=(), model=None):
    service.home_service.create_home(
        ProviderHomeSpec(
            provider_type="grok",
            home_id=name,
            provider_options=GrokHomeOptions(
                model=model, reasoning_effort="low", tools=tools
            ),
            mcp_servers=servers,
            instructions=instructions,
        )
    )
    return service.create_agent("validation", "grok-validation", home_id=name)


def turn(agent, key, prompt):
    service.start_agent(agent.agent_id, prompt=prompt, workdir=str(WORK))
    result = service.wait_agent(agent.agent_id, timeout_s=180)
    value = result.provider_result
    REPORT["checks"][key] = to_jsonable(value)
    save()
    print(key, value.status, repr(value.final_text), flush=True)
    assert value.status.value == "completed", value.error
    return value


try:
    mode = sys.argv[1] if len(sys.argv) > 1 else "memory"
    if mode not in {
        "memory",
        "stdio",
        "http",
        "cancel",
        "write_tools",
        "scope_snapshot",
        "instructions",
        "required_missing",
    }:
        raise ValueError("Unknown or unsupported acceptance mode")
    if mode == "memory":
        a = home("memory")
        result = turn(
            a,
            "fresh",
            "Remember the secret code CITRUS_8172. Reply exactly SAVED. Do not use tools.",
        )
        assert "SAVED" in result.final_text
        session = result.session_locator
        capture = bundle.artifacts.capture(
            ArtifactCaptureRequest(
                session=session, snapshot_root=str(ROOT / "snapshot")
            )
        )
        REPORT["capture"] = to_jsonable(capture)
        result = turn(
            a,
            "resume",
            "What is the secret code I told you? Reply only the code. Do not use tools.",
        )
        assert "CITRUS_8172" in result.final_text
        result = turn(
            a,
            "advance",
            "Replace the secret code with VIOLET_9436. Forget the previous code. Reply exactly UPDATED. Do not use tools.",
        )
        result = turn(
            a,
            "advanced_recall",
            "What is the current secret code? Reply only it. Do not use tools.",
        )
        assert "VIOLET_9436" in result.final_text
        bundle.artifacts.prepare_restore(
            ArtifactRestoreRequest(
                manifest=capture.manifest, snapshot_root=capture.snapshot_root
            )
        )
        restored = bundle.artifacts.restore(
            ArtifactRestoreRequest(
                manifest=capture.manifest, snapshot_root=capture.snapshot_root
            )
        )
        REPORT["restore"] = to_jsonable(restored)
        result = turn(
            a,
            "restored_recall",
            "What is the secret code I told you? Reply only the code. Do not use tools.",
        )
        assert (
            "CITRUS_8172" in result.final_text
            and "VIOLET_9436" not in result.final_text
        )
        b = home("fresh-isolation")
        result = turn(
            b,
            "isolation",
            "Have I told you a secret code earlier in this conversation? Reply exactly UNKNOWN if none. Do not use tools.",
        )
        assert "UNKNOWN" in result.final_text
        (WORK / "marker.txt").write_text("ARK_READ_6283")
        result = turn(
            b,
            "builtin_tool",
            "Read marker.txt using read_file and report its exact contents.",
        )
        assert "ARK_READ_6283" in result.final_text and result.tool_calls
    elif mode in ("stdio", "http", "sse"):
        server = McpServerSpec(
            name="ark_probe",
            transport=mode,
            required=True,
            command=sys.executable if mode == "stdio" else None,
            args=[str(Path(__file__).with_name("mcp_fixture.py").resolve())]
            if mode == "stdio"
            else [],
            env={"ARK_MCP_CALL_LOG": str(ROOT / "mcp_calls.txt")}
            if mode == "stdio"
            else {},
            url="http://127.0.0.1:18976/mcp"
            if mode == "http"
            else "http://127.0.0.1:18977/sse"
            if mode == "sse"
            else None,
        )
        a = home(mode, servers=(server,))
        result = turn(
            a,
            "mcp_" + mode,
            "Find ark_probe probe_echo with search_tool and call it using use_tool with token ARK_"
            + mode.upper()
            + ". Return its exact output.",
        )
        assert "ARK_MCP_OK:ARK_" + mode.upper() in result.final_text
        assert result.tool_calls
        if mode == "stdio":
            assert "ARK_STDIO" in (ROOT / "mcp_calls.txt").read_text()
    elif mode == "cancel":
        a = home("cancel", tools=("read_file", "run_terminal_cmd"))
        service.start_agent(
            a.agent_id,
            prompt="Use run_terminal_cmd to run exactly this shell command: echo $$ > started; sleep 60; touch should_not_exist . Run it in foreground and wait. Do not use background execution.",
            workdir=str(WORK),
        )
        deadline = time.monotonic() + 120
        while not (WORK / "started").exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        assert (WORK / "started").exists(), "real tool never started"
        shell_pid = int((WORK / "started").read_text().strip())
        interrupted = service.interrupt_agent(a.agent_id, timeout_s=30)
        stat = Path(f"/proc/{shell_pid}/stat")
        alive = stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
        REPORT["checks"]["interrupt"] = {
            "accepted": interrupted,
            "late_write": (WORK / "should_not_exist").exists(),
            "tool_pid": shell_pid,
            "tool_alive": alive,
        }
        assert interrupted and not (WORK / "should_not_exist").exists()
        assert not alive, "real tool shell survived cancellation"
        save()
    elif mode == "write_tools":
        a = home(
            "write",
            tools=(
                "read_file",
                "list_dir",
                "grep",
                "search_replace",
                "run_terminal_cmd",
            ),
        )
        (WORK / "edit.txt").write_text("OLD_3927\n")
        result = turn(
            a,
            "write_tools",
            "Use list_dir to list this directory, grep to find OLD_3927, read_file to read edit.txt, search_replace to replace OLD_3927 with NEW_5841, and run_terminal_cmd to run: cat edit.txt . Perform all five tool calls, then report DONE.",
        )
        assert (WORK / "edit.txt").read_text().strip() == "NEW_5841"
        REPORT["tool_names"] = [t.tool_name for t in result.tool_calls]
        assert len(result.tool_calls) >= 5
    elif mode == "scope_snapshot":
        a = home("scope")
        turn(
            a,
            "initial",
            "Remember the code SCOPE_ORIGINAL_852. Reply SAVED. Do not use tools.",
        )
        snapshots = AgentSnapshotService(
            ROOT / "runtime", store=service.store, agent_service=service
        )
        captured = snapshots.create_scope_snapshot("validation")
        REPORT["snapshot"] = to_jsonable(captured)
        assert captured.status == "created", captured
        turn(
            a,
            "advance",
            "The code is now SCOPE_CHANGED_194. Reply UPDATED. Do not use tools.",
        )
        restored = snapshots.restore_scope_snapshot(
            captured.snapshot_id, leave_paused=False
        )
        REPORT["restore"] = to_jsonable(restored)
        assert restored.status == "created", restored
        result = turn(
            a,
            "scope_restored",
            "What is the code I told you? Reply only it. Do not use tools.",
        )
        assert "SCOPE_ORIGINAL_852" in result.final_text
    elif mode == "instructions":
        a = home(
            "instructions",
            model="grok-4.6",
            instructions=(
                TextFragment("marker", "When asked for HOME_TOKEN, reply HOME_4739."),
            ),
        )
        service.start_agent(
            a.agent_id,
            prompt="Give HOME_TOKEN and RUN_TOKEN values only. Do not use tools.",
            developer_instructions_template_override="When asked for RUN_TOKEN, reply RUN_6518.",
            workdir=str(WORK),
        )
        result = service.wait_agent(a.agent_id, timeout_s=180).provider_result
        REPORT["checks"]["instructions"] = to_jsonable(result)
        assert result.session_locator.backend_identity.requested_model == "grok-4.6"
        assert result.session_locator.backend_identity.reasoning_effort == "low"
        assert "HOME_4739" in result.final_text and "RUN_6518" in result.final_text
    elif mode == "required_missing":
        a = home(
            "missing",
            servers=(
                McpServerSpec(
                    name="missing",
                    transport="http",
                    url="http://127.0.0.1:9/mcp",
                    required=True,
                    startup_timeout_sec=2,
                ),
            ),
        )
        service.start_agent(
            a.agent_id, prompt="Reply UNEXPECTED_MODEL_CALL.", workdir=str(WORK)
        )
        try:
            service.wait_agent(a.agent_id, timeout_s=90)
        except RuntimeError as exc:
            assert "required Grok MCP" in str(exc)
            REPORT["checks"]["required_failure"] = {
                "error": str(exc),
                "session_locator": service.store.get_agent(a.agent_id).session_locator,
            }
            assert service.store.get_agent(a.agent_id).session_locator is None
        else:
            raise AssertionError("unhealthy required MCP was accepted")
    REPORT["passed"] = True
except BaseException as exc:
    REPORT["passed"] = False
    REPORT["error"] = repr(exc)
    raise
finally:
    service.close()
    save()
    print("REPORT", ROOT / "acceptance.json", flush=True)
