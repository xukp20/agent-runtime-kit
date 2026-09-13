"""Explicit real Grok probes for the LC Home contract; no production services."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

from agent_runtime_kit.agent.homes import McpServerSpec
from agent_runtime_kit.agent.provider_contracts import ProviderHomeSpec, ProviderRegistry
from agent_runtime_kit.agent.providers import build_grok_provider_bundle, GrokHomeOptions
from agent_runtime_kit.agent.service import AgentService, AgentType, AgentTypeRegistry
from agent_runtime_kit.agent.skills import SkillSpec
from agent_runtime_kit.agent.models import to_jsonable

if os.environ.get("ARK_RUN_REAL_GROK") != "1":
    raise SystemExit("Set ARK_RUN_REAL_GROK=1")
root = Path(tempfile.mkdtemp(prefix="ark-grok-lc-compat-"))
work = root / "work"
work.mkdir()
report = {"root": str(root), "checks": {}}
print("ROOT", root, flush=True)

class Worker(AgentType):
    agent_type = "probe"
    provider_type = "grok"

registry = AgentTypeRegistry()
registry.register(Worker())
bundle = build_grok_provider_bundle(runtime_root=root / "runtime")
service = AgentService(root / "runtime", agent_types=registry, provider_registry=ProviderRegistry((bundle,)))

def save():
    (root / "acceptance.json").write_text(json.dumps(report, indent=2, default=str))

def wait(agent, label):
    result = service.wait_agent(agent.agent_id, timeout_s=240).provider_result
    report["checks"][label] = to_jsonable(result)
    save()
    print(label, result.status, repr(result.final_text), flush=True)
    assert result.status.value == "completed", result.error
    return result

mode = sys.argv[1]
server = None
try:
    if mode == "web":
        service.home_service.create_home(ProviderHomeSpec(provider_type="grok", home_id=mode,
            provider_options=GrokHomeOptions(reasoning_effort="low", tools=("read_file", "web_search", "web_fetch"))))
        agent = service.create_agent("probe", "probe", home_id=mode)
        service.start_agent(agent.agent_id, workdir=str(work), prompt="Use web_search to find the official Lean theorem prover website, then use web_fetch on https://lean-lang.org/ . Report its title and links. Actually call both tools; do not substitute prior knowledge.")
        wait(agent, "web")
    else:
        fixture = Path(__file__).with_name("mcp_fixture.py")
        if mode == "http":
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            server = subprocess.Popen([sys.executable, str(fixture)], env={**os.environ, "ARK_MCP_PORT": str(port), "ARK_MCP_TRANSPORT": "streamable-http"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(100):
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.1):
                        break
                except OSError:
                    time.sleep(.1)
            mcp = McpServerSpec(name="lc_probe", url=f"http://127.0.0.1:{port}/mcp", required=True,
                env_http_headers={"x-step": "ARK_STEP_ID", "x-round": "ARK_ROUND"}, result_profile="content_only")
        elif mode == "stdio":
            mcp = McpServerSpec(name="lc_probe", command=sys.executable, args=[str(fixture)], transport="stdio", required=True,
                env_vars=["ARK_STEP_ID", "ARK_ROUND"], result_profile="content_only")
        else:
            raise ValueError(mode)
        service.home_service.create_home(ProviderHomeSpec(provider_type="grok", home_id=mode, mcp_servers=(mcp,),
            skills=(SkillSpec(name="identity-check", description="How to verify identity", body="Read resource.txt in this skill directory and report its exact content.", files={"resource.txt": "SKILL_LC_7291"}),),
            provider_options=GrokHomeOptions(reasoning_effort="low")))
        agents = [service.create_agent("probe", "probe", home_id=mode) for _ in range(2)]
        prompt = "Read the identity-check skill and its resource.txt. Use search_tool and use_tool to call lc_probe probe_identity with empty arguments. Return the exact skill code and all identity fields in your final answer. Do not guess identity."
        for i, agent in enumerate(agents):
            service.start_agent(agent.agent_id, workdir=str(work), env={"ARK_STEP_ID": f"STEP_{i}_3917", "ARK_ROUND": f"ROUND_{i}_7293"}, prompt=prompt)
        for i, agent in enumerate(agents):
            result = wait(agent, f"parallel-{i}")
            assert f"STEP_{i}_3917" in result.final_text and f"ROUND_{i}_7293" in result.final_text
            assert "SKILL_LC_7291" in result.final_text and "content_only" in result.final_text
        service.start_agent(agents[0].agent_id, workdir=str(work), env={"ARK_STEP_ID": "STEP_RESUME_8327"}, prompt=prompt + " This is a new step; call the tool again, ignore previous identity.")
        result = wait(agents[0], "resume-clear-round")
        assert "STEP_RESUME_8327" in result.final_text and "ROUND_0_7293" not in result.final_text
    report["passed"] = True
finally:
    save()
    if server is not None:
        server.terminate()
        server.wait(timeout=10)
