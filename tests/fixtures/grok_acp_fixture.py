from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote


_write_lock = threading.Lock()
_sessions: dict[str, str] = {}
_cancelled: set[str] = set()


def _write(value: object) -> None:
    with _write_lock:
        sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
        sys.stdout.flush()


def _response(request: dict[str, object], result: object) -> None:
    _write({"jsonrpc": "2.0", "id": request["id"], "result": result})


def _profile_tools() -> list[str]:
    try:
        index = sys.argv.index("--agent-profile")
        text = Path(sys.argv[index + 1]).read_text(encoding="utf-8")
        frontmatter = text.split("---", 2)[1].strip()
        profile = json.loads(frontmatter)
        return [item["id"].split(":", 1)[1] for item in profile["toolConfig"]["tools"]]
    except Exception:
        return ["read_file", "list_dir", "grep"]


def _session_dir(cwd: str, session_id: str) -> Path:
    return Path(os.environ["GROK_HOME"]) / "sessions" / quote(cwd, safe="") / session_id


def _persist(cwd: str, session_id: str, prompt: str) -> None:
    root = _session_dir(cwd, session_id)
    root.mkdir(parents=True, exist_ok=True)
    history = root / "chat_history.jsonl"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"role": "user", "content": prompt}) + "\n")
        stream.write(json.dumps({"role": "assistant", "content": "ARK_GROK_DONE"}) + "\n")
    (root / "summary.json").write_text(json.dumps({"last": prompt}), encoding="utf-8")
    (root / "usage.json").write_text(json.dumps({"inputTokens": 7, "outputTokens": 3}), encoding="utf-8")


def _run_prompt(request: dict[str, object], session_id: str, prompt: str, cwd: str) -> None:
    if "permission" in prompt:
        _write(
            {
                "jsonrpc": "2.0",
                "id": "permission-1",
                "method": "session/request_permission",
                "params": {
                    "sessionId": session_id,
                    "toolCall": {"kind": "execute", "rawInput": {"command": "true"}},
                    "options": [
                        {"kind": "allow_once", "optionId": "yes"},
                        {"kind": "reject_once", "optionId": "no"},
                    ],
                },
            }
        )
        return
    if "cancel" in prompt:
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        deadline = time.monotonic() + 60
        while session_id not in _cancelled and time.monotonic() < deadline:
            time.sleep(0.02)
        _response(request, {"stopReason": "cancelled", "_meta": {"promptId": "cancel-turn"}})
        return
    stop = "refusal" if "refuse" in prompt else "end_turn"
    _persist(cwd, session_id, prompt)
    _write(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "ARK_GROK_DONE"},
                },
            },
        }
    )
    _response(
        request,
        {
            "stopReason": stop,
            "_meta": {
                "requestId": "request-1",
                "promptId": f"turn-{uuid.uuid4().hex}",
                "modelId": "grok-fixture",
                "usage": {
                    "inputTokens": 7,
                    "outputTokens": 3,
                    "totalTokens": 10,
                    "cachedReadTokens": 2,
                    "reasoningTokens": 1,
                    "modelCalls": 3,
                    "apiDurationMs": 12,
                    "costUsdTicks": 999,
                },
            },
        },
    )


def main() -> int:
    if "--version" in sys.argv:
        print("grok 1.0.30 (fixture)")
        return 0
    if sys.argv[1:4] == ["mcp", "doctor", "--json"]:
        time.sleep(float(os.environ.get("GROK_FIXTURE_DOCTOR_DELAY", "0")))
        print(json.dumps({"result": {"servers": []}}))
        return 0
    cwd_by_session: dict[str, str] = {}
    for line in sys.stdin:
        request = json.loads(line)
        method = request.get("method")
        params = request.get("params") or {}
        if "id" not in request:
            if method == "session/cancel":
                _cancelled.add(str(params.get("sessionId")))
            continue
        if method is None:
            continue
        if method == "initialize":
            _response(
                request,
                {
                    "protocolVersion": 1,
                    "agentCapabilities": {"loadSession": True},
                    "_meta": {"agentVersion": "1.0.30"},
                },
            )
        elif method == "session/new":
            session_id = str(uuid.uuid4())
            cwd_by_session[session_id] = str(params["cwd"])
            _response(request, {"sessionId": session_id})
        elif method == "session/load":
            session_id = str(params["sessionId"])
            cwd_by_session[session_id] = str(params["cwd"])
            _response(request, {})
        elif method == "_x.ai/commands/list":
            _response(request, {"tools": _profile_tools()})
        elif method == "session/prompt":
            session_id = str(params["sessionId"])
            prompt = str(params["prompt"][0]["text"])
            threading.Thread(
                target=_run_prompt,
                args=(request, session_id, prompt, cwd_by_session[session_id]),
                daemon=True,
            ).start()
        else:
            _write({"jsonrpc": "2.0", "id": request["id"], "error": {"message": "unknown"}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
