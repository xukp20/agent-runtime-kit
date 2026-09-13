"""Native idle-session operations shared by Grok fork and compaction."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

from .grok_acp import GrokAcpProcess, GrokAcpError
from .grok_home import GROK_CLI_VERSION, validate_grok_workdir


def session_directory(context, session):  # noqa: ANN001, ANN201
    native = session.native_locator
    if (
        context.provider_type != "grok"
        or session.provider_type != "grok"
        or context.home_id != session.home_id
    ):
        raise ValueError("Grok operation requires the same provider and Home")
    home = context.home_root
    uuid.UUID(session.session_id)
    if native.get("grok_home") != str(home / ".grok"):
        raise ValueError("Grok session belongs to a different Home")
    cwd = Path(native["workdir"]).resolve(strict=True)
    if context.workdir is not None and Path(context.workdir).resolve() != cwd:
        raise ValueError("Grok operation cannot change the session cwd")
    path = home / ".grok" / "sessions" / quote(str(cwd), safe="") / session.session_id
    if str(path.relative_to(home.parents[2])) != native.get("session_relpath"):
        raise ValueError("Grok session path does not match its identity")
    if not path.is_dir() or path.resolve() != path:
        raise ValueError("Grok native session directory is missing or linked")
    validate_grok_workdir(
        cwd, managed_skills=bool(context.runtime_payload.get("skills"))
    )
    return cwd, path


def updates(path: Path) -> list[dict]:
    source = path / "updates.jsonl"
    if not source.exists():
        return []
    return [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]


def latest_prompt_id(path: Path) -> str | None:
    result = None
    for record in updates(path):
        value = record.get("params", {}).get("_meta", {}).get("promptId")
        if value:
            result = str(value)
    return result


def completed_compactions(path: Path, session_id: str) -> list[str]:
    result = []
    checkpoint = None
    for record in updates(path):
        params = record.get("params", {})
        if params.get("sessionId") != session_id:
            continue
        update = params.get("update", {})
        if update.get("sessionUpdate") == "compaction_checkpoint":
            checkpoint = update.get("checkpoint_id")
        elif update.get("sessionUpdate") == "auto_compact_completed" and checkpoint:
            result.append(str(checkpoint))
            checkpoint = None
    return result


@contextmanager
def native_session(runtime, context, session):  # noqa: ANN001, ANN201
    from .grok_runtime import build_grok_command

    cwd, path = session_directory(context, session)
    with runtime.maintenance(session.session_id):
        process = GrokAcpProcess(
            build_grok_command(context, model=session.backend_identity),
            cwd=cwd,
            env=context.process_environment,
        )
        try:
            init = process.request(
                "initialize", {"protocolVersion": 1, "clientCapabilities": {}}
            )
            if (
                init.get("protocolVersion") != 1
                or init.get("_meta", {}).get("agentVersion") != GROK_CLI_VERSION
            ):
                raise GrokAcpError(
                    "Grok maintenance requires pinned ACP v1 / CLI 1.0.30"
                )
            process.request(
                "session/load",
                {"sessionId": session.session_id, "cwd": str(cwd), "mcpServers": []},
            )
            yield process, path
        finally:
            try:
                process.close_process_group()
            except BaseException:
                runtime.mark_unstable(session.session_id)
                raise
